/*
Matrix Multiplication for SYCL (ported from llmc/matmul.cuh).
Row-major GEMMs with fp32 accumulation and bf16 round-to-nearest outputs,
plus bias and GELU handling as separate kernels (no cublasLt epilogues).
*/
#ifndef SYCL_MATMUL_HPP
#define SYCL_MATMUL_HPP

#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"
// GELU kernels
#include "gelu.hpp"

// ----------------------------------------------------------------------------
// Row-major GEMM kernels, fp32 accumulate.
//
// Convention (row-major, matching llm.c logical shapes):
//   forward:  O[M,N] = A[M,K] @ B[N,K]^T   (+ bias[N] optionally)
//   dinp:     DI[M,K] = W[N,K]^T @ DO[M,N]^T  i.e. DI[m,k] = sum_n W[n,k]*DO[m,n]
//   dweight:  DW[N,K] = DO[M,N]^T @ A[M,K]    i.e. DW[n,k] = sum_m DO[m,n]*A[m,k]
// All reads convert bf16->fp32 elementwise; accumulation order is k-inner,
// deterministic for fixed shapes.

// O[M,N] = A[M,K] @ B[N,K]^T
inline void gemm_forward_kernel(const floatX* A, const floatX* B, floatX* O,
                                int M, int N, int K, sycl::queue& q) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    constexpr int TM = 4, TN = 16;
    int GM = CEIL_DIV(M, TM), GN = CEIL_DIV(N, TN);
    q.parallel_for(sycl::range<2>(GM, GN), [=](sycl::id<2> id) {
        int r0 = int(id[0]) * TM, c0 = int(id[1]) * TN;
        sycl::vec<float, 16> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
        for (int k = 0; k < K; k++) {
            float a0 = (r0 + 0 < M) ? (float)A[size_t(r0 + 0) * K + k] : 0.f;
            float a1 = (r0 + 1 < M) ? (float)A[size_t(r0 + 1) * K + k] : 0.f;
            float a2 = (r0 + 2 < M) ? (float)A[size_t(r0 + 2) * K + k] : 0.f;
            float a3 = (r0 + 3 < M) ? (float)A[size_t(r0 + 3) * K + k] : 0.f;
            sycl::vec<unsigned short, 16> bu;
            for (int j = 0; j < 16; j++) {
                unsigned short u = 0;
                if (c0 + j < N) {
                    const unsigned short* bp = reinterpret_cast<const unsigned short*>(&B[size_t(c0 + j) * K + k]);
                    u = *bp;
                }
                bu[j] = u;
            }
            sycl::vec<unsigned int, 16> bi = bu.convert<unsigned int>() << 16;
            sycl::vec<float, 16> bv = bi.as<sycl::vec<float, 16>>();
            acc0 = sycl::fma(sycl::vec<float, 16>(a0), bv, acc0);
            acc1 = sycl::fma(sycl::vec<float, 16>(a1), bv, acc1);
            acc2 = sycl::fma(sycl::vec<float, 16>(a2), bv, acc2);
            acc3 = sycl::fma(sycl::vec<float, 16>(a3), bv, acc3);
        }
        for (int j = 0; j < 16; j++) {
            if (c0 + j >= N) break;
            if (r0 + 0 < M) O[size_t(r0 + 0) * N + c0 + j] = (floatX)acc0[j];
            if (r0 + 1 < M) O[size_t(r0 + 1) * N + c0 + j] = (floatX)acc1[j];
            if (r0 + 2 < M) O[size_t(r0 + 2) * N + c0 + j] = (floatX)acc2[j];
            if (r0 + 3 < M) O[size_t(r0 + 3) * N + c0 + j] = (floatX)acc3[j];
        }
    }).wait();
}

// Specialization for float (fp32 path, kept for completeness)
inline void gemm_forward_kernel_f32(const float* A, const float* B, float* O,
                                    int M, int N, int K, sycl::queue& q) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    constexpr int TM = 4, TN = 16;
    int GM = CEIL_DIV(M, TM), GN = CEIL_DIV(N, TN);
    q.parallel_for(sycl::range<2>(GM, GN), [=](sycl::id<2> id) {
        int r0 = int(id[0]) * TM, c0 = int(id[1]) * TN;
        sycl::vec<float, 16> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
        for (int k = 0; k < K; k++) {
            float a0 = (r0 + 0 < M) ? A[size_t(r0 + 0) * K + k] : 0.f;
            float a1 = (r0 + 1 < M) ? A[size_t(r0 + 1) * K + k] : 0.f;
            float a2 = (r0 + 2 < M) ? A[size_t(r0 + 2) * K + k] : 0.f;
            float a3 = (r0 + 3 < M) ? A[size_t(r0 + 3) * K + k] : 0.f;
            sycl::vec<float, 16> bv(0.f);
            for (int j = 0; j < 16; j++) {
                if (c0 + j < N) bv[j] = B[size_t(c0 + j) * K + k];
            }
            acc0 = sycl::fma(sycl::vec<float, 16>(a0), bv, acc0);
            acc1 = sycl::fma(sycl::vec<float, 16>(a1), bv, acc1);
            acc2 = sycl::fma(sycl::vec<float, 16>(a2), bv, acc2);
            acc3 = sycl::fma(sycl::vec<float, 16>(a3), bv, acc3);
        }
        for (int j = 0; j < 16; j++) {
            if (c0 + j >= N) break;
            if (r0 + 0 < M) O[size_t(r0 + 0) * N + c0 + j] = acc0[j];
            if (r0 + 1 < M) O[size_t(r0 + 1) * N + c0 + j] = acc1[j];
            if (r0 + 2 < M) O[size_t(r0 + 2) * N + c0 + j] = acc2[j];
            if (r0 + 3 < M) O[size_t(r0 + 3) * N + c0 + j] = acc3[j];
        }
    }).wait();
}

// DI[M,K] = DO[M,N] @ W[N,K]  (backward to input, overwrites)
inline void gemm_dinp_kernel(const floatX* DO_, const floatX* W, floatX* DI,
                             int M, int N, int K, sycl::queue& q) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    constexpr int TM = 4, TN = 16;
    int GM = CEIL_DIV(M, TM), GN = CEIL_DIV(K, TN);
    q.parallel_for(sycl::range<2>(GM, GN), [=](sycl::id<2> id) {
        int r0 = int(id[0]) * TM, c0 = int(id[1]) * TN;
        sycl::vec<float, 16> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
        for (int n = 0; n < N; n++) {
            float d0 = (r0 + 0 < M) ? (float)DO_[size_t(r0 + 0) * N + n] : 0.f;
            float d1 = (r0 + 1 < M) ? (float)DO_[size_t(r0 + 1) * N + n] : 0.f;
            float d2 = (r0 + 2 < M) ? (float)DO_[size_t(r0 + 2) * N + n] : 0.f;
            float d3 = (r0 + 3 < M) ? (float)DO_[size_t(r0 + 3) * N + n] : 0.f;
            sycl::vec<float, 16> wv(0.f);
            for (int j = 0; j < 16; j++) {
                if (c0 + j < K) wv[j] = (float)W[size_t(n) * K + c0 + j];
            }
            acc0 = sycl::fma(sycl::vec<float, 16>(d0), wv, acc0);
            acc1 = sycl::fma(sycl::vec<float, 16>(d1), wv, acc1);
            acc2 = sycl::fma(sycl::vec<float, 16>(d2), wv, acc2);
            acc3 = sycl::fma(sycl::vec<float, 16>(d3), wv, acc3);
        }
        for (int j = 0; j < 16; j++) {
            if (c0 + j >= K) break;
            if (r0 + 0 < M) DI[size_t(r0 + 0) * K + c0 + j] = (floatX)acc0[j];
            if (r0 + 1 < M) DI[size_t(r0 + 1) * K + c0 + j] = (floatX)acc1[j];
            if (r0 + 2 < M) DI[size_t(r0 + 2) * K + c0 + j] = (floatX)acc2[j];
            if (r0 + 3 < M) DI[size_t(r0 + 3) * K + c0 + j] = (floatX)acc3[j];
        }
    }).wait();
}

// DW[N,K] += DO[M,N]^T @ A[M,K]  (backward to weight, accumulates)
inline void gemm_dweight_kernel(const floatX* DO_, const floatX* A, floatX* DW,
                                int M, int N, int K, sycl::queue& q) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    constexpr int TM = 4, TN = 16;
    int GM = CEIL_DIV(N, TM), GN = CEIL_DIV(K, TN);
    q.parallel_for(sycl::range<2>(GM, GN), [=](sycl::id<2> id) {
        int r0 = int(id[0]) * TM, c0 = int(id[1]) * TN;
        sycl::vec<float, 16> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
        for (int m = 0; m < M; m++) {
            float d0 = (r0 + 0 < N) ? (float)DO_[size_t(m) * N + r0 + 0] : 0.f;
            float d1 = (r0 + 1 < N) ? (float)DO_[size_t(m) * N + r0 + 1] : 0.f;
            float d2 = (r0 + 2 < N) ? (float)DO_[size_t(m) * N + r0 + 2] : 0.f;
            float d3 = (r0 + 3 < N) ? (float)DO_[size_t(m) * N + r0 + 3] : 0.f;
            sycl::vec<float, 16> av(0.f);
            for (int j = 0; j < 16; j++) {
                if (c0 + j < K) av[j] = (float)A[size_t(m) * K + c0 + j];
            }
            acc0 = sycl::fma(sycl::vec<float, 16>(d0), av, acc0);
            acc1 = sycl::fma(sycl::vec<float, 16>(d1), av, acc1);
            acc2 = sycl::fma(sycl::vec<float, 16>(d2), av, acc2);
            acc3 = sycl::fma(sycl::vec<float, 16>(d3), av, acc3);
        }
        for (int j = 0; j < 16; j++) {
            if (c0 + j >= K) break;
            if (r0 + 0 < N) DW[size_t(r0 + 0) * K + c0 + j] = (floatX)((float)DW[size_t(r0 + 0) * K + c0 + j] + acc0[j]);
            if (r0 + 1 < N) DW[size_t(r0 + 1) * K + c0 + j] = (floatX)((float)DW[size_t(r0 + 1) * K + c0 + j] + acc1[j]);
            if (r0 + 2 < N) DW[size_t(r0 + 2) * K + c0 + j] = (floatX)((float)DW[size_t(r0 + 2) * K + c0 + j] + acc2[j]);
            if (r0 + 3 < N) DW[size_t(r0 + 3) * K + c0 + j] = (floatX)((float)DW[size_t(r0 + 3) * K + c0 + j] + acc3[j]);
        }
    }).wait();
}

// Batched variant used by attention: for each batch b:
//   O_b[M,N] = A_b[M,K] @ B_b[K,N]  (B_b given K-major: B[k*N+n])
// A stride strideA (elements), B stride strideB, O stride strideO.
inline void gemm_batched_kernel(const floatX* A, const floatX* B, floatX* O,
                                int M, int N, int K, int batch,
                                size_t strideA, size_t strideB, size_t strideO,
                                bool transB_is_KxN, sycl::queue& q) {
    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) return;
    // attention shapes: (M=HS or T) x (N=T) with K=T or HS. Use simple per-output kernels
    // with sequential k-accumulation; work items are independent so order is deterministic.
    q.parallel_for(sycl::range<2>(size_t(batch) * M, N), [=](sycl::id<2> id) {
        size_t bm = id[0]; int n = int(id[1]);
        int b = int(bm / (size_t)M), m = int(bm % (size_t)M);
        const floatX* Ab = A + size_t(b) * strideA;
        const floatX* Bb = B + size_t(b) * strideB;
        floatX* Ob = O + size_t(b) * strideO;
        float acc = 0.f;
        for (int k = 0; k < K; k++) {
            float a = (float)Ab[size_t(m) * K + k];
            float bv = transB_is_KxN ? (float)Bb[size_t(k) * N + n]
                                     : (float)Bb[size_t(n) * K + k];
            acc += a * bv;
        }
        Ob[size_t(m) * N + n] = (floatX)acc;
    }).wait();
}

// add bias (row vector, length OC) to every row of out [rows x OC]
inline void matmul_add_bias(floatX* out, const floatX* bias, int rows, int OC, sycl::queue& q) {
    if (rows <= 0 || OC <= 0 || bias == nullptr) return;
    q.parallel_for(sycl::range<2>(rows, OC), [=](sycl::id<2> id) {
        size_t r = id[0], c = id[1];
        out[r * (size_t)OC + c] = (floatX)((float)out[r * (size_t)OC + c] + (float)bias[c]);
    }).wait();
}

// dbias[OC] += sum over rows of dout[rows x OC]
inline void matmul_backward_bias(floatX* dbias, const floatX* dout, int rows, int OC, sycl::queue& q) {
    if (rows <= 0 || OC <= 0) return;
    q.parallel_for(sycl::range<1>(OC), [=](sycl::id<1> id) {
        int c = int(id[0]);
        float acc = (float)dbias[c];
        for (int r = 0; r < rows; r++) { acc += (float)dout[size_t(r) * OC + c]; }
        dbias[c] = (floatX)acc;
    }).wait();
}

// ----------------------------------------------------------------------------
// launcher API (mirrors the CUDA file's launchers, minus streams/epilogues)

// forward: out[B*T, OC] = inp[B*T, C] @ weight[OC, C]^T (+ bias)
inline void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, sycl::queue& q,
                     floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    NVTX_RANGE_FN();
    int M = B * T;
    gemm_forward_kernel(inp, weight, out, M, OC, C, q);
    if (bias != nullptr) { matmul_add_bias(out, bias, M, OC, q); }
    if (pre_gelu != nullptr) {
        // save pre-gelu activations then apply gelu in place on out
        q.memcpy(pre_gelu, out, size_t(M) * OC * sizeof(floatX)).wait();
        if (gelu_fusion >= 1) { gelu_forward(out, out, M * OC, q); }
    }
}

inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, sycl::queue& q,
                     floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    NVTX_RANGE_FN();
    (void)dbias_buffer;
    int M = B * T;
    // backward to bias (+=)
    if (dbias != nullptr) {
        matmul_backward_bias(dbias, dout, M, OC, q);
    }
    // backward to input (overwrite)
    // optional fused dgelu: dinp = (W^T @ dout) .* dgelu(pre_gelu)
    gemm_dinp_kernel(dout, weight, dinp, M, OC, C, q);
    if (pre_gelu != nullptr && gelu_fusion >= 2) {
        gelu_backward_inplace(dinp, pre_gelu, M * C, q);
    } else if (pre_gelu != nullptr) {
        gelu_backward_inplace(dinp, pre_gelu, M * C, q);
    }
    // backward to weight (+=)
    gemm_dweight_kernel(dout, inp, dweight, M, OC, C, q);
}

#endif
