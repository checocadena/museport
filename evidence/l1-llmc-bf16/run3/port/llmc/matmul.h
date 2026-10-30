/*
Matrix Multiplication with oneMath BLAS (ported from matmul.cuh).
Row-major convention, matching the llm.c layout:
  forward:  out(BT x OC) = inp(BT x C) * weight(OC x C)^T (+ bias)
  din:      dinp(BT x C) = dout(BT x OC) * weight(OC x C)
  dweight:  dweight(C x OC) += inp(BT x C)^T * dout(BT x OC)   [stored row-major C x OC]
Bias and GELU epilogues are separate SYCL kernels.
*/
#ifndef MATMUL_SYCL_H
#define MATMUL_SYCL_H

#include <assert.h>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <oneapi/math/blas.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
#include "blas_common.h"
#include "gelu.h"

// ----------------------------------------------------------------------------
// oneMath GEMM wrappers (row-major tensors via column_major transposed view)
//
// Forward: Y(BT x OC) = X(BT x C) * W(OC x C)^T
//   Viewed column-major: Y^T(OC x BT) = W^T^T... we use:
//   gemm(transA=trans, transB=nontrans, m=OC, n=BT, k=C, A=W, lda=C, B=X, ldb=C, C=Y, ldc=OC)
//   All pointers are to row-major data; the (trans, nontrans) pair plus swapped
//   m/n implements exactly the row-major product.

inline void sycl_gemm_forward(float* Y, const floatX* X, const floatX* W,
                              int64_t BT, int64_t OC, int64_t C, float beta, sycl::queue& q) {
    oneapi::math::blas::column_major::gemm(q,
        oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
        OC, BT, C, 1.0f, W, C, X, C, beta, Y, OC);
    q.wait();
}

// Backward to input: DIN(BT x C) = DOUT(BT x OC) * W(OC x C)
//   Column-major view: DIN^T(C x BT) = W^T... precisely C_col = op(A)*op(B) with
//   op(A) = W^T-as-stored (C x OC, i.e. W memory as col-major, nontrans),
//   op(B) = DOUT^T (OC x BT, i.e. DOUT memory as col-major, nontrans).
//   gemm(nontrans, nontrans, m=C, n=BT, k=OC, A=W, lda=C, B=DOUT, ldb=OC, C=DIN, ldc=C)
inline void sycl_gemm_din(float* DIN, const floatX* DOUT, const floatX* W,
                          int64_t BT, int64_t C, int64_t OC, sycl::queue& q) {
    oneapi::math::blas::column_major::gemm(q,
        oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
        C, BT, OC, 1.0f, W, C, DOUT, OC, 0.0f, DIN, C);
    q.wait();
}

// Backward to weight: DW(C x OC) += X(BT x C)^T * DOUT(BT x OC)
//   DW is stored row-major (OC x C)-compatible: its memory viewed column-major
//   is (C x OC), exactly like cuBLASLt's D(m=C, n=OC) layout.
//   gemm(nontrans, trans, m=C, n=OC, k=BT, A=X, lda=C, B=DOUT, ldb=OC, C=DW, ldc=C)
inline void sycl_gemm_dweight(float* DW, const floatX* X, const floatX* DOUT,
                              int64_t BT, int64_t C, int64_t OC, float beta, sycl::queue& q) {
    oneapi::math::blas::column_major::gemm(q,
        oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
        C, OC, BT, 1.0f, X, C, DOUT, OC, beta, DW, C);
    q.wait();
}

// Batched attention GEMMs, all writing fp32 output.
// Layouts (row-major, batch BNH):
//   preatt(b, i, j) = sum_p K(b, j, p) * Q(b, i, p)   [T x T per batch]
//     column-major per batch: P^T(T x T) = K^T(T x HS)^T... use:
//     gemm(transA=trans, transB=nontrans, m=T, n=T, k=HS, A=K, lda=HS, B=Q, ldb=HS, C=P, ldc=T)
//     with strided batches: strideA = strideB = T*HS, strideC = T*T.
//   vaccum(b, i, p) = sum_j ATT(b, i, j) * V(b, j, p)  [T x HS]
//     gemm(nontrans, nontrans, m=HS, n=T, k=T, A=V, lda=HS, B=ATT, ldb=T, C=Y, ldc=HS)
//     strides: A: T*HS, B: T*T, C: T*HS.
//   datt(b, i, j) = sum_p V(b, j, p) * DOUT(b, i, p)
//     gemm(trans, nontrans, m=T, n=T, k=HS, A=V, lda=HS, B=DOUT, ldb=HS, C=DATT, ldc=T)
//   dv(b, j, p) = sum_i ATT(b, i, j) * DOUT(b, i, p)
//     gemm(trans, nontrans, m=HS, n=T, k=T, A=DOUT... careful:
//     dv^T per batch (HS x T) = DOUT^T... let's derive:
//     dv(j,p) = sum_i ATT(i,j) * DOUT(i,p) = sum_i ATT^T(j,i) * DOUT(i,p)
//     column-major: dv is (HS x T)?? dv row-major (T x HS) == col-major (HS x T).
//     dv_col = DOUT_col^T... simplest: dv_col(HS x T) = DOUT_col... use gemm with
//     transA=trans on DOUT? DOUT col-view is (p? ...). Instead compute via:
//     gemm(transA=nontrans, transB=trans, m=HS, n=T, k=T): A = V?? no.
//     We use: dv_col = DOUT_row^T... Let me just use explicit loops-free mapping:
//     dv(j,p) = sum_i ATT(i,j)*DOUT(i,p). With A=ATT^T col-view (T x T stored col-major
//     as ATT(j*? )). Easiest verified mapping: treat as
//       gemm(trans, trans, m=HS, n=T, k=T, A=DOUT(TxHS row) lda=HS, B=ATT(TxT row) ldb=T, C=DV(TxHS row) ldc=HS)
//     Check: C_col(HS x T) = A_col^T? For Y = A*B row-major, col view Y^T = B^T * A^T.
//     dv_row = DOUT?? no: dv = ATT^T * DOUT row-major (T x T)^T (T x T)... ATT^T (T x T) * DOUT (T x HS) = DV (T x HS). Yes!
//     So dv_row = ATT_row^T * DOUT_row, col view: dv_col = DOUT_col^T?? dv^T = DOUT^T * ATT.
//       gemm(transA=trans, transB=nontrans, m=HS, n=T, k=T, A=DOUT, lda=HS, B=ATT, ldb=T, C=DV, ldc=HS).
//   dq(i,p) = sum_j DPRE(j? ) ... dq(b,i,p) = sum_j DPRE(b,i,j) * K(b,j,p)
//     dq_row = DPRE_row * K_row: gemm(nontrans, nontrans, m=HS, n=T, k=T, A=K lda=HS, B=DPRE ldb=T, C=DQ ldc=HS).
//   dk(j,p) = sum_i DPRE(b,i,j) * Q(b,i,p) = DPRE^T * Q:
//     dk_col = Q_col^T?? dk_row = DPRE_row^T * Q_row -> col: dk_col = Q_col^T... use
//       gemm(trans, nontrans, m=HS, n=T, k=T, A=Q lda=HS, B=DPRE ldb=T, C=DK ldc=HS).
// Batched attention GEMMs, all writing fp32 output.
// oneMath has no bfloat16 gemm_batch USM overload on this toolchain, so the
// batched calls below loop over per-batch gemm calls (each still hardware
// accelerated). Batch counts here are small (B*NH <= ~192 in the test).
inline void sycl_gemm_batch_preatt(float* P, const floatX* K, const floatX* Q,
                                   int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // preatt(b, i, j) = sum_p K(b, j, p) * Q(b, i, p): P^T = gemm(T, trans=K, N, B=Q)
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
            T, T, HS, 1.0f, K + b * T * HS, HS, Q + b * T * HS, HS, 0.0f, P + b * T * T, T);
    }
    q.wait();
}
inline void sycl_gemm_batch_vaccum(float* Y, const floatX* V, const floatX* ATT,
                                   int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // vaccum(b, i, p) = sum_j ATT(b, i, j) * V(b, j, p)  (row-major Y = ATT * V)
    // Column-major C = A*B reads back row-major as B*A, so (N,N,A=V,B=ATT)
    // yields Y_row = ATT_row * V_row. Verified by derivation + unit test.
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
            HS, T, T, 1.0f, V + b * T * HS, HS, ATT + b * T * T, T, 0.0f, Y + b * T * HS, HS);
    }
    q.wait();
}
inline void sycl_gemm_batch_datt(float* DATT, const floatX* V, const floatX* DOUT,
                                 int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // datt(b, i, j) = sum_p V(b, j, p) * DOUT(b, i, p)
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
            T, T, HS, 1.0f, V + b * T * HS, HS, DOUT + b * T * HS, HS, 0.0f, DATT + b * T * T, T);
    }
    q.wait();
}
inline void sycl_gemm_batch_dv(float* DV, const floatX* DOUT, const floatX* ATT,
                               int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // dv(b, j, p) = sum_i ATT(b, i, j) * DOUT(b, i, p) = ATT^T * DOUT (row-major)
    // col view: DV^T(HS x T) = DOUT^T(HS x T) * ATT(T x T):
    // gemm(nontrans, trans, m=HS, n=T, k=T, A=DOUT, lda=HS, B=ATT, ldb=T, C=DV, ldc=HS)
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
            HS, T, T, 1.0f, DOUT + b * T * HS, HS, ATT + b * T * T, T, 0.0f, DV + b * T * HS, HS);
    }
    q.wait();
}
inline void sycl_gemm_batch_dq(float* DQ, const floatX* K, const floatX* DPRE,
                               int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // dq(b, i, p) = sum_j DPRE(b, i, j) * K(b, j, p)  (row-major DQ = DPRE * K)
    // Column-major C = A*B reads back row-major as B*A, so (N,N,A=K,B=DPRE)
    // yields DQ_row = DPRE_row * K_row.
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
            HS, T, T, 1.0f, K + b * T * HS, HS, DPRE + b * T * T, T, 0.0f, DQ + b * T * HS, HS);
    }
    q.wait();
}
inline void sycl_gemm_batch_dk(float* DK, const floatX* Q, const floatX* DPRE,
                               int64_t T, int64_t HS, int64_t batch, sycl::queue& q) {
    // dk(b, j, p) = sum_i DPRE(b, i, j) * Q(b, i, p)  (row-major DK = DPRE^T * Q)
    // col view: DK^T(HS x T) = Q^T(HS x T) * DPRE(T x T):
    // gemm(nontrans, trans, m=HS, n=T, k=T, A=Q, lda=HS, B=DPRE, ldb=T, C=DK, ldc=HS)
    for (int64_t b = 0; b < batch; b++) {
        oneapi::math::blas::column_major::gemm(q,
            oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
            HS, T, T, 1.0f, Q + b * T * HS, HS, DPRE + b * T * T, T, 0.0f, DK + b * T * HS, HS);
    }
    q.wait();
}

// ----------------------------------------------------------------------------
// Bias epilogue kernels (forward: add bias; backward: sum bias grads)

// out[i*OC+j] = f32[i*OC+j] + bias[j]; then round-to-nearest to `outX`
// (matches cuBLASLt epilogue: plain RNE, no stochastic rounding here;
// llm.c uses stochastic rounding only for wte/wpe grads and the AdamW update)
inline void bias_add_forward(floatX* outX, const float* f32, const floatX* bias,
                             size_t BT, size_t OC, sycl::queue& q, unsigned int seed) {
    (void)seed;
    constexpr size_t BLOCK = 256;
    size_t total = BT * OC;
    size_t nblocks = CEIL_DIV(total, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= total) { return; }
        size_t j = lin % OC;
        float v = f32[lin] + (float)bias[j];
        if constexpr (std::is_same<floatX, float>::value) {
            outX[lin] = v;
        } else {
            outX[lin] = sycl_float_to_bf16_rn(v);
        }
    }).wait();
}

// out[i*OC+j] = f32[i*OC+j] (convert fp32 GEMM output to floatX, no bias; RNE)
inline void convert_forward(floatX* outX, const float* f32, size_t N, sycl::queue& q, unsigned int seed) {
    (void)seed;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= N) { return; }
        if constexpr (std::is_same<floatX, float>::value) {
            outX[lin] = f32[lin];
        } else {
            outX[lin] = sycl_float_to_bf16_rn(f32[lin]);
        }
    }).wait();
}

// dbias[j] += sum_i dout[i*OC+j]  (deterministic: single work-group tree per j via atomics-free staged sum)
// Simple deterministic implementation: each j reduced by a single group sequentially is too slow;
// instead do a two-stage deterministic sum: partial sums per block, then single-group final sum.
inline void bias_backward_sum(floatX* dbias, const floatX* dout, size_t BT, size_t OC, sycl::queue& q) {
    // stage 1: partial sums, PB blocks
    constexpr size_t BLOCK = 256;
    size_t PB = 64;
    if (BT < 64) { PB = 1; }
    float* partial = sycl::malloc_device<float>(PB * OC, q);
    q.memset(partial, 0, PB * OC * sizeof(float)).wait();
    // each block handles a shard of rows; all OC columns
    q.parallel_for(sycl::range<1>(PB * BLOCK), [=](sycl::id<1> id) {
        size_t tid = id[0] % BLOCK;
        size_t pb = id[0] / BLOCK;
        size_t rows_per = (BT + PB - 1) / PB;
        size_t r0 = pb * rows_per;
        size_t r1 = sycl::min(r0 + rows_per, BT);
        for (size_t j = tid; j < OC; j += BLOCK) {
            float s = 0.f;
            for (size_t i = r0; i < r1; i++) {
                s += (float)dout[i * OC + j];
            }
            partial[pb * OC + j] = s;
        }
    }).wait();
    // stage 2: deterministic sum over PB partials + accumulate into dbias
    {
        size_t nblocks = CEIL_DIV(OC, BLOCK);
        q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t j = id[0];
            if (j >= OC) { return; }
            float s = 0.f;
            for (size_t pb = 0; pb < PB; pb++) { s += partial[pb * OC + j]; }
            dbias[j] = (floatX)((float)dbias[j] + s);
        }).wait();
    }
    sycl::free(partial, q);
}

// reduce fp32 partial rows (rows x OC) into floatX dbias (+=)
inline void reduce_add_sum_kernel_sycl(floatX* dst, const float* src, size_t n, size_t m, sycl::queue& q) {
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(n, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= n) { return; }
        float acc = 0.f;
        for (size_t l = 0; l < m; l++) { acc += src[idx + n * l]; }
        dst[idx] = (floatX)((float)dst[idx] + acc);
    }).wait();
}

// ----------------------------------------------------------------------------
// High-level matmul forward/backward (API mirrors the CUDA file)

// scratchF32 must hold BT*OC floats (caller provides, e.g. acts.output as float*)
inline void matmul_forward_sycl(floatX* out, float* scratchF32,
                                const floatX* inp, const floatX* weight, const floatX* bias,
                                int B, int T, int C, int OC, sycl::queue& q, unsigned int seed=0) {
    NVTX_RANGE_FN();
    size_t BT = (size_t)B * T;
    sycl_gemm_forward(scratchF32, inp, weight, BT, OC, C, 0.0f, q);
    if (bias != nullptr) {
        bias_add_forward(out, scratchF32, bias, BT, OC, q, seed);
    } else {
        convert_forward(out, scratchF32, BT * OC, q, seed);
    }
}

// GELU-fused forward: computes pre_gelu = inp*W+bias (fp32->floatX), then gelu into out
inline void matmul_forward_gelu_sycl(floatX* out, floatX* pre_gelu, float* scratchF32,
                                     const floatX* inp, const floatX* weight, const floatX* bias,
                                     int B, int T, int C, int OC, sycl::queue& q, unsigned int seed=0) {
    NVTX_RANGE_FN();
    size_t BT = (size_t)B * T;
    sycl_gemm_forward(scratchF32, inp, weight, BT, OC, C, 0.0f, q);
    if (bias != nullptr) {
        bias_add_forward(pre_gelu, scratchF32, bias, BT, OC, q, seed);
    } else {
        convert_forward(pre_gelu, scratchF32, BT * OC, q, seed);
    }
    gelu_forward(out, pre_gelu, BT * OC, q);
}

// dout: (BT x OC) floatX; inp: (BT x C); weight: (OC x C)
// dinp: (BT x C) floatX (set); dweight: (C x OC) floatX (+=); dbias: (OC) floatX (+=, optional)
inline void matmul_backward_sycl(floatX* dinp, floatX* dweight, floatX* dbias,
                                 const floatX* dout, const floatX* inp, const floatX* weight,
                                 int B, int T, int C, int OC, sycl::queue& q,
                                 const floatX* pre_gelu = nullptr) {
    NVTX_RANGE_FN();
    size_t BT = (size_t)B * T;
    // backward to bias (+=), deterministic
    if (dbias != nullptr) {
        bias_backward_sum(dbias, dout, BT, OC, q);
    }
    // backward to input (=): need fp32 intermediate then convert
    {
        float* dinF = sycl::malloc_device<float>(BT * (size_t)C, q);
        sycl_gemm_din(dinF, dout, weight, BT, C, OC, q);
        if (pre_gelu != nullptr) {
            // apply gelu backward inplace on fp32? No: gelu backward needs floatX pre and din.
            // Convert dinF to floatX first, then apply gelu_backward_inplace.
            convert_forward(dinp, dinF, BT * (size_t)C, q, 0);
            gelu_backward_inplace(dinp, pre_gelu, BT * (size_t)C, q);
        } else {
            convert_forward(dinp, dinF, BT * (size_t)C, q, 0);
        }
        sycl::free(dinF, q);
    }
    // backward to weight (+=): fp32 accumulate then round-add into dweight
    {
        float* dwF = sycl::malloc_device<float>((size_t)C * OC, q);
        q.memset(dwF, 0, (size_t)C * OC * sizeof(float)).wait();
        sycl_gemm_dweight(dwF, inp, dout, BT, C, OC, 0.0f, q);
        // dwF is row-major (C x OC); dweight same layout: dweight += round(dwF)
        size_t N = (size_t)C * OC;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            float v = (float)dweight[i] + dwF[i];
            if constexpr (std::is_same<floatX, float>::value) {
                dweight[i] = v;
            } else {
                dweight[i] = sycl_float_to_bf16_rn(v);
            }
        }).wait();
        sycl::free(dwF, q);
    }
}

#endif
