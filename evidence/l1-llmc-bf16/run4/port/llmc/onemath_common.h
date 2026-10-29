/*
oneMath BLAS helpers. Replaces llmc/cublas_common.h.

cuBLAS is column-major; the port keeps the exact same call pattern:
column_major::gemm with bf16 A/B inputs and an fp32 C output, then an
explicit conversion kernel to bf16 where the CUDA code produced bf16.
*/
#ifndef ONEMATH_COMMON_H
#define ONEMATH_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "oneapi/math/blas.hpp"

#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// Convert an fp32 device buffer to floatX in place-ish: dst[i] = (floatX)src[i]
// with plain round-to-nearest (NOT stochastic); used for matmul outputs.
inline void convert_fp32_to_floatX(floatX* dst, const float* src, size_t n) {
    if (n == 0) { return; }
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(n, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_linear_id();
            if (i < n) { dst[i] = (floatX)src[i]; }
        }).wait();
}

// out[i] = acc[i] + (float)bias[i % cols] for an (rows x cols) row-major matrix
inline void add_bias_rowwise_fp32(float* out, const float* acc, const floatX* bias,
                                  size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) { return; }
    size_t n = rows * cols;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(n, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_linear_id();
            if (i < n) { out[i] = acc[i] + (float)bias[i % cols]; }
        }).wait();
}

// ----------------------------------------------------------------------------
// Forward matmul used across the port:
//
//   out[b*OC:(b+1)*OC] rows: out^T is (OC, BT) col-major computed as
//     acc = W^T? ... (see below) then optional bias add, then convert to bf16.
//
// Given the CUDA layout (row-major inp (BT,C), weight (OC,C), out (BT,OC)):
//   acc(col-major OC x BT) = gemm(transA=W, transB=inp, m=OC, n=BT, k=C,
//                                 lda=C, ldb=C, ldc=OC)
// which computes exactly out[bt*OC+oc] = bias[oc] + sum_k inp[bt*C+k]*W[oc*C+k].
// fp32 accumulation in oneMath matches CUBLAS_COMPUTE_32F; beta selects
// overwrite vs accumulate (CUDA: beta = accumulate ? 1 : 0).
inline void sycl_matmul_forward(floatX* out, const floatX* inp, const floatX* weight,
                                const floatX* bias, float* tmp_fp32,
                                size_t OC, size_t BT, size_t C, bool accumulate) {
    using namespace oneapi::math;
    if (OC == 0 || BT == 0 || C == 0) { return; }
    auto ta = transpose::trans;
    auto tb = transpose::nontrans;
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    auto evt = blas::column_major::gemm(sycl_queue, ta, tb,
        (int64_t)OC, (int64_t)BT, (int64_t)C, alpha,
        weight, (int64_t)C, inp, (int64_t)C, beta, tmp_fp32, (int64_t)OC, {});
    evt.wait();
    if (bias != nullptr) {
        add_bias_rowwise_fp32(tmp_fp32, tmp_fp32, bias, BT, OC);
    }
    convert_fp32_to_floatX(out, tmp_fp32, OC * BT);
}

// Backward-to-input: dinp(BT,C) = dout(BT,OC) @ W(OC,C), overwrite.
// Row-major identity: dinp_c(C,BT) = W_c(C,OC) @ dout_c(OC,BT), both nontrans.
inline void sycl_matmul_bwd_input(floatX* dinp, const floatX* dout, const floatX* weight,
                                  float* tmp_fp32, size_t BT, size_t OC, size_t C) {
    using namespace oneapi::math;
    if (BT == 0 || OC == 0 || C == 0) { return; }
    auto evt = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::nontrans,
        (int64_t)C, (int64_t)BT, (int64_t)OC, 1.0f,
        weight, (int64_t)C, dout, (int64_t)OC, 0.0f, tmp_fp32, (int64_t)C, {});
    evt.wait();
    convert_fp32_to_floatX(dinp, tmp_fp32, BT * C);
}

// Backward-to-weight: dw(OC,C) += inp^T @ dout  (accumulate).
// Row-major identity: dw_c(C,OC) += inp_c(C,BT) @ dout_c^T(BT,OC).
inline void sycl_matmul_bwd_weight(floatX* dweight, const floatX* inp, const floatX* dout,
                                   float* tmp_fp32, size_t OC, size_t BT, size_t C) {
    using namespace oneapi::math;
    if (OC == 0 || BT == 0 || C == 0) { return; }
    auto evt = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::trans,
        (int64_t)C, (int64_t)OC, (int64_t)BT, 1.0f,
        inp, (int64_t)C, dout, (int64_t)OC, 0.0f, tmp_fp32, (int64_t)C, {});
    evt.wait();
    // accumulate into dweight (stochastic rounding not used for grads in llm.c backward;
    // plain conversion-add like the CUDA fp32-accumulate path)
    size_t n = OC * C;
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(n, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t i = it.get_global_linear_id();
            if (i < n) { dweight[i] = (floatX)((float)dweight[i] + tmp_fp32[i]); }
        }).wait();
}

#endif // ONEMATH_COMMON_H
