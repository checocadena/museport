/*
Portable strided-batched GEMM helpers (SYCL 2020 + oneMath only).

Dense GEMMs use oneMath bf16 -> fp32 gemm plus a separate convert epilogue.
Batched attention GEMMs use a loop of oneMath bf16 -> fp32 gemms (one launch
per (batch, head) pair, fp32 accumulation). Callers fuse the fp32 -> bf16
convert into their consumer (softmax / unpermute / permute), so no extra
rounding stage is introduced.

Single code path for every shape and every device: no size, shape or device
queries here. Small shapes and large shapes run exactly the same kernels.
*/
#ifndef SYCL_BATCHED_GEMM_H
#define SYCL_BATCHED_GEMM_H

#include <stdint.h>
#include <oneapi/math/blas.hpp>
// llmc internal imports
#include "sycl_common.h"

// Strided-batched bf16 -> fp32 GEMM with the same row-major convention used
// everywhere in this port: for each batch b,
//   Out(m x n) = op_row(A)(m x k) * op_row(B)(k x n),
// fp32 accumulation into tmp (batch-major, m*n floats per batch), one oneMath
// launch per batch via the transpose identity
// Out^T (n x m) = op(B)^T (n x k) * op(A)^T (k x m):
// C_col(n x m) = U(n x k) * V(k x m) with U sourced from B and V from A.
// A row-major (a x b) aliases a column-major (b x a) with leading dim b,
// hence the transpose flags / leading dims below.
inline void bf16_strided_batched_gemm(float* tmp,
                     const floatX* A, const floatX* B,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideA, size_t strideB, int batch_count,
                     sycl::queue& q) {
    if (batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) { return; }
    using namespace oneapi::math;
    transpose tu = transB ? transpose::trans : transpose::nontrans;
    transpose tv = transA ? transpose::trans : transpose::nontrans;
    int64_t ldu = transB ? (int64_t)k : (int64_t)n;
    int64_t ldv = transA ? (int64_t)m : (int64_t)k;
    int64_t ldc = (int64_t)n;
    for (int b = 0; b < batch_count; b++) {
        const floatX* Ab = A + (size_t)b * strideA;
        const floatX* Bb = B + (size_t)b * strideB;
        float* Cb = tmp + (size_t)b * (size_t)m * n; // col (n x m) == row (m x n)
        blas::column_major::gemm(q, tu, tv, (int64_t)n, (int64_t)m, (int64_t)k,
            1.0f, Bb, ldu, Ab, ldv, 0.0f, Cb, ldc);
    }
}

#endif
