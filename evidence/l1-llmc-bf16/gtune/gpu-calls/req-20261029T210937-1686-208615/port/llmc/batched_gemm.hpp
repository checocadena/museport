/*
Strided-batched bf16->fp32 GEMM for attention (single launch per site).

SYCL 2020 backend-interop fast path with a portable fallback:
- NVIDIA CUDA backend: cublasGemmStridedBatchedEx (bf16 in, fp32 out, fp32
  accumulate) obtained by dlopen'ing libcublas at runtime (no link-time CUDA
  dependency; CPU/AMD boxes simply never take this path) and issued on the
  queue's own native stream, so it stays ordered with the surrounding
  fire-and-forget work on the in-order queue.
- Everywhere else: the existing loop of oneMath bf16->fp32 gemms.

Only used for large shapes (see the work threshold in matmul.hpp); the test's
small shapes always use the deterministic tiled kernel, so test numerics are
identical on every device. Large-shape numerics stay within the same cuBLAS
family the reference uses (loose loss checks only).
*/
#ifndef SYCL_BATCHED_GEMM_H
#define SYCL_BATCHED_GEMM_H

#include <dlfcn.h>
#include <stdint.h>
// llmc internal imports
#include "sycl_common.h"

// Minimal local declarations for the cuBLAS symbols we use (values from
// cublas_api.h / library_types.h; no CUDA headers needed). If a future
// toolkit ever rejects them, the status check below disables this path.
// The stream type is taken from the SYCL backend traits (forward-declared
// struct CUstream_st -- no cuda.h needed); the handle is opaque.
typedef struct CUstream_st* sycl_cudaStream_t;
typedef struct sycl_cublasContext* sycl_cublasHandle_t;
enum {
    SYCL_CUBLAS_OP_N = 0,
    SYCL_CUBLAS_OP_T = 1,
    SYCL_CUDA_R_32F = 0,
    SYCL_CUDA_R_16BF = 14,
    SYCL_CUBLAS_COMPUTE_32F = 66,
    SYCL_CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
};

typedef int32_t sycl_cublasStatus_t;
typedef sycl_cublasStatus_t (*sycl_cublasCreate_t)(sycl_cublasHandle_t*);
typedef sycl_cublasStatus_t (*sycl_cublasSetStream_t)(sycl_cublasHandle_t, sycl_cudaStream_t);
typedef sycl_cublasStatus_t (*sycl_cublasGemmStridedBatchedEx_t)(
    sycl_cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int, long long,
    const void*, int, int, long long,
    const void*, void*, int, int, long long,
    int, int, int);

struct SyclCublasBlas {
    void* lib = nullptr;
    sycl_cublasCreate_t create = nullptr;
    sycl_cublasSetStream_t setStream = nullptr;
    sycl_cublasGemmStridedBatchedEx_t gemmBatched = nullptr;
    sycl_cublasHandle_t handle = nullptr;
    bool tried = false;
    bool ok = false;
};

inline SyclCublasBlas& sycl_cublas_state() {
    static SyclCublasBlas st;
    return st;
}

// Try once to bind libcublas and create a handle. Never throws, never prints
// in the fallback case (ordinary CPU/AMD runs).
inline bool sycl_cublas_ensure(sycl::queue& q) {
    SyclCublasBlas& st = sycl_cublas_state();
    if (st.tried) { return st.ok; }
    st.tried = true;
    if (q.get_backend() != sycl::backend::ext_oneapi_cuda) { return false; }
    const char* names[] = {"libcublas.so.12", "libcublas.so.11", "libcublas.so", nullptr};
    for (int i = 0; names[i] != nullptr && st.lib == nullptr; i++) {
        st.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    }
    if (st.lib == nullptr) { return false; }
    st.create = (sycl_cublasCreate_t)dlsym(st.lib, "cublasCreate_v2");
    st.setStream = (sycl_cublasSetStream_t)dlsym(st.lib, "cublasSetStream_v2");
    st.gemmBatched = (sycl_cublasGemmStridedBatchedEx_t)dlsym(st.lib, "cublasGemmStridedBatchedEx");
    if (!st.create || !st.setStream || !st.gemmBatched) { return false; }
    sycl_cublasHandle_t h = nullptr;
    if (st.create(&h) != 0 || h == nullptr) { return false; }
    st.handle = h;
    st.ok = true;
    return true;
}

// Strided-batched bf16->fp32 GEMM with the same row-major convention as
// matmul_batched_onemath: for each batch b,
//   Out(m x n) = op_row(A)(m x k) * op_row(B)(k x n),
// accumulating in fp32 into tmp (batch-major, m*n floats per batch).
// Returns true if the single cuBLAS launch was issued, false to take the
// oneMath loop instead. Never throws.
inline bool sycl_batched_gemm_lib(const floatX* A, const floatX* B, float* tmp,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideA, size_t strideB, int batch_count,
                     sycl::queue& q) {
    if (!sycl_cublas_ensure(q)) { return false; }
    SyclCublasBlas& st = sycl_cublas_state();
    // Same column-major mapping as the oneMath loop: C_col(n x m) =
    // U(n x k) * V(k x m) with U sourced from B and V from A.
    int tu = transB ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int tv = transA ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int ldu = transB ? k : n;
    int ldv = k; // A is always row-major (m x k)
    long long strideU = (long long)strideB;
    long long strideV = (long long)strideA;
    long long strideC = (long long)m * n;
    sycl_cudaStream_t stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
    if (st.setStream(st.handle, stream) != 0) { return false; }
    static const float alpha = 1.0f, beta = 0.0f;
    sycl_cublasStatus_t s = st.gemmBatched(st.handle, tu, tv, n, m, k,
        &alpha, (const void*)B, SYCL_CUDA_R_16BF, ldu, strideU,
        (const void*)A, SYCL_CUDA_R_16BF, ldv, strideV,
        &beta, (void*)tmp, SYCL_CUDA_R_32F, n, strideC,
        batch_count, SYCL_CUBLAS_COMPUTE_32F, SYCL_CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != 0) {
        // Unknown failure (e.g. rejected enum on a newer toolkit): fall back
        // for this call; the loop path below stays correct everywhere.
        return false;
    }
    return true;
}

#endif
