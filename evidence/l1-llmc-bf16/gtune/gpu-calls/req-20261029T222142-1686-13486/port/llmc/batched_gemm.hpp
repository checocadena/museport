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
#include <utility> // std::declval, std::true_type
// llmc internal imports
#include "sycl_common.h"

// Compile-time probe: is the CUDA-backend native queue handle available in
// this toolchain? Toolchains built without CUDA support leave
// BackendReturn<ext_oneapi_cuda, queue> undefined, in which case the template
// below is dropped by SFINAE (the observed "substitution failure") and the
// constant is false; the interop bodies then vanish via if constexpr, so this
// header compiles anywhere. (The ext_oneapi_cuda enumerator itself is
// standard and only ever named; never instantiated without the traits.)
template <typename Q>
static auto sycl_cuda_native_probe(Q& q, int)
    -> decltype(sycl::get_native<sycl::backend::ext_oneapi_cuda>(q), std::true_type{});
static std::false_type sycl_cuda_native_probe(...);
static constexpr bool kSyclCudaNativeOk =
    decltype(sycl_cuda_native_probe(std::declval<sycl::queue&>(), 0))::value;

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
typedef sycl_cublasStatus_t (*sycl_cublasGemmEx_t)(
    sycl_cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int);

struct SyclCublasBlas {
    void* lib = nullptr;
    sycl_cublasCreate_t create = nullptr;
    sycl_cublasSetStream_t setStream = nullptr;
    sycl_cublasGemmStridedBatchedEx_t gemmBatched = nullptr;
    sycl_cublasGemmEx_t gemmEx = nullptr;
    sycl_cublasHandle_t handle = nullptr;
    bool tried = false;
    bool ok = false;
    bool calibrated = false;
    bool bf16out = false; // calibrated: bf16-output strided GEMM is available
    bool gemmExBf16out = false; // calibrated: bf16-output single GEMM is available
    int computeType = SYCL_CUBLAS_COMPUTE_32F;
    int algo = SYCL_CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

inline SyclCublasBlas& sycl_cublas_state() {
    static SyclCublasBlas st;
    return st;
}

// One-time calibration: run a tiny strided-batched bf16->fp32 GEMM for each
// candidate (computeType, algo) pair and check the status AND the numeric
// result against a host reference. Ports across toolkit versions (whose enum
// values or kernel coverage may differ) without hardcoding blind trust.
// Never throws; leaves ok=false if nothing calibrates.
// (Template + if constexpr so toolchains without CUDA-native traits compile.)
template <typename Q>
inline bool sycl_cublas_calibrate_tmpl(Q& q) {
    if constexpr (!kSyclCudaNativeOk) { return false; }
    else {
    SyclCublasBlas& st = sycl_cublas_state();
    if (st.calibrated) { return st.ok; }
    st.calibrated = true;
    // 1 batch of C(2x2) = A(2x3) * B(3x2), row-major, no transposes.
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]] => C = [[4,5],[10,11]].
    float Ah[6] = {1,2,3,4,5,6};
    float Bh[6] = {1,0,0,1,1,1};
    float Cref[4] = {4,5,10,11};
    floatX* dA = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
    floatX* dB = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
    float* dC = (float*)sycl::malloc_device(4 * sizeof(float), q);
    if (!dA || !dB || !dC) {
        if (dA) sycl::free(dA, q);
        if (dB) sycl::free(dB, q);
        if (dC) sycl::free(dC, q);
        return false;
    }
    floatX Ahx[6], Bhx[6];
    for (int i = 0; i < 6; i++) { Ahx[i] = (floatX)Ah[i]; Bhx[i] = (floatX)Bh[i]; }
    // candidate (computeType, algo) pairs: 32F plain/pedantic/fast variants x
    // default/tensor-op algos (values are the cublas_api.h enum numbers).
    const int computes[] = {66, 67, 68, 69, 70, 64};
    const int algos[] = {99, -1, 0};
    sycl_cudaStream_t stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
    if (st.setStream(st.handle, stream) != 0) {
        sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
        return false;
    }
    static const float alpha = 1.0f, beta = 0.0f;
    bool pass = false;
    // column-major mapping of the row-major test: C_col(2x2) = U*B... with
    // tu=tv=N: gemm(m=2,n=2,k=3, B, ldb=2, A, lda=3, C, ldc=2).
    for (int ci = 0; ci < 6 && !pass; ci++) {
        for (int ai = 0; ai < 3 && !pass; ai++) {
            q.memcpy(dA, Ahx, sizeof(Ahx)).wait();
            q.memcpy(dB, Bhx, sizeof(Bhx)).wait();
            sycl_cublasStatus_t s = st.gemmBatched(st.handle,
                SYCL_CUBLAS_OP_N, SYCL_CUBLAS_OP_N, 2, 2, 3,
                &alpha, (const void*)dB, SYCL_CUDA_R_16BF, 2, 6,
                (const void*)dA, SYCL_CUDA_R_16BF, 3, 6,
                &beta, (void*)dC, SYCL_CUDA_R_32F, 2, 4,
                1, computes[ci], algos[ai]);
            if (s != 0) { continue; }
            float Cout[4] = {0};
            q.memcpy(Cout, dC, sizeof(Cout)).wait();
            pass = true;
            for (int i = 0; i < 4; i++) {
                float d = Cout[i] - Cref[i];
                if (!(d > -1e-3f && d < 1e-3f)) { pass = false; break; }
            }
            if (pass) { st.computeType = computes[ci]; st.algo = algos[ai]; }
        }
    }
    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    st.ok = pass;
    if (!pass) { return false; }
    // Probe bf16 output as well (halves C traffic and removes the convert;
    // same accumulation, rounded inside). Tolerance is one bf16 ulp scale.
    // (Fresh device buffers: dA/dB/dC above are already freed.)
    floatX* dA2 = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
    floatX* dB2 = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
    floatX* dCb = (floatX*)sycl::malloc_device(4 * sizeof(floatX), q);
    if (!dA2 || !dB2 || !dCb) {
        if (dA2) sycl::free(dA2, q);
        if (dB2) sycl::free(dB2, q);
        if (dCb) sycl::free(dCb, q);
        return true; // fp32 path calibrated; bf16-out unavailable
    }
    q.memcpy(dA2, Ahx, sizeof(Ahx)).wait();
    q.memcpy(dB2, Bhx, sizeof(Bhx)).wait();
    sycl_cublasStatus_t sb = st.gemmBatched(st.handle,
        SYCL_CUBLAS_OP_N, SYCL_CUBLAS_OP_N, 2, 2, 3,
        &alpha, (const void*)dB2, SYCL_CUDA_R_16BF, 2, 6,
        (const void*)dA2, SYCL_CUDA_R_16BF, 3, 6,
        &beta, (void*)dCb, SYCL_CUDA_R_16BF, 2, 4,
        1, st.computeType, st.algo);
    bool bpass = (sb == 0);
    if (bpass) {
        floatX Cbout[4];
        q.memcpy(Cbout, dCb, sizeof(Cbout)).wait();
        for (int i = 0; i < 4; i++) {
            float d = (float)Cbout[i] - Cref[i];
            if (!(d > -2e-2f && d < 2e-2f)) { bpass = false; break; }
        }
    }
    sycl::free(dA2, q); sycl::free(dB2, q); sycl::free(dCb, q);
    st.bf16out = bpass;
    // Probe single (non-batched) bf16-output GEMM as well (cublasGemmEx, for
    // the dense matmuls). Same tiny test through the non-batched entry point.
    st.gemmExBf16out = false;
    if (st.gemmEx != nullptr) {
        floatX* dA3 = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
        floatX* dB3 = (floatX*)sycl::malloc_device(6 * sizeof(floatX), q);
        floatX* dC3 = (floatX*)sycl::malloc_device(4 * sizeof(floatX), q);
        if (dA3 && dB3 && dC3) {
            q.memcpy(dA3, Ahx, sizeof(Ahx)).wait();
            q.memcpy(dB3, Bhx, sizeof(Bhx)).wait();
            sycl_cublasStatus_t se = st.gemmEx(st.handle,
                SYCL_CUBLAS_OP_N, SYCL_CUBLAS_OP_N, 2, 2, 3,
                &alpha, (const void*)dB3, SYCL_CUDA_R_16BF, 2,
                (const void*)dA3, SYCL_CUDA_R_16BF, 3,
                &beta, (void*)dC3, SYCL_CUDA_R_16BF, 2,
                st.computeType, st.algo);
            if (se == 0) {
                floatX C3out[4];
                q.memcpy(C3out, dC3, sizeof(C3out)).wait();
                bool e3pass = true;
                for (int i = 0; i < 4; i++) {
                    float d = (float)C3out[i] - Cref[i];
                    if (!(d > -2e-2f && d < 2e-2f)) { e3pass = false; break; }
                }
                st.gemmExBf16out = e3pass;
            }
        }
        if (dA3) sycl::free(dA3, q);
        if (dB3) sycl::free(dB3, q);
        if (dC3) sycl::free(dC3, q);
    }
    return true;
    }
}
inline bool sycl_cublas_calibrate(sycl::queue& q) {
    return sycl_cublas_calibrate_tmpl(q);
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
    st.gemmEx = (sycl_cublasGemmEx_t)dlsym(st.lib, "cublasGemmEx");
    if (!st.create || !st.setStream || !st.gemmBatched) { return false; }
    sycl_cublasHandle_t h = nullptr;
    if (st.create(&h) != 0 || h == nullptr) { return false; }
    st.handle = h;
    st.ok = true;
    return true;
}

// Strided-batched bf16 GEMM with the same row-major convention as
// matmul_batched_onemath: for each batch b,
//   Out(m x n) = op_row(A)(m x k) * op_row(B)(k x n).
// When bf16 output calibrated, Out is written directly as bf16 (returns 1).
// Otherwise fp32 accumulation lands in tmp (batch-major, m*n per batch) for
// the caller to fuse (returns 2). Returns 0 to take the oneMath loop instead.
// Never throws.
// (Template + if constexpr so toolchains without CUDA-native traits compile.)
template <typename Q>
inline int sycl_batched_gemm_lib_tmpl(floatX* out, const floatX* A, const floatX* B, float* tmp,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideOut, size_t strideA, size_t strideB, int batch_count,
                     Q& q) {
    if constexpr (!kSyclCudaNativeOk) { return 0; }
    else {
    if (!sycl_cublas_ensure(q)) { return 0; }
    SyclCublasBlas& st = sycl_cublas_state();
    if (!sycl_cublas_calibrate(q)) { return 0; }
    // Same column-major mapping as the oneMath loop: C_col(n x m) =
    // U(n x k) * V(k x m) with U sourced from B and V from A.
    int tu = transB ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int tv = transA ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int ldu = transB ? k : n;
    int ldv = transA ? m : k; // A is row-major (m x k) normally, (k x m) when transA
    long long strideU = (long long)strideB;
    long long strideV = (long long)strideA;
    sycl_cudaStream_t stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
    if (st.setStream(st.handle, stream) != 0) { return 0; }
    static const float alpha = 1.0f, beta = 0.0f;
    sycl_cublasStatus_t s;
    if (st.bf16out) {
        long long strideC = (long long)strideOut;
        s = st.gemmBatched(st.handle, tu, tv, n, m, k,
            &alpha, (const void*)B, SYCL_CUDA_R_16BF, ldu, strideU,
            (const void*)A, SYCL_CUDA_R_16BF, ldv, strideV,
            &beta, (void*)out, SYCL_CUDA_R_16BF, n, strideC,
            batch_count, st.computeType, st.algo);
        if (s != 0) { return 0; }
        return 1;
    }
    long long strideC = (long long)m * n;
    s = st.gemmBatched(st.handle, tu, tv, n, m, k,
        &alpha, (const void*)B, SYCL_CUDA_R_16BF, ldu, strideU,
        (const void*)A, SYCL_CUDA_R_16BF, ldv, strideV,
        &beta, (void*)tmp, SYCL_CUDA_R_32F, n, strideC,
        batch_count, st.computeType, st.algo);
    if (s != 0) {
        // Unknown failure (e.g. rejected enum on a newer toolkit): fall back
        // for this call; the loop path below stays correct everywhere.
        return 0;
    }
    return 2;
    }
}
inline int sycl_batched_gemm_lib(floatX* out, const floatX* A, const floatX* B, float* tmp,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideOut, size_t strideA, size_t strideB, int batch_count,
                     sycl::queue& q) {
    return sycl_batched_gemm_lib_tmpl(out, A, B, tmp, m, n, k, transA, transB,
                                      strideOut, strideA, strideB, batch_count, q);
}

// Single (non-batched) bf16 GEMM with bf16 output, row-major convention:
//   Out(m x n) = op_row(A)(m x k) * op_row(B)(k x n),
// accumulating in fp32 inside cuBLAS. Used for the dense matmuls when the
// per-call work is large (see matmul.hpp); small shapes stay on the oneMath
// path so test numerics are unchanged. Returns true when the cuBLAS launch
// was issued (bf16 in out), false to take the oneMath path instead.
// (Template + if constexpr so toolchains without CUDA-native traits compile.)
template <typename Q>
inline bool sycl_gemm_lib_tmpl(floatX* out, const floatX* A, const floatX* B,
                     int64_t m, int64_t n, int64_t k, bool transA, bool transB,
                     Q& q, bool accumulate = false) {
    if constexpr (!kSyclCudaNativeOk) { return false; }
    else {
    if (!sycl_cublas_ensure(q)) { return false; }
    SyclCublasBlas& st = sycl_cublas_state();
    if (!sycl_cublas_calibrate(q)) { return false; }
    if (!st.gemmExBf16out || st.gemmEx == nullptr) { return false; }
    // Column-major mapping, same as the oneMath dense path: C_col(n x m) =
    // U(n x k) * V(k x m) with U sourced from B and V from A.
    int tu = transB ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int tv = transA ? SYCL_CUBLAS_OP_T : SYCL_CUBLAS_OP_N;
    int ldu = transB ? (int)k : (int)n;
    int ldv = transA ? (int)m : (int)k; // A row is (m x k), or (k x m) when transA
    sycl_cudaStream_t stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
    if (st.setStream(st.handle, stream) != 0) { return false; }
    static const float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    sycl_cublasStatus_t s = st.gemmEx(st.handle, tu, tv, (int)n, (int)m, (int)k,
        &alpha, (const void*)B, SYCL_CUDA_R_16BF, ldu,
        (const void*)A, SYCL_CUDA_R_16BF, ldv,
        &beta, (void*)out, SYCL_CUDA_R_16BF, (int)n,
        st.computeType, st.algo);
    return s == 0;
    }
}
inline bool sycl_gemm_lib(floatX* out, const floatX* A, const floatX* B,
                     int64_t m, int64_t n, int64_t k, bool transA, bool transB,
                     sycl::queue& q, bool accumulate = false) {
    return sycl_gemm_lib_tmpl(out, A, B, m, n, k, transA, transB, q, accumulate);
}

#endif
