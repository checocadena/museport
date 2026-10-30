/*
Matrix Multiplication, with help from oneMath BLAS (port of matmul.cuh).
Row-major logical layout everywhere (llm.c tensors are row-major):
  out(BT x OC) = inp(BT x C) * W^T + bias,  W is (OC x C) row-major.
Realised as oneMath column_major gemm with bf16 A/B and fp32 C output
(written to an fp32 buffer, then converted to bf16 with optional bias add):
  O_colm(OC x BT) = gemm(A=Wbuf trans, B=Inbuf nontrans),
  A is (C x OC) col-major == W row-major (OC x C), lda=C;
  B is (C x BT) col-major == In row-major (BT x C), ldb=C;
  C is (OC x BT) col-major == Out row-major (BT x OC), ldc=OC.

All oneMath calls are fire-and-forget on the in-order queue: successive
kernels/transfers are submitted in dependency order and never need a wait;
only host reads (via memcpy_d2h) block. Batched attention matmuls use a loop
of oneMath bf16->fp32 gemms over all (batch, head) pairs; callers fuse the
fp32->bf16 convert into their consumer. One code path for every shape and
device.
*/
#ifndef SYCL_MATMUL_H
#define SYCL_MATMUL_H

#include <assert.h>
#include <type_traits>      // std::bool_constant
#include <oneapi/math/blas.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"
#include "batched_gemm.hpp"
// GELU can be either fused or non-fused (kept non-fused here; fusion flags accepted for API compat)
#include "gelu.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels (implementations; declarations above for the self-test)

// dbias[OC] += sum over BT rows of dout (deterministic: fixed-grid ordered
// partial sums over rows per OC column into `aux` (needs 256*OC floats),
// then an ordered accumulation per column). Fire-and-forget.
template<typename OutFloat>
inline void matmul_backward_bias(floatX* dbias_out, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             bool use_aux, float* aux, int aux_rows, sycl::queue& q) {
    size_t N = (size_t)B * T;
    if (N == 0 || OC <= 0) { return; }
    // Deterministic path (always): fixed grid of partial sums + ordered tail.
    // `aux` must hold aux_rows*OC floats; callers pass the fp32 scratch buffer.
    const int P = (use_aux && aux_rows > 0) ? aux_rows : 256;
    float* partials = aux;
    size_t chunk = (N + (size_t)P - 1) / (size_t)P;
    q.parallel_for(sycl::range<2>((size_t)P, (size_t)OC), [=](sycl::id<2> id) {
        size_t r = id[0]; size_t oc = id[1];
        size_t b = r * chunk;
        size_t e = b + chunk;
        if (e > N) { e = N; }
        float acc = 0.0f;
        for (size_t i = b; i < e; i++) {
            acc += (float)dout[i * (size_t)OC + oc];
        }
        partials[r * (size_t)OC + oc] = acc;
    });
    q.parallel_for(sycl::range<1>((size_t)OC), [=](sycl::id<1> id) {
        size_t oc = id[0];
        float acc = (float)dbias[oc];
        for (int r = 0; r < P; r++) { acc += partials[(size_t)r * OC + oc]; }
        dbias_out[oc] = (floatX)acc;
    });
}

// dst[n] += sum over m rows of src (fp32), result rounded once to bf16 (deterministic order).
// Fire-and-forget; single kernel, ordered accumulation per column.
inline void reduce_add_sum(floatX* dst, const float* src, size_t n, size_t m, sycl::queue& q) {
    if (n == 0 || m == 0) { return; }
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        size_t i = id[0];
        float acc = 0.0f;
        for (size_t l = 0; l < m; l++) { acc += src[i + n * l]; }
        dst[i] = (floatX)((float)dst[i] + acc);
    });
}

// out(BT x OC) = inp(BT x C) @ W^T(OC x C) [+ bias], via oneMath bf16->fp32 gemm + convert kernel.
// Single path for every shape and device: oneMath always runs the fp32
// accumulate, then the epilogue below rounds once to bf16 (+bias, +gelu).
// The epilogue reads the fp32 GEMM output with a 2-D index (row-major
// BT x OC) so consecutive work-items in a sub-group touch consecutive
// addresses; bias loads are hoisted per row automatically by the compiler.
inline void matmul_onemath(floatX* out, const floatX* weight, const floatX* inp, const floatX* bias,
                     int OC, int BT, int C, sycl::queue& q, float* tmp_fp32,
                     bool accumulate = false, floatX* pre_gelu = nullptr, bool backward_gelu = false,
                     floatX* gelu_src = nullptr) {
    NVTX_RANGE_FN();
    bool has_bias = (bias != nullptr);
    bool has_gelu_fwd = (pre_gelu != nullptr && !backward_gelu);
    using namespace oneapi::math;
    // oneMath gemm writes fp32 C (no bf16 C support): use provided fp32 scratch
    oneapi::math::blas::column_major::gemm(q, transpose::trans, transpose::nontrans,
        (int64_t)OC, (int64_t)BT, (int64_t)C,
        1.0f, weight, (int64_t)C, inp, (int64_t)C,
        accumulate ? 1.0f : 0.0f, tmp_fp32, (int64_t)OC);
    // epilogue kernel: convert fp32 -> bf16 (+bias, +gelu), or dgelu-gated backward epilogue.
    // tmp_fp32 holds col-major (OC x BT) == row-major (BT x OC): tmp[r*OC+c].
    size_t nBT = (size_t)BT;
    size_t nOC = (size_t)OC;
    q.parallel_for(sycl::range<2>(nBT, nOC), [=](sycl::id<2> id) {
        size_t r = id[0];
        size_t oc = id[1];
        size_t i = r * nOC + oc;
        float v = tmp_fp32[i];
        if (backward_gelu && gelu_src != nullptr) {
            // DGELU epilogue used in QKV-backward path when fusion enabled (kept for API compat;
            // our default gelu_fusion=0 never takes this path)
            float x = (float)gelu_src[r * (size_t)C + oc % (size_t)C];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = sycl::tanh(tanh_arg);
            float coshf_out = sycl::cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            v = v * local_grad;
        } else {
            if (has_bias) { v += (float)bias[oc]; }
            if (has_gelu_fwd) { gelu_src[i] = (floatX)v; /* pre-activation aux */ }
        }
        float o = v;
        if (has_gelu_fwd) {
            float xi = v;
            float cube = 0.044715f * xi * xi * xi;
            o = 0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
        }
        if (accumulate) {
            o += (float)out[i];
        }
        out[i] = (floatX)o;
    });
}

// Batched attention matmul as a loop of oneMath bf16 -> fp32 gemms (one launch
// per batch, fp32 accumulation into tmp_fp32 batch-major, m*n floats/batch).
// The caller fuses the fp32 -> bf16 convert into its consumer (softmax /
// unpermute / permute), so there is no extra rounding stage.
// Always returns 2 (fp32 result in tmp_fp32 for the caller to fuse).
// Same code path for every shape and every device.
inline int matmul_batched_onemath(floatX* out, const floatX* A, const floatX* B,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideOut, size_t strideA, size_t strideB, int batch_count,
                     bool accumulate, float* tmp_fp32, sycl::queue& q) {
    if (batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) { return 2; }
    (void)out; (void)strideOut; (void)accumulate;
    bf16_strided_batched_gemm(tmp_fp32, A, B, m, n, k, transA, transB,
                              strideA, strideB, batch_count, q);
    return 2;
}

// Raw fp32 GEMM used by fused consumers (QKV permute, residual+layernorm):
// O_colm(OC x BT) = gemm(W trans, In nontrans) into tmp_fp32, no epilogue.
// The caller fuses bias/convert into its own kernel (same arithmetic as the
// epilogue: same order, one rounding).
inline void matmul_gemm_fp32(float* tmp_fp32, const floatX* weight, const floatX* inp,
                     int OC, int BT, int C, sycl::queue& q) {
    using namespace oneapi::math;
    oneapi::math::blas::column_major::gemm(q, transpose::trans, transpose::nontrans,
        (int64_t)OC, (int64_t)BT, (int64_t)C,
        1.0f, weight, (int64_t)C, inp, (int64_t)C,
        0.0f, tmp_fp32, (int64_t)OC);
}

// ----------------------------------------------------------------------------
// API mirroring matmul.cuh (stream arg replaced by queue)

inline void matmul_forward_onemath(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, sycl::queue& q, float* tmp_fp32,
                     floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_onemath(pre_gelu, weight, inp, bias, OC, B*T, C, q, tmp_fp32);
        gelu_forward(out, pre_gelu, (size_t)B*T*OC, q);
    } else {
        matmul_onemath(out, weight, inp, bias, OC, B*T, C, q, tmp_fp32, false, pre_gelu, false, pre_gelu);
    }
}

template<typename OutFloat = floatX>
inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, sycl::queue& q, float* tmp_fp32,
                     floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
    if (dbias != nullptr) {
        // deterministic fixed-grid reduction path (no host reads)
        matmul_backward_bias(dbias, dbias, dout, B, T, OC, true, dbias_buffer, 256, q);
    }

    // backward to input: dinp(BT x C) = dout(BT x OC) * W(OC x C)
    // Single path: oneMath bf16->fp32 accumulate into tmp, then one rounding
    // convert to bf16 (+ optional dgelu epilogue below).
    {
        using namespace oneapi::math;
        // Out_col(C x BT) = gemm(W trans, dout nontrans): A=Wbuf(C x OC col), B=doutbuf(C? ...)
        // dout row (BT x OC) == col (OC x BT). want Bop(C? ) hmm derive: Out_row(BT x C) = dout(BT x OC) * W(OC x C).
        // Out_col(C x BT) = W_col^T?? W row (OC x C) == col (C x OC): nontrans gives (C x OC). dout col (OC x BT): nontrans gives (OC x BT).
        // Out_col = W_col * dout_col: gemm(nontrans, nontrans, C, BT, OC, W, C, dout, OC).
        float* tmp = tmp_fp32; // BT*C*? need C*BT floats
        oneapi::math::blas::column_major::gemm(q, transpose::nontrans, transpose::nontrans,
            (int64_t)C, (int64_t)(B*T), (int64_t)OC, 1.0f, weight, (int64_t)C, dout, (int64_t)OC,
            0.0f, tmp, (int64_t)C);
        size_t N = (size_t)(B*T) * C;
        if (gelu_fusion >= 2 && pre_gelu) {
            // fused dgelu epilogue
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
                size_t i = id[0];
                float x = (float)pre_gelu[i];
                float cube = 0.044715f * x * x * x;
                float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
                float tanh_out = sycl::tanh(tanh_arg);
                float coshf_out = sycl::cosh(tanh_arg);
                float sech_out = 1.0f / (coshf_out * coshf_out);
                float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
                dinp[i] = (floatX)(tmp[i] * local_grad);
            });
        } else {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) { dinp[id[0]] = (floatX)tmp[id[0]]; });
            if (gelu_fusion < 2 && pre_gelu) {
                gelu_backward_inplace(dinp, pre_gelu, (size_t)B*T*C, q);
            }
        }
    }

    // backward to weight: dW(OC x C) += inp^T x dout  (accumulate)
    // Single path: oneMath bf16->fp32 into tmp, then accumulate-convert.
    // Row-major form: Out(OC x C) = dout^T(BT x OC)^T (OC x BT) * inp(BT x C):
    // A=dout transposed, B=inp plain.
    {
        using namespace oneapi::math;
        // dW_col(C x OC) = In_col * dout_col^T?? In row (BT x C) == col (C x BT) nontrans (C x BT); dout row (BT x OC) == col (OC x BT): trans gives (BT x OC).
        // dW_col(C x OC) = gemm(nontrans, trans, C, OC, BT, In, C, dout, OC).
        float* tmp = tmp_fp32; // C*OC floats (fits: tmp sized max(OC*BT, C*OC))
        oneapi::math::blas::column_major::gemm(q, transpose::nontrans, transpose::trans,
            (int64_t)C, (int64_t)OC, (int64_t)(B*T), 1.0f, inp, (int64_t)C, dout, (int64_t)OC,
            0.0f, tmp, (int64_t)C);
        size_t N = (size_t)OC * C;
        // tmp col (C x OC) == dW row (OC x C): accumulate-convert
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            dweight[id[0]] = (floatX)((float)dweight[id[0]] + tmp[id[0]]);
        });
    }
    (void)dbias_buffer;
}

#endif
