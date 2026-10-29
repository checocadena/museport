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
*/
#ifndef SYCL_MATMUL_H
#define SYCL_MATMUL_H

#include <assert.h>
#include <type_traits>      // std::bool_constant
#include <oneapi/math/blas.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"
// GELU can be either fused or non-fused (kept non-fused here; fusion flags accepted for API compat)
#include "gelu.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

// dbias[OC] += sum over BT rows of dout (deterministic: ordered accumulation per OC)
template<typename OutFloat>
inline void matmul_backward_bias(floatX* dbias_out, OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             bool use_aux, float* aux, int aux_rows, sycl::queue& q) {
    size_t N = (size_t)B * T;
    if (use_aux) {
        // each aux row r holds partial sums over a strided subset (mirrors grid_size_y path)
        q.parallel_for(sycl::range<2>((size_t)aux_rows, (size_t)OC), [=](sycl::id<2> id) {
            size_t r = id[0]; size_t oc = id[1];
            float acc = 0.0f;
            for (size_t i = r; i < N; i += (size_t)aux_rows) {
                acc += (float)dout[i * (size_t)OC + oc];
            }
            aux[r * (size_t)OC + oc] = acc;
        }).wait();
        // deterministic reduction over aux rows in order
        q.wait();
        for (size_t oc = 0; oc < (size_t)OC; oc++) {
            float acc = (float)dbias[oc];
            for (int r = 0; r < aux_rows; r++) { acc += aux[(size_t)r * OC + oc]; }
            dbias_out[oc] = (floatX)acc;
        }
    } else {
        q.parallel_for(sycl::range<1>((size_t)OC), [=](sycl::id<1> id) {
            size_t oc = id[0];
            float acc = 0.0f;
            for (size_t i = 0; i < N; i++) {
                acc += (float)dout[i * (size_t)OC + oc];
            }
            dbias_out[oc] = (floatX)(acc + (float)dbias[oc]);
        }).wait();
    }
}

// dst[n] += sum over m rows of src (fp32), result rounded once to bf16 (deterministic order)
inline void reduce_add_sum(floatX* dst, const float* src, size_t n, size_t m, sycl::queue& q) {
    q.wait();
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (size_t l = 0; l < m; l++) { acc += src[i + n * l]; }
        dst[i] = (floatX)((float)dst[i] + acc);
    }
}

// out(BT x OC) = inp(BT x C) @ W^T(OC x C) [+ bias], via oneMath bf16->fp32 gemm + convert kernel
inline void matmul_onemath(floatX* out, const floatX* weight, const floatX* inp, const floatX* bias,
                     int OC, int BT, int C, sycl::queue& q, float* tmp_fp32,
                     bool accumulate = false, floatX* pre_gelu = nullptr, bool backward_gelu = false,
                     floatX* gelu_src = nullptr) {
    NVTX_RANGE_FN();
    using namespace oneapi::math;
    // oneMath gemm writes fp32 C (no bf16 C support): use provided fp32 scratch
    oneapi::math::blas::column_major::gemm(q, transpose::trans, transpose::nontrans,
        (int64_t)OC, (int64_t)BT, (int64_t)C,
        1.0f, weight, (int64_t)C, inp, (int64_t)C,
        accumulate ? 1.0f : 0.0f, tmp_fp32, (int64_t)OC).wait();
    size_t N = (size_t)OC * BT;
    bool has_bias = (bias != nullptr);
    bool has_gelu_fwd = (pre_gelu != nullptr && !backward_gelu);
    // epilogue kernel: convert fp32 -> bf16 (+bias, +gelu), or dgelu-gated backward epilogue
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t oc = i % (size_t)OC;
        float v = tmp_fp32[i];
        if (backward_gelu && gelu_src != nullptr) {
            // DGELU epilogue used in QKV-backward path when fusion enabled (kept for API compat;
            // our default gelu_fusion=0 never takes this path)
            float x = (float)gelu_src[(i / (size_t)OC) * (size_t)C + oc % (size_t)C];
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
    }).wait();
}

// Batched variant for attention: outTPL described per-batch below (row-major batches).
// Computes, for each batch b: Out(m x n) = op(A)(m x k) * op(B)(k x n) with fp32 accumulate,
// where the A/B/Out pointers are batch-strided views of row-major tensors.
inline void matmul_batched_onemath(floatX* out, const floatX* A, const floatX* B,
                     int m, int n, int k, bool transA, bool transB,
                     size_t strideOut, size_t strideA, size_t strideB, int batch_count,
                     bool accumulate, float* tmp_fp32, sycl::queue& q) {
    using namespace oneapi::math;
    // Implement via per-batch oneMath gemm calls on column-major views.
    // Row-major (m x k) == col-major (k x m). We compute Out_row(m x n) via
    // Out_col(n x m) = B_col_op * A_col_op with appropriate transpose flags:
    //   Out_row = A_row * B_row            => Out_col = B_col^T? Let's derive per case below.
    // Cases used by attention (all row-major logical):
    //  F1: preatt(T x T) = Q(T x HS) * K^T  i.e. A=Q(m=T,k=HS) N, B=K(n=T,k=HS) T
    //  F2: vaccum(T x HS) = Att(T x T) * V(T x HS): A=Att(m=T,k=T) N, B=V(n=HS,k=T) N with roles swapped
    //  B1: datt(T x T) = dOut(T x HS) * V^T : A=dOut(m=T,k=HS) N, B=V(n=T,k=HS) T
    //  B2: dv(HS x T)?? dv(T x HS) = Att^T * dOut: A=Att(m=T,k=T) T, B=dOut(n=HS,k=T) N
    //  B3: dq(T x HS) = dpreatt(T x T) * K: A=dpreatt(m=T,k=T) N, B=K(n=HS,k=T) N swapped
    //  B4: dk(T x HS) = dpreatt^T * Q: A=dpreatt(m=T,k=T) T, B=Q(n=HS,k=T) N swapped
    // The generic identity: Out_row(m x n) = Aop(m x k) * Bop(k x n)
    //   <=> Out_col(n x m) = Bop_col^T... simplest: Out_col(n x m) = gemm(Bop_as_col, Aop_as_col)
    //   with Bop_as_col = (n x k) col-major? Bop row (k x n) == col (n x k), Aop row (m x k) == col (k x m).
    //   Out_col = Bop_col * Aop_col with op flags: Bop_col is (n x k): use transB? and Aop_col is (k x m): use transA?
    //   gemm_col(C(n x m) = op1(Bb (n x k)) * op2(Ab (k x m))): op1 = nontrans if Bop row is (k x n)... Bop row-major (k x n) memory == col-major (n x k): op1=nontrans gives (n x k). op2: Aop row (m x k) == col (k x m): nontrans gives (k x m). So: C = gemm(nontrans, nontrans, n, m, k, Bbuf, n, Abuf, k).
    // With transpose variants on the row-major side, adjust flags accordingly:
    //   if transA(row): Aop = A^T where A row (k x m) i.e. requested A buffer is (m x k)?? We handle by swapping lda/flags:
    // Easiest correct approach: implement each batch with explicit transpose flags on the col-major views:
    //   Acol: buffer A interpreted col-major with dims (transA ? (m x k) : (k x m))... Instead of deriving further,
    //   implement via two tiny transpose-free paths used in practice (see attention.hpp which calls
    //   specialised wrappers). This generic function supports transA/transB on row-major semantics:
    //   Out(m x n) = op_row(A) * op_row(B), A:(m x k) or (k x m) if transA, B:(k x n) or (n x k) if transB.
    for (int b = 0; b < batch_count; b++) {
        const floatX* Ab = A + (size_t)b * strideA;
        const floatX* Bb = B + (size_t)b * strideB;
        float* Cb = tmp_fp32 + (size_t)b * (size_t)m * n;
        // col-major: Cc(n x m) = Bop^T(n x k) * Aop^T(k x m); both buffers viewed col-major.
        transpose fB = transB ? transpose::trans : transpose::nontrans;
        transpose fA = transA ? transpose::trans : transpose::nontrans;
        int64_t ldB = transB ? (int64_t)k : (int64_t)n; // col rows of Bbuf as col-major: (n x k)[N] rows=n; (k x n)[T] rows=k
        int64_t ldA = transA ? (int64_t)m : (int64_t)k; // col rows of Abuf as col-major: (k x m)[N] rows=k; (m x k)[T] rows=m
        // NOTE: with swapped argument order, m/n swap: gemm(m=n_row? ...). Cc is (n x m): gemm(n, m, k, Bb, Ab).
        oneapi::math::blas::column_major::gemm(q, fB, fA, (int64_t)n, (int64_t)m, (int64_t)k,
            1.0f, Bb, ldB, Ab, ldA, 0.0f, Cb, (int64_t)n).wait();
        // Cb col-major (n x m) == Out row-major (m x n): copy/convert out (with accumulate)
        floatX* Ob = out + (size_t)b * strideOut;
        size_t N = (size_t)m * n;
        if (!accumulate) {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) { Ob[id[0]] = (floatX)Cb[id[0]]; }).wait();
        } else {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) { Ob[id[0]] = (floatX)((float)Ob[id[0]] + Cb[id[0]]); }).wait();
        }
    }
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
        // single deterministic reduction path (no cross-block aux needed)
        matmul_backward_bias(dbias, dbias, dout, B, T, OC, false, nullptr, 0, q);
    }

    // backward to input: dinp(BT x C) = dout(BT x OC) * W(OC x C)
    {
        using namespace oneapi::math;
        // Out_col(C x BT) = gemm(W trans, dout nontrans): A=Wbuf(C x OC col), B=doutbuf(C? ...)
        // dout row (BT x OC) == col (OC x BT). want Bop(C? ) hmm derive: Out_row(BT x C) = dout(BT x OC) * W(OC x C).
        // Out_col(C x BT) = W_col^T?? W row (OC x C) == col (C x OC): nontrans gives (C x OC). dout col (OC x BT): nontrans gives (OC x BT).
        // Out_col = W_col * dout_col: gemm(nontrans, nontrans, C, BT, OC, W, C, dout, OC).
        float* tmp = tmp_fp32; // BT*C*? need C*BT floats
        oneapi::math::blas::column_major::gemm(q, transpose::nontrans, transpose::nontrans,
            (int64_t)C, (int64_t)(B*T), (int64_t)OC, 1.0f, weight, (int64_t)C, dout, (int64_t)OC,
            0.0f, tmp, (int64_t)C).wait();
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
            }).wait();
        } else {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) { dinp[id[0]] = (floatX)tmp[id[0]]; }).wait();
            if (gelu_fusion < 2 && pre_gelu) {
                gelu_backward_inplace(dinp, pre_gelu, (size_t)B*T*C, q);
            }
        }
    }

    // backward to weight: dW(OC x C) += inp^T x dout  (accumulate)
    {
        using namespace oneapi::math;
        // dW_col(C x OC) = In_col * dout_col^T?? In row (BT x C) == col (C x BT) nontrans (C x BT); dout row (BT x OC) == col (OC x BT): trans gives (BT x OC).
        // dW_col(C x OC) = gemm(nontrans, trans, C, OC, BT, In, C, dout, OC).
        float* tmp = tmp_fp32; // C*OC floats (fits: tmp sized max(OC*BT, C*OC))
        oneapi::math::blas::column_major::gemm(q, transpose::nontrans, transpose::trans,
            (int64_t)C, (int64_t)OC, (int64_t)(B*T), 1.0f, inp, (int64_t)C, dout, (int64_t)OC,
            0.0f, tmp, (int64_t)C).wait();
        size_t N = (size_t)OC * C;
        // tmp col (C x OC) == dW row (OC x C): accumulate-convert
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            dweight[id[0]] = (floatX)((float)dweight[id[0]] + tmp[id[0]]);
        }).wait();
    }
    (void)dbias_buffer;
}

#endif
