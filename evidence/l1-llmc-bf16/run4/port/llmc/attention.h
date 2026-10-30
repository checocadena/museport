/*
Attention (own kernels, like llm.c's default non-cuDNN path).
SYCL 2020 port of llmc/attention.cuh. Single device.
*/
#ifndef ATTENTION_H
#define ATTENTION_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
#include "onemath_common.h"
#include "matmul.h"

// ----------------------------------------------------------------------------
// SYCL kernels

// Q[b][nh][n][d] = inp[b][n][0][nh][d] (and K/V with offsets 1,2)
inline void permute_kernel(floatX* q, floatX* k, floatX* v,
                           const floatX* inp,
                           size_t B, size_t N, size_t NH, size_t d) {
    size_t total = B * NH * N * d;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= total) { return; }
            size_t b = idx / (NH * N * d);
            size_t rest = idx % (NH * N * d);
            size_t nh_ = rest / (N * d);
            rest = rest % (N * d);
            size_t n = rest / d;
            size_t d_ = rest % d;
            size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh_ * d) + d_;
            q[idx] = inp[inp_idx];
            k[idx] = inp[inp_idx + NH * d];
            v[idx] = inp[inp_idx + 2 * (NH * d)];
        }).wait();
}

inline void permute_kernel_backward(floatX* dinp,
                                    const floatX* dq, const floatX* dk, const floatX* dv,
                                    size_t B, size_t N, size_t NH, size_t d) {
    size_t total = B * NH * N * d;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= total) { return; }
            size_t b = idx / (NH * N * d);
            size_t rest = idx % (NH * N * d);
            size_t nh_ = rest / (N * d);
            rest = rest % (N * d);
            size_t n = rest / d;
            size_t d_ = rest % d;
            size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh_ * d) + d_;
            dinp[inp_idx] = dq[idx];
            dinp[inp_idx + NH * d] = dk[idx];
            dinp[inp_idx + 2 * (NH * d)] = dv[idx];
        }).wait();
}

inline void unpermute_kernel(floatX* inp, floatX *out, size_t B, size_t N, size_t NH, size_t d) {
    size_t total = B * NH * N * d;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= total) { return; }
            size_t b = idx / (NH * N * d);
            size_t rest = idx % (NH * N * d);
            size_t nh_ = rest / (N * d);
            rest = rest % (N * d);
            size_t n = rest / d;
            size_t d_ = rest % d;
            size_t other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
            out[other_idx] = inp[idx];
        }).wait();
}

inline void unpermute_kernel_backward(floatX* dinp, const floatX *dout,
                                      size_t B, size_t N, size_t NH, size_t d) {
    size_t total = B * NH * N * d;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= total) { return; }
            size_t b = idx / (NH * N * d);
            size_t rest = idx % (NH * N * d);
            size_t nh_ = rest / (N * d);
            rest = rest % (N * d);
            size_t n = rest / d;
            size_t d_ = rest % d;
            size_t other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
            dinp[idx] = (floatX)dout[other_idx];
        }).wait();
}

// inp, out shape: (rows, T, T), rows = B*NH. Autoregressive online softmax,
// one work-group per row.
inline void softmax_forward_kernel(floatX* out, float inv_temperature, const floatX* inp,
                                   size_t rows, size_t T) {
    if (rows == 0) { return; }
    constexpr size_t WG = 128;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            size_t own_pos = idx % T;
            const floatX* x = inp + idx * T;
            const float flt_max = 340282346638528859811704183484516925440.0f;
            float maxval = -flt_max;
            float sumval = 0.0f;
            // online softmax over causal prefix
            for (size_t i = lid; i <= own_pos; i += lsz) {
                float v = (float)x[i] * 1.0f; // scale applied below via inv_temperature
                (void)v;
                break;
            }
            // scalar online pass over the row prefix (T<=1024, cheap and exactly ordered)
            if (lid == 0) {
                for (size_t i = 0; i <= own_pos; i++) {
                    float v = inv_temperature * (float)x[i];
                    float old = maxval;
                    if (v > maxval) { maxval = v; }
                    sumval *= sycl::exp(old - maxval);
                    sumval += sycl::exp(v - maxval);
                }
            }
            maxval = sycl::group_broadcast(grp, maxval, 0);
            sumval = sycl::group_broadcast(grp, sumval, 0);
            float norm = 1.0f / sumval;
            for (size_t i = lid; i <= own_pos; i += lsz) {
                float ev = sycl::exp(inv_temperature * (float)x[i] - maxval);
                out[idx * T + i] = (floatX)(ev * norm);
            }
        }).wait();
}

inline void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att,
                                                           size_t B, size_t T, float scale) {
    size_t rows = B; // caller passes B*NH flattened as B here
    if (rows == 0) { return; }
    constexpr size_t WG = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            const floatX* att_bt = att + idx * T * T;
            const floatX* datt_bt_ro = datt + idx * T * T;
            floatX* datt_bt = datt + idx * T * T;
            for (size_t t = 0; t < T; t++) {
                // local_sum = sum_{t2<=t} att*datt (deterministic order via group sum of strided lanes is
                // still deterministic for fixed WG; do scalar in lane 0 for exactness)
                float local_sum = 0.0f;
                if (lid == 0) {
                    for (size_t t2 = 0; t2 <= t; t2++) {
                        local_sum += (float)att_bt[t * T + t2] * (float)datt_bt_ro[t * T + t2];
                    }
                }
                local_sum = sycl::group_broadcast(grp, local_sum, 0);
                for (size_t t3 = lid; t3 < T; t3 += lsz) {
                    if (t3 <= t) {
                        float acc = (float)att_bt[t * T + t3] * ((float)datt_bt_ro[t * T + t3] - local_sum);
                        datt_bt[t * T + t3] = (floatX)(scale * acc);
                    } else {
                        datt_bt[t * T + t3] = (floatX)0.f;
                    }
                }
                it.barrier(sycl::access::fence_space::global_space);
            }
        }).wait();
}

// Batched attention GEMMs via oneMath (one call per head).
// All buffers are row-major per head; the column_major oneMath calls below
// were each verified numerically against the llm.c cuBLAS call they replace
// (same linear memory, i.e. C == R^T as matrices, read back as R rows).
inline void att_batched_gemm_qk(floatX* preatt, const floatX* k, const floatX* q,
                                size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* kh = k + h * T * HS;
        const floatX* qh = q + h * T * HS;
        float* ch = tmp + h * T * T;
        // Rule A (outer form): R[a,b] = sum_h K[a,h]*Q[b,h]
        auto e = blas::column_major::gemm(sycl_queue, transpose::trans, transpose::nontrans,
            (int64_t)T, (int64_t)T, (int64_t)HS, 1.0f, qh, (int64_t)HS, kh, (int64_t)HS,
            0.0f, ch, (int64_t)T, {});
        e.wait();
    }
    convert_fp32_to_floatX(preatt, tmp, BNH * T * T);
}
inline void att_batched_gemm_av(floatX* y, const floatX* v, const floatX* att,
                                size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* vh = v + h * T * HS;
        const floatX* ah = att + h * T * T;
        float* ch = tmp + h * T * HS;
        // Rule B: R = A@V -> R^T = V^T @ A^T
        auto e = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::nontrans,
            (int64_t)HS, (int64_t)T, (int64_t)T, 1.0f, vh, (int64_t)HS, ah, (int64_t)T,
            0.0f, ch, (int64_t)HS, {});
        e.wait();
    }
    convert_fp32_to_floatX(y, tmp, BNH * T * HS);
}
inline void att_batched_gemm_datt(floatX* datt, const floatX* v, const floatX* scratch,
                                  size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* vh = v + h * T * HS;
        const floatX* sh = scratch + h * T * HS;
        float* ch = tmp + h * T * T;
        // Rule A (outer form): R[a,b] = sum_h S[a,h]*V[b,h]
        auto e = blas::column_major::gemm(sycl_queue, transpose::trans, transpose::nontrans,
            (int64_t)T, (int64_t)T, (int64_t)HS, 1.0f, vh, (int64_t)HS, sh, (int64_t)HS,
            0.0f, ch, (int64_t)T, {});
        e.wait();
    }
    convert_fp32_to_floatX(datt, tmp, BNH * T * T);
}
inline void att_batched_gemm_dv(floatX* dv, const floatX* scratch, const floatX* att,
                                size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* sh = scratch + h * T * HS;
        const floatX* ah = att + h * T * T;
        float* ch = tmp + h * T * HS;
        // R = S^T-view @ att (NT form): R[a,b] = C[b,a], C = S^T @ att
        auto e = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::trans,
            (int64_t)HS, (int64_t)T, (int64_t)T, 1.0f, sh, (int64_t)HS, ah, (int64_t)T,
            0.0f, ch, (int64_t)HS, {});
        e.wait();
    }
    convert_fp32_to_floatX(dv, tmp, BNH * T * HS);
}
inline void att_batched_gemm_dq(floatX* dq, const floatX* k, const floatX* dpreatt,
                                size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* kh = k + h * T * HS;
        const floatX* dh = dpreatt + h * T * T;
        float* ch = tmp + h * T * HS;
        // R = K^T-view @ d (NN form)
        auto e = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::nontrans,
            (int64_t)HS, (int64_t)T, (int64_t)T, 1.0f, kh, (int64_t)HS, dh, (int64_t)T,
            0.0f, ch, (int64_t)HS, {});
        e.wait();
    }
    convert_fp32_to_floatX(dq, tmp, BNH * T * HS);
}
inline void att_batched_gemm_dk(floatX* dk, const floatX* q, const floatX* dpreatt,
                                size_t BNH, size_t T, size_t HS, float* tmp) {
    using namespace oneapi::math;
    for (size_t h = 0; h < BNH; h++) {
        const floatX* qh = q + h * T * HS;
        const floatX* dh = dpreatt + h * T * T;
        float* ch = tmp + h * T * HS;
        // R = Q^T-view @ d^T-view (NT form)
        auto e = blas::column_major::gemm(sycl_queue, transpose::nontrans, transpose::trans,
            (int64_t)HS, (int64_t)T, (int64_t)T, 1.0f, qh, (int64_t)HS, dh, (int64_t)T,
            0.0f, ch, (int64_t)HS, {});
        e.wait();
    }
    convert_fp32_to_floatX(dk, tmp, BNH * T * HS);
}

// ----------------------------------------------------------------------------
// launchers

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                              floatX* inp,
                              size_t B, size_t T, size_t C, size_t NH) {
    NVTX_RANGE_FN();
    const size_t HS = C / NH; // head size

    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    permute_kernel(q, k, v, inp, B, T, NH, HS);

    floatX* preatt = inp; // reuse inp as scratch buffer
    matmul_ensure_tmp(B * NH * T * T > B * NH * T * HS ? B * NH * T * T : B * NH * T * HS);
    att_batched_gemm_qk(preatt, k, q, B * NH, T, HS, matmul_tmp);

    float scale = 1.f / sqrtf((float)HS);
    softmax_forward_kernel(att, scale, preatt, B * NH, T);

    floatX* vaccum = inp;
    att_batched_gemm_av(vaccum, v, att, B * NH, T, HS, matmul_tmp);

    unpermute_kernel(vaccum, out, B, T, NH, HS);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
inline void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                               const floatX* dout,
                               const floatX* qkvr, const floatX* att,
                               size_t B, size_t T, size_t C, size_t NH) {
    NVTX_RANGE_FN();
    const size_t HS = C / NH; // head size

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    unpermute_kernel_backward(scratch, dout, B, T, NH, HS);
    matmul_ensure_tmp(B * NH * T * T > B * NH * T * HS ? B * NH * T * T : B * NH * T * HS);
    // backward into datt
    att_batched_gemm_datt(datt, v, scratch, B * NH, T, HS, matmul_tmp);
    // backward into dv
    att_batched_gemm_dv(dv, scratch, att, B * NH, T, HS, matmul_tmp);
    const float scale = 1.0f / sqrtf((float)HS);
    // backward into preatt. in-place; datt turns into dpreatt here
    softmax_autoregressive_backward_inplace_kernel(datt, att, B * NH, T, scale);
    const floatX* dpreatt = datt;
    // backward into q
    att_batched_gemm_dq(dq, k, dpreatt, B * NH, T, HS, matmul_tmp);
    // backward into k
    att_batched_gemm_dk(dk, q, dpreatt, B * NH, T, HS, matmul_tmp);
    // backward into inp
    permute_kernel_backward(dinp, dq, dk, dv, B, T, NH, HS);
}

#endif
