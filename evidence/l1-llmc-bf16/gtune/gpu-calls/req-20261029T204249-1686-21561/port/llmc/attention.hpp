/*
Attention (own kernels, no cuDNN), SYCL 2020 port of attention.cuh.
Batched matmuls run as a single kernel launch over all (batch, head) pairs
(see matmul.hpp). All launches are fire-and-forget on the in-order queue.
*/
#ifndef SYCL_ATTENTION_H
#define SYCL_ATTENTION_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"
#include "matmul.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

inline void permute(floatX* qo, floatX* ko, floatX* vo,
                               const floatX* inp,
                               int B, int N, int NH, int d, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t total = (size_t)B * NH * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh_ = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t d_ = rest % (size_t)d;
        size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh_ * d) + d_;
        qo[idx] = inp[inp_idx];
        ko[idx] = inp[inp_idx + (size_t)NH * d];
        vo[idx] = inp[inp_idx + 2 * (size_t)NH * d];
    });
}

inline void permute_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t total = (size_t)B * NH * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh_ = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t d_ = rest % (size_t)d;
        size_t inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + (size_t)NH * d] = dk[idx];
        dinp[inp_idx + 2 * (size_t)NH * d] = dv[idx];
    });
}

inline void unpermute(floatX* inp, floatX *out, int B, int N, int NH, int d, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t total = (size_t)B * NH * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh_ = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t d_ = rest % (size_t)d;
        size_t other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    });
}

inline void unpermute_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t total = (size_t)B * NH * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh_ = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t d_ = rest % (size_t)d;
        size_t other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = (floatX)dout[other_idx];
    });
}

// inp, out shape: (NT, T, T) row-major batches, where NT = B * NH; autoregressive causal.
// One work-group (128) per row; lanes cooperate on an online softmax over the
// valid prefix (strided vec4 loads), then normalize. The max reduction is
// exact; the sum reduction order differs from a single-thread loop in the last
// ulp only. The upper triangle is left untouched: the att buffer is zeroed at
// allocation and no kernel ever writes above the diagonal, so it stays zero
// (same as llm.c, which never writes it either).
inline void softmax_forward(floatX* out, float inv_temperature, const floatX* inp, int NT, int T, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t rows = (size_t)NT * T;
    if (rows == 0 || T <= 0) { return; }
    constexpr size_t WG = 128;
    const float flt_max = 340282346638528859811704183484516925440.0f;
    q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        size_t idx = it.get_group(0);
        size_t lid = it.get_local_id(0);
        sycl::group<1> g = it.get_group();
        size_t own_pos = idx % (size_t)T;
        const floatX* x = inp + idx * (size_t)T;
        floatX* o = out + idx * (size_t)T;
        size_t n = own_pos + 1; // number of valid elements
        float maxval = -flt_max;
        float sumval = 0.0f;
        if ((((uintptr_t)x) & 15) == 0) {
            size_t n4 = n / 4;
            for (size_t i = lid; i < n4; i += WG) {
                const floatX* p = x + i * 4;
                float r0 = (float)p[0], r1 = (float)p[1], r2 = (float)p[2], r3 = (float)p[3];
                float old = maxval;
                float m4 = fmaxf(fmaxf(r0, r1), fmaxf(r2, r3));
                maxval = fmaxf(maxval, m4);
                sumval *= sycl::exp(inv_temperature * (old - maxval));
                sumval += sycl::exp(inv_temperature * (r0 - maxval))
                        + sycl::exp(inv_temperature * (r1 - maxval))
                        + sycl::exp(inv_temperature * (r2 - maxval))
                        + sycl::exp(inv_temperature * (r3 - maxval));
            }
            for (size_t i = n4 * 4 + lid; i < n; i += WG) {
                float r = (float)x[i];
                float old = maxval;
                maxval = fmaxf(maxval, r);
                sumval *= sycl::exp(inv_temperature * (old - maxval));
                sumval += sycl::exp(inv_temperature * (r - maxval));
            }
        } else {
            for (size_t i = lid; i < n; i += WG) {
                float r = (float)x[i];
                float old = maxval;
                maxval = fmaxf(maxval, r);
                sumval *= sycl::exp(inv_temperature * (old - maxval));
                sumval += sycl::exp(inv_temperature * (r - maxval));
            }
        }
        float gmax = sycl::reduce_over_group(g, maxval, sycl::maximum<float>());
        sumval *= sycl::exp(inv_temperature * (maxval - gmax));
        float gsum = sycl::reduce_over_group(g, sumval, sycl::plus<float>());
        float norm = 1.0f / gsum;
        for (size_t i = lid; i < n; i += WG) {
            float ev = sycl::exp(inv_temperature * ((float)x[i] - gmax));
            o[i] = (floatX)(ev * norm);
        }
    });
}

// In-place causal softmax backward: dpreatt[t3] = scale * att[t3] * (datt[t3] -
// sum_{t2<=t} att*datt), zeroed for t3 > t. One work-group (256) per row with
// a barrier between the reduction and the (in-place) store. The dot-product
// reduction order differs from a single-thread loop in the last ulp only.
inline void softmax_autoregressive_backward_inplace(floatX* datt, const floatX* att,
                                                               int B, int T, int NH, float scale, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t rows = (size_t)B * NH * T;
    if (rows == 0 || T <= 0) { return; }
    constexpr size_t WG = 256;
    q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        size_t r = it.get_group(0);
        size_t lid = it.get_local_id(0);
        sycl::group<1> g = it.get_group();
        size_t t = r % (size_t)T;
        const floatX* att_bth = att + r * (size_t)T;
        const floatX* datt_bth = datt + r * (size_t)T;
        floatX* dpreatt_bth = datt + r * (size_t)T;
        float local = 0.0f;
        for (size_t t2 = lid; t2 <= t; t2 += WG) {
            local += (float)att_bth[t2] * (float)datt_bth[t2];
        }
        float local_sum = sycl::reduce_over_group(g, local, sycl::plus<float>());
        sycl::group_barrier(g); // store below overwrites datt in place
        for (size_t t3 = lid; t3 < (size_t)T; t3 += WG) {
            if (t3 <= t) {
                float d = (float)datt_bth[t3];
                float a = (float)att_bth[t3];
                dpreatt_bth[t3] = (floatX)(scale * (a * (d - local_sum)));
            } else {
                dpreatt_bth[t3] = (floatX)0.f;
            }
        }
    });
}

// ----------------------------------------------------------------------------
// launchers

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, sycl::queue& q, float* tmp_fp32) {
    NVTX_RANGE_FN();
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    const int HS = C / NH; // head size

    floatX *qq_, *kk, *vv;
    qq_ = qkvr + 0 * (size_t)B * T * C;
    kk = qkvr + 1 * (size_t)B * T * C;
    vv = qkvr + 2 * (size_t)B * T * C;
    permute(qq_, kk, vv, inp, B, T, NH, HS, q);

    floatX* preatt = inp; // reuse inp as scratch buffer
    // preatt(b) (T x T) = Q(b) (T x HS) * K(b)^T : m=T,n=T,k=HS, transA=false, transB=true
    matmul_batched_onemath(preatt, qq_, kk, T, T, HS, false, true,
        (size_t)T*T, (size_t)T*HS, (size_t)T*HS, B*NH, false, tmp_fp32, q);

    float scale = 1.f / sqrtf((float)HS);
    softmax_forward(att, scale, preatt, B * NH, T, q);

    floatX* vaccum = inp;
    // vaccum(b) (T x HS) = Att(b) (T x T) * V(b) (T x HS): m=T,n=HS,k=T, N,N
    matmul_batched_onemath(vaccum, att, vv, T, HS, T, false, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_fp32, q);

    unpermute(vaccum, out, B, T, NH, HS, q);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
inline void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, sycl::queue& q, float* tmp_fp32) {
    NVTX_RANGE_FN();
    (void)scratch;
    const int HS = C / NH; // head size

    const floatX *qq_, *kk, *vv;
    qq_ = qkvr + 0 * (size_t)B * T * C;
    kk = qkvr + 1 * (size_t)B * T * C;
    vv = qkvr + 2 * (size_t)B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * (size_t)B * T * C;
    dk = dqkvr + 1 * (size_t)B * T * C;
    dv = dqkvr + 2 * (size_t)B * T * C;

    // backward through the unpermute operation
    unpermute_backward(dqkvr /*reuse as scratch*/, dout, B, T, NH, HS, q);
    floatX* scratch_b = dqkvr; // (B,NH,T,HS) == (B,T,C)
    // backward into datt: datt(b) (T x T) = scratch(b) (T x HS) * V(b)^T : m=T,n=T,k=HS, N,T
    matmul_batched_onemath(datt, scratch_b, vv, T, T, HS, false, true,
        (size_t)T*T, (size_t)T*HS, (size_t)T*HS, B*NH, false, tmp_fp32, q);
    // backward into dv: dv(b) (T x HS) = Att(b)^T * scratch(b): m=T,n=HS,k=T with A transposed:
    // dv = Att^T @ scratch => m=T,n=HS,k=T, transA=true, transB=false
    matmul_batched_onemath(dv, att, scratch_b, T, HS, T, true, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_fp32, q);
    const float scale = 1.0f / sqrtf((float)HS);
    softmax_autoregressive_backward_inplace(datt, att, B, T, NH, scale, q);
    const floatX* dpreatt = datt;
    // backward into q: dq(b) (T x HS) = dpreatt(b) * K(b): m=T,n=HS,k=T, N,N
    matmul_batched_onemath(dq, dpreatt, kk, T, HS, T, false, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_fp32, q);
    // backward into k: dk(b) (T x HS) = dpreatt(b)^T * Q(b): m=T,n=HS,k=T, T,N
    matmul_batched_onemath(dk, dpreatt, qq_, T, HS, T, true, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_fp32, q);
    // backward into inp
    permute_backward(dinp, dq, dk, dv, B, T, NH, HS, q);
}

#endif
