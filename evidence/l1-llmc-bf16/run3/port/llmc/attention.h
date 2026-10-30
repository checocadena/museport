/*
Attention (SYCL port of attention.cuh, non-flash path = llm.c default).
Uses oneMath batched GEMM with fp32 intermediates + SYCL softmax/permute kernels.
*/
#ifndef ATTENTION_SYCL_H
#define ATTENTION_SYCL_H

#include <assert.h>
#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
#include "blas_common.h"
#include "matmul.h"

// ----------------------------------------------------------------------------
// SYCL kernels

// split (B, T, 3*C) inp into q,k,v each (B, NH, T, HS)
inline void permute_qkv(floatX* q, floatX* k, floatX* v, const floatX* inp,
                        int B, int T, int NH, int HS, sycl::queue& q_) {
    size_t N = (size_t)B * NH * T * HS;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t d = HS;
        size_t b = idx / ((size_t)NH * T * d);
        size_t rest = idx % ((size_t)NH * T * d);
        size_t nh = rest / ((size_t)T * d);
        rest = rest % ((size_t)T * d);
        size_t n = rest / d;
        size_t dd = rest % d;
        size_t inp_idx = (b * (size_t)T * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + dd;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + (size_t)NH * d];
        v[idx] = inp[inp_idx + 2 * (size_t)NH * d];
    }).wait();
}

inline void permute_qkv_backward(floatX* dinp, const floatX* dq, const floatX* dk, const floatX* dv,
                                 int B, int T, int NH, int HS, sycl::queue& q_) {
    size_t N = (size_t)B * NH * T * HS;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t d = HS;
        size_t b = idx / ((size_t)NH * T * d);
        size_t rest = idx % ((size_t)NH * T * d);
        size_t nh = rest / ((size_t)T * d);
        rest = rest % ((size_t)T * d);
        size_t n = rest / d;
        size_t dd = rest % d;
        size_t inp_idx = (b * (size_t)T * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + dd;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + (size_t)NH * d] = dk[idx];
        dinp[inp_idx + 2 * (size_t)NH * d] = dv[idx];
    }).wait();
}

// vaccum (B, NH, T, HS) -> out (B, T, C)
inline void unpermute_out(floatX* out, const floatX* vaccum, int B, int T, int NH, int HS, sycl::queue& q_) {
    size_t N = (size_t)B * T * NH * HS;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t d = HS;
        size_t b = idx / ((size_t)NH * T * d);
        size_t rest = idx % ((size_t)NH * T * d);
        size_t nh = rest / ((size_t)T * d);
        rest = rest % ((size_t)T * d);
        size_t n = rest / d;
        size_t dd = rest % d;
        size_t other_idx = (b * (size_t)NH * T * d) + (n * NH * d) + (nh * d) + dd;
        out[other_idx] = vaccum[idx];
    }).wait();
}

inline void unpermute_backward(floatX* dinp, const floatX* dout, int B, int T, int NH, int HS, sycl::queue& q_) {
    size_t N = (size_t)B * T * NH * HS;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t d = HS;
        size_t b = idx / ((size_t)NH * T * d);
        size_t rest = idx % ((size_t)NH * T * d);
        size_t nh = rest / ((size_t)T * d);
        rest = rest % ((size_t)T * d);
        size_t n = rest / d;
        size_t dd = rest % d;
        size_t other_idx = (b * (size_t)NH * T * d) + (n * NH * d) + (nh * d) + dd;
        dinp[idx] = (floatX)dout[other_idx];
    }).wait();
}

// autoregressive causal softmax: out(N*T*T) = softmax(inp * scale), per row causal. inp fp32, out floatX.
inline void softmax_forward_sycl(floatX* out, float scale, const float* inp, int N, int T, sycl::queue& q_) {
    // one work-group per row
    constexpr size_t WG = 32;
    q_.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)N * T * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
        size_t idx = item.get_group(0);
        if (idx >= (size_t)N * T) { return; }
        size_t lid = item.get_local_id(0);
        auto grp = item.get_group();
        size_t own_pos = idx % (size_t)T;
        const float* x = inp + idx * (size_t)T;
        const float FLT_MAXF = 340282346638528859811704183484516925440.0f;
        float maxval = -FLT_MAXF;
        float sumval = 0.f;
        // online softmax over causal prefix
        for (size_t i = lid; i <= own_pos; i += WG) {
            float v = x[i] * scale;
            float old = maxval;
            maxval = sycl::fmax(maxval, v);
            sumval *= sycl::exp(old - maxval);
            sumval += sycl::exp(v - maxval);
        }
        float gmax = sycl::reduce_over_group(grp, maxval, sycl::maximum<float>());
        sumval *= sycl::exp(maxval - gmax);
        float gsum = sycl::reduce_over_group(grp, sumval, sycl::plus<float>());
        float norm = 1.f / gsum;
        floatX* o = out + idx * (size_t)T;
        for (size_t i = lid; i <= own_pos; i += WG) {
            float ev = sycl::exp(x[i] * scale - gmax);
            o[i] = (floatX)(ev * norm);
        }
    }).wait();
}

// datt (B*NH*T*T, floatX) inplace -> dpreatt; att floatX; scale applied.
inline void softmax_backward_sycl(floatX* datt, const floatX* att, int BNH, int T, float scale, sycl::queue& q_) {
    constexpr size_t WG = 32;
    size_t rows = (size_t)BNH * T;
    q_.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
        size_t idx = item.get_group(0);
        if (idx >= rows) { return; }
        size_t lid = item.get_local_id(0);
        auto grp = item.get_group();
        size_t t = idx % (size_t)T;
        const floatX* att_row = att + idx * (size_t)T;
        const floatX* datt_row = datt + idx * (size_t)T;
        float local = 0.f;
        for (size_t t2 = lid; t2 <= t; t2 += WG) {
            local += (float)att_row[t2] * (float)datt_row[t2];
        }
        local = sycl::reduce_over_group(grp, local, sycl::plus<float>());
        floatX* dp = datt + idx * (size_t)T;
        for (size_t t3 = lid; t3 < (size_t)T; t3 += WG) {
            if (t3 <= t) {
                float acc = (float)att_row[t3] * ((float)datt_row[t3] - local);
                dp[t3] = (floatX)(scale * acc);
            } else {
                dp[t3] = (floatX)0.f;
            }
        }
    }).wait();
}

// ----------------------------------------------------------------------------
// launchers

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                              floatX* inp, float* preatt, float* vaccumF,
                              int B, int T, int C, int NH, sycl::queue& q_) {
    NVTX_RANGE_FN();
    const int HS = C / NH;
    floatX *qq, *kk, *vv;
    qq = qkvr + 0 * (size_t)B * T * C;
    kk = qkvr + 1 * (size_t)B * T * C;
    vv = qkvr + 2 * (size_t)B * T * C;
    permute_qkv(qq, kk, vv, inp, B, T, NH, HS, q_);

    // preatt (fp32, dedicated staging)
    sycl_gemm_batch_preatt(preatt, kk, qq, T, HS, (int64_t)B * NH, q_);

    float scale = 1.f / sqrtf((float)HS);
    softmax_forward_sycl(att, scale, preatt, B * NH, T, q_);

    // vaccum: fp32 staging then convert to floatX reusing inp buffer
    floatX* vaccum = inp;
    sycl_gemm_batch_vaccum(vaccumF, vv, att, T, HS, (int64_t)B * NH, q_);
    {
        size_t N = (size_t)B * NH * T * HS;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            vaccum[i] = (floatX)vaccumF[i];
        }).wait();
    }
    unpermute_out(out, vaccum, B, T, NH, HS, q_);
}

inline void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                               float* dpreattF, float* tmpF,
                               const floatX* dout,
                               const floatX* qkvr, const floatX* att,
                               int B, int T, int C, int NH, sycl::queue& q_) {
    NVTX_RANGE_FN();
    const int HS = C / NH;
    const floatX *qq, *kk, *vv;
    qq = qkvr + 0 * (size_t)B * T * C;
    kk = qkvr + 1 * (size_t)B * T * C;
    vv = qkvr + 2 * (size_t)B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * (size_t)B * T * C;
    dk = dqkvr + 1 * (size_t)B * T * C;
    dv = dqkvr + 2 * (size_t)B * T * C;

    // scratch (floatX, B*T*C) holds unpermuted dout
    unpermute_backward(scratch, dout, B, T, NH, HS, q_);
    // datt (floatX BNH*T*T): fp32 staging in dpreattF, then convert
    sycl_gemm_batch_datt(dpreattF, vv, scratch, T, HS, (int64_t)B * NH, q_);
    {
        size_t N = (size_t)B * NH * T * T;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            datt[i] = (floatX)dpreattF[i];
        }).wait();
    }
#ifdef ATTN_DEBUG
#define ATTN_WTE_CHECK(tag) do {} while(0)
#else
#define ATTN_WTE_CHECK(tag) do {} while(0)
#endif
    // dv: fp32 staging in tmpF
    sycl_gemm_batch_dv(tmpF, scratch, att, T, HS, (int64_t)B * NH, q_);
    {
        size_t N = (size_t)B * NH * T * HS;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            dv[i] = (floatX)tmpF[i];
        }).wait();
    }
    const float scale = 1.0f / sqrtf((float)HS);
    softmax_backward_sycl(datt, att, B * NH, T, scale, q_);
    const floatX* dpreatt = datt;
    // dq: fp32 staging in tmpF
    sycl_gemm_batch_dq(tmpF, kk, dpreatt, T, HS, (int64_t)B * NH, q_);
    {
        size_t N = (size_t)B * NH * T * HS;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            dq[i] = (floatX)tmpF[i];
        }).wait();
    }
    // dk: fp32 staging in tmpF
    sycl_gemm_batch_dk(tmpF, qq, dpreatt, T, HS, (int64_t)B * NH, q_);
    {
        size_t N = (size_t)B * NH * T * HS;
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV(N, BLOCK);
        q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t i = id[0];
            if (i >= N) { return; }
            dk[i] = (floatX)tmpF[i];
        }).wait();
    }
    permute_qkv_backward(dinp, dq, dk, dv, B, T, NH, HS, q_);
}

#endif
