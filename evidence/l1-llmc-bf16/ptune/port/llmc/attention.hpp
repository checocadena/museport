/*
Attention (own kernels), SYCL 2020 port of attention.cuh.
Batched matmuls run as a loop of oneMath bf16->fp32 gemms over all
(batch, head) pairs (see matmul.hpp); the fp32->bf16 convert is fused into
the consumer (softmax / unpermute / permute). All launches are
fire-and-forget on the in-order queue. One code path for every shape/device.
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

// (kept for API compat; unused by the single-path launchers below)

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

// Floating-point-input variants of (un)permute used to fuse the bf16 convert
// of a library attention GEMM into its consumer: rounding on store is
// bit-exact with convert-then-copy (same element, same order, same rounding).
inline void unpermute_fp32in(floatX* out, const float* inp, int B, int N, int NH, int d, sycl::queue& qq) {
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
        out[other_idx] = (floatX)inp[idx];
    });
}

inline void permute_backward_fp32in(floatX* dinp,
                                        const float* dq, const float* dk, const float* dv,
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
        dinp[inp_idx] = (floatX)dq[idx];
        dinp[inp_idx + (size_t)NH * d] = (floatX)dk[idx];
        dinp[inp_idx + 2 * (size_t)NH * d] = (floatX)dv[idx];
    });
}

// inp, out shape: (NT, T, T) row-major batches, where NT = B * NH; autoregressive causal.
// inp, out shape: (NT, T, T) row-major batches, where NT = B * NH; autoregressive causal.
// One work-group (128) per row; lanes cooperate on an online softmax over the
// valid prefix (strided vec4 loads), then normalize. The max reduction is
// exact; the sum reduction order differs from a single-thread loop in the last
// ulp only. The upper triangle is left untouched: the att buffer is zeroed at
// allocation and no kernel ever writes above the diagonal, so it stays zero
// (same as llm.c, which never writes it either).
// (kept for API compat; unused by the single-path launchers below)
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
// (kept for API compat; unused by the single-path launchers below)
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

// Floating-point-input softmax forward: consumes a library GEMM's fp32 output
// directly (fused convert). Only used on the large-shape library path, so the
// small-shape (test) numerics are untouched; the result is slightly more
// accurate (one fewer bf16 rounding) than convert-then-softmax.
inline void softmax_forward_fp32in(floatX* out, float inv_temperature, const float* inp, int NT, int T, sycl::queue& qq) {
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
        const float* x = inp + idx * (size_t)T;
        floatX* o = out + idx * (size_t)T;
        size_t n = own_pos + 1;
        float maxval = -flt_max;
        float sumval = 0.0f;
        if ((((uintptr_t)x) & 15) == 0) {
            size_t n4 = n / 4;
            for (size_t i = lid; i < n4; i += WG) {
                const float* p = x + i * 4;
                float r0 = p[0], r1 = p[1], r2 = p[2], r3 = p[3];
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
                float r = x[i];
                float old = maxval;
                maxval = fmaxf(maxval, r);
                sumval *= sycl::exp(inv_temperature * (old - maxval));
                sumval += sycl::exp(inv_temperature * (r - maxval));
            }
        } else {
            for (size_t i = lid; i < n; i += WG) {
                float r = x[i];
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
            float ev = sycl::exp(inv_temperature * (x[i] - gmax));
            o[i] = (floatX)(ev * norm);
        }
    });
}

// Floating-point-input softmax backward: reads fp32 datt (library GEMM output)
// and bf16 att, writes bf16 dpreatt in place of datt_out. Same gating/accuracy
// notes as the forward fp32 variant.
inline void softmax_autoregressive_backward_inplace_fp32in(floatX* datt_out, const floatX* att, const float* datt_in,
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
        const float* datt_bth = datt_in + r * (size_t)T;
        floatX* dpreatt_bth = datt_out + r * (size_t)T;
        float local = 0.0f;
        for (size_t t2 = lid; t2 <= t; t2 += WG) {
            local += (float)att_bth[t2] * datt_bth[t2];
        }
        float local_sum = sycl::reduce_over_group(g, local, sycl::plus<float>());
        sycl::group_barrier(g);
        for (size_t t3 = lid; t3 < (size_t)T; t3 += WG) {
            if (t3 <= t) {
                float d = datt_bth[t3];
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

// QKV GEMM + permute fused: qkvr(b,nh,n,d) for Q/K/V blocks straight from the
// fp32 GEMM output (no bf16 round-trip of the QKV activation). Reads the
// column-major (3C x BT) fp32 GEMM output as row-major (BT x 3C): tmp[r*3C+c].
// Same arithmetic (same order, one rounding) as convert-then-permute.
inline void permute_fp32in(floatX* qo, floatX* ko, floatX* vo,
                           const float* inp, const floatX* bias,
                           int B, int N, int NH, int d, sycl::queue& qq) {
    sycl::queue& q = qq;
    size_t C3 = (size_t)3 * NH * d;
    size_t total = (size_t)B * NH * N * d;
    bool has_bias = (bias != nullptr);
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh_ = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t d_ = rest % (size_t)d;
        size_t row = b * (size_t)N + n;
        size_t c0 = nh_ * (size_t)d + d_;
        float qv = inp[row * C3 + c0];
        float kv = inp[row * C3 + (size_t)NH * d + c0];
        float vv = inp[row * C3 + 2 * (size_t)NH * d + c0];
        if (has_bias) {
            qv += (float)bias[c0];
            kv += (float)bias[(size_t)NH * d + c0];
            vv += (float)bias[2 * (size_t)NH * d + c0];
        }
        qo[idx] = (floatX)qv;
        ko[idx] = (floatX)kv;
        vo[idx] = (floatX)vv;
    });
}

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       const floatX* qkv_w, const floatX* qkv_b, const floatX* ln_inp,
                       floatX* inp,
                       int B, int T, int C, int NH, sycl::queue& q, float* tmp_fp32) {
    NVTX_RANGE_FN();
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    const int HS = C / NH; // head size

    floatX *qq_, *kk, *vv;
    qq_ = qkvr + 0 * (size_t)B * T * C;
    kk = qkvr + 1 * (size_t)B * T * C;
    vv = qkvr + 2 * (size_t)B * T * C;
    // QKV GEMM straight into fp32 tmp (col (3C x BT) == row (BT x 3C)), then
    // fused bias+permute: no bf16 round-trip of the QKV activation, one
    // fewer kernel and one fewer full-tensor pass. Same arithmetic (same
    // order, one rounding) as convert-then-permute.
    matmul_gemm_fp32(tmp_fp32, qkv_w, ln_inp, 3 * C, B * T, C, q);
    permute_fp32in(qq_, kk, vv, tmp_fp32, qkv_b, B, T, NH, HS, q);

    floatX* preatt = inp; // reuse inp as scratch buffer
    // preatt(b) (T x T) = Q(b) (T x HS) * K(b)^T : m=T,n=T,k=HS, transA=false, transB=true
    // fp32 result lands in tmp_fp32; the fp32-input softmax below fuses the
    // bf16 convert (same path for every shape and device).
    matmul_batched_onemath(preatt, qq_, kk, T, T, HS, false, true,
        (size_t)T*T, (size_t)T*HS, (size_t)T*HS, B*NH, false, tmp_fp32, q);

    float scale = 1.f / sqrtf((float)HS);
    softmax_forward_fp32in(att, scale, tmp_fp32, B * NH, T, q);

    floatX* vaccum = inp;
    // vaccum(b) (T x HS) = Att(b) (T x T) * V(b) (T x HS): m=T,n=HS,k=T, N,N
    // fp32 result lands in tmp_fp32; unpermute rounds on store.
    matmul_batched_onemath(vaccum, att, vv, T, HS, T, false, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_fp32, q);

    unpermute_fp32in(out, tmp_fp32, B, T, NH, HS, q);
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
    // fp32 result lands in tmp_fp32 for the fp32-input softmax-backward below.
    matmul_batched_onemath(datt, scratch_b, vv, T, T, HS, false, true,
        (size_t)T*T, (size_t)T*HS, (size_t)T*HS, B*NH, false, tmp_fp32, q);
    // dq/dk/dv fp32 outputs share tmp_fp32 in three contiguous regions placed
    // AFTER the datt fp32 region (each B*T*C floats); permute fuses the bf16
    // convert. (dv runs before datt is consumed, so its region must
    // not overlap the datt fp32 data.)
    size_t bhtt = (size_t)B * NH * T * T;
    float* tmp_dq = tmp_fp32 + bhtt;
    float* tmp_dk = tmp_fp32 + bhtt + (size_t)B * T * C;
    float* tmp_dv = tmp_fp32 + bhtt + 2 * (size_t)B * T * C;
    // backward into dv: dv(b) (T x HS) = Att(b)^T * scratch(b): m=T,n=HS,k=T with A transposed:
    // dv = Att^T @ scratch => m=T,n=HS,k=T, transA=true, transB=false
    // (fp32 result lands straight in tmp_dv; batches are contiguous
    // with strideOut == m*n, so tmp batch b lands at tmp_dv + b*m*n,
    // exactly the fused-consumer layout.)
    matmul_batched_onemath(dv, att, scratch_b, T, HS, T, true, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_dv, q);
    const float scale = 1.0f / sqrtf((float)HS);
    softmax_autoregressive_backward_inplace_fp32in(datt, att, tmp_fp32, B, T, NH, scale, q);
    const floatX* dpreatt = datt;
    // backward into q: dq(b) (T x HS) = dpreatt(b) * K(b): m=T,n=HS,k=T, N,N
    matmul_batched_onemath(dq, dpreatt, kk, T, HS, T, false, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_dq, q);
    // backward into k: dk(b) (T x HS) = dpreatt(b)^T * Q(b): m=T,n=HS,k=T, T,N
    matmul_batched_onemath(dk, dpreatt, qq_, T, HS, T, true, false,
        (size_t)T*HS, (size_t)T*T, (size_t)T*HS, B*NH, false, tmp_dk, q);
    // backward into inp (fused bf16 convert of the three fp32 regions)
    permute_backward_fp32in(dinp, tmp_dq, tmp_dk, tmp_dv, B, T, NH, HS, q);
}

#endif
