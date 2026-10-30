/*
Attention (ported from llmc/attention.cuh to SYCL 2020).
Own non-flash attention kernels (llm.c default): permute, batched GEMMs,
causal online softmax, unpermute; plus the full backward.
*/
#ifndef SYCL_ATTENTION_HPP
#define SYCL_ATTENTION_HPP

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"
#include "matmul.hpp"

// ----------------------------------------------------------------------------
// helpers: permute / unpermute

// Q,K,V of shape (B, NH, N, d) from inp (B, N, 3, NH, d):
// Q[b][nh][n][dd] = inp[((b*N+n)*3+0)*NH*d + nh*d + dd] (same for K/V with 1/2)
inline void permute_qkv(floatX* q, floatX* k, floatX* v, const floatX* inp,
                        int B, int N, int NH, int d, sycl::queue& q_) {
    size_t total = size_t(B) * NH * N * d;
    if (total == 0) return;
    q_.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / (size_t)(NH * N * d);
        size_t rest = idx % (size_t)(NH * N * d);
        size_t nh = rest / (size_t)(N * d);
        rest = rest % (size_t)(N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t inp_idx = ((b * (size_t)N + n) * 3 * (size_t)NH * d) + (nh * (size_t)d) + dd;
        floatX qv = inp[inp_idx];
        floatX kv = inp[inp_idx + NH * d];
        floatX vv = inp[inp_idx + 2 * NH * d];
        q[idx] = qv; k[idx] = kv; v[idx] = vv;
    }).wait();
}

inline void permute_backward(floatX* dinp, const floatX* dq, const floatX* dk, const floatX* dv,
                             int B, int N, int NH, int d, sycl::queue& q_) {
    size_t total = size_t(B) * NH * N * d;
    if (total == 0) return;
    q_.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / (size_t)(NH * N * d);
        size_t rest = idx % (size_t)(NH * N * d);
        size_t nh = rest / (size_t)(N * d);
        rest = rest % (size_t)(N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t inp_idx = ((b * (size_t)N + n) * 3 * (size_t)NH * d) + (nh * (size_t)d) + dd;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * NH * d] = dv[idx];
    }).wait();
}

// out (B, N, NH, d) from inp (B, NH, N, d):
// out[b][n][nh][dd] <- inp[b][nh][n][dd]
// (CUDA names the first arg `inp` = vaccum (B,NH,T,HS) and second `out`;
//  idx decomposes the INPUT, other_idx the OUTPUT.)
inline void unpermute_forward(floatX* out, const floatX* inp,
                              int B, int N, int NH, int d, sycl::queue& q_) {
    size_t total = size_t(B) * NH * N * d;
    if (total == 0) return;
    q_.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        // decompose idx as INPUT (B, NH, N, d)
        size_t b = idx / (size_t)(NH * N * d);
        size_t rest = idx % (size_t)(NH * N * d);
        size_t nh = rest / (size_t)(N * d);
        rest = rest % (size_t)(N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        // OUTPUT (B, N, NH, d): ((b*N + n)*NH + nh)*d + dd
        size_t other_idx = ((b * (size_t)N + n) * NH + nh) * (size_t)d + dd;
        out[other_idx] = inp[idx];
    }).wait();
}

inline void unpermute_backward(floatX* dinp, const floatX* dout,
                               int B, int N, int NH, int d, sycl::queue& q_) {
    size_t total = size_t(B) * NH * N * d;
    if (total == 0) return;
    q_.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / (size_t)(NH * N * d);
        size_t rest = idx % (size_t)(NH * N * d);
        size_t nh = rest / (size_t)(N * d);
        rest = rest % (size_t)(N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t other_idx = ((b * (size_t)N + n) * NH + nh) * (size_t)d + dd;
        dinp[idx] = (floatX)dout[other_idx];
    }).wait();
}

// ----------------------------------------------------------------------------
// causal softmax forward: out[N*T, T] rows, one work-group per row

inline void softmax_forward(floatX* out, float inv_temperature, const floatX* inp,
                            int rows, int T, sycl::queue& q_) {
    if (rows <= 0) return;
    constexpr size_t LS = 128;
    floatX* o = out; const floatX* x = inp;
    float invT = inv_temperature;
    q_.parallel_for(sycl::nd_range<1>(size_t(rows) * LS, LS), [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        auto grp = it.get_group();
        size_t own_pos = row % (size_t)T;
        const floatX* xr = x + row * (size_t)T;
        // online softmax over [0..own_pos], causal
        float maxval = -INFINITY;
        float sumval = 0.f;
        for (size_t i = lid; i <= own_pos; i += LS) {
            float v = (float)xr[i];
            float old = maxval;
            maxval = sycl::fmax(maxval, v);
            sumval *= sycl::exp(invT * (old - maxval));
            sumval += sycl::exp(invT * (v - maxval));
        }
        float gmax = sycl::reduce_over_group(grp, maxval, sycl::maximum<float>());
        sumval *= sycl::exp(invT * (maxval - gmax));
        float gsum = sycl::reduce_over_group(grp, sumval, sycl::plus<float>());
        float norm = 1.f / gsum;
        floatX* orow = o + row * (size_t)T;
        for (size_t i = lid; i <= own_pos; i += LS) {
            float ev = sycl::exp(invT * ((float)xr[i] - gmax));
            orow[i] = (floatX)(ev * norm);
        }
    }).wait();
}

// ----------------------------------------------------------------------------
// launchers

inline void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, sycl::queue& q_) {
    NVTX_RANGE_FN();
    const int HS = C / NH;
    // qkvr is (B, T, 3*C): row r holds Q-part-written per (head,hs) interleave.
    // permute_qkv scatters it into Q,K,V blocks of shape (B,NH,T,HS).
    floatX *qq, *kk, *vv;
    // NOTE: qkvr aliases as Q/K/V block storage: Q block = first B*T*C elements
    // (= (B,NH,T,HS)), K block next, V block last. inp is the SEPARATE QKV
    // (B,T,3C) scratch holding the matmul output to be permuted in.
    // Q/K/V blocks are (B*NH*T*HS) == (B*T*C) elements each. Correct.
    qq = qkvr + 0 * size_t(B) * T * C;
    kk = qkvr + 1 * size_t(B) * T * C;
    vv = qkvr + 2 * size_t(B) * T * C;
    permute_qkv(qq, kk, vv, inp, B, T, NH, HS, q_);

    floatX* preatt = inp; // reuse inp as scratch buffer
    // preatt[b, i, j] = sum_hs q[b,h,i,hs] * k[b,h,j,hs]; per-batch (M=T,N=T,K=HS)
    // NOTE: q,k are (B,NH,T,HS); batch index b runs over B*NH heads.
    // Q[b] rows are query positions i, K[b] rows are key positions j.
    // preatt aliases inp (the QKV (B,T,3C) buffer, no longer needed), while
    // Q/K/V live in the separate qkvr buffer, so this is safe with no copy.
    {
        int batch = B * NH;
        // q layout per batch: (T,HS) row-major; k per batch: (T,HS) row-major
        // preatt[b](i,j) = sum_hs q[b](i,hs)*k[b](j,hs)
        const floatX* Q = qq; const floatX* K = kk; floatX* P = preatt;
        size_t sQ = size_t(T) * HS, sK = size_t(T) * HS, sP = size_t(T) * T;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, T), [=](sycl::id<2> id) {
            size_t bt = id[0]; int j = int(id[1]);
            int b = int(bt / (size_t)T), i = int(bt % (size_t)T);
            const floatX* Qb = Q + size_t(b) * sQ;
            const floatX* Kb = K + size_t(b) * sK;
            floatX* Pb = P + size_t(b) * sP;
            float acc = 0.f;
            for (int hs = 0; hs < HS; hs++) {
                acc += (float)Qb[size_t(i) * HS + hs] * (float)Kb[size_t(j) * HS + hs];
            }
            Pb[size_t(i) * T + j] = (floatX)acc;
        }).wait();
    }

    float scale = 1.f / sqrtf((float)HS);
    softmax_forward(att, scale, preatt, B * NH * T, T, q_);

    // vaccum[b](i,hs) = sum_j att[b](i,j) * v[b](j,hs)
    // vaccum aliases inp (already consumed: preatt->att done). V points into
    // qkvr (separate buffer) so no read/write hazard with O. Write directly.
    // NOTE: batch stride: att batch b covers heads of batch b/NH... heads are
    // flattened as b = batch_idx (0..B*NH-1); V batch b is at vv + b*T*HS. Same.
    {
        floatX* vaccum = inp;
        int batch = B * NH;
        const floatX* A = att; const floatX* V = vv; floatX* O = vaccum;
        size_t sA = size_t(T) * T, sV = size_t(T) * HS, sO = size_t(T) * HS;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, HS), [=](sycl::id<2> id) {
            size_t bt = id[0]; int hs = int(id[1]);
            int b = int(bt / (size_t)T), i = int(bt % (size_t)T);
            const floatX* Ab = A + size_t(b) * sA;
            const floatX* Vb = V + size_t(b) * sV;
            floatX* Ob = O + size_t(b) * sO;
            float acc = 0.f;
            for (int j = 0; j < T; j++) {
                acc += (float)Ab[size_t(i) * T + j] * (float)Vb[size_t(j) * HS + hs];
            }
            Ob[size_t(i) * HS + hs] = (floatX)acc;
        }).wait();
        unpermute_forward(out, vaccum, B, T, NH, HS, q_);
    }
}

inline void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, sycl::queue& q_) {
    NVTX_RANGE_FN();
    const int HS = C / NH;

    const floatX *qq, *kk, *vv;
    qq = qkvr + 0 * size_t(B) * T * C;
    kk = qkvr + 1 * size_t(B) * T * C;
    vv = qkvr + 2 * size_t(B) * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * size_t(B) * T * C;
    dk = dqkvr + 1 * size_t(B) * T * C;
    dv = dqkvr + 2 * size_t(B) * T * C;

    // backward through unpermute: scratch (B,NH,T,HS) = unpermute^T(dout)
    unpermute_backward(scratch, dout, B, T, NH, HS, q_);

    int batch = B * NH;
    // datt[b](i,j) = sum_hs v[b](j,hs) * scratch[b](i,hs)
    {
        const floatX* V = vv; const floatX* S = scratch; floatX* DA = datt;
        size_t sV = size_t(T) * HS, sS = size_t(T) * HS, sDA = size_t(T) * T;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, T), [=](sycl::id<2> id) {
            size_t bt = id[0]; int j = int(id[1]);
            int b = int(bt / (size_t)T), i = int(bt % (size_t)T);
            const floatX* Vb = V + size_t(b) * sV;
            const floatX* Sb = S + size_t(b) * sS;
            floatX* DAb = DA + size_t(b) * sDA;
            float acc = 0.f;
            for (int hs = 0; hs < HS; hs++) {
                acc += (float)Vb[size_t(j) * HS + hs] * (float)Sb[size_t(i) * HS + hs];
            }
            DAb[size_t(i) * T + j] = (floatX)acc;
        }).wait();
    }
    // dv[b](j,hs) = sum_i att[b](i,j) * scratch[b](i,hs)
    {
        const floatX* S = scratch; const floatX* A = att; floatX* DV = dv;
        size_t sS = size_t(T) * HS, sA = size_t(T) * T, sDV = size_t(T) * HS;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, HS), [=](sycl::id<2> id) {
            size_t bt = id[0]; int hs = int(id[1]);
            int b = int(bt / (size_t)T), j = int(bt % (size_t)T);
            const floatX* Sb = S + size_t(b) * sS;
            const floatX* Ab = A + size_t(b) * sA;
            floatX* DVb = DV + size_t(b) * sDV;
            float acc = 0.f;
            for (int i = 0; i < T; i++) {
                acc += (float)Ab[size_t(i) * T + j] * (float)Sb[size_t(i) * HS + hs];
            }
            DVb[size_t(j) * HS + hs] = (floatX)acc;
        }).wait();
    }
    // softmax backward in place: datt -> dpreatt (causal), scale = 1/sqrt(HS)
    {
        const float scale = 1.0f / sqrtf((float)HS);
        floatX* DA = datt; const floatX* A = att;
        size_t sM = size_t(T) * T;
        q_.parallel_for(sycl::range<1>(size_t(batch) * T), [=](sycl::id<1> id) {
            size_t bt = id[0];
            int b = int(bt / (size_t)T), t = int(bt % (size_t)T);
            const floatX* ab = A + size_t(b) * sM + size_t(t) * T;
            floatX* db = DA + size_t(b) * sM + size_t(t) * T;
            float dot = 0.f;
            for (int j = 0; j <= t; j++) { dot += (float)ab[j] * (float)db[j]; }
            for (int j = 0; j < T; j++) {
                if (j <= t) {
                    float acc = (float)ab[j] * ((float)db[j] - dot);
                    db[j] = (floatX)(scale * acc);
                } else {
                    db[j] = (floatX)0.f;
                }
            }
        }).wait();
    }
    const floatX* dpreatt = datt;
    // dq[b](i,hs) = sum_j k[b](j,hs) * dpreatt[b](i,j)
    {
        const floatX* K = kk; const floatX* DP = dpreatt; floatX* DQ = dq;
        size_t sK = size_t(T) * HS, sDP = size_t(T) * T, sDQ = size_t(T) * HS;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, HS), [=](sycl::id<2> id) {
            size_t bt = id[0]; int hs = int(id[1]);
            int b = int(bt / (size_t)T), i = int(bt % (size_t)T);
            const floatX* Kb = K + size_t(b) * sK;
            const floatX* DPb = DP + size_t(b) * sDP;
            floatX* DQb = DQ + size_t(b) * sDQ;
            float acc = 0.f;
            for (int j = 0; j < T; j++) {
                acc += (float)Kb[size_t(j) * HS + hs] * (float)DPb[size_t(i) * T + j];
            }
            DQb[size_t(i) * HS + hs] = (floatX)acc;
        }).wait();
    }
    // dk[b](j,hs) = sum_i q[b](i,hs) * dpreatt[b](i,j)
    {
        const floatX* Q = qq; const floatX* DP = dpreatt; floatX* DK = dk;
        size_t sQ = size_t(T) * HS, sDP = size_t(T) * T, sDK = size_t(T) * HS;
        q_.parallel_for(sycl::range<2>(size_t(batch) * T, HS), [=](sycl::id<2> id) {
            size_t bt = id[0]; int hs = int(id[1]);
            int b = int(bt / (size_t)T), j = int(bt % (size_t)T);
            const floatX* Qb = Q + size_t(b) * sQ;
            const floatX* DPb = DP + size_t(b) * sDP;
            floatX* DKb = DK + size_t(b) * sDK;
            float acc = 0.f;
            for (int i = 0; i < T; i++) {
                acc += (float)Qb[size_t(i) * HS + hs] * (float)DPb[size_t(i) * T + j];
            }
            DKb[size_t(j) * HS + hs] = (floatX)acc;
        }).wait();
    }
    permute_backward(dinp, dq, dk, dv, B, T, NH, HS, q_);
}

#endif
