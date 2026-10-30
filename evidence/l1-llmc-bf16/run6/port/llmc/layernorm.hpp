/*
LayerNorm + Residual kernels (ported from llmc/layernorm.cuh to SYCL 2020).
1-D launches (one work item per row for forward, per element/row for backward).
Reductions use SYCL group algorithms (reduce_over_group with plus/f maximum).
*/
#ifndef SYCL_LAYERNORM_HPP
#define SYCL_LAYERNORM_HPP

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// forward: one work-group per row, sub-group reduction via group algorithms

inline void layernorm_forward(floatX* out, float* mean, float* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = size_t(B) * T;
    if (N == 0) return;
    // one work-group per row; 32-wide sub-groups assumed, use local size 128
    constexpr size_t LS = 128;
    floatX* o = out; floatX* x = inp;
    const floatX* w = weight; const floatX* b = bias;
    float* m = mean; float* r = rstd;
    const float eps = 1e-5f;
    q.parallel_for(sycl::nd_range<1>(N * LS, LS), [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        const floatX* xr = x + row * (size_t)C;
        // mean
        float sum = 0.f;
        for (int c = int(lid); c < C; c += int(LS)) { sum += (float)xr[c]; }
        auto grp = it.get_group();
        float gsum = sycl::reduce_over_group(grp, sum, sycl::plus<float>());
        float mm = gsum / (float)C;
        // var
        float vsum = 0.f;
        for (int c = int(lid); c < C; c += int(LS)) {
            float d = (float)xr[c] - mm;
            vsum += d * d;
        }
        float gvar = sycl::reduce_over_group(grp, vsum, sycl::plus<float>());
        float s = sycl::rsqrt(gvar / (float)C + eps);
        if (lid == 0) {
            if (m != nullptr) m[row] = mm;
            if (r != nullptr) r[row] = s;
        }
        floatX* orow = o + row * (size_t)C;
        for (int c = int(lid); c < C; c += int(LS)) {
            float n = s * ((float)xr[c] - mm);
            orow[c] = (floatX)(n * (float)w[c] + (float)b[c]);
        }
    }).wait();
}

inline void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    if (N == 0) return;
    floatX* o = out; const floatX* a = inp1; const floatX* b = inp2;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t i = id[0];
        o[i] = (floatX)((float)a[i] + (float)b[i]);
    }).wait();
}

inline void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    if (N <= 0) return;
    // residual = inp1 + inp2 (1-D), then layernorm(residual)
    residual_forward(residual, inp1, inp2, size_t(N) * C, q);
    layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, q);
}

// ----------------------------------------------------------------------------
// backward: deterministic, matches layernorm_backward math.
//
// CUDA kernel10 accumulates dweight/dbias in fp32 shared memory over all rows
// then adds old values with RN conversion. dinp[row,c] += f(row,c).
// We compute per-row terms with two-pass (deterministic order) and accumulate
// parameter grads via ordered host reduction over per-row partials.

inline void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const float* mean, const float* rstd,
                        int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    (void)scratch;
    size_t N = size_t(B) * T;
    if (N == 0) return;
    // Pass 1: per-row dnorm_mean, dnorm_norm_mean (sequential, deterministic)
    float* row_a = sycl::malloc_device<float>(N, q); // dnorm_mean
    float* row_b = sycl::malloc_device<float>(N, q); // dnorm_norm_mean
    {
        floatX* di = dinp; (void)di;
        const floatX* dout_ = dout; const floatX* inp_ = inp; const floatX* w_ = weight;
        const float* mean_ = mean; const float* rstd_ = rstd;
        float* ra = row_a; float* rb = row_b;
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
            size_t row = id[0];
            const floatX* dr = dout_ + row * (size_t)C;
            const floatX* xr = inp_ + row * (size_t)C;
            float dm = 0.f, dnm = 0.f;
            for (int c = 0; c < C; c++) {
                float dnorm = (float)w_[c] * (float)dr[c];
                dm += dnorm;
                dnm += dnorm * (float)xr[c];
            }
            float rs = rstd_[row], mn = mean_[row];
            ra[row] = dm / (float)C;
            rb[row] = dnm / (float)C * rs - (dm / (float)C) * mn * rs;
        }).wait();
    }
    // Pass 2: dinp += ..., per-row partial sums for dweight/dbias in fp32 device buffers
    float* part_w = sycl::malloc_device<float>(N * size_t(C), q);
    float* part_b = sycl::malloc_device<float>(N * size_t(C), q);
    {
        floatX* di = dinp;
        const floatX* dout_ = dout; const floatX* inp_ = inp; const floatX* w_ = weight;
        const float* mean_ = mean; const float* rstd_ = rstd;
        const float* ra = row_a; const float* rb = row_b;
        float* pw = part_w; float* pb = part_b;
        q.parallel_for(sycl::range<2>(N, size_t(C)), [=](sycl::id<2> id) {
            size_t row = id[0]; int c = int(id[1]);
            float dout_c = (float)dout_[row * (size_t)C + c];
            float norm = ((float)inp_[row * (size_t)C + c] - mean_[row]) * rstd_[row];
            pb[row * (size_t)C + c] = dout_c;
            pw[row * (size_t)C + c] = norm * dout_c;
            float dval = (float)w_[c] * dout_c - ra[row] - norm * rb[row];
            dval *= rstd_[row];
            di[row * (size_t)C + c] = (floatX)((float)di[row * (size_t)C + c] + dval);
        }).wait();
    }
    // Pass 3: deterministic ordered reduction over rows on host, RN add of old grads
    {
        float* pw_h = (float*)malloc(N * size_t(C) * sizeof(float));
        float* pb_h = (float*)malloc(N * size_t(C) * sizeof(float));
        q.memcpy(pw_h, part_w, N * size_t(C) * sizeof(float)).wait();
        q.memcpy(pb_h, part_b, N * size_t(C) * sizeof(float)).wait();
        float* sw = (float*)calloc(size_t(C), sizeof(float));
        float* sb = (float*)calloc(size_t(C), sizeof(float));
        for (size_t row = 0; row < N; row++) {
            for (int c = 0; c < C; c++) {
                sb[c] += pb_h[row * C + c];
                sw[c] += pw_h[row * C + c];
            }
        }
        free(pw_h); free(pb_h);
        float* sw_d = sycl::malloc_device<float>(C, q);
        float* sb_d = sycl::malloc_device<float>(C, q);
        q.memcpy(sw_d, sw, size_t(C) * sizeof(float)).wait();
        q.memcpy(sb_d, sb, size_t(C) * sizeof(float)).wait();
        free(sw); free(sb);
        floatX* dw = dweight; floatX* db = dbias;
        q.parallel_for(sycl::range<1>(size_t(C)), [=](sycl::id<1> id) {
            int c = int(id[0]);
            db[c] = (floatX)(sb_d[c] + (float)db[c]);
            dw[c] = (floatX)(sw_d[c] + (float)dw[c]);
        }).wait();
        sycl::free(sw_d, q); sycl::free(sb_d, q);
    }
    sycl::free(row_a, q); sycl::free(row_b, q);
    sycl::free(part_w, q); sycl::free(part_b, q);
}

#endif
