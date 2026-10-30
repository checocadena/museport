/*
Global norm, used in gradient clipping. SYCL port of global_norm.cuh.
Deterministic: fixed-grid ordered partial sums per slice (in device memory),
then an ordered accumulation per slice. `tmp` needs >= num_slices*65536 floats
of device scratch. Fire-and-forget on the in-order queue.
*/
#ifndef SYCL_GLOBAL_NORM_H
#define SYCL_GLOBAL_NORM_H

#include <assert.h>
#include <stddef.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// launcher

// Helper function determines the maximum number of block sums (kept for API compat)
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    // SYCL port: single partial sum per call; return total slices as the bound.
    int max_num_block_sums = 0;
    for (int i = 0; i < numel; i++) {
        max_num_block_sums = std::max(max_num_block_sums, num_slices_all[i]);
    }
    return std::max(max_num_block_sums, 1);
}

// out[0..slices) accumulates sum of squares (each slice covers count elements
// at values + s*stride). Deterministic ordered accumulation over a fixed grid
// of G partial sums per slice (G depends only on count): thousands of workers
// feed an ordered tail, so the result is bit-deterministic run to run.
// `tmp` needs >= num_slices * G floats of device scratch (G <= 65536).
template<typename T>
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, sycl::queue& q, float* tmp) {
    (void)max_num_block_sums;
    constexpr int GMAX = 65536;
    if (num_slices <= 0) { return; }
    if (reset) {
        q.parallel_for(sycl::range<1>((size_t)num_slices), [=](sycl::id<1> id) {
            out[id[0]] = 0.0f;
        });
    }
    if (count == 0) { return; }
    int g = (count < (size_t)GMAX) ? (int)count : GMAX;
    size_t chunk = (count + (size_t)g - 1) / (size_t)g;
    float* partials = tmp; // num_slices * g floats of device scratch
    // vectorized per-worker accumulation (4 elements at a time; tail handles
    // the remainder) then ordered tail per slice in double precision.
    q.parallel_for(sycl::range<2>((size_t)num_slices, (size_t)g), [=](sycl::id<2> id) {
        size_t s = id[0];
        size_t j = id[1];
        const T* v = values + (ptrdiff_t)s * stride;
        size_t b = j * chunk;
        size_t e = b + chunk;
        if (e > count) { e = count; }
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        size_t i = b;
        size_t e4 = b + (e - b) / 4 * 4;
        for (; i < e4; i += 4) {
            float x0 = (float)v[i], x1 = (float)v[i+1], x2 = (float)v[i+2], x3 = (float)v[i+3];
            acc0 += x0 * x0; acc1 += x1 * x1; acc2 += x2 * x2; acc3 += x3 * x3;
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < e; i++) {
            float x = (float)v[i];
            acc += x * x;
        }
        partials[s * (size_t)g + j] = acc;
    });
    q.parallel_for(sycl::range<1>((size_t)num_slices), [=](sycl::id<1> id) {
        size_t s = id[0];
        double acc = (double)out[s];
        for (int j = 0; j < g; j++) { acc += (double)partials[s * (size_t)g + j]; }
        out[s] = (float)acc;
    });
}

#endif
