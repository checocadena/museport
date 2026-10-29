/*
Global norm, used in gradient clipping. SYCL port of global_norm.cuh.
Deterministic: fixed-grid ordered partial sums per slice (in device memory),
then an ordered accumulation per slice. `tmp` needs >= num_slices*256 floats
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
// at values + s*stride). Deterministic ordered accumulation.
template<typename T>
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, sycl::queue& q, float* tmp) {
    (void)max_num_block_sums;
    constexpr int P = 256;
    if (num_slices <= 0) { return; }
    if (reset) {
        q.parallel_for(sycl::range<1>((size_t)num_slices), [=](sycl::id<1> id) {
            out[id[0]] = 0.0f;
        });
    }
    if (count == 0) { return; }
    int p = (count < (size_t)P) ? (int)count : P;
    size_t chunk = (count + (size_t)p - 1) / (size_t)p;
    float* partials = tmp; // num_slices * p floats of device scratch
    q.parallel_for(sycl::range<2>((size_t)num_slices, (size_t)p), [=](sycl::id<2> id) {
        size_t s = id[0];
        size_t j = id[1];
        const T* v = values + (ptrdiff_t)s * stride;
        size_t b = j * chunk;
        size_t e = b + chunk;
        if (e > count) { e = count; }
        float acc = 0.0f;
        for (size_t i = b; i < e; i++) {
            float x = (float)v[i];
            acc += x * x;
        }
        partials[s * (size_t)p + j] = acc;
    });
    q.parallel_for(sycl::range<1>((size_t)num_slices), [=](sycl::id<1> id) {
        size_t s = id[0];
        double acc = (double)out[s];
        for (int j = 0; j < p; j++) { acc += (double)partials[s * (size_t)p + j]; }
        out[s] = (float)acc;
    });
}

#endif
