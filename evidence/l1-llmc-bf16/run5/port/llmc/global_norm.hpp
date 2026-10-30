/*
Global norm, used in gradient clipping. SYCL port of global_norm.cuh.
Deterministic: ordered accumulation over the gradient vector.
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
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, sycl::queue& q) {
    (void)max_num_block_sums;
    q.wait();
    if (reset) {
        for (int s = 0; s < num_slices; s++) { out[s] = 0.0f; }
    }
    for (int s = 0; s < num_slices; s++) {
        const T* v = values + (ptrdiff_t)s * stride;
        double acc = 0.0;
        for (size_t i = 0; i < count; i++) {
            double x = (double)(float)v[i];
            acc += x * x;
        }
        out[s] += (float)acc;
    }
}

#endif
