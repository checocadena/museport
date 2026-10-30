/*
Global norm, used in gradient clipping (ported from llmc/global_norm.cuh).
Deterministic: per-element squares accumulated in fixed order.
*/
#ifndef SYCL_GLOBAL_NORM_HPP
#define SYCL_GLOBAL_NORM_HPP

#include <assert.h>
#include <stddef.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------

// Helper function determines the maximum number of block sums (kept for API compat)
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    int mx = 0;
    for (int i = 0; i < numel; i++) { mx = std::max(mx, num_slices_all[i]); }
    return std::max(mx, 1);
}

template<typename T>
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, sycl::queue& q_) {
    if (reset) {
        q_.memset(out, 0, size_t(max_num_block_sums) * sizeof(float)).wait();
    }
    if (count == 0 || num_slices <= 0) return;
    // one partial sum per slice, deterministic accumulation order inside kernel
    const T* v = values; float* o = out;
    q_.parallel_for(sycl::range<1>(size_t(num_slices)), [=](sycl::id<1> id) {
        size_t s = id[0];
        const T* d = v + s * (size_t)stride;
        float acc = 0.f;
        for (size_t i = 0; i < count; i++) {
            float x = (float)d[i];
            acc += x * x;
        }
        o[s] = o[s] + acc;
    }).wait();
}

#endif
