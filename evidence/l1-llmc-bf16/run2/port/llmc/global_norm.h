/*
Global norm, used in gradient clipping (SYCL port of global_norm.cuh).
Deterministic: two-stage sum (per-shard partials, then ordered final sum).
*/
#ifndef GLOBAL_NORM_SYCL_H
#define GLOBAL_NORM_SYCL_H

#include <assert.h>
#include <stddef.h>
#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// maximum number of partial sums (single group can hold this)
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    int max_num_block_sums = 0;
    for (int i = 0; i < numel; i++) {
        int num_slices = num_slices_all[i];
        max_num_block_sums = (num_slices > max_num_block_sums) ? num_slices : max_num_block_sums;
    }
    if (max_num_block_sums < 1) { max_num_block_sums = 1; }
    return max_num_block_sums * 64; // match usage: out has room for gx*gy entries; we use num_slices*64
}

template<typename T>
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride,
                                int num_slices, int max_num_block_sums, bool reset, sycl::queue& q_) {
    // partial sums: num_slices x 64
    const int P = 64;
    if (reset) {
        q_.memset(out, 0, (size_t)max_num_block_sums * sizeof(float)).wait();
    }
    size_t total = (size_t)num_slices * P;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(total, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= total) { return; }
        size_t slice = lin / P;
        size_t pi = lin % P;
        const T* data = values + slice * (size_t)stride;
        float acc = 0.f;
        for (size_t i = pi; i < count; i += P) {
            float v = (float)data[i];
            acc += v * v;
        }
        out[slice * P + pi] = out[slice * P + pi] + acc;
    }).wait();
}

#endif
