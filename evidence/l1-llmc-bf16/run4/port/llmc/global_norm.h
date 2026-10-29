/*
Global norm, used in gradient clipping. SYCL 2020 port of llmc/global_norm.cuh.
*/
#ifndef GLOBAL_NORM_H
#define GLOBAL_NORM_H

#include <assert.h>
#include <stddef.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// deterministic host-side accumulation (single device, exact order)

// Helper function determines the maximum number of block sums
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    (void)num_slices_all; (void)numel;
    return 1;
}

template<typename T>
inline void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride,
                                int num_slices, int max_num_block_sums, bool reset) {
    (void)stride; (void)num_slices; (void)max_num_block_sums;
    if (reset) {
        float zero = 0.0f;
        sycl_memcpy_h2d(out, &zero, sizeof(float));
    }
    if (count == 0) { return; }
    // deterministic: single ordered host accumulation over all slices
    constexpr size_t CHUNK = 1 << 20;
    std::vector<T> staging;
    double acc = 0.0; // accumulate in double, add to float `out` at the end (matches CUDA sum order qualitatively)
    float current = 0.0f;
    sycl_memcpy_d2h(&current, out, sizeof(float));
    for (int s = 0; s < num_slices; s++) {
        const T* base = values + (size_t)stride * (size_t)s;
        size_t done = 0;
        while (done < count) {
            size_t chunk = count - done < CHUNK ? count - done : CHUNK;
            staging.resize(chunk);
            sycl_memcpy_d2h(staging.data(), base + done, chunk * sizeof(T));
            for (size_t i = 0; i < chunk; i++) {
                float v = (float)staging[i];
                acc += (double)v * (double)v;
            }
            done += chunk;
        }
    }
    float total = current + (float)acc;
    sycl_memcpy_h2d(out, &total, sizeof(float));
}

#endif
