/*
Common utilities for the SYCL 2020 port of llm.c (Intel oneAPI DPC++).
Replaces llmc/cuda_common.h. Single device, in-order queue, USM device memory.
*/
#ifndef SYCL_COMMON_H
#define SYCL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <sycl/sycl.hpp>

#include "utils.h"

// ----------------------------------------------------------------------------
// Global SYCL device queue (single device, in-order, like the single CUDA stream)

inline sycl::queue sycl_queue{sycl::default_selector_v, sycl::property::queue::in_order{}};

// Human readable name of the selected device, filled in by common_start()
inline std::string sycl_device_name = "SYCL device";

// ----------------------------------------------------------------------------
// Global defines and settings

// convenience macro for calculating grid dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<false> False;

// plain 4x32-bit struct replacing CUDA int4 (used for encoder bucket info)
struct Int4 {
    int x, y, z, w;
};

// ----------------------------------------------------------------------------
// USM device memory management (synchronous, like the CUDA code with syncs)

template <typename T>
inline T* sycl_malloc_device(size_t n, const char* file = nullptr, int line = 0) {
    T* p = sycl::malloc_device<T>(n, sycl_queue);
    if (p == nullptr && n > 0) {
        printf("[SYCL ERROR] malloc_device failed for %zu elements", n);
        if (file != nullptr) { printf(" at file %s:%d", file, line); }
        printf("\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

// like cudaFree, but resets the pointer.
template<class T>
inline void syclFreeCheck(T** ptr, const char *file, int line) {
    (void)file; (void)line;
    sycl::free(*ptr, sycl_queue);
    *ptr = nullptr;
}
#define syclFreeCheck(ptr) (syclFreeCheck(ptr, __FILE__, __LINE__))

inline void sycl_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) { return; }
    sycl_queue.memcpy(dst, src, bytes).wait();
}
inline void sycl_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) { return; }
    sycl_queue.memcpy(dst, src, bytes).wait();
}
inline void sycl_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) { return; }
    sycl_queue.memcpy(dst, src, bytes).wait();
}
inline void sycl_memset_sync(void* dst, int value, size_t bytes) {
    if (bytes == 0) { return; }
    sycl_queue.memset(dst, value, bytes).wait();
}

// ----------------------------------------------------------------------------
// SYCL Precision settings and defines

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
typedef sycl::ext::oneapi::bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Profiler utils (no-op in the SYCL port)

class NvtxRange {
 public:
    NvtxRange(const char* s) { (void)s; }
    NvtxRange(const std::string& base_str, int number) { (void)base_str; (void)number; }
};
#define NVTX_RANGE_FN()

// ----------------------------------------------------------------------------
// Utilities to Read & Write between SYCL device memory <-> files

// copy num_bytes from device pointer src into file dest, via a host staging buffer.
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size) {
    if (num_bytes == 0) { return; }
    char* staging = (char*)mallocCheck(buffer_size);
    char* dev_ptr = (char*)src;
    size_t remaining = num_bytes;
    while (remaining > 0) {
        size_t chunk = remaining < buffer_size ? remaining : buffer_size;
        sycl_memcpy_d2h(staging, dev_ptr, chunk);
        fwriteCheck(staging, 1, chunk, dest);
        dev_ptr += chunk;
        remaining -= chunk;
    }
    free(staging);
}

// copy num_bytes from file src into device pointer dest, via a host staging buffer.
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size) {
    if (num_bytes == 0) { return; }
    char* staging = (char*)mallocCheck(buffer_size);
    char* dev_ptr = (char*)dest;
    size_t remaining = num_bytes;
    while (remaining > 0) {
        size_t chunk = remaining < buffer_size ? remaining : buffer_size;
        freadCheck(staging, 1, chunk, src);
        sycl_memcpy_h2d(dev_ptr, staging, chunk);
        dev_ptr += chunk;
        remaining -= chunk;
    }
    free(staging);
}

#endif // SYCL_COMMON_H
