/*
Common utilities for SYCL code (port of cuda_common.h).
Single SYCL device, USM shared allocations, in-order queue.
*/
#ifndef SYCL_COMMON_H
#define SYCL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Single SYCL device queue used by the whole program (created in train file).
extern sycl::queue sycl_queue;

// Block size used for 1-D kernels (tunable, no CUDA meaning)
#define SYCL_BLOCK_SIZE 256u

// convenience macro for calculating grid dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking helpers for SYCL allocations

template<class T>
inline void syclFreeCheck(T* &ptr) {
    if (ptr) { sycl::free(ptr, sycl_queue); ptr = nullptr; }
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
#elif defined(ENABLE_FP16)
typedef sycl::half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
typedef sycl::ext::oneapi::bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Profiler utils (no-op in SYCL port)

class NvtxRange {
 public:
    NvtxRange(const char* s) { (void)s; }
    NvtxRange(const std::string& base_str, int number) { (void)base_str; (void)number; }
    ~NvtxRange() { }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between SYCL (USM-shared) memory <-> files
// USM shared pointers are directly accessible from the host, so plain fread/fwrite
// of the memory contents preserves llm.c's file formats bit-for-bit.

inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    (void)buffer_size;
    q.wait();
    fwriteCheck(src, 1, num_bytes, dest);
}

inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    (void)buffer_size;
    freadCheck(dest, 1, num_bytes, src);
    q.wait();
}

#endif // SYCL_COMMON_H
