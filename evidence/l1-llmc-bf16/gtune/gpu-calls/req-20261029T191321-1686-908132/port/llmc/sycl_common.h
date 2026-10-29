/*
Common utilities for SYCL code (port of cuda_common.h).
Single SYCL device, USM device allocations, in-order queue.
All model/activation/optimizer memory lives in sycl::malloc_device memory so
that pages never migrate on discrete GPUs; every host access goes through an
explicit q.memcpy (blocking only where the host immediately consumes data).
*/
#ifndef SYCL_COMMON_H
#define SYCL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <algorithm>            // std::min
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
// USM device memory movement.
//
// The queue is in-order, so device work and transfers submitted in order never
// overtake each other. Fire-and-forget transfers need no wait; transfers whose
// result the host consumes immediately use the blocking variant.

inline void dev_zero(void* ptr, size_t bytes, sycl::queue& q) {
    if (bytes != 0 && ptr != nullptr) { q.memset(ptr, 0, bytes); }
}

// host -> device, ordered after/before surrounding device work, no host wait
inline void memcpy_h2d(void* dst, const void* src, size_t bytes, sycl::queue& q) {
    if (bytes != 0) { q.memcpy(dst, src, bytes); }
}

// device -> host, host consumes the data immediately: blocking
inline void memcpy_d2h(void* dst, const void* src, size_t bytes, sycl::queue& q) {
    if (bytes != 0) { q.memcpy(dst, src, bytes).wait(); }
}

// ----------------------------------------------------------------------------
// Utilities to Read & Write between SYCL device memory <-> files.
// A small host staging buffer preserves llm.c's file formats bit-for-bit.

inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    if (num_bytes == 0) { return; }
    size_t chunk = buffer_size != 0 ? buffer_size : (size_t)(32 * 1024 * 1024);
    char* staging = (char*)mallocCheck(chunk < num_bytes ? chunk : num_bytes);
    size_t off = 0;
    while (off < num_bytes) {
        size_t n = std::min(chunk, num_bytes - off);
        q.memcpy(staging, (const char*)src + off, n).wait();
        fwriteCheck(staging, 1, n, dest);
        off += n;
    }
    free(staging);
}

inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    if (num_bytes == 0) { return; }
    size_t chunk = buffer_size != 0 ? buffer_size : (size_t)(32 * 1024 * 1024);
    char* staging = (char*)mallocCheck(chunk < num_bytes ? chunk : num_bytes);
    size_t off = 0;
    while (off < num_bytes) {
        size_t n = std::min(chunk, num_bytes - off);
        freadCheck(staging, 1, n, src);
        // blocking: the staging buffer is reused for the next chunk
        q.memcpy((char*)dest + off, staging, n).wait();
        off += n;
    }
    free(staging);
}

#endif // SYCL_COMMON_H
