/*
Common utilities for SYCL code (ported from llmc/cuda_common.h).
Single SYCL device, USM device memory, in-order queue.
*/
#ifndef SYCL_COMMON_H
#define SYCL_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Single SYCL device queue, created in train_gpt2.cpp (like deviceProp/main_stream)
extern sycl::queue q_sycl;
inline sycl::queue& sycl_queue() { return q_sycl; }

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// convenience macro for calculating grid dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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

class SyclRange {
 public:
    SyclRange(const char* s) { (void)s; }
    SyclRange(const std::string& base_str, int number) { (void)base_str; (void)number; }
    ~SyclRange() {}
};
#define NVTX_RANGE_FN() SyclRange sycl_range(__FUNCTION__)
class NvtxRange {
 public:
    NvtxRange(const char* s) { (void)s; }
    NvtxRange(const std::string& base_str, int number) { (void)base_str; (void)number; }
    ~NvtxRange() {}
};

// ----------------------------------------------------------------------------
// Device/host queries

inline void sycl_check_last_error() {}

// ----------------------------------------------------------------------------
// Utilities to Read & Write between SYCL device memory <-> files

// copy num_bytes from device pointer src into file dest.
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    (void)buffer_size;
    char* host_buf = (char*)malloc(num_bytes);
    if (host_buf == nullptr) { fprintf(stderr, "device_to_file: malloc failed\n"); exit(EXIT_FAILURE); }
    q.memcpy(host_buf, src, num_bytes).wait();
    fwriteCheck(host_buf, 1, num_bytes, dest);
    free(host_buf);
}

// copy num_bytes from file src into device pointer dest.
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    (void)buffer_size;
    char* host_buf = (char*)malloc(num_bytes);
    if (host_buf == nullptr) { fprintf(stderr, "file_to_device: malloc failed\n"); exit(EXIT_FAILURE); }
    freadCheck(host_buf, 1, num_bytes, src);
    q.memcpy(dest, host_buf, num_bytes).wait();
    free(host_buf);
}

#endif
