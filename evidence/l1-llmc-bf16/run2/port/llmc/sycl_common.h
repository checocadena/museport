/*
Common utilities for SYCL code (ported from cuda_common.h).
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

// SYCL queue used by this process (single device). Created in common_start().
extern sycl::queue sycl_queue;

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<false> False;

// Sub-group size used for reductions (no warp-size-32 idioms in launches)
static constexpr int SYCL_SUBGROUP_SIZE = 16;

// ----------------------------------------------------------------------------
// Error checking helpers for SYCL allocations

template<class T>
inline void syclFreeCheck(T** ptr, const char *file, int line) {
    if (*ptr != nullptr) {
        try {
            sycl::free(*ptr, sycl_queue);
        } catch (sycl::exception& e) {
            printf("[SYCL ERROR] at file %s:%d:\n%s\n", file, line, e.what());
            exit(EXIT_FAILURE);
        }
        *ptr = nullptr;
    }
}
#define syclFreeCheck(ptr) (syclFreeCheck(ptr, __FILE__, __LINE__))

#define CHECK_SYCL_TRY(stmt) do { try { stmt; } catch (sycl::exception& e) { \
    printf("[SYCL ERROR] at file %s:%d:\n%s\n", __FILE__, __LINE__, e.what()); \
    exit(EXIT_FAILURE); } } while(0)

// wait on the main queue (like cudaDeviceSynchronize for our single queue)
inline void syclDeviceSynchronize() { sycl_queue.wait(); }

// ----------------------------------------------------------------------------
// Precision settings and defines

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
// Profiler utils (no-op stand-ins for NVTX)

class SyclRange {
 public:
    SyclRange(const char*) {}
    SyclRange(const std::string&, int) {}
    ~SyclRange() {}
};
#define NVTX_RANGE_FN() SyclRange sycl_range(__FUNCTION__)
class NvtxRange {
 public:
    NvtxRange(const char*) {}
    NvtxRange(const std::string&, int) {}
    ~NvtxRange() {}
};

// ----------------------------------------------------------------------------
// Utilities to Read & Write between device memory <-> files

// copy num_bytes from device pointer src into file dest
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    char* buffer_space = (char*)mallocCheck(2 * buffer_size);
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;
    char* dev_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    q.memcpy(read_buffer, dev_read_ptr, copy_amount).wait();
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    dev_read_ptr += copy_amount;
    std::swap(read_buffer, write_buffer);
    while (rest_bytes > 0) {
        copy_amount = std::min(buffer_size, rest_bytes);
        q.memcpy(read_buffer, dev_read_ptr, copy_amount).wait();
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        dev_read_ptr += copy_amount;
    }
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    free(buffer_space);
}

// copy num_bytes from file src into device pointer dest
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, sycl::queue& q) {
    char* buffer_space = (char*)mallocCheck(2 * buffer_size);
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;
    char* dev_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);
    while (rest_bytes > 0) {
        copy_amount = std::min(buffer_size, rest_bytes);
        q.memcpy(dev_write_ptr, write_buffer, write_buffer_size).wait();
        dev_write_ptr += write_buffer_size;
        freadCheck(read_buffer, 1, copy_amount, src);
        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }
    q.memcpy(dev_write_ptr, write_buffer, write_buffer_size).wait();
    free(buffer_space);
}

#endif // SYCL_COMMON_H
