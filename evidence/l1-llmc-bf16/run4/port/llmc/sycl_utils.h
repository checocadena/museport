// Utilities for use in SYCL device code. Replaces llmc/cuda_utils.cuh.
#ifndef SYCL_UTILS_H
#define SYCL_UTILS_H

#include "sycl_common.h"

// ----------------------------------------------------------------------------
// DType support

// enumerator to identify the datatype of a tensor.
enum class DType : uint8_t {
    FP32, FP16, BF16
};

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
inline size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(sycl::half);
        case DType::BF16:
            return sizeof(sycl::ext::oneapi::bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

inline DType dtype_of(float* f) { (void)f; return DType::FP32; }
inline DType dtype_of(sycl::ext::oneapi::bfloat16* f) { (void)f; return DType::BF16; }
inline DType dtype_of(sycl::half* f) { (void)f; return DType::FP16; }

// ----------------------------------------------------------------------------
// Cast helpers between float and floatX (host + device)

template<typename Td, typename Ts>
inline Td cast_value(Ts val);

template<>
inline float cast_value<float, float>(float val) {
    return val;
}

template<>
inline float cast_value<float, sycl::half>(sycl::half val) {
    return (float)val;
}

template<>
inline float cast_value<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
    return (float)val;
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire device
// http://eiserloh.net/noise/SquirrelNoise5.hpp
constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f; // 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197; // 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB; // 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5; // 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// bit cast helpers (usable on host and device)
inline unsigned int float_as_uint(float f) {
    return sycl::bit_cast<unsigned int>(f);
}
inline float uint_as_float(unsigned int u) {
    return sycl::bit_cast<float>(u);
}
// round-to-nearest-even float -> bf16
inline sycl::ext::oneapi::bfloat16 float_to_bf16_rn(float f) {
    unsigned int bits = float_as_uint(f);
    // round to nearest even: add bias then truncate top 16 bits
    unsigned int rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    uint16_t out = (uint16_t)((bits + rounding_bias) >> 16);
    return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(out);
}

// stochastic rounding (host + device). For 1-D launches, pos_x is the global
// linear id and pos_y must be 0; for 2-D launches pos_x/pos_y are the item ids
// so that rounding matches llm.c's CUDA indexing scheme.
inline void stochastic_rounding(float in, sycl::ext::oneapi::bfloat16 *out, unsigned int seed,
                                int pos_x, int pos_y) {
    unsigned int random = Get2dNoiseUint(pos_x, pos_y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = float_to_bf16_rn(uint_as_float(float_bits));
}
inline void stochastic_rounding(float in, sycl::half *out, unsigned int random) {
    (void)random;
    *out = (float)in; // todo - implement this...
}
inline void stochastic_rounding(float in, float *out, unsigned int random) {
    (void)random;
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

// ----------------------------------------------------------------------------
// SYCL reductions (replacing CUDA warp/block reductions)
//
// Conventions: work-group sizes are 1-D. Sub-group size on CPU is typically 8
// (but may be 4/16/32); we never hardcode 32 anywhere.

template <typename T, typename GroupT>
inline T group_sum(GroupT g, T val) {
    return sycl::reduce_over_group(g, val, sycl::plus<T>());
}

template <typename T, typename GroupT>
inline T group_max(GroupT g, T val) {
    return sycl::reduce_over_group(g, val, sycl::maximum<T>());
}

// Deterministic single-work-group sum reduction over `count` elements.
template<class Float>
inline void global_sum_deterministic(float* result, const Float* values, size_t count) {
    float host_sum = 0.0f;
    constexpr size_t CHUNK = 1 << 20;
    std::vector<Float> staging;
    staging.reserve(count < CHUNK ? (size_t)count : CHUNK);
    size_t done = 0;
    while (done < count) {
        size_t chunk = count - done < CHUNK ? count - done : CHUNK;
        staging.resize(chunk);
        sycl_memcpy_d2h(staging.data(), values + done, chunk * sizeof(Float));
        for (size_t i = 0; i < chunk; i++) {
            host_sum += (float)staging[i];
        }
        done += chunk;
    }
    sycl_memcpy_h2d(result, &host_sum, sizeof(float));
}

// ----------------------------------------------------------------------------
// memory management

// allocate device memory; single-process port always uses device memory.
inline int syclMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    (void)file; (void)line;
    *out = sycl::malloc_device(bytes, sycl_queue);
    if (*out == nullptr && bytes > 0) {
        printf("[SYCL ERROR] malloc_device failed for %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return 0;
}

#define syclMallocConditionallyManaged(out, bytes)\
(syclMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// Copy-and-cast kernel: dst[i + stride_dst * group] = (Td)src[i + stride_src * group].
// Matches llm.c's copy_and_cast_kernel (used to init master weights), but with
// SYCL 1-D launches instead of CUDA dim3.
template<typename Td, typename Ts>
inline void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n,
                                 ptrdiff_t stride_dst, ptrdiff_t stride_src,
                                 int num_slices) {
    if (n == 0 || num_slices == 0) { return; }
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(n, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            size_t grp = it.get_group_linear_id();
            if (idx < n && grp < (size_t)num_slices) {
                dst[idx + (size_t)stride_dst * grp] = (Td)src[idx + (size_t)stride_src * grp];
            }
        }).wait();
}

#endif
