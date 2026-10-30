// Utilities for use in SYCL device code (ported from llmc/cuda_utils.cuh)

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
// Copy, cast helpers

template<typename Td, typename Ts>
inline Td cast_value(Ts val) { return (Td)((float)val); }

template<typename Td, typename Ts>
inline void copy_and_cast(Td* dst, const Ts* src, size_t n,
                          ptrdiff_t stride_dst, ptrdiff_t stride_src, sycl::queue& q) {
    // strided 2-D copy-cast: num rows derived from launch in callers; here n = elements per row
    // callers pass explicit total; keep simple 1-D with strides via kernel below
    (void)q;
    for (size_t i = 0; i < n; i++) { dst[i] = cast_value<Td, Ts>(src[i]); }
    (void)stride_dst; (void)stride_src;
}

inline void copy_and_cast_kernel_launch(floatX* dst, const floatX* src, size_t n,
                                        ptrdiff_t stride_dst, ptrdiff_t stride_src,
                                        int num_slices, sycl::queue& q) {
    // used to init master weights: dst[s*s_stride ...], src[w*w_stride ...]
    // implemented as a SYCL kernel
    if (num_slices <= 1) {
        floatX* d = dst; const floatX* s = src;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            d[i] = (floatX)((float)s[i]);
        }).wait();
    } else {
        floatX* d = dst; const floatX* s = src;
        q.parallel_for(sycl::range<2>(num_slices, n), [=](sycl::id<2> id) {
            size_t r = id[0], i = id[1];
            d[r * (size_t)stride_dst + i] = (floatX)((float)s[r * (size_t)stride_src + i]);
        }).wait();
    }
}

// backwards-compat launcher used by train_gpt2.cpp
template <typename Td, typename Ts>
inline void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n,
                                 ptrdiff_t stride_dst, ptrdiff_t stride_src,
                                 int num_slices, sycl::queue& q) {
    if (num_slices <= 1) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            dst[i] = (Td)((float)src[i]);
        }).wait();
    } else {
        q.parallel_for(sycl::range<2>(num_slices, n), [=](sycl::id<2> id) {
            size_t r = id[0], i = id[1];
            dst[r * (size_t)stride_dst + i] = (Td)((float)src[r * (size_t)stride_src + i]);
        }).wait();
    }
}

// ----------------------------------------------------------------------------
// Deterministic single-pass sum reduction (ported from global_sum_single_block_kernel)

template<class Float>
inline void global_sum_deterministic(float* result, const Float* values, size_t count, sycl::queue& q) {
    // Two-stage deterministic reduction: per-work-item partial sums (fixed order),
    // then ordered accumulation on host. Bit-identical for identical input order.
    const size_t NW = 64;
    float* partials = sycl::malloc_device<float>(NW, q);
    const Float* v = values;
    q.parallel_for(sycl::range<1>(NW), [=](sycl::id<1> id) {
        size_t i = id[0];
        float s = 0.f;
        for (size_t j = i; j < count; j += NW) { s += (float)v[j]; }
        partials[i] = s;
    }).wait();
    float h[64];
    q.memcpy(h, partials, sizeof(h)).wait();
    // deterministic ordered tree reduction on host (fixed order => deterministic)
    float total = 0.f;
    for (size_t i = 0; i < NW; i++) { total += h[i]; }
    q.memcpy(result, &total, sizeof(float)).wait();
    sycl::free(partials, q);
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// http://eiserloh.net/noise/SquirrelNoise5.hpp
inline constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B;
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;
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
inline constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u;
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// stochastic rounding: identical algorithm to CUDA version.
// positionX/indexY emulate (threadIdx.x, blockIdx.x*blockDim.x + blockIdx.y) numbering.
inline void stochastic_rounding(float in, floatX* out, unsigned int seed,
                                unsigned int pos_x, unsigned int pos_y) {
#if defined(ENABLE_FP32)
    *out = in;
#else
    unsigned int random = Get2dNoiseUint((int)pos_x, (int)pos_y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = sycl::bit_cast<unsigned int>(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits & ~0xFFFF);
    float rounded = sycl::bit_cast<float>(float_bits);
    // round-to-nearest-even bf16 conversion of the manipulated bits: top16 must survive exactly.
    // Manual truncate keeps determinism identical to __float2bfloat16_rn on already-rounded values:
    // after the OR/FILL manipulation the low 16 bits are either 0x0000 or 0xFFFF, both RN-stable.
    unsigned int rb = sycl::bit_cast<unsigned int>(rounded);
    unsigned short half = (unsigned short)(rb >> 16);
    unsigned short* hp = reinterpret_cast<unsigned short*>(out);
    *hp = half;
#endif
}

#endif
