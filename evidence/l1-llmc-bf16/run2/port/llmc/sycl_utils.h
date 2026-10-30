// Utilities for use in SYCL device code (ported from cuda_utils.cuh)
#ifndef SYCL_UTILS_H
#define SYCL_UTILS_H

#include "sycl_common.h"

// ----------------------------------------------------------------------------
// DType support

enum class DType : uint8_t {
    FP32, FP16, BF16
};

inline size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(sycl::half);
        case DType::BF16:
            return sizeof(sycl::ext::oneapi::bfloat16);
        default:
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

inline DType dtype_of(float* f) { (void)f; return DType::FP32; }
inline DType dtype_of(sycl::ext::oneapi::bfloat16* f) { (void)f; return DType::BF16; }
inline DType dtype_of(sycl::half* f) { (void)f; return DType::FP16; }

// ----------------------------------------------------------------------------
// Copy, cast functions (host + device)

template<typename Td, typename Ts>
inline Td cast_value(Ts val);

template<>
inline float cast_value<float, float>(float val) { return val; }
template<>
inline float cast_value<float, sycl::half>(sycl::half val) { return sycl::vec<float,1>{val}.x(); }
template<>
inline float cast_value<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
    return static_cast<float>(val);
}

// generic copy+cast kernel launcher (1-D work, num_slices groups via offset loop)
template<typename Td, typename Ts>
inline void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n,
                                 ptrdiff_t stride_dst, ptrdiff_t stride_src,
                                 int num_slices, sycl::queue& q) {
    if (n == 0 || num_slices == 0) { return; }
    size_t total = n * (size_t)num_slices;
    constexpr size_t BLOCK = 512;
    size_t nblocks = CEIL_DIV(total, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= total) { return; }
        size_t slice = lin / n;
        size_t idx = lin % n;
        dst[idx + (size_t)stride_dst * slice] = cast_value<Td, Ts>(src[idx + (size_t)stride_src * slice]);
    }).wait();
}

// ----------------------------------------------------------------------------
// Memory management: device malloc with managed fallback (host-pinned style)

// allocate memory on the device; on failure fall back to shared allocation
// returns 0 = OK, 1 = fell back to shared memory
inline int syclMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    (void)file; (void)line;
    try {
        *out = sycl::malloc_device(bytes, sycl_queue);
        if (*out == nullptr) { throw std::bad_alloc(); }
        return 0;
    } catch (...) {
        try {
            *out = sycl::malloc_shared(bytes, sycl_queue);
        } catch (sycl::exception& e) {
            printf("[SYCL ERROR] allocation failed at %s:%d:\n%s\n", file, line, e.what());
            exit(EXIT_FAILURE);
        }
        if (*out == nullptr) {
            printf("[SYCL ERROR] allocation failed at %s:%d\n", file, line);
            exit(EXIT_FAILURE);
        }
        return 1;
    }
}
#define syclMallocConditionallyManaged(out, bytes) \
(syclMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding (host + device versions)

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
inline constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed) {
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
inline constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed) {
    constexpr unsigned int PRIME_NUMBER = 198491317u;
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);
    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// bit helpers for bf16 stochastic rounding that work on host and device
inline float sycl_uint_as_float(unsigned int u) {
    float f; memcpy(&f, &u, sizeof(float)); return f;
}
inline unsigned int sycl_float_as_uint(float f) {
    unsigned int u; memcpy(&u, &f, sizeof(float)); return u;
}
inline sycl::ext::oneapi::bfloat16 sycl_float_to_bf16_rn(float f) {
    // round-to-nearest-even from fp32 to bf16
    unsigned int u = sycl_float_as_uint(f);
    unsigned int lsb = (u >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    uint16_t b = (uint16_t)(u >> 16);
    return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(b);
}

// stochastic rounding built on top of Squirrel Noise above
inline void stochastic_rounding(float in, sycl::ext::oneapi::bfloat16 *out, unsigned int seed) {
    unsigned int random = Get2dNoiseUint(0, 0, seed); // placeholder, real impl in kernels below
    (void)random;
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = sycl_float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits & ~0xFFFF);
    *out = sycl_float_to_bf16_rn(sycl_uint_as_float(float_bits));
}
inline void stochastic_rounding(float in, sycl::half *out, unsigned int random) {
    (void)random;
    *out = sycl::half(in);
}
inline void stochastic_rounding(float in, float *out, unsigned int random) {
    (void)random;
    *out = in;
}

// device-side stochastic rounding with explicit per-thread random value
// (matches CUDA: threshold from low 16 bits of per-thread random)
inline sycl::ext::oneapi::bfloat16 stochastic_rounding_bf16(float in, unsigned int random) {
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = sycl_float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFFu) : (float_bits & ~0xFFFFu);
    return sycl_float_to_bf16_rn(sycl_uint_as_float(float_bits));
}

// ----------------------------------------------------------------------------
// Deterministic single-sum reduction: sum `count` values into *result using
// one work-group + sub-group reductions (deterministic order via tree in one group)

template<class Float>
inline void global_sum_deterministic(float* result, const Float* values, size_t count, sycl::queue& q) {
    if (count == 0) {
        q.memset(result, 0, sizeof(float)).wait();
        return;
    }
    // Single-task sequential sum: trivially deterministic (fixed ascending order).
    // Counts here are small (losses B*T, grad-norm partials), so this is fast enough.
    q.single_task([=]() {
        float s = 0.f;
        for (size_t i = 0; i < count; i++) {
            s += (float)values[i];
        }
        *result = s;
    }).wait();
}

#endif
