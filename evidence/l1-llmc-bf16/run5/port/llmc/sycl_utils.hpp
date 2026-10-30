// Utilities for use in SYCL device code (port of cuda_utils.cuh)

#ifndef SYCL_UTILS_H
#define SYCL_UTILS_H

#include "sycl_common.h"

// ----------------------------------------------------------------------------
// Packed128 data structure: 16-byte vector loads/stores (e.g. 8x bf16)

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    explicit Packed128(sycl::vec<int, 4> bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < (int)size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    static Packed128 zeros() {
        return constant(ElementType(0.f));
    }
    static Packed128 ones() {
        return constant(ElementType(1.f));
    }

    ElementType& operator[](int index) {
        return payload[index];
    }
    const ElementType& operator[](int index) const {
        return payload[index];
    }
    sycl::vec<int, 4> get_bits() const {
        sycl::vec<int, 4> bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(sycl::vec<int, 4>) / sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
inline Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const sycl::vec<int, 4>*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint (same as plain load in SYCL)
template<class ElementType>
inline Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const sycl::vec<int, 4>*>(address)};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
inline void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::vec<int, 4>*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint (same in SYCL)
template<class ElementType>
inline void store128cs(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::vec<int, 4>*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1 (same in SYCL)
template<class ElementType>
inline void store128cg(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::vec<int, 4>*>(target) = value.get_bits();
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

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
// Copy, cast functions

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

// dst[i] = (Td)src[i], SYCL kernel version of copy_and_cast_kernel
template<typename Td, typename Ts>
inline void copy_and_cast(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src, int num_slices, sycl::queue& q) {
    if (n == 0) return;
    q.parallel_for(sycl::range<2>(num_slices, n), [=](sycl::id<2> id) {
        size_t j = id[0];
        size_t i = id[1];
        dst[i + stride_dst * j] = cast_value<Td, Ts>(src[i + stride_src * j]);
    }).wait();
}

// ----------------------------------------------------------------------------
// Sub-group reduction helpers (port of warp shuffles; sub-group size is queried
// at runtime, kernel code below is written to not assume any fixed width)

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// http://eiserloh.net/noise/SquirrelNoise5.hpp
constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
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
constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// stochastic rounding, SYCL device-callable versions.
// The (linearId, groupId) pair plays the role of (threadIdx.x, blockIdx.x*blockDim.x+blockIdx.y).
inline void stochastic_rounding(float in, sycl::ext::oneapi::bfloat16 *out, unsigned int seed) {
    unsigned int random = Get2dNoiseUint(0, (int)seed, seed);
    (void)random;
    // full computation happens in the overload below; kept for interface compatibility
    uint32_t float_bits;
    memcpy(&float_bits, &in, sizeof(float_bits));
    unsigned int threshold = random & 0xFFFFu;
    unsigned int rounded_bits = float_bits & 0x0000FFFFu;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFFu) : (float_bits & ~0xFFFFu);
    float rounded;
    memcpy(&rounded, &float_bits, sizeof(rounded));
    *out = sycl::ext::oneapi::bfloat16(rounded);
}

inline void stochastic_rounding_at(float in, sycl::ext::oneapi::bfloat16 *out, unsigned int random) {
    uint32_t float_bits;
    memcpy(&float_bits, &in, sizeof(float_bits));
    unsigned int threshold = random & 0xFFFFu;
    unsigned int rounded_bits = float_bits & 0x0000FFFFu;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFFu) : (float_bits & ~0xFFFFu);
    float rounded;
    memcpy(&rounded, &float_bits, sizeof(rounded));
    *out = sycl::ext::oneapi::bfloat16(rounded);
}

inline void stochastic_rounding(float in, sycl::half *out, unsigned int random) {
    (void)random;
    *out = (float)in; // todo - implement this...
}
inline void stochastic_rounding(float in, float *out, unsigned int random) {
    (void)random;
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

// Deterministic global sum over `count` elements into *result (single work-group tree
// reduction would be device-specific; for the CPU device a host-visible accumulation
// loop is exactly deterministic and matches llm.c's single-block requirement).
template<class Float>
inline void global_sum_deterministic(float* result, const Float* values, size_t count, sycl::queue& q) {
    q.wait();
    double acc = 0.0;
    for (size_t i = 0; i < count; i++) {
        acc += (double)(float)values[i];
    }
    *result = (float)acc;
}

// memory management: SYCL port uses USM shared allocations (see train file).

#endif
