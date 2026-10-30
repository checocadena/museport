/*
The GPT-2 Encoder: token + position encodings (SYCL port of encoder.cuh)
*/
#ifndef ENCODER_SYCL_H
#define ENCODER_SYCL_H

#include <assert.h>
#include <stdint.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

inline void encoder_forward(floatX* out,
                            const int* inp, const floatX* wte, const floatX* wpe,
                            int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T * C;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t bt = idx / (size_t)C;
        size_t b = bt / (size_t)T;
        size_t t = bt % (size_t)T;
        size_t c = idx % (size_t)C;
        int ix = inp[b * (size_t)T + t];
        out[idx] = (floatX)((float)wte[(size_t)ix * C + c] + (float)wpe[t * (size_t)C + c]);
    }).wait();
}

// Fully deterministic wte backward using CPU bucketing (same algorithm as CUDA version).
// dout: (B*T*C), dwte: (Vp*C) updated with stochastic rounding read-modify-write.
inline void wte_backward_sycl(floatX* dwte,
                              const int* bucket_info, const int* workload_indices,
                              const floatX* dout,
                              unsigned int seed, int B, int T, int C, int num_buckets, sycl::queue& q) {
    // One work-group (sub-group cooperatively) per bucket-channel-group.
    // Simplify vs CUDA: each (bucket) handled by 256 threads processing items in order.
    constexpr size_t BLOCK = 256;
    // copy bucket info to device already done by caller; here bucket_info layout: [start, size, ix, c] x num_buckets
    // For determinism, accumulate per (bucket) in fixed item order, then stochastic round once.
    // Each bucket may span multiple channel-vectors; handle per-lane below.
    // We launch num_buckets * Cgroups? No: bucket already encodes c-group. Each bucket -> BLOCK threads.
    // Within a bucket, thread t handles channel offset (t % 32)*8+k style for bf16 (x8 per 128-bit). Generalize: 8-wide chunks.
    size_t total = (size_t)num_buckets * BLOCK;
    // Deterministic two-pass approach without local memory:
    // Pass 1: each thread accumulates its strided items into a per-thread slot
    // of a device staging buffer (fixed order per thread).
    // Pass 2: single-task ordered reduction per bucket-channel into dwte.
    // Staging size: num_buckets * BLOCK * CHMAX floats.
    constexpr int CHMAX = 8;
    float* stage = sycl::malloc_device<float>(total * CHMAX, q);
    q.memset(stage, 0, total * CHMAX * sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t lin = id[0];
        size_t bucket = lin / BLOCK;
        size_t tid = lin % BLOCK;
        int bstart = bucket_info[bucket * 4 + 0];
        int bsize = bucket_info[bucket * 4 + 1];
        int bix = bucket_info[bucket * 4 + 2];
        int bc = bucket_info[bucket * 4 + 3];
        // channels per thread chunk: 8 for bf16 (matches x128::size=8), 4 for fp32
        constexpr int CH = (sizeof(floatX) == 2) ? 8 : 4;
        constexpr int LANES = 32;
        int c_per_group = LANES * CH;
        int lane = (int)(tid % LANES);
        int c = bc * c_per_group + lane * CH;
        if (c >= C) { return; }
        // accumulate over items assigned round-robin deterministically then tree-reduce in order
        // First: each thread sums its strided items in index order.
        int nwarps_items = (int)((BLOCK / LANES)); // 8 "warps" per block like CUDA (256/32)
        int warp_id = (int)(tid / LANES);
        float accum[8] = {0,0,0,0,0,0,0,0};
        for (int item_i = warp_id; item_i < bsize; item_i += nwarps_items) {
            int bt = workload_indices[bstart + item_i];
            const floatX* row = dout + (size_t)bt * C + c;
            for (int k = 0; k < CH; k++) {
                if (c + k < C) { accum[k] += (float)row[k]; }
            }
        }
        // Pass 1: accumulate this thread's strided items into staging (fixed order).
        for (int k = 0; k < CH; k++) { stage[lin * CHMAX + k] = accum[k]; }
    }).wait();
    // Pass 2: ordered reduction per bucket lane-channel, then stochastic round once.
    // One thread per (bucket, lane, k): sums the 8 warp slots in warp order.
    {
        size_t total2 = (size_t)num_buckets * 32 * CHMAX;
        q.parallel_for(sycl::range<1>(total2), [=](sycl::id<1> id) {
            size_t lin2 = id[0];
            size_t bucket = lin2 / (32 * CHMAX);
            size_t rest = lin2 % (32 * CHMAX);
            size_t lane = rest / CHMAX;
            size_t k = rest % CHMAX;
            int bsize = bucket_info[bucket * 4 + 1];
            int bix = bucket_info[bucket * 4 + 2];
            int bc = bucket_info[bucket * 4 + 3];
            constexpr int CH = (sizeof(floatX) == 2) ? 8 : 4;
            constexpr int LANES = 32;
            if (k >= (size_t)CH) { return; }
            int c_per_group = LANES * CH;
            int c = bc * c_per_group + (int)lane * CH + (int)k;
            if (c >= C) { return; }
            int nwarps_items = (int)(BLOCK / LANES);
            float s = 0.f;
            for (int w = 0; w < nwarps_items; w++) {
                size_t tid = (size_t)w * LANES + lane;
                s += stage[(bucket * BLOCK + tid) * CHMAX + k];
            }
            (void)bsize;
            floatX* drow = dwte + (size_t)bix * C + c;
            float v = s + (float)(*drow);
            if constexpr (std::is_same<floatX, float>::value) {
                *drow = v;
            } else {
                unsigned int random = Get2dNoiseUint((int)((bucket * 32 + lane) % 1024),
                                                     (int)((bucket * 32 + lane) / 1024 + k),
                                                     seed + (unsigned int)bucket * 32u + (unsigned int)lane + (unsigned int)k);
                *drow = stochastic_rounding_bf16(v, random);
            }
        }).wait();
    }
    sycl::free(stage, q);
}

inline void wpe_backward_sycl(floatX* dwpe, const floatX* dout,
                              int B, int T, int C, unsigned int seed, sycl::queue& q) {
    size_t N = (size_t)T * C;
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        size_t t = idx / (size_t)C;
        size_t c = idx % (size_t)C;
        float acc = 0.f;
        for (int b = 0; b < B; b++) {
            acc += (float)dout[(size_t)b * T * C + t * (size_t)C + c];
        }
        floatX* p = dwpe + t * (size_t)C + c;
        float v = acc + (float)(*p);
        if constexpr (std::is_same<floatX, float>::value) {
            *p = v;
        } else {
            unsigned int random = Get2dNoiseUint((int)(idx % 1024), (int)(idx / 1024), seed + (unsigned int)idx);
            *p = stochastic_rounding_bf16(v, random);
        }
    }).wait();
}

// ----------------------------------------------------------------------------
// launchers

// Fully deterministic (see comments in CUDA file for details)
inline void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // device outputs & scratch
                             int* workload_indices, int* bucket_info,      // cpu scratch buffers (int[4] per bucket)
                             const floatX* dout, const int* inp, const int* inputs_cpu,
                             int B, int T, int C, unsigned int seed, sycl::queue& q) {
    NVTX_RANGE_FN();

    // Launch wpe first (runs on device while CPU pre-processes wte buckets)
    wpe_backward_sycl(dwpe, dout, B, T, C, seed, q);

    constexpr int LANES = 32;
    constexpr int CH = (sizeof(floatX) == 2) ? 8 : 4;
    int num_c_groups = (int)CEIL_DIV(C, CH * LANES);

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < (uint64_t)(B * T); bt++) {
        for (uint64_t c_group = 0; c_group < (uint64_t)num_c_groups; c_group++) {
            uint64_t data = bt + (c_group << 32ULL) + ((uint64_t)(uint32_t)inputs_cpu[bt] << 42ULL);
            buckets[c_group + (uint64_t)num_c_groups * (uint64_t)(uint32_t)inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a,
                 const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = (int)buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index * 4 + 0] = workload_index;
        bucket_info[bucket_index * 4 + 1] = (int)bucket.second.size();
        bucket_info[bucket_index * 4 + 2] = (int)((bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1));
        bucket_info[bucket_index * 4 + 3] = (int)((bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1));
        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL << 31ULL) - 1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data host->device
    int* d_bucket_info = (int*)scratch;
    int* d_workload_indices = (int*)(scratch + (size_t)B * T * num_c_groups * sizeof(int) * 4 / sizeof(floatX));
    // NOTE: scratch is floatX*; compute byte offsets carefully:
    d_bucket_info = (int*)scratch;
    d_workload_indices = (int*)((char*)scratch + (size_t)B * T * num_c_groups * sizeof(int) * 4);
    q.memcpy(d_bucket_info, bucket_info, (size_t)num_buckets * 4 * sizeof(int)).wait();
    q.memcpy(d_workload_indices, workload_indices, (size_t)total_items * sizeof(int)).wait();

    wte_backward_sycl(dwte, d_bucket_info, d_workload_indices, dout, seed, B, T, C, num_buckets, q);
    (void)inp;
}

#endif
