/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
SYCL 2020 port of encoder.cuh (single device, deterministic).
All launches are fire-and-forget on the in-order queue; bucket accumulation
for wte runs on the device (no host reads of device memory).
*/
#ifndef SYCL_ENCODER_H
#define SYCL_ENCODER_H

#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

inline void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T * C;
    size_t vec = x128::size;
    size_t nvec = CEIL_DIV(N, vec);
    q.parallel_for(sycl::range<1>(nvec), [=](sycl::id<1> id) {
        size_t vidx = id[0];
        size_t idx = vidx * vec;
        if (idx >= N) { return; }
        size_t bt = idx / (size_t)C;
        size_t b = bt / (size_t)T;
        size_t t = bt % (size_t)T;
        size_t c = idx % (size_t)C;
        int ix = inp[b * T + t];
        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + (size_t)ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;
        x128 packed_out;
        x128 wte128 = load128cs(wte_ix);
        x128 wpe128 = load128cs(wpe_tc);
        for (size_t k = 0; k < x128::size; k++) {
            packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
        }
        store128(out_btc, packed_out);
    });
}

// wte backward: deterministic bucketed accumulation.
// Same bucketing as CUDA version, but computed on the HOST from the host-side
// input copy (no device reads). Accumulation runs ON THE DEVICE, in bucket
// order, so discrete-GPU memory is never read directly by the host:
//  - `scratch` (device, >= B*T*C floats) holds per-(bt) scaled dout rows.
//  - pass 1 (device): scatter dout rows into scratch grouped by bucket with
//    fixed work assignment (one work-item per (bucket-item, channel vec)).
//  - pass 2 (device): ordered accumulation per bucket-channel on device +
//    stochastic rounding with the same per-parameter seeds as before.
// wpe backward stays a device kernel (batch-order accumulation per (t,c)).
inline void encoder_backward(floatX* dwte, floatX* dwpe, float* scratch, // device outputs & scratch
                      int* workload_indices, int* bucket_info, int* item_cgroup, // cpu scratch buffers
                      int* d_workload, int* d_buckets, int* d_cgroup, // persistent device tables
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/device inputs
                      int B, int T, int C, unsigned int seed, sycl::queue& q) {
    NVTX_RANGE_FN();
    (void)inp;

    // ---- wpe backward (deterministic): each (t,c) accumulated in batch order ----
    {
        size_t vec = x128::size;
        size_t N = (size_t)T * C;
        size_t nvec = CEIL_DIV(N, vec);
        q.parallel_for(sycl::range<1>(nvec), [=](sycl::id<1> id) {
            size_t vidx = id[0];
            size_t idx = vidx * vec;
            if (idx >= N) { return; }
            size_t t = idx / (size_t)C;
            size_t c = idx % (size_t)C;
            float accum[x128::size] = {0.0f};
            for (int b = 0; b < B; b++) {
                x128 packed_dout = load128cs(dout + ((size_t)b * T * C) + (t * C) + c);
                for (size_t k = 0; k < x128::size; k++) {
                    accum[k] += (float)packed_dout[k];
                }
            }
            floatX* dwpe_tc = dwpe + (t * C) + c;
            x128 packed_dwpe = load128(dwpe_tc);
            for (size_t k = 0; k < x128::size; k++) {
                // deterministic per-parameter seed, identical formula to CUDA version
                unsigned int rnd = Get2dNoiseUint((int)(vidx % 1024), (int)(vidx / 1024 + 1), seed + (unsigned int)(idx + k));
                float cur = accum[k] + (float)packed_dwpe[k];
                stochastic_rounding_at(cur, &packed_dwpe[k], rnd);
            }
            store128(dwpe_tc, packed_dwpe);
        });
    }

    // ---- Step 1: Sort inputs into buckets (host, identical to CUDA version) ----
    // Uses inputs_cpu (host copy), never reads device memory.
    size_t c_per_group = 32 * x128::size; // WARP_SIZE(32) * vec
    size_t num_c_groups = CEIL_DIV((size_t)C, c_per_group);

    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    buckets.reserve((size_t)B * T);
    for (uint64_t bt = 0; bt < (uint64_t)(B * T); bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)(uint32_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * (uint64_t)(uint32_t)inputs_cpu[bt]].push_back(data);
        }
    }

    // ---- Step 2: Sort buckets by size in descending order ----
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = (int)sortedBuckets.size();
    int workload_index = 0;
    for (int bucket_index = 0; bucket_index < num_buckets; bucket_index++) {
        const auto& bucket = sortedBuckets[bucket_index];
        bucket_info[bucket_index*4 + 0] = workload_index; // bucket start
        bucket_info[bucket_index*4 + 1] = (int)bucket.second.size(); // bucket size
        bucket_info[bucket_index*4 + 2] = (int)((bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1)); // bucket ix
        bucket_info[bucket_index*4 + 3] = (int)((bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1)); // bucket c
        for (uint64_t v : bucket.second) {
            workload_indices[workload_index++] = (int)(v & ((1ULL<<31ULL)-1ULL));
        }
    }

    // ---- Step 3: device-side scatter + deterministic ordered accumulation ----
    // Copy bucket tables to device (small, fire-and-forget), then:
    //  pass A: gather dout rows into per-(bucket item) staging in `scratch`
    //          (float). Staging layout: workload order, C floats per item.
    //  pass B: ordered accumulation per (bucket, channel) over the bucket's
    //          items in workload order + stochastic rounding (same seeds as
    //          the previous host path: seed + bucket*32 + lane + k).
    size_t total_items = (size_t)workload_index; // == B*T*num_c_groups
    if (total_items == 0) { return; }
    // Per-item channel group, precomputed on host into the persistent buffer so
    // the gather kernel is O(1). The buffer lives in the model (not on the
    // stack), so the device copy below can stay fire-and-forget.
    for (int bucket = 0; bucket < num_buckets; bucket++) {
        int bstart = bucket_info[bucket*4+0];
        int bsize = bucket_info[bucket*4+1];
        int bc = bucket_info[bucket*4+3];
        for (int item = 0; item < bsize; item++) {
            item_cgroup[(size_t)bstart + item] = bc;
        }
    }
    float* stage = scratch; // reduce_scratch covers the worst case (see allocation)
    size_t vec = x128::size;
    size_t vecs_per_group = c_per_group / vec; // 32
    q.memcpy(d_workload, workload_indices, total_items * sizeof(int));
    q.memcpy(d_buckets, bucket_info, (size_t)num_buckets * 4 * sizeof(int));
    // item_cgroup lives in the model: fire-and-forget, ordered before the kernels below
    q.memcpy(d_cgroup, item_cgroup, total_items * sizeof(int));
    const floatX* dout_ptr = dout;
    float* stage_ptr = stage;
    // Pass A: gather (one work-item per (item, lane-vec))
    q.parallel_for(sycl::range<2>(total_items, vecs_per_group), [=](sycl::id<2> id) {
        size_t item = id[0];
        size_t lane = id[1];
        int bt = d_workload[item];
        int bc = d_cgroup[item];
        size_t c = (size_t)bc * c_per_group + lane * vec;
        // rows beyond C in the last group are padding: still gather-safe via mask
        float* dst = stage_ptr + item * c_per_group + lane * vec;
        if (c + vec <= (size_t)C) {
            const floatX* src = dout_ptr + (size_t)bt * C + c;
            for (size_t k = 0; k < vec; k++) { dst[k] = (float)src[k]; }
        } else {
            for (size_t k = 0; k < vec; k++) {
                dst[k] = (c + k < (size_t)C) ? (float)dout_ptr[(size_t)bt * C + c + k] : 0.0f;
            }
        }
    });
    // Pass B: ordered accumulation + stochastic rounding, one item per (bucket, lane, k)
    floatX* dwte_ptr = dwte;
    int* d_buckets_ptr = d_buckets;
    float* stage_ptr2 = stage_ptr;
    int* d_workload_ptr = d_workload;
    (void)d_workload_ptr;
    q.parallel_for(sycl::range<1>((size_t)num_buckets * 32 * vec), [=](sycl::id<1> id) {
        size_t gid = id[0];
        size_t k = gid % vec;
        size_t lane = (gid / vec) % 32;
        size_t bucket = gid / (32 * vec);
        int bstart = d_buckets_ptr[bucket*4+0];
        int bsize = d_buckets_ptr[bucket*4+1];
        int bix = d_buckets_ptr[bucket*4+2];
        int bc = d_buckets_ptr[bucket*4+3];
        size_t c = (size_t)bc * c_per_group + lane * vec + k;
        if (c >= (size_t)C) { return; }
        float acc = 0.0f;
        for (int item = 0; item < bsize; item++) {
            acc += stage_ptr2[(size_t)(bstart + item) * c_per_group + lane * vec + k];
        }
        floatX* dst = dwte_ptr + (size_t)bix * C + c;
        unsigned int rnd = Get2dNoiseUint((int)lane, (int)(bucket * 32 + lane + 1), seed + (unsigned int)(bucket * 32 + lane + k));
        float cur = acc + (float)(*dst);
        stochastic_rounding_at(cur, dst, rnd);
    });
    // device tables persist in the model (no per-step alloc/free, no host task)
}

#endif
