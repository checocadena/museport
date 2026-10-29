/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
SYCL 2020 port of encoder.cuh (single device, deterministic).
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
    }).wait();
}

// wte backward: deterministic bucketed accumulation.
// Same bucketing as CUDA version (computed on host); each bucket is accumulated
// in bucket order with a sequential, deterministic summation per channel group.
inline void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // device outputs & scratch
                      int* workload_indices, int* bucket_info,    // cpu scratch buffers (4 ints per bucket)
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
        }).wait();
    }

    // ---- Step 1: Sort inputs into buckets (host, identical to CUDA version) ----
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

    // ---- Step 3: accumulate wte buckets deterministically on host-visible USM ----
    // dout/dwte are USM shared: read directly on host in bucket order. Each channel
    // group is summed in workload order, then stochastic rounding uses the same
    // per-parameter seed formula as the CUDA kernel (seed + bucket*32 + lane + k).
    q.wait();
    {
        size_t vec = x128::size;
        for (int bucket = 0; bucket < num_buckets; bucket++) {
            int bstart = bucket_info[bucket*4+0];
            int bsize = bucket_info[bucket*4+1];
            int bix = bucket_info[bucket*4+2];
            int bc = bucket_info[bucket*4+3];
            // process in chunks of 8 items (BLOCK_SIZE/WARP_SIZE like CUDA) but accumulate
            // item order sequentially for determinism
            for (int c_lane = 0; c_lane < 32; c_lane++) {
                size_t c = (size_t)bc * c_per_group + (size_t)c_lane * vec;
                if (c >= (size_t)C) { continue; }
                float accum[x128::size] = {0.0f};
                for (int item = 0; item < bsize; item++) {
                    int bt = workload_indices[bstart + item];
                    const floatX* dout_btc = dout + (size_t)bt * C + c;
                    for (size_t k = 0; k < vec; k++) {
                        accum[k] += (float)dout_btc[k];
                    }
                }
                floatX* dwte_ix = dwte + (size_t)bix * C + c;
                for (size_t k = 0; k < vec; k++) {
                    unsigned int rnd = Get2dNoiseUint(c_lane, bucket * 32 + c_lane + 1, seed + (unsigned int)(bucket * 32 + c_lane + k));
                    float cur = accum[k] + (float)dwte_ix[k];
                    stochastic_rounding_at(cur, &dwte_ix[k], rnd);
                }
            }
        }
    }
    (void)scratch;
}

#endif
