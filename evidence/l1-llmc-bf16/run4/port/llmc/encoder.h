/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
SYCL 2020 port of llmc/encoder.cuh. Single device, no NCCL/MPI.
*/
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

inline void encoder_forward_kernel(floatX* out,
                                   const int* inp, const floatX* wte, const floatX* wpe,
                                   size_t B, size_t T, size_t C) {
    size_t N = B * T * C;
    constexpr size_t BLOCK = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(N, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= N) { return; }
            size_t bt = idx / C;
            size_t b = bt / T;
            size_t t = bt % T;
            int ix = inp[b * T + t];
            out[idx] = (floatX)((float)wte[(size_t)ix * C + (idx % C)] + (float)wpe[t * C + (idx % C)]);
        }).wait();
}

inline void encoder_backward(floatX* dwte, floatX* dwpe,
                             int* workload_indices, Int4* bucket_info,    // cpu scratch buffers
                             const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/device inputs
                             size_t B, size_t T, size_t C, unsigned int seed) {
    NVTX_RANGE_FN();
    constexpr size_t BLOCK = 256;

    // ---- wpe backward: deterministic sum over batch per (t, c) ----
    {
        size_t N = T * C;
        sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(N, BLOCK) * BLOCK),
                                                  sycl::range<1>(BLOCK)),
            [=](sycl::nd_item<1> it) {
                size_t idx = it.get_global_linear_id();
                if (idx >= N) { return; }
                size_t t = idx / C;
                size_t c = idx % C;
                float accum = 0.0f;
                for (size_t b = 0; b < B; b++) {
                    accum += (float)dout[(b * T + t) * C + c];
                }
                float old = (float)dwpe[idx];
                stochastic_rounding(accum + old, &dwpe[idx], seed + (unsigned int)idx,
                                    (int)(idx & 0x7fffffff), 0);
            }).wait();
    }

    // ---- wte backward: deterministic bucketed accumulation on host staging ----
    // The CUDA kernel buckets (token, channel-group) pairs and reduces each bucket
    // deterministically on the device. On the CPU device we do the identical
    // deterministic reduction on the host (bit-exact fp32 accumulation order is
    // defined by sorted buckets), then stochastic-round once per element.
    size_t num_c_groups = CEIL_DIV(C, (size_t)8);
    // download dout once
    size_t dout_n = B * T * C;
    std::vector<float> dout_f(dout_n);
    {
        std::vector<floatX> tmp(dout_n);
        sycl_memcpy_d2h(tmp.data(), dout, dout_n * sizeof(floatX));
        for (size_t i = 0; i < dout_n; i++) { dout_f[i] = (float)tmp[i]; }
    }
    // download current dwte
    // (need V = num rows; recover from bucket token ids below; instead read full dwte sized B*T*C? no.)
    // We read dwte per-bucket below: first gather bucket structure on host.
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    buckets.reserve(B * T * 2);
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            uint64_t data = bt + (c_group << 32ULL) + ((uint64_t)(uint32_t)inputs_cpu[bt] << 42ULL);
            buckets[c_group + num_c_groups * (uint64_t)(uint32_t)inputs_cpu[bt]].push_back(data);
        }
    }
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a,
                 const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });
    (void)workload_indices; (void)bucket_info; (void)inp; // device copies not needed in SYCL port
    // determine vocab rows needed
    size_t Vrows = 0;
    for (auto& bk : sortedBuckets) {
        uint64_t ix = (bk.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1);
        if (ix + 1 > Vrows) { Vrows = ix + 1; }
    }
    size_t dwte_n = Vrows * C;
    std::vector<floatX> dwte_tmp(dwte_n);
    sycl_memcpy_d2h(dwte_tmp.data(), dwte, dwte_n * sizeof(floatX));

    int bucket_index = 0;
    for (const auto& bucket : sortedBuckets) {
        uint64_t b0 = bucket.second[0];
        uint64_t bucket_ix = (b0 >> 42ULL) & ((1ULL << 20ULL) - 1);
        uint64_t bucket_c = (b0 >> 32ULL) & ((1ULL << 10ULL) - 1);
        size_t c_start = (size_t)bucket_c * 8;
        size_t c_end = c_start + 8 < C ? c_start + 8 : C;
        // deterministic order: workload order within bucket = insertion order (bt ascending,
        // c_group fixed), matching CUDA's bucketed loop
        for (size_t c = c_start; c < c_end; c++) {
            float accum = 0.0f;
            for (uint64_t item : bucket.second) {
                uint64_t bt = item & ((1ULL << 31ULL) - 1ULL);
                accum += dout_f[bt * C + c];
            }
            size_t idx = (size_t)bucket_ix * C + c;
            float old = (float)dwte_tmp[idx];
            // seed must be unique per parameter like CUDA's
            // (seed + bucket*WARP + lane + k); use deterministic equivalent:
            unsigned int s = seed + (unsigned int)bucket_index * 8u + (unsigned int)(c - c_start);
            stochastic_rounding(accum + old, &dwte_tmp[idx], s, (int)(idx & 0x7fffffff), 0);
        }
        bucket_index++;
    }
    sycl_memcpy_h2d(dwte, dwte_tmp.data(), dwte_n * sizeof(floatX));
}

// ----------------------------------------------------------------------------
// kernel launchers

inline void encoder_forward(floatX* out,
                            const int* inp, const floatX* wte, const floatX* wpe,
                            size_t B, size_t T, size_t C) {
    NVTX_RANGE_FN();
    encoder_forward_kernel(out, inp, wte, wpe, B, T, C);
}
