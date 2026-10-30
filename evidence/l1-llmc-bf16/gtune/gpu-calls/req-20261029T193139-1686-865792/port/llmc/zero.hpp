/*
Single-device stand-in for zero.cuh (ZeRO sharding / NCCL / MPI).
Keeps the single-process path: no-ops for reductions, rank 0 of 1.
SYCL port: no NCCL, MPI or multi-GPU.
*/

#ifndef LLMC_ZERO_HPP
#define LLMC_ZERO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

// defines: fcloseCheck, fwriteCheck
#include "utils.h"

// ----------------------------------------------------------------------------
// Parameters specific to training on multiple GPUs (single-device version).
typedef struct {
    int process_rank;      // 0
    int num_processes;     // 1
    int local_device_idx;  // 0
    int zero_stage;        // 0
    size_t shard_num_parameters;
} MultiGpuConfig;

// one global variable to hold the multi-GPU configuration for this process
inline MultiGpuConfig multi_gpu_config;

// ----------------------------------------------------------------------------

inline MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node, char* server_ip, char* fs_path, char* init_method) {
    (void)num_processes; (void)process_rank; (void)gpus_per_node;
    (void)server_ip; (void)fs_path; (void)init_method;
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    result.zero_stage = 0;
    result.shard_num_parameters = 0;
    return result;
}

inline void multi_gpu_config_free(MultiGpuConfig* config) {
    (void)config;
}

inline void multi_gpu_barrier(const MultiGpuConfig* config) {
    (void)config;
}

// Offset and size of a tensor shard
typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;

// Get info about sharding for a tensor of elements many numbers
inline ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* config, int shard_at_stage) {
    (void)config; (void)shard_at_stage;
    return {0, elements};
}

// Single process: nothing to reduce.
template<int N>
inline void multi_gpu_async_reduce_gradient(
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* config, sycl::queue& q) {
    (void)pointers; (void)pointers_sizes; (void)config; (void)q;
    return;
}

// convenience macro that only prints if the rank of process is zero
#define printf0(...) if (::multi_gpu_config.process_rank == 0) { printf(__VA_ARGS__); }

inline void set_zero_configs(MultiGpuConfig* config, int zero_stage, size_t total_parameters) {
    config->zero_stage = 0;
    config->shard_num_parameters = total_parameters;
    // Check the Zero Stage and define sharding parameters
    if (zero_stage == 0) {
        printf0("| Zero Optimization is disabled                                              |\n");
    }
    else if (zero_stage == 1) {
        if (total_parameters % (size_t)config->num_processes != 0) {
            printf0("| Zero Optimization is disabled, Can't equally partition parameters          |\n");
            config->zero_stage = 0;
        }
        else {
            config->zero_stage = 1;
            config->shard_num_parameters = total_parameters / (size_t)config->num_processes;
        }
    }
    else{
        printf0("| Disabling Zero Optimization, Zero Stage2 and Stage3 are not yet supported  |\n");
        config->zero_stage = 0;
    }
}

// Compute sum of a single CPU value across all GPU processes. No-op when multi-GPU is disabled.
inline float multi_gpu_cpu_float_sum(float value, MultiGpuConfig* config) {
    (void)config;
    return value;
}

#endif
