/*
Single-process configuration. SYCL 2020 port of llmc/zero.cuh (single-process
path only: no NCCL, MPI or multi-GPU).
*/

#ifndef LLMC_ZERO_H
#define LLMC_ZERO_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

// defines: fcloseCheck, fwriteCheck
#include "utils.h"

// ----------------------------------------------------------------------------
// Parameters specific to training on multiple GPUs (single-process here).
typedef struct {
    int process_rank;      // Rank of this process among all processes. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int local_device_idx;  // This process device index on current machine. 0 if no multi-GPU.

    // Zero Redundancy Optimizer stage (only 0 supported in this port)
    int zero_stage;
    size_t shard_num_parameters;
} MultiGpuConfig;

// one global variable to hold the multi-GPU configuration for this process
// inline, so we can include this header multiple times without getting multiple definitions
inline MultiGpuConfig multi_gpu_config;

// Offset and size of a tensor shard
typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;

// Get info about sharding for a tensor of elements many numbers
inline ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* config, int shard_at_stage) {
    const int nproc = config->num_processes;
    if(config->zero_stage >= shard_at_stage) {
        if (elements % (size_t)nproc != 0) {
            fprintf(stderr, "Number of elements %zu must be a multiple of the number of processes %d\n", elements, nproc);
            exit(EXIT_FAILURE);
        }
        return {(ptrdiff_t) (config->process_rank * (elements / (size_t)nproc)), elements / (size_t)nproc};
    } else {
        return {0, elements};
    }
}

inline MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node,
                                            char* server_ip, char* fs_path, char* init_method) {
    (void)num_processes; (void)process_rank; (void)gpus_per_node;
    (void)server_ip; (void)fs_path; (void)init_method;
    printf("Multi-GPU support is disabled. Using a single device.\n");
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

// Single process: no-op.
template<int N>
inline void multi_gpu_async_reduce_gradient(
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* config) {
    (void)pointers; (void)pointers_sizes; (void)config;
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

// Compute sum of a single CPU value across all processes. No-op when multi-GPU is disabled.
inline float multi_gpu_cpu_float_sum(float value, MultiGpuConfig* config) {
    (void)config;
    return value;
}

#endif
