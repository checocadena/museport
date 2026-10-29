/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_forward.cu -o crossentropy_forward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_forward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference crossentropy_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void crossentropy_forward_kernel1(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_forward1(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V,
                            const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(losses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_forward(int kernel_num,
                          float* losses,
                          const float* probs, const int* targets,
                          int B, int T, int V,
                          const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_forward1(losses, probs, targets, B, T, V, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char** argv) {
    setup_main();
    if (argc != 6) {
        fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int V = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* probs = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    if (fread(probs, sizeof(float), (size_t)(B*T*V), fin) != (size_t)(B*T*V)) {
        fprintf(stderr, "input too short at probs\n"); return 2;
    }
    int* targets = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(targets, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at targets\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_probs;
    cudaCheck(cudaMalloc(&d_probs, sizeof(float) * (size_t)(B*T*V)));
    cudaCheck(cudaMemcpy(d_probs, probs, sizeof(float) * (size_t)(B*T*V), cudaMemcpyHostToDevice));
    int* d_targets;
    cudaCheck(cudaMalloc(&d_targets, sizeof(int) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_targets, targets, sizeof(int) * (size_t)(B*T), cudaMemcpyHostToDevice));
    float* out = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_out;
    cudaCheck(cudaMalloc(&d_out, sizeof(float) * (size_t)(B*T)));
    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_out, 0, sizeof(float) * (size_t)(B*T)));
        crossentropy_forward(k, d_out, d_probs, d_targets, B, T, V, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out, d_out, sizeof(float) * (size_t)(B*T), cudaMemcpyDeviceToHost));
        fwrite(out, sizeof(float), (size_t)(B*T), fout);
    }
    fclose(fout);
    return 0;
}
