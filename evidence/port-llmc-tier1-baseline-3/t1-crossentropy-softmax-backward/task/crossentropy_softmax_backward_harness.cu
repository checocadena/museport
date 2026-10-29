/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_softmax_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference crossentropy_softmax_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel that just parallelizes over B,T,V
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_softmax_backward1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    const int N = B * T * V;
    const int grid_size = ceil_div(N, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dlogits, dlosses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_softmax_backward(int kernel_num,
                           float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_softmax_backward1(dlogits, dlosses, probs, targets, B, T, V, block_size);
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
    float* dlosses = (float*)malloc(sizeof(float) * (size_t)(B*T));
    if (fread(dlosses, sizeof(float), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at dlosses\n"); return 2;
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
    float* d_dlosses;
    cudaCheck(cudaMalloc(&d_dlosses, sizeof(float) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, sizeof(float) * (size_t)(B*T), cudaMemcpyHostToDevice));
    float* dlogits = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    float* d_dlogits;
    cudaCheck(cudaMalloc(&d_dlogits, sizeof(float) * (size_t)(B*T*V)));
    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_dlogits, 0, sizeof(float) * (size_t)(B*T*V)));
        crossentropy_softmax_backward(k, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(dlogits, d_dlogits, sizeof(float) * (size_t)(B*T*V), cudaMemcpyDeviceToHost));
        fwrite(dlogits, sizeof(float), (size_t)(B*T*V), fout);
    }
    fclose(fout);
    return 0;
}
