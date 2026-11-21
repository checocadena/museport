/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_backward.cu -o encoder_backward

version 1 is naive port from CPU code to kernel
parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward 1

version 2 is another naive port
parallelizes over C, loops over B,T; much slower than version 1
./encoder_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
// museport: CPU reference encoder_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
__global__ void encoder_backward_kernel1(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

// naive implementation that parallelizes over C and loops over B,T
// but it gets rid of atomics
__global__ void encoder_backward_kernel2(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) { return; } // guard
    int BT = B * T;
    for (int i = 0; i < BT; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_backward1(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_backward_kernel1<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward2(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int grid_size = ceil_div(C, block_size);
    encoder_backward_kernel2<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void encoder_backward(int kernel_num,
                     float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(dwte, dwpe, dout, inp, B, T, C, block_size);
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
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int V = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    int* inp = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(inp, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_dout;
    cudaCheck(cudaMalloc(&d_dout, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_dout, dout, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    int* d_inp;
    cudaCheck(cudaMalloc(&d_inp, sizeof(int) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_inp, inp, sizeof(int) * (size_t)(B*T), cudaMemcpyHostToDevice));
    float* dwte = (float*)malloc(sizeof(float) * (size_t)(V*C));
    float* d_dwte;
    cudaCheck(cudaMalloc(&d_dwte, sizeof(float) * (size_t)(V*C)));
    float* dwpe = (float*)malloc(sizeof(float) * (size_t)(T*C));
    float* d_dwpe;
    cudaCheck(cudaMalloc(&d_dwpe, sizeof(float) * (size_t)(T*C)));
    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_dwte, 0, sizeof(float) * (size_t)(V*C)));
        cudaCheck(cudaMemset(d_dwpe, 0, sizeof(float) * (size_t)(T*C)));
        encoder_backward(k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(dwte, d_dwte, sizeof(float) * (size_t)(V*C), cudaMemcpyDeviceToHost));
        fwrite(dwte, sizeof(float), (size_t)(V*C), fout);
        cudaCheck(cudaMemcpy(dwpe, d_dwpe, sizeof(float) * (size_t)(T*C), cudaMemcpyDeviceToHost));
        fwrite(dwpe, sizeof(float), (size_t)(T*C), fout);
    }
    fclose(fout);
    return 0;
}
