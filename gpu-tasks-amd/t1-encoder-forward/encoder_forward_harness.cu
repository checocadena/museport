/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_forward.cu -o encoder_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./encoder_forward 1

version 2 is more optimized, parallelizes over all of B,T,C
./encoder_forward 2

version 3 is like version 2 but uses float4 reads/writes
./encoder_forward 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>

// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
// museport: CPU reference encoder_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation into kernel, parallelize over B,T, loop over C
__global__ void encoder_forward_kernel1(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;

    if (idx < N) {
        int b = idx / T;
        int t = idx % T;
        floatX* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C;
        const floatX* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++) {
            out_bt[i] = (floatX)((float)wte_ix[i] + (float)wpe_t[i]);
        }
    }
}

// optimized implementation: parallelize over all of B,T,C
__global__ void encoder_forward_kernel2(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;
        *out_btc = (floatX)((float)*wte_ix + (float)*wpe_tc);
    }
}

__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;

        x128 packed_out;
        x128 wte = load128cs(wte_ix);
        x128 wpe = load128cs(wpe_tc);
        #pragma unroll
        for (int k = 0; k < wte.size; k++) {
            packed_out[k] = (floatX)((float)wte[k] + (float)wpe[k]);
        }
        store128(out_btc, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward1(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward2(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward3(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void encoder_forward(int kernel_num,
                     floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 3:
            encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
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
    int* inp = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(inp, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* wte = (float*)malloc(sizeof(float) * (size_t)(V*C));
    if (fread(wte, sizeof(float), (size_t)(V*C), fin) != (size_t)(V*C)) {
        fprintf(stderr, "input too short at wte\n"); return 2;
    }
    float* wpe = (float*)malloc(sizeof(float) * (size_t)(T*C));
    if (fread(wpe, sizeof(float), (size_t)(T*C), fin) != (size_t)(T*C)) {
        fprintf(stderr, "input too short at wpe\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    int* d_inp;
    cudaCheck(cudaMalloc(&d_inp, sizeof(int) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_inp, inp, sizeof(int) * (size_t)(B*T), cudaMemcpyHostToDevice));
    float* d_wte;
    cudaCheck(cudaMalloc(&d_wte, sizeof(float) * (size_t)(V*C)));
    cudaCheck(cudaMemcpy(d_wte, wte, sizeof(float) * (size_t)(V*C), cudaMemcpyHostToDevice));
    float* d_wpe;
    cudaCheck(cudaMalloc(&d_wpe, sizeof(float) * (size_t)(T*C)));
    cudaCheck(cudaMemcpy(d_wpe, wpe, sizeof(float) * (size_t)(T*C), cudaMemcpyHostToDevice));
    float* out = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_out;
    cudaCheck(cudaMalloc(&d_out, sizeof(float) * (size_t)(B*T*C)));
    const int variants[] = {1, 2, 3};
    for (int vi = 0; vi < 3; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_out, 0, sizeof(float) * (size_t)(B*T*C)));
        encoder_forward(k, d_out, d_inp, d_wte, d_wpe, B, T, C, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out, d_out, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(out, sizeof(float), (size_t)(B*T*C), fout);
    }
    fclose(fout);
    return 0;
}
