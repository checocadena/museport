/*
Kernels for residual forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt residual_forward.cu -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
version 2 packs input into 128 bit memory reads
./residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

// museport: CPU reference residual_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

__global__ void residual_forward_kernel2(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx);
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k)
        {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
        }
        store128(out + idx, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    residual_forward_kernel2<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void residual_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp1,
                  const floatX* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
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
        fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp1 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp1, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp1\n"); return 2;
    }
    float* inp2 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp2, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp2\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_inp1;
    cudaCheck(cudaMalloc(&d_inp1, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_inp1, inp1, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    float* d_inp2;
    cudaCheck(cudaMalloc(&d_inp2, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_inp2, inp2, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    float* out = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_out;
    cudaCheck(cudaMalloc(&d_out, sizeof(float) * (size_t)(B*T*C)));
    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_out, 0, sizeof(float) * (size_t)(B*T*C)));
        residual_forward(k, d_out, d_inp1, d_inp2, B*T*C, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out, d_out, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(out, sizeof(float), (size_t)(B*T*C), fout);
    }
    fclose(fout);
    return 0;
}
