/*
Kernels for gelu backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_backward.cu -o gelu_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_backward 1

version 2 uses the Packed128 data structure
./gelu_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// museport: CPU reference gelu_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

__global__ void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }

        store128(dinp + i, packed_dinp);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                  floatX* dinp, 
                  const floatX* inp, 
                  const floatX* dout,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B * T * C, block_size);
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
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_inp;
    cudaCheck(cudaMalloc(&d_inp, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_inp, inp, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    float* d_dout;
    cudaCheck(cudaMalloc(&d_dout, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_dout, dout, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    float* dinp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_dinp;
    cudaCheck(cudaMalloc(&d_dinp, sizeof(float) * (size_t)(B*T*C)));
    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_dinp, 0, sizeof(float) * (size_t)(B*T*C)));
        gelu_backward(k, d_dinp, d_inp, d_dout, B, T, C, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(dinp, d_dinp, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
    }
    fclose(fout);
    return 0;
}
