// Synthetic llm.c-style dev file for museport's tests: out = SCALE * inp + shift[0].
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define ENABLE_BF16
#include "common.h"

#define SCALE 2.0f

void scale_cpu(float* out, const float* inp, const float* shift, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = SCALE * inp[i] + shift[0];
    }
}

__global__ void scale_kernel1(float* out, const float* inp, const float* shift, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = SCALE * inp[i] + shift[0]; }
}

void scale(int kernel_num, float* out, const float* inp, const float* shift, int N, int bs) {
    switch (kernel_num) {
        case 1: scale_kernel1<<<(N + bs - 1) / bs, bs>>>(out, inp, shift, N); break;
        default: exit(1);
    }
}

int main(int argc, char** argv) {
    setup_main();
    float* inp = (float*)malloc(16 * sizeof(float));
    scale_cpu(inp, inp, inp, 16);
    return 0;
}
