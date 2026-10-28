/*
Kernels for the AdamW optimizer.

References:
  * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
nvcc -lcublas -lcublasLt adamw.cu -o adamw
nvcc -O3 --use_fast_math -lcublas -lcublasLt adamw.cu -o adamw

./adamw

TODO(general):
amsgrad=True

TODO(perf):
dtype
thread coarsening/ILP
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"


// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference adamw_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// utility functions

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

// naive fused kernel
__global__ void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   // update the first moment (momentum)
   m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
   // update the second moment (RMSprop)
   v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];
   float m_hat = m_memory[i] / beta1_correction;
   float v_hat = v_memory[i] / beta2_correction;
   params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params_memory[i]);
}

// Slightly more optimized AdamW kernel by:
// * loading data that is accessed more than once into registers,
// * using optimized linear interpolation for the moment updates.
__global__ void adamw_kernel2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}


// ----------------------------------------------------------------------------
// kernel launcher

// version 1: naive dispatch to naive kernel
void adamw_dispatch1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel1<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

// version 2: naive dispatch to slightly optimized kernel
void adamw_dispatch2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel2<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void adamw(int kernel_num,
           float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
           float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num) {
        case 1:
            adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
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
    if (argc != 5) {
        fprintf(stderr, "usage: %s <input> <output> N T\n", argv[0]);
        return 2;
    }
    const int N = atoi(argv[3]);
    const int T = atoi(argv[4]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* params_init = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(params_init, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at params_init\n"); return 2;
    }
    float* grads_memory = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(grads_memory, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at grads_memory\n"); return 2;
    }
    float* m_init = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(m_init, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at m_init\n"); return 2;
    }
    float* v_init = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(v_init, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at v_init\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_params_init;
    cudaCheck(cudaMalloc(&d_params_init, sizeof(float) * (size_t)(N)));
    cudaCheck(cudaMemcpy(d_params_init, params_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
    float* d_grads_memory;
    cudaCheck(cudaMalloc(&d_grads_memory, sizeof(float) * (size_t)(N)));
    cudaCheck(cudaMemcpy(d_grads_memory, grads_memory, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
    float* d_m_init;
    cudaCheck(cudaMalloc(&d_m_init, sizeof(float) * (size_t)(N)));
    cudaCheck(cudaMemcpy(d_m_init, m_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
    float* d_v_init;
    cudaCheck(cudaMalloc(&d_v_init, sizeof(float) * (size_t)(N)));
    cudaCheck(cudaMemcpy(d_v_init, v_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
    float* params_memory = (float*)malloc(sizeof(float) * (size_t)(N));
    float* d_params_memory;
    cudaCheck(cudaMalloc(&d_params_memory, sizeof(float) * (size_t)(N)));
    float* m_memory = (float*)malloc(sizeof(float) * (size_t)(N));
    float* d_m_memory;
    cudaCheck(cudaMalloc(&d_m_memory, sizeof(float) * (size_t)(N)));
    float* v_memory = (float*)malloc(sizeof(float) * (size_t)(N));
    float* d_v_memory;
    cudaCheck(cudaMalloc(&d_v_memory, sizeof(float) * (size_t)(N)));
    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemcpy(d_params_memory, params_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_m_memory, m_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_v_memory, v_init, sizeof(float) * (size_t)(N), cudaMemcpyHostToDevice));
        adamw(k, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, T, (long)N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(params_memory, d_params_memory, sizeof(float) * (size_t)(N), cudaMemcpyDeviceToHost));
        fwrite(params_memory, sizeof(float), (size_t)(N), fout);
        cudaCheck(cudaMemcpy(m_memory, d_m_memory, sizeof(float) * (size_t)(N), cudaMemcpyDeviceToHost));
        fwrite(m_memory, sizeof(float), (size_t)(N), fout);
        cudaCheck(cudaMemcpy(v_memory, d_v_memory, sizeof(float) * (size_t)(N), cudaMemcpyDeviceToHost));
        fwrite(v_memory, sizeof(float), (size_t)(N), fout);
    }
    fclose(fout);
    return 0;
}
