/*
Kernels for residual forward pass fused with layernorm

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_forward.cu -o fused_residual_forward

version 1 is naive port from CPU code to kernel
./fused_residual_forward 1
version 2 packs input into 128 bit memory reads
./fused_residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <cuda_runtime.h>

// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

// museport: CPU reference residual_forward_cpu() removed (it is the hidden check)

// museport: CPU reference layernorm_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
__global__ void layernorm_forward_kernel1(floatX* out, floatX* mean, floatX* rstd,
                                          const floatX* inp, const floatX* weight, const floatX* bias,
                                          int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// naive fusion; uncoalesced access pattern leads to terrible performance
__global__ void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = (floatX)o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = (floatX)m;
    rstd[idx] = (floatX)s;
}

// handle one token per warp for coalesced access
__global__ void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = warpReduceSum(m);

    m = m / C;
    float v = 0.0f;
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }

    v = warpReduceSum(v);
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// vectorized loading, single pass stats, streaming access and zigzag loop
__global__ void fused_residual_forward_kernel4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = threadIdx.x * x128::size;
    for(; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
            sum_sq += (float)out[k] * (float)out[k];
        }
        store128(residual + c, out);
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    float m = sum / C;
    float v = sum_sq / C - m * m;
    float s = rsqrtf(v + eps);

    c -= WarpSize * x128::size;
    for(; c >= 0; c -= WarpSize * x128::size) {
        const x128 res = load128cs(residual + c);
        const x128 w = load128(weight + c);
        const x128 b = load128(bias + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// what do you want in shared memory? EVERYTHING!
// thus, we no longer require zigzag loops and can do the numerically more stable variance estimation
// needs special attention in the kernel launcher to ensure we have enough smem.
__global__ void fused_residual_forward_kernel5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WarpSize * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}


// using multiple warps per token, and keep threads persistent, so we never have to reload weights and biases
// if we had one warp per token, though, this would require us to use a huge amount of shared memory. Therefore,
// we use multiple warps per token; but generally we cannot use the entire block, because that would give too
// little work per warp to be effective (each warp processes 256 bfloat16 elements, so for C=768 more than 3 warps
// will just mean idle). Therefore, we add a z dimension, where warps with different z handle different tokens.
// all this makes the launcher logic more complicated :(
__global__ void fused_residual_forward_kernel6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    // weights and biases are  shared among all tokens
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // residual output (input to layernorm) is independent for each sub-block indicates by threadIdx.z
    x128* s_res = reinterpret_cast<x128*>(params + (2 + threadIdx.z) * C * sizeof(floatX));
    // similarly, each sub-block needs its own reduction buffers
    float* s_mean = reinterpret_cast<float*>(params + (2 + blockDim.z) * C * sizeof(floatX) + threadIdx.z * 32 * sizeof(float));
    float* s_var = reinterpret_cast<float*>(params + (2 + blockDim.z) * C * sizeof(floatX) + 32 * sizeof(float) * (blockDim.z + threadIdx.z));

    int cidx = (threadIdx.x + WarpSize * threadIdx.y) * x128::size;
    int step = blockDim.y * WarpSize * x128::size;

    for(int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size] = load128(bias + c);
    }

    // the block-level reductions will cause sync before the first time we read these
    // => no syncthreads needed here

    // loop over all tokens
    for(int tidx = blockIdx.x * blockDim.z + threadIdx.z; tidx < N; tidx += gridDim.x * blockDim.z) {
        // adjust pointers to current token
        floatX* residual_bt = residual + C * tidx;
        floatX* normed_bt = normed + C * tidx;
        const floatX* inp1_bt = inp1 + C * tidx;
        const floatX* inp2_bt = inp2 + C * tidx;

        const float eps = 1e-5f;
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (float) in1[k] + (float) in2[k];
                sum += (float) out[k];
            }
            store128cs(residual_bt + c, out);
            s_res[c / x128::size] = out;
        }
        sum = warpReduceSum(sum);
        if(threadIdx.x == 0) {
            s_mean[threadIdx.y] = sum;
        }
        __syncthreads();
        float m = warpReduceSum(threadIdx.x < blockDim.y ? s_mean[threadIdx.x] : 0.f) / C;
        // normally, we'd syncthread here to make sure that no warp is already at the next
        // iteration of the loop, messing with s_mean. The fact that we interleave s_mean and s_var means
        // we don't need these additional syncs.
        float v = 0.f;

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float) res[k] - m) * ((float) res[k] - m);
            }
        }

        v = warpReduceSum(v);
        if(threadIdx.x == 0) {
            s_var[threadIdx.y] = v;
        }
        __syncthreads();
        v = warpReduceSum(threadIdx.x < blockDim.y ? s_var[threadIdx.x] : 0.f) / C;
        float s = rsqrtf(v + eps);

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float) res[k] - m); // normalized output
                float o = n * (float) w[k] + (float) b[k]; // scale and shift it
                out[k] = o;
            }

            store128(normed_bt + c, out);
        }
        // cache the mean and rstd for the backward pass later
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            mean[tidx] = m;
            rstd[tidx] = s;
        }
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void fused_residual_forward1(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    residual_forward_kernel1<<<grid_size_resid, block_size>>>(residual, inp1, inp2, N*C);
    cudaCheck(cudaGetLastError());
    const int grid_size_ln = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size_ln, block_size>>>(normed, mean, rstd, residual, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size));
    fused_residual_forward2<<<grid_size, block_size>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_residual_forward3<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_residual_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if(status == cudaSuccess) {
        fused_residual_forward_kernel5<<<grid_size, dim3(32, block_y), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                               weight, bias, N, C);
    } else {
        fused_residual_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                         weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int warps_per_token = max(1, C / Packed128<floatX>::size / 32);
    int total_warps = block_size / 32;
    int block_z = max(1, total_warps / warps_per_token);
    int block_y = max(1, total_warps / block_z);
    size_t smem = (2 + block_z) * C * sizeof(floatX) + 64 * sizeof(float) * block_z;

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if(status == cudaSuccess) {
        const int num_blocks = max(1, cuda_threads_per_SM * cuda_num_SMs / block_size);
        fused_residual_forward_kernel6<<<num_blocks, dim3(32, block_y, block_z), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                               weight, bias, N, C);
    } else {
        const int grid_size = ceil_div(N, total_warps);
        fused_residual_forward_kernel4<<<grid_size, dim3(32, total_warps)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                         weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 4:
            fused_residual_forward4(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 5:
            fused_residual_forward5(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 6:
            fused_residual_forward6(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
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
    float* weight = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(weight, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* bias = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(bias, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at bias\n"); return 2;
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
    float* d_weight;
    cudaCheck(cudaMalloc(&d_weight, sizeof(float) * (size_t)(C)));
    cudaCheck(cudaMemcpy(d_weight, weight, sizeof(float) * (size_t)(C), cudaMemcpyHostToDevice));
    float* d_bias;
    cudaCheck(cudaMalloc(&d_bias, sizeof(float) * (size_t)(C)));
    cudaCheck(cudaMemcpy(d_bias, bias, sizeof(float) * (size_t)(C), cudaMemcpyHostToDevice));
    float* residual = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_residual;
    cudaCheck(cudaMalloc(&d_residual, sizeof(float) * (size_t)(B*T*C)));
    float* normed = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_normed;
    cudaCheck(cudaMalloc(&d_normed, sizeof(float) * (size_t)(B*T*C)));
    float* mean = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_mean;
    cudaCheck(cudaMalloc(&d_mean, sizeof(float) * (size_t)(B*T)));
    float* rstd = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_rstd;
    cudaCheck(cudaMalloc(&d_rstd, sizeof(float) * (size_t)(B*T)));
    const int variants[] = {1, 2, 3, 4, 5, 6};
    for (int vi = 0; vi < 6; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_residual, 0, sizeof(float) * (size_t)(B*T*C)));
        cudaCheck(cudaMemset(d_normed, 0, sizeof(float) * (size_t)(B*T*C)));
        cudaCheck(cudaMemset(d_mean, 0, sizeof(float) * (size_t)(B*T)));
        cudaCheck(cudaMemset(d_rstd, 0, sizeof(float) * (size_t)(B*T)));
        fused_residual_forward(k, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias, B*T, C, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(residual, d_residual, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(residual, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(cudaMemcpy(normed, d_normed, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(normed, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(cudaMemcpy(mean, d_mean, sizeof(float) * (size_t)(B*T), cudaMemcpyDeviceToHost));
        fwrite(mean, sizeof(float), (size_t)(B*T), fout);
        cudaCheck(cudaMemcpy(rstd, d_rstd, sizeof(float) * (size_t)(B*T), cudaMemcpyDeviceToHost));
        fwrite(rstd, sizeof(float), (size_t)(B*T), fout);
    }
    fclose(fout);
    return 0;
}
