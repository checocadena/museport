/*
Kernels for matmul backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// museport: #include <omp.h> removed (only the hidden CPU reference used it)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference matmul_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
__global__ void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < OC) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

// use shared memory and coarsening + reductions
__global__ void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = shared[0];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // backward to input
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));
    // backward to weight
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &alpha, inp, C, dout, OC, &beta, dweight, C));
    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        dim3 block_dim(block_size);
        dim3 grid_dim(OC);
        size_t shared_mem_size = block_size * sizeof(float);
        matmul_backward_bias_kernel_faster<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, B, T, OC);
    }
}

void matmul_backward(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
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
        fprintf(stderr, "usage: %s <input> <output> B T C OC\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int OC = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*OC));
    if (fread(dout, sizeof(float), (size_t)(B*T*OC), fin) != (size_t)(B*T*OC)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* weight = (float*)malloc(sizeof(float) * (size_t)(OC*C));
    if (fread(weight, sizeof(float), (size_t)(OC*C), fin) != (size_t)(OC*C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* ones = (float*)malloc(sizeof(float) * (size_t)(OC));
    if (fread(ones, sizeof(float), (size_t)(OC), fin) != (size_t)(OC)) {
        fprintf(stderr, "input too short at ones\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_dout;
    cudaCheck(cudaMalloc(&d_dout, sizeof(float) * (size_t)(B*T*OC)));
    cudaCheck(cudaMemcpy(d_dout, dout, sizeof(float) * (size_t)(B*T*OC), cudaMemcpyHostToDevice));
    float* d_inp;
    cudaCheck(cudaMalloc(&d_inp, sizeof(float) * (size_t)(B*T*C)));
    cudaCheck(cudaMemcpy(d_inp, inp, sizeof(float) * (size_t)(B*T*C), cudaMemcpyHostToDevice));
    float* d_weight;
    cudaCheck(cudaMalloc(&d_weight, sizeof(float) * (size_t)(OC*C)));
    cudaCheck(cudaMemcpy(d_weight, weight, sizeof(float) * (size_t)(OC*C), cudaMemcpyHostToDevice));
    float* d_ones;
    cudaCheck(cudaMalloc(&d_ones, sizeof(float) * (size_t)(OC)));
    cudaCheck(cudaMemcpy(d_ones, ones, sizeof(float) * (size_t)(OC), cudaMemcpyHostToDevice));
    float* dinp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_dinp;
    cudaCheck(cudaMalloc(&d_dinp, sizeof(float) * (size_t)(B*T*C)));
    float* dweight = (float*)malloc(sizeof(float) * (size_t)(OC*C));
    float* d_dweight;
    cudaCheck(cudaMalloc(&d_dweight, sizeof(float) * (size_t)(OC*C)));
    float* dbias = (float*)malloc(sizeof(float) * (size_t)(OC));
    float* d_dbias;
    cudaCheck(cudaMalloc(&d_dbias, sizeof(float) * (size_t)(OC)));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_dinp, 0, sizeof(float) * (size_t)(B*T*C)));
        cudaCheck(cudaMemset(d_dweight, 0, sizeof(float) * (size_t)(OC*C)));
        cudaCheck(cudaMemset(d_dbias, 0, sizeof(float) * (size_t)(OC)));
        matmul_backward(k, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(dinp, d_dinp, sizeof(float) * (size_t)(B*T*C), cudaMemcpyDeviceToHost));
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(cudaMemcpy(dweight, d_dweight, sizeof(float) * (size_t)(OC*C), cudaMemcpyDeviceToHost));
        fwrite(dweight, sizeof(float), (size_t)(OC*C), fout);
        cudaCheck(cudaMemcpy(dbias, d_dbias, sizeof(float) * (size_t)(OC), cudaMemcpyDeviceToHost));
        fwrite(dbias, sizeof(float), (size_t)(OC), fout);
    }
    fclose(fout);
    return 0;
}
