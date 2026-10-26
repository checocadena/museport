/*
Kernels for matmul backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dpct/blas_utils.hpp>

// museport: #include <omp.h> removed (only the hidden CPU reference used it)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference matmul_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
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
void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC,
                                        uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local;
    int o = item_ct1.get_group(2);      // range [0, OC)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    int block_size = item_ct1.get_local_range(2);
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
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
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, C, B * T, OC, alpha, weight, C, dout,
        OC, beta, dinp, C)));
    // backward to weight
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, C, OC, B * T, alpha, inp, C, dout, OC,
        beta, dweight, C)));
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
        dpct::dim3 block_dim(block_size);
        dpct::dim3 grid_dim(OC);
        /*
        DPCT1083:3: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = block_size * sizeof(float);
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(
                dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                    [=](sycl::nd_item<3> item_ct1) {
                        matmul_backward_bias_kernel_faster(
                            dbias, dout, B, T, OC,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
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
    cudaCheck(DPCT_CHECK_ERROR(
        d_dout = sycl::malloc_device<float>((size_t)(B * T * OC),
                                            dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_dout, dout, sizeof(float) * (size_t)(B * T * OC))
            .wait()));
    float* d_inp;
    cudaCheck(
        DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_inp, inp, sizeof(float) * (size_t)(B * T * C))
            .wait()));
    float* d_weight;
    cudaCheck(
        DPCT_CHECK_ERROR(d_weight = sycl::malloc_device<float>(
                             (size_t)(OC * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_weight, weight, sizeof(float) * (size_t)(OC * C))
            .wait()));
    float* d_ones;
    cudaCheck(DPCT_CHECK_ERROR(d_ones = sycl::malloc_device<float>(
                                   (size_t)(OC), dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_ones, ones, sizeof(float) * (size_t)(OC))
                             .wait()));
    float* dinp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_dinp;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dinp = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    float* dweight = (float*)malloc(sizeof(float) * (size_t)(OC*C));
    float* d_dweight;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dweight = sycl::malloc_device<float>(
                             (size_t)(OC * C), dpct::get_in_order_queue())));
    float* dbias = (float*)malloc(sizeof(float) * (size_t)(OC));
    float* d_dbias;
    cudaCheck(DPCT_CHECK_ERROR(d_dbias = sycl::malloc_device<float>(
                                   (size_t)(OC), dpct::get_in_order_queue())));
    cublasCheck(DPCT_CHECK_ERROR(
        cublas_handle->set_math_mode(dpct::blas::math_mode::mm_default)));
    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
        const int k = variants[vi];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dinp, 0, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dweight, 0, sizeof(float) * (size_t)(OC * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dbias, 0, sizeof(float) * (size_t)(OC))
                .wait()));
        matmul_backward(k, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);
        /*
        DPCT1010:18: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dinp, d_dinp, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dweight, d_dweight, sizeof(float) * (size_t)(OC * C))
                .wait()));
        fwrite(dweight, sizeof(float), (size_t)(OC*C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dbias, d_dbias, sizeof(float) * (size_t)(OC))
                .wait()));
        fwrite(dbias, sizeof(float), (size_t)(OC), fout);
    }
    fclose(fout);
    return 0;
}
