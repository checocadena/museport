// Synthetic stand-in for llm.c's dev/cuda/common.h (same layout, written for museport's tests).
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) { exit(EXIT_FAILURE); }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) { exit(EXIT_FAILURE); }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
int cuda_num_SMs = 0;

void setup_main() {
    srand(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cuda_num_SMs = deviceProp.multiProcessorCount;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
}
