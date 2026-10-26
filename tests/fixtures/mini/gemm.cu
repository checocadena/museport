#include <cublas_v2.h>
#include <thrust/device_vector.h>
void run(cublasHandle_t h, float* a, float* b, float* c) {
    float one = 1.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &one, a, 4, b, 4, &one, c, 4);
    thrust::device_vector<float> v(4);
}
