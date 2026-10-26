#include "common.h"
// cublasSgemm in a comment does not count; neither does __shfl_down_sync here
/* __global__ void commented_out(float* x) {} */
__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
template <typename T>
__global__ void __launch_bounds__(256) scale_kernel(T* x, T s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&x[0], x[i] * s);
}
