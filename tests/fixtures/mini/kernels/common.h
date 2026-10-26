#include <cublas_v2.h>
#define WARP_SIZE 32U
__device__ float warpReduceSum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}
