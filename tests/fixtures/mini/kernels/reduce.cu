#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void sum_kernel(float* out, const float* in, int n) {
    __shared__ float tile[32];
    cg::thread_block block = cg::this_thread_block();
    float v = __shfl_down_sync(0xFFFFFFFF, in[threadIdx.x], 1);
    tile[threadIdx.x % 32] = v;
}
