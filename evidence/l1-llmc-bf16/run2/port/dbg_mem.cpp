#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdio>
#include <vector>
using bf16 = sycl::ext::oneapi::bfloat16;
int main() {
  sycl::queue q(sycl::default_selector_v);
  printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("sizeof bf16=%zu\n", sizeof(bf16));
  std::vector<float> xf(8);
  for (size_t i = 0; i < 8; i++) xf[i] = 0.3f*((int)((i*29)%13)-6);
  printf("host xf[0]=%.6f xf[1]=%.6f\n", xf[0], xf[1]);
  std::vector<bf16> x(8);
  for (size_t i = 0; i < 8; i++) x[i] = (bf16)xf[i];
  printf("host x[0]=%.6f x[1]=%.6f\n", (float)x[0], (float)x[1]);
  bf16* dx = sycl::malloc_device<bf16>(8, q);
  q.memcpy(dx, x.data(), 8*sizeof(bf16)).wait();
  // device kernel: y = 2*x+1
  bf16* dy = sycl::malloc_device<bf16>(8, q);
  q.parallel_for(sycl::range<1>(8), [=](sycl::id<1> id) {
    size_t i = id[0];
    dy[i] = (bf16)((float)dx[i]*2.0f+1.0f);
  }).wait();
  std::vector<bf16> y(8);
  q.memcpy(y.data(), dy, 8*sizeof(bf16)).wait();
  for (int i = 0; i < 8; i++) printf("i=%d in=%.4f out=%.4f expect=%.4f\n", i, (float)x[i], (float)y[i], xf[i]*2+1);
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}
