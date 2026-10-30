#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>
#include <cstdio>
using bf16 = sycl::ext::oneapi::bfloat16;
int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  const std::int64_t m = 4, n = 3, k = 5;
  bf16* a = sycl::malloc_shared<bf16>(m*k, q); bf16* b = sycl::malloc_shared<bf16>(k*n, q);
  float* c = sycl::malloc_shared<float>(m*n, q); bf16* c16 = sycl::malloc_shared<bf16>(m*n, q);
  for (int i = 0; i < m*k; i++) a[i] = bf16(0.5f * (i % 7));
  for (int i = 0; i < k*n; i++) b[i] = bf16(0.25f * (i % 5));
  using namespace oneapi::math;
  try { blas::column_major::gemm(q, transpose::nontrans, transpose::nontrans, m, n, k, 1.0f, (const bf16*)a, m, (const bf16*)b, k, 0.0f, c, m).wait();
        printf("bf16 x bf16 -> fp32 gemm: ok, c[0]=%f\n", c[0]); } catch (std::exception& e) { printf("bf16->fp32: %s\n", e.what()); }
  float ref = 0; for (int p = 0; p < k; p++) ref += (float)a[p*m] * (float)b[p]; printf("ref c[0]=%f on %s\n", ref, q.get_device().get_info<sycl::info::device::name>().c_str());
}
