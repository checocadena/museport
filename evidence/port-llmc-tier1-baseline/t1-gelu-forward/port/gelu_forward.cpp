// GELU forward pass, fp32 build — idiomatic SYCL 2020 port of
// llm.c dev/cuda/gelu_forward.cu (gelu_forward_harness.cu).
//
// Two kernel variants, each its own SYCL kernel executed on the SYCL device:
//   k=1: naive scalar elementwise GELU (tanh approximation)
//   k=2: 4-wide vectorized elementwise GELU (fp32 Packed128 equivalent)
//
// Contract: build/candidate <input> <output> B T C reads B*T*C float32 inputs
// and appends each variant's B*T*C float32 outputs in order (k = 1, 2).

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using floatX = float;

// sqrtf(2 / pi); precomputed so it is usable in device code without
// host/device math header subtleties.
constexpr float GELU_SCALING_FACTOR = 0.7978845608028654f;
constexpr int VEC_WIDTH = 4; // sizeof(int4)/sizeof(float): one 128-bit word

namespace {

template <typename T>
T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

inline float gelu_tanh(float xi) {
  float cube = 0.044715f * xi * xi * xi;
  return 0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
}

class GeluKernel1;
class GeluKernel2;

void gelu_forward1(sycl::queue &q, floatX *out, const floatX *inp, std::size_t n,
                   std::size_t block_size) {
  if (n == 0) return;
  std::size_t local = block_size;
  std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (local > max_wg) local = max_wg;
  if (local == 0) local = 1;
  std::size_t global = ceil_div(n, local) * local;
  q.parallel_for<GeluKernel1>(sycl::nd_range<1>{global, local},
                              [=](sycl::nd_item<1> it) {
                                std::size_t i = it.get_global_linear_id();
                                if (i < n) out[i] = gelu_tanh(inp[i]);
                              });
}

void gelu_forward2(sycl::queue &q, floatX *out, const floatX *inp, std::size_t n,
                   std::size_t block_size) {
  if (n == 0) return;
  std::size_t local = block_size;
  std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (local > max_wg) local = max_wg;
  if (local == 0) local = 1;
  // One work-item computes one 128-bit (4x fp32) word, mirroring the CUDA
  // x128 Packed128 kernel. Per-lane bounds checks keep arbitrary N exact.
  std::size_t words = ceil_div(n, static_cast<std::size_t>(VEC_WIDTH));
  std::size_t global = ceil_div(words, local) * local;
  q.parallel_for<GeluKernel2>(sycl::nd_range<1>{global, local},
                              [=](sycl::nd_item<1> it) {
                                std::size_t base =
                                    it.get_global_linear_id() * VEC_WIDTH;
                                if (base >= n) return;
                                // Vector load/store for the full tile, scalar
                                // tail for the partial last word.
                                if (base + VEC_WIDTH <= n) {
                                  sycl::vec<float, VEC_WIDTH> v;
                                  v.load(0, inp + base);
                                  for (int k = 0; k < VEC_WIDTH; ++k)
                                    v[k] = gelu_tanh(v[k]);
                                  v.store(0, out + base);
                                } else {
                                  for (int k = 0; k < VEC_WIDTH; ++k) {
                                    std::size_t idx = base + (std::size_t)k;
                                    if (idx < n) out[idx] = gelu_tanh(inp[idx]);
                                  }
                                }
                              });
}

void gelu_forward(sycl::queue &q, int kernel_num, floatX *out,
                  const floatX *inp, int B, int T, int C,
                  std::size_t block_size) {
  std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                  static_cast<std::size_t>(C);
  switch (kernel_num) {
    case 1:
      gelu_forward1(q, out, inp, n, block_size);
      break;
    case 2:
      gelu_forward2(q, out, inp, n, block_size);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const std::size_t n =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (n > 0 &&
      std::fread(inp, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp = sycl::malloc_device<float>(n == 0 ? 1 : n, q);
  float *d_out = sycl::malloc_device<float>(n == 0 ? 1 : n, q);
  if (!d_inp || !d_out) {
    std::fprintf(stderr, "device allocation failed\n");
    return 1;
  }
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n == 0 ? 1 : n)));

  if (n > 0) q.memcpy(d_inp, inp, sizeof(float) * n).wait();

  const int variants[] = {1, 2};
  const std::size_t block_size = 256;
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) q.memset(d_out, 0, sizeof(float) * n).wait();
    gelu_forward(q, k, d_out, d_inp, B, T, C, block_size);
    q.wait();
    if (n > 0) {
      q.memcpy(out, d_out, sizeof(float) * n).wait();
      std::fwrite(out, sizeof(float), n, fout);
    }
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(out);
  return 0;
}
