#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using floatX = float;

inline constexpr float GELU_SCALING_FACTOR_VAL = 0.7978845608028654f; // sqrtf(2/M_PI)

constexpr int VEC_WIDTH = 4;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// SYCL kernels: one per variant, computed on the device.

class GeluBackwardKernel1;
class GeluBackwardKernel2;

void gelu_backward1(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, int N, int block_size) {
  const std::size_t local = static_cast<std::size_t>(block_size);
  const std::size_t grid =
      static_cast<std::size_t>(ceil_div(N, block_size)) * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackwardKernel1>(
        sycl::nd_range<1>(sycl::range<1>(grid), sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
          const int i = static_cast<int>(it.get_global_id(0));
          if (i < N) {
            const float x = static_cast<float>(inp[i]);
            const float cube = 0.044715f * x * x * x;
            const float tanh_arg =
                GELU_SCALING_FACTOR_VAL * (x + cube);
            const float tanh_out = sycl::tanh(tanh_arg);
            const float coshf_out = sycl::cosh(tanh_arg);
            const float sech_out = 1.0f / (coshf_out * coshf_out);
            const float local_grad =
                0.5f * (1.0f + tanh_out) +
                x * 0.5f * sech_out * GELU_SCALING_FACTOR_VAL *
                    (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] = static_cast<floatX>(local_grad *
                                          static_cast<float>(dout[i]));
          }
        });
  });
  q.wait();
}

void gelu_backward2(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, int N, int block_size) {
  const std::size_t local = static_cast<std::size_t>(block_size);
  const int nvec = ceil_div(N, VEC_WIDTH);
  const std::size_t grid =
      static_cast<std::size_t>(ceil_div(nvec, block_size)) * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackwardKernel2>(
        sycl::nd_range<1>(sycl::range<1>(grid), sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
          const int base = static_cast<int>(it.get_global_id(0)) * VEC_WIDTH;
          if (base + VEC_WIDTH <= N) {
            // Vectorized path: 128-bit (4 x fp32) transaction.
            const sycl::float4 packed_inp =
                reinterpret_cast<const sycl::float4 *>(inp)[base / VEC_WIDTH];
            const sycl::float4 packed_dout =
                reinterpret_cast<const sycl::float4 *>(dout)[base / VEC_WIDTH];
            sycl::float4 packed_dinp;
            const float xs[4] = {packed_inp.x(), packed_inp.y(),
                                 packed_inp.z(), packed_inp.w()};
            const float ds[4] = {packed_dout.x(), packed_dout.y(),
                                 packed_dout.z(), packed_dout.w()};
            float outs[4];
            for (int k = 0; k < VEC_WIDTH; ++k) {
              const float x = xs[k];
              const float cube = 0.044715f * x * x * x;
              const float tanh_arg =
                  GELU_SCALING_FACTOR_VAL * (x + cube);
              const float tanh_out = sycl::tanh(tanh_arg);
              const float coshf_out = sycl::cosh(tanh_arg);
              const float sech_out = 1.0f / (coshf_out * coshf_out);
              const float local_grad =
                  0.5f * (1.0f + tanh_out) +
                  x * 0.5f * sech_out * GELU_SCALING_FACTOR_VAL *
                      (1.0f + 3.0f * 0.044715f * x * x);
              outs[k] = local_grad * ds[k];
            }
            packed_dinp =
                sycl::float4(outs[0], outs[1], outs[2], outs[3]);
            reinterpret_cast<sycl::float4 *>(dinp)[base / VEC_WIDTH] =
                packed_dinp;
          } else {
            // Scalar tail for N not divisible by 4.
            for (int k = 0; k < VEC_WIDTH; ++k) {
              const int i = base + k;
              if (i < N) {
                const float x = static_cast<float>(inp[i]);
                const float cube = 0.044715f * x * x * x;
                const float tanh_arg =
                    GELU_SCALING_FACTOR_VAL * (x + cube);
                const float tanh_out = sycl::tanh(tanh_arg);
                const float coshf_out = sycl::cosh(tanh_arg);
                const float sech_out = 1.0f / (coshf_out * coshf_out);
                const float local_grad =
                    0.5f * (1.0f + tanh_out) +
                    x * 0.5f * sech_out * GELU_SCALING_FACTOR_VAL *
                        (1.0f + 3.0f * 0.044715f * x * x);
                dinp[i] = static_cast<floatX>(local_grad *
                                              static_cast<float>(dout[i]));
              }
            }
          }
        });
  });
  q.wait();
}

void gelu_backward(int kernel_num, sycl::queue &q, floatX *dinp,
                   const floatX *inp, const floatX *dout, int B, int T, int C,
                   int block_size) {
  switch (kernel_num) {
  case 1:
    gelu_backward1(q, dinp, inp, dout, B * T * C, block_size);
    break;
  case 2:
    gelu_backward2(q, dinp, inp, dout, B * T * C, block_size);
    break;
  default:
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// Contract main: <input> <output> B T C; runs every variant in order.
int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (std::fread(inp, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (std::fread(dout, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_inp = sycl::malloc_device<float>(N, q);
  float *d_dout = sycl::malloc_device<float>(N, q);
  float *d_dinp = sycl::malloc_device<float>(N, q);
  if (!d_inp || !d_dout || !d_dinp) {
    std::fprintf(stderr, "device allocation failed\n");
    return 1;
  }

  q.memcpy(d_inp, inp, sizeof(float) * N).wait();
  q.memcpy(d_dout, dout, sizeof(float) * N).wait();

  float *dinp = static_cast<float *>(std::malloc(sizeof(float) * N));

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * N).wait();
    gelu_backward(k, q, d_dinp, d_inp, d_dout, B, T, C, 256);
    q.memcpy(dinp, d_dinp, sizeof(float) * N).wait();
    std::fwrite(dinp, sizeof(float), N, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_dout, q);
  sycl::free(d_dinp, q);
  std::free(inp);
  std::free(dout);
  std::free(dinp);
  return 0;
}
