#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// fp32 build: floatX is plain float.
using floatX = float;

// sqrt(2/pi) — matches GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI) in the CUDA source.
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;
inline constexpr float kGeluCubicCoeff = 0.044715f;
// 128-bit vector width for float: 16 bytes / 4 bytes.
inline constexpr int kPack128Floats = 4;

struct GeluBackwardKernel1 {};
struct GeluBackwardKernel2 {};

inline float gelu_local_grad(float x) {
  float cube = kGeluCubicCoeff * x * x * x;
  float tanh_arg = kGeluScalingFactor * (x + cube);
  float tanh_out = sycl::tanh(tanh_arg);
  float coshf_out = sycl::cosh(tanh_arg);
  float sech_out = 1.0f / (coshf_out * coshf_out);
  return 0.5f * (1.0f + tanh_out) +
         x * 0.5f * sech_out * kGeluScalingFactor *
             (1.0f + 3.0f * kGeluCubicCoeff * x * x);
}

void gelu_backward_v1(sycl::queue &q, floatX *dinp, const floatX *inp,
                      const floatX *dout, std::size_t n) {
  if (n == 0) return;
  q.parallel_for<GeluBackwardKernel1>(sycl::range<1>(n),
                                      [=](sycl::id<1> idx) {
                                        std::size_t i = idx[0];
                                        float x = static_cast<float>(inp[i]);
                                        float g = gelu_local_grad(x);
                                        dinp[i] = static_cast<floatX>(
                                            g * static_cast<float>(dout[i]));
                                      })
      .wait();
}

void gelu_backward_v2(sycl::queue &q, floatX *dinp, const floatX *inp,
                      const floatX *dout, std::size_t n) {
  if (n == 0) return;
  // Vectorized variant: each work-item handles up to 4 contiguous elements
  // (one 128-bit transaction per input/output stream), mirroring the
  // Packed128 kernel in the CUDA source.
  std::size_t items = (n + kPack128Floats - 1) / kPack128Floats;
  q.parallel_for<GeluBackwardKernel2>(sycl::range<1>(items),
                                      [=](sycl::id<1> idx) {
                                        std::size_t base =
                                            idx[0] * kPack128Floats;
                                        // Plain loads; SYCL coalesces the
                                        // contiguous accesses per work-item.
                                        for (int k = 0; k < kPack128Floats;
                                             ++k) {
                                          std::size_t i = base + k;
                                          if (i >= n) break;
                                          float x =
                                              static_cast<float>(inp[i]);
                                          float g = gelu_local_grad(x);
                                          dinp[i] = static_cast<floatX>(
                                              g * static_cast<float>(dout[i]));
                                        }
                                      })
      .wait();
}

void gelu_backward(int kernel_num, sycl::queue &q, floatX *dinp,
                   const floatX *inp, const floatX *dout, std::size_t n) {
  switch (kernel_num) {
    case 1:
      gelu_backward_v1(q, dinp, inp, dout, n);
      break;
    case 2:
      gelu_backward_v2(q, dinp, inp, dout, n);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                  static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * n + (n == 0)));
  float *dout_h = static_cast<float *>(std::malloc(sizeof(float) * n + (n == 0)));
  if (n > 0) {
    if (std::fread(inp, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at inp\n");
      return 2;
    }
    if (std::fread(dout_h, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at dout\n");
      return 2;
    }
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_inp = nullptr;
  float *d_dout = nullptr;
  float *d_dinp = nullptr;
  float *dinp = static_cast<float *>(std::malloc(sizeof(float) * n + (n == 0)));
  if (n > 0) {
    d_inp = sycl::malloc_device<float>(n, q);
    d_dout = sycl::malloc_device<float>(n, q);
    d_dinp = sycl::malloc_device<float>(n, q);
    q.memcpy(d_inp, inp, sizeof(float) * n).wait();
    q.memcpy(d_dout, dout_h, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.memset(d_dinp, 0, sizeof(float) * n).wait();
      gelu_backward(k, q, d_dinp, d_inp, d_dout, n);
      q.wait();
      q.memcpy(dinp, d_dinp, sizeof(float) * n).wait();
      std::fwrite(dinp, sizeof(float), n, fout);
    }
  }

  std::fclose(fout);
  if (n > 0) {
    sycl::free(d_inp, q);
    sycl::free(d_dout, q);
    sycl::free(d_dinp, q);
  }
  std::free(inp);
  std::free(dout_h);
  std::free(dinp);
  return 0;
}
