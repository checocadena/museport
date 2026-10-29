#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using floatX = float;

// sqrt(2/pi), matching GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI) in llm.c
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;

template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

inline float gelu_local_grad(float x) {
  float cube = 0.044715f * x * x * x;
  float tanh_arg = kGeluScalingFactor * (x + cube);
  float tanh_out = sycl::tanh(tanh_arg);
  float coshf_out = sycl::cosh(tanh_arg);
  float sech_out = 1.0f / (coshf_out * coshf_out);
  return 0.5f * (1.0f + tanh_out) +
         x * 0.5f * sech_out * kGeluScalingFactor *
             (1.0f + 3.0f * 0.044715f * x * x);
}

class GeluBackward1Kernel;
class GeluBackward2Kernel;

void gelu_backward1(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr std::size_t block_size = 256;
  std::size_t global = ceil_div(n, block_size) * block_size;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackward1Kernel>(
        sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{block_size}},
        [=](sycl::nd_item<1> it) {
          std::size_t i = it.get_global_id(0);
          if (i < n) {
            float x = static_cast<float>(inp[i]);
            float g = gelu_local_grad(x);
            dinp[i] = static_cast<floatX>(g * static_cast<float>(dout[i]));
          }
        });
  });
  q.wait();
}

void gelu_backward2(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t n) {
  if (n == 0) {
    return;
  }
  // Packed-128 (float4) path: one work-item handles 4 consecutive elements,
  // mirroring x128::size == 4 for fp32.
  constexpr std::size_t kPack = 4;
  constexpr std::size_t block_size = 256;
  std::size_t num_packs = ceil_div(n, kPack);
  std::size_t global = ceil_div(num_packs, block_size) * block_size;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackward2Kernel>(
        sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{block_size}},
        [=](sycl::nd_item<1> it) {
          std::size_t pack = it.get_global_id(0);
          if (pack >= num_packs) {
            return;
          }
          std::size_t base = pack * kPack;
          // Vector load/store when fully in bounds; scalar tail otherwise.
          if (base + kPack <= n) {
            const sycl::vec<float, 4> *in4 =
                reinterpret_cast<const sycl::vec<float, 4> *>(inp);
            const sycl::vec<float, 4> *dout4 =
                reinterpret_cast<const sycl::vec<float, 4> *>(dout);
            sycl::vec<float, 4> *out4 =
                reinterpret_cast<sycl::vec<float, 4> *>(dinp);
            sycl::vec<float, 4> vin = in4[pack];
            sycl::vec<float, 4> vdout = dout4[pack];
            sycl::vec<float, 4> vout;
            vout.x() = gelu_local_grad(vin.x()) * vdout.x();
            vout.y() = gelu_local_grad(vin.y()) * vdout.y();
            vout.z() = gelu_local_grad(vin.z()) * vdout.z();
            vout.w() = gelu_local_grad(vin.w()) * vdout.w();
            out4[pack] = vout;
          } else {
            for (std::size_t k = 0; k < kPack; ++k) {
              std::size_t i = base + k;
              if (i < n) {
                float x = static_cast<float>(inp[i]);
                float g = gelu_local_grad(x);
                dinp[i] = static_cast<floatX>(g * static_cast<float>(dout[i]));
              }
            }
          }
        });
  });
  q.wait();
}

void gelu_backward(sycl::queue &q, int kernel_num, floatX *dinp,
                   const floatX *inp, const floatX *dout, int B, int T, int C) {
  std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                  static_cast<std::size_t>(C);
  switch (kernel_num) {
    case 1:
      gelu_backward1(q, dinp, inp, dout, n);
      break;
    case 2:
      gelu_backward2(q, dinp, inp, dout, n);
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
  std::size_t n =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *dinp = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!inp || !dout || !dinp) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (std::fread(dout, sizeof(float), n, fin) != n) {
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
  float *d_inp = sycl::malloc_device<float>(n, q);
  float *d_dout = sycl::malloc_device<float>(n, q);
  float *d_dinp = sycl::malloc_device<float>(n, q);
  if ((n > 0) && (!d_inp || !d_dout || !d_dinp)) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  if (n > 0) {
    q.memcpy(d_inp, inp, sizeof(float) * n).wait();
    q.memcpy(d_dout, dout, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.memset(d_dinp, 0, sizeof(float) * n).wait();
    }
    gelu_backward(q, k, d_dinp, d_inp, d_dout, B, T, C);
    if (n > 0) {
      q.memcpy(dinp, d_dinp, sizeof(float) * n).wait();
    }
    std::fwrite(dinp, sizeof(float), n, fout);
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
