// gelu_backward SYCL 2020 port (fp32) of llm.c dev/cuda/gelu_backward.cu.
// Two kernel variants, each its own SYCL kernel, computed on the SYCL device.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

using floatX = float;

// Matches the CUDA macro GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI).
// Evaluated in device code with float sqrt, exactly like the original.
inline float gelu_scaling_factor() {
  return sycl::sqrt(static_cast<float>(2.0f / 3.14159265358979323846));
}

inline constexpr int kVecSize = 4;
inline constexpr std::size_t kWorkGroupSize = 256;

template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

inline float gelu_local_grad(float x, float scaling) {
  const float cube = 0.044715f * x * x * x;
  const float tanh_arg = scaling * (x + cube);
  const float tanh_out = sycl::tanh(tanh_arg);
  const float coshf_out = sycl::cosh(tanh_arg);
  const float sech_out = 1.0f / (coshf_out * coshf_out);
  return 0.5f * (1.0f + tanh_out) +
         x * 0.5f * sech_out * scaling *
             (1.0f + 3.0f * 0.044715f * x * x);
}

class GeluBackward1Kernel;
class GeluBackward2Kernel;

void gelu_backward_variant1(sycl::queue &q, floatX *dinp, const floatX *inp,
                            const floatX *dout, std::size_t n) {
  if (n == 0) return;
  const std::size_t g = ceil_div(n, kWorkGroupSize) * kWorkGroupSize;
  q.parallel_for<GeluBackward1Kernel>(
       sycl::nd_range<1>(sycl::range<1>(g), sycl::range<1>(kWorkGroupSize)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i < n) {
           const float scale = gelu_scaling_factor();
           const float x = static_cast<float>(inp[i]);
           const float dy = static_cast<float>(dout[i]);
           dinp[i] = static_cast<floatX>(gelu_local_grad(x, scale) * dy);
         }
       });
  q.wait();
}

void gelu_backward_variant2(sycl::queue &q, floatX *dinp, const floatX *inp,
                            const floatX *dout, std::size_t n) {
  if (n == 0) return;
  const std::size_t items = ceil_div(n, static_cast<std::size_t>(kVecSize));
  const std::size_t g = ceil_div(items, kWorkGroupSize) * kWorkGroupSize;
  q.parallel_for<GeluBackward2Kernel>(
       sycl::nd_range<1>(sycl::range<1>(g), sycl::range<1>(kWorkGroupSize)),
       [=](sycl::nd_item<1> it) {
         const std::size_t base = it.get_global_id(0) * kVecSize;
         if (base >= n) return;
         const float scale = gelu_scaling_factor();
         // Fast path: whole 128-bit group in bounds, one vector load/store.
         if (base + kVecSize <= n) {
           const sycl::float4 vin =
               reinterpret_cast<const sycl::float4 *>(inp)[base / kVecSize];
           const sycl::float4 vdout =
               reinterpret_cast<const sycl::float4 *>(dout)[base / kVecSize];
           sycl::float4 vout;
           vout.x() = gelu_local_grad(vin.x(), scale) * vdout.x();
           vout.y() = gelu_local_grad(vin.y(), scale) * vdout.y();
           vout.z() = gelu_local_grad(vin.z(), scale) * vdout.z();
           vout.w() = gelu_local_grad(vin.w(), scale) * vdout.w();
           reinterpret_cast<sycl::float4 *>(dinp)[base / kVecSize] = vout;
         } else {
           for (int k = 0; k < kVecSize; ++k) {
             const std::size_t i = base + static_cast<std::size_t>(k);
             if (i >= n) break;
             const float x = static_cast<float>(inp[i]);
             const float dy = static_cast<float>(dout[i]);
             dinp[i] = static_cast<floatX>(gelu_local_grad(x, scale) * dy);
           }
         }
       });
  q.wait();
}

void gelu_backward(sycl::queue &q, int kernel_num, floatX *dinp,
                   const floatX *inp, const floatX *dout, std::size_t n) {
  switch (kernel_num) {
    case 1:
      gelu_backward_variant1(q, dinp, inp, dout, n);
      break;
    case 2:
      gelu_backward_variant2(q, dinp, inp, dout, n);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  std::srand(0);
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const long B = std::atol(argv[3]);
  const long T = std::atol(argv[4]);
  const long C = std::atol(argv[5]);
  const std::size_t n = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  if (!inp || !dout) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (n > 0) {
    if (std::fread(inp, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at inp\n");
      return 2;
    }
    if (std::fread(dout, sizeof(float), n, fin) != n) {
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

  float *d_inp = sycl::malloc_device<float>(n ? n : 1, q);
  float *d_dout = sycl::malloc_device<float>(n ? n : 1, q);
  float *d_dinp = sycl::malloc_device<float>(n ? n : 1, q);
  float *dinp = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  if (!d_inp || !d_dout || !d_dinp || !dinp) {
    std::fprintf(stderr, "allocation failed\n");
    return 2;
  }
  if (n > 0) {
    q.memcpy(d_inp, inp, sizeof(float) * n).wait();
    q.memcpy(d_dout, dout, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) q.memset(d_dinp, 0, sizeof(float) * n).wait();
    gelu_backward(q, k, d_dinp, d_inp, d_dout, n);
    if (n > 0) q.memcpy(dinp, d_dinp, sizeof(float) * n).wait();
    if (n > 0) std::fwrite(dinp, sizeof(float), n, fout);
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
