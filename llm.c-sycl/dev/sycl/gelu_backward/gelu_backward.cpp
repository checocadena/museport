#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

using floatX = float;

namespace {
// sqrt(2/pi)
constexpr float kGeluScaling = 0.7978845608028654f;

constexpr std::size_t kBlockSize = 256;
constexpr int kVecWidth = 4;

inline float gelu_local_grad(float x) {
  float cube = 0.044715f * x * x * x;
  float tanh_arg = kGeluScaling * (x + cube);
  float tanh_out = sycl::tanh(tanh_arg);
  float coshf_out = sycl::cosh(tanh_arg);
  float sech_out = 1.0f / (coshf_out * coshf_out);
  return 0.5f * (1.0f + tanh_out) +
         x * 0.5f * sech_out * kGeluScaling *
             (1.0f + 3.0f * 0.044715f * x * x);
}

class GeluBackwardKernel1;
class GeluBackwardKernel2;

template <typename T>
T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

void run_variant1(sycl::queue &q, floatX *dinp, const floatX *inp,
                  const floatX *dout, std::size_t n) {
  if (n == 0) return;
  std::size_t num_groups = ceil_div(n, kBlockSize);
  sycl::nd_range<1> launch(sycl::range<1>(num_groups * kBlockSize),
                           sycl::range<1>(kBlockSize));
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackwardKernel1>(
        launch, [=](sycl::nd_item<1> item) {
          std::size_t i = item.get_global_id(0);
          if (i < n) {
            float x = static_cast<float>(inp[i]);
            float g = gelu_local_grad(x);
            dinp[i] = static_cast<floatX>(g * static_cast<float>(dout[i]));
          }
        });
  });
  q.wait();
}

void run_variant2(sycl::queue &q, floatX *dinp, const floatX *inp,
                  const floatX *dout, std::size_t n) {
  if (n == 0) return;
  using Vec = sycl::vec<float, kVecWidth>;
  std::size_t num_vecs =
      ceil_div(n, static_cast<std::size_t>(kVecWidth));
  std::size_t num_groups = ceil_div(num_vecs, kBlockSize);
  sycl::nd_range<1> launch(sycl::range<1>(num_groups * kBlockSize),
                           sycl::range<1>(kBlockSize));
  const auto *in_vecs = reinterpret_cast<const Vec *>(inp);
  const auto *dout_vecs = reinterpret_cast<const Vec *>(dout);
  auto *out_vecs = reinterpret_cast<Vec *>(dinp);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluBackwardKernel2>(
        launch, [=](sycl::nd_item<1> item) {
          std::size_t vid = item.get_global_id(0);
          std::size_t base = vid * kVecWidth;
          if (base >= n) return;
          if (base + kVecWidth <= n) {
            // Aligned 128-bit vector path (one float4 per work-item).
            Vec vin = in_vecs[vid];
            Vec vdout = dout_vecs[vid];
            Vec vout;
            for (int k = 0; k < kVecWidth; ++k) {
              float g = gelu_local_grad(vin[k]);
              vout[k] = g * vdout[k];
            }
            out_vecs[vid] = vout;
          } else {
            for (std::size_t k = base; k < n; ++k) {
              float x = static_cast<float>(inp[k]);
              float g = gelu_local_grad(x);
              dinp[k] =
                  static_cast<floatX>(g * static_cast<float>(dout[k]));
            }
          }
        });
  });
  q.wait();
}

void dispatch(sycl::queue &q, int kernel_num, floatX *dinp, const floatX *inp,
              const floatX *dout, std::size_t n) {
  switch (kernel_num) {
    case 1:
      run_variant1(q, dinp, inp, dout, n);
      break;
    case 2:
      run_variant2(q, dinp, inp, dout, n);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}
}  // namespace

int main(int argc, char **argv) {
  std::srand(0);
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const long B = std::atol(argv[3]);
  const long T = std::atol(argv[4]);
  const long C = std::atol(argv[5]);
  const auto n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                 static_cast<std::size_t>(C);
  const std::size_t bytes = n * sizeof(float);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(bytes));
  float *dout_h = static_cast<float *>(std::malloc(bytes));
  if (std::fread(inp, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (std::fread(dout_h, sizeof(float), n, fin) != n) {
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
  floatX *d_inp = sycl::malloc_device<floatX>(n, q);
  floatX *d_dout = sycl::malloc_device<floatX>(n, q);
  floatX *d_dinp = sycl::malloc_device<floatX>(n, q);
  float *dinp = static_cast<float *>(std::malloc(bytes));
  q.memcpy(d_inp, inp, bytes).wait();
  q.memcpy(d_dout, dout_h, bytes).wait();

  const int variants[] = {1, 2};
  for (int k : variants) {
    q.memset(d_dinp, 0, bytes).wait();
    dispatch(q, k, d_dinp, d_inp, d_dout, n);
    q.memcpy(dinp, d_dinp, bytes).wait();
    std::fwrite(dinp, sizeof(float), n, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_dout, q);
  sycl::free(d_dinp, q);
  std::free(inp);
  std::free(dout_h);
  std::free(dinp);
  return 0;
}
