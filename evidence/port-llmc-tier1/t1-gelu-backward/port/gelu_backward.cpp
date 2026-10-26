#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using floatX = float;

// sqrt(2/pi)
inline constexpr float GELU_SCALING_FACTOR = 0.7978845608028654f;
inline constexpr float GELU_CUBIC_COEFF = 0.044715f;

template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// 128-bit packed vector: 4 x float. Forces 128-bit loads/stores.
template <typename ElementType>
struct alignas(16) Packed128 {
  static constexpr int size = 16 / sizeof(ElementType);
  ElementType payload[size];
  ElementType &operator[](int index) { return payload[index]; }
  const ElementType &operator[](int index) const { return payload[index]; }
};

using x128 = Packed128<floatX>;

inline x128 load128cs(const floatX *address) {
  x128 v;
  for (int k = 0; k < x128::size; ++k) {
    v[k] = address[k];
  }
  return v;
}

inline void store128(floatX *target, x128 value) {
  for (int k = 0; k < x128::size; ++k) {
    target[k] = value[k];
  }
}

inline float gelu_local_grad(float x) {
  float cube = GELU_CUBIC_COEFF * x * x * x;
  float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
  float tanh_out = sycl::tanh(tanh_arg);
  float coshf_out = sycl::cosh(tanh_arg);
  float sech_out = 1.0f / (coshf_out * coshf_out);
  return 0.5f * (1.0f + tanh_out) +
         x * 0.5f * sech_out * GELU_SCALING_FACTOR *
             (1.0f + 3.0f * GELU_CUBIC_COEFF * x * x);
}

class GeluBackwardK1;
class GeluBackwardK2;

void gelu_backward1(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t N) {
  constexpr std::size_t block_size = 256;
  std::size_t grid = ceil_div(N, block_size);
  if (grid == 0) {
    return;
  }
  q.parallel_for<GeluBackwardK1>(
      sycl::nd_range<1>(sycl::range<1>(grid * block_size),
                        sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> item) {
        std::size_t i = item.get_global_id(0);
        if (i < N) {
          float x = static_cast<float>(inp[i]);
          float local_grad = gelu_local_grad(x);
          dinp[i] = static_cast<floatX>(local_grad * static_cast<float>(dout[i]));
        }
      });
}

void gelu_backward2(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t N) {
  constexpr std::size_t block_size = 256;
  constexpr int vec = x128::size;
  std::size_t nvec = ceil_div(N, static_cast<std::size_t>(vec));
  std::size_t grid = ceil_div(nvec, block_size);
  if (grid == 0) {
    return;
  }
  q.parallel_for<GeluBackwardK2>(
      sycl::nd_range<1>(sycl::range<1>(grid * block_size),
                        sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> item) {
        std::size_t base = item.get_global_id(0) * static_cast<std::size_t>(vec);
        if (base >= N) {
          return;
        }
        if (base + vec <= N) {
          x128 packed_dinp;
          x128 packed_inp = load128cs(inp + base);
          x128 packed_dout = load128cs(dout + base);
          for (int k = 0; k < vec; ++k) {
            float x = static_cast<float>(packed_inp[k]);
            float local_grad = gelu_local_grad(x);
            packed_dinp[k] = static_cast<floatX>(
                local_grad * static_cast<float>(packed_dout[k]));
          }
          store128(dinp + base, packed_dinp);
        } else {
          for (int k = 0; k < vec; ++k) {
            std::size_t i = base + static_cast<std::size_t>(k);
            if (i < N) {
              float x = static_cast<float>(inp[i]);
              float local_grad = gelu_local_grad(x);
              dinp[i] = static_cast<floatX>(local_grad * static_cast<float>(dout[i]));
            }
          }
        }
      });
}

void gelu_backward(int kernel_num, sycl::queue &q, floatX *dinp,
                   const floatX *inp, const floatX *dout, int B, int T, int C) {
  std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                  static_cast<std::size_t>(C);
  switch (kernel_num) {
    case 1:
      gelu_backward1(q, dinp, inp, dout, N);
      break;
    case 2:
      gelu_backward2(q, dinp, inp, dout, N);
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
  std::size_t N =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!inp || !dout) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
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

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(N, q);
  float *d_dout = sycl::malloc_device<float>(N, q);
  float *d_dinp = sycl::malloc_device<float>(N, q);
  if (!d_inp || !d_dout || !d_dinp) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_inp, inp, sizeof(float) * N).wait();
  q.memcpy(d_dout, dout, sizeof(float) * N).wait();

  float *dinp = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!dinp) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * N).wait();
    gelu_backward(k, q, d_dinp, d_inp, d_dout, B, T, C);
    q.wait();
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
