#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// fp32 build
using floatX = float;

// tanh-based GELU (llm.c): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
inline float gelu_elem(float xi) {
  constexpr float kScaling = 0.7978845608028654f; // sqrtf(2/pi)
  float cube = 0.04715f * xi * xi * xi;
  return 0.5f * xi * (1.0f + sycl::tanh(kScaling * (xi + cube)));
}

class GeluForwardKernel1;
class GeluForwardKernel2;

void gelu_forward1(sycl::queue &q, floatX *out, const floatX *inp, std::size_t n) {
  constexpr std::size_t kLocal = 256;
  std::size_t n_groups = (n + kLocal - 1) / kLocal;
  if (n_groups == 0) {
    return;
  }
  sycl::nd_range<1> launch(sycl::range<1>(n_groups * kLocal),
                           sycl::range<1>(kLocal));
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluForwardKernel1>(launch, [=](sycl::nd_item<1> it) {
      std::size_t i = it.get_global_id(0);
      if (i < n) {
        out[i] = gelu_elem(inp[i]);
      }
    });
  });
}

void gelu_forward2(sycl::queue &q, floatX *out, const floatX *inp, std::size_t n) {
  // Vectorized: one work-item handles 4 consecutive floats (128-bit chunk).
  constexpr std::size_t kVec = 4;
  constexpr std::size_t kLocal = 256;
  std::size_t n_vec = (n + kVec - 1) / kVec;
  std::size_t n_groups = (n_vec + kLocal - 1) / kLocal;
  if (n_groups == 0) {
    return;
  }
  sycl::nd_range<1> launch(sycl::range<1>(n_groups * kLocal),
                           sycl::range<1>(kLocal));
  q.submit([&](sycl::handler &h) {
    h.parallel_for<GeluForwardKernel2>(launch, [=](sycl::nd_item<1> it) {
      std::size_t v = it.get_global_id(0);
      if (v >= n_vec) {
        return;
      }
      std::size_t base = v * kVec;
      // Unaligned-safe 128-bit chunk copy via int4-width move when in bounds.
      if (base + kVec <= n) {
        sycl::vec<float, 4> chunk;
        std::memcpy(&chunk, inp + base, sizeof(chunk));
        chunk[0] = gelu_elem(chunk[0]);
        chunk[1] = gelu_elem(chunk[1]);
        chunk[2] = gelu_elem(chunk[2]);
        chunk[3] = gelu_elem(chunk[3]);
        std::memcpy(out + base, &chunk, sizeof(chunk));
      } else {
        for (std::size_t k = 0; k < kVec; ++k) {
          if (base + k < n) {
            out[base + k] = gelu_elem(inp[base + k]);
          }
        }
      }
    });
  });
}

void gelu_forward(sycl::queue &q, int kernel_num, floatX *out,
                  const floatX *inp, std::size_t n) {
  switch (kernel_num) {
    case 1:
      gelu_forward1(q, out, inp, n);
      break;
    case 2:
      gelu_forward2(q, out, inp, n);
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
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (n > 0 &&
      std::fread(inp, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp\n");
    std::fclose(fin);
    std::free(inp);
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    std::free(inp);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp = sycl::malloc_device<float>(n, q);
  float *d_out = sycl::malloc_device<float>(n, q);
  if ((n > 0) && (d_inp == nullptr || d_out == nullptr)) {
    std::fprintf(stderr, "device allocation failed\n");
    std::fclose(fout);
    std::free(inp);
    return 1;
  }

  q.memcpy(d_inp, inp, sizeof(float) * n).wait();

  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * n).wait();
    gelu_forward(q, k, d_out, d_inp, n);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * n).wait();
    std::fwrite(out, sizeof(float), n, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(out);
  std::free(inp);
  return 0;
}
