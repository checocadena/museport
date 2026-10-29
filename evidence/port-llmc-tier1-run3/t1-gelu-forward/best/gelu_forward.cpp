#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// GELU tanh approximation scaling factor: sqrt(2/pi).
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;
inline constexpr float kGeluCubeCoeff = 0.044715f;

inline float gelu_elem(float xi) {
  float cube = kGeluCubeCoeff * xi * xi * xi;
  return 0.5f * xi *
         (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube)));
}

class GeluKernel1;
class GeluKernel2;

void gelu_forward1(sycl::queue &q, float *out, const float *inp,
                   std::size_t n) {
  if (n == 0) return;
  q.parallel_for<GeluKernel1>(sycl::range<1>(n), [=](sycl::id<1> idx) {
     std::size_t i = idx[0];
     float xi = inp[i];
     out[i] = gelu_elem(xi);
   }).wait();
}

void gelu_forward2(sycl::queue &q, float *out, const float *inp,
                   std::size_t n) {
  if (n == 0) return;
  constexpr std::size_t kVec = 4;
  const std::size_t nvec = (n + kVec - 1) / kVec;
  q.parallel_for<GeluKernel2>(sycl::range<1>(nvec), [=](sycl::id<1> idx) {
     const std::size_t base = idx[0] * kVec;
     // Scalar loads/stores; the compiler vectorizes to 128-bit accesses
     // when aligned. Per-lane guard preserves the tail behavior.
     for (std::size_t k = 0; k < kVec; ++k) {
       const std::size_t i = base + k;
       if (i < n) {
         out[i] = gelu_elem(inp[i]);
       }
     }
   }).wait();
}

void gelu_forward(sycl::queue &q, int kernel_num, float *out,
                  const float *inp, std::size_t n) {
  switch (kernel_num) {
    case 1:
      gelu_forward1(q, out, inp, n);
      break;
    case 2:
      gelu_forward2(q, out, inp, n);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
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
  std::vector<float> host_in(n);
  if (n > 0 &&
      std::fread(host_in.data(), sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp = sycl::malloc_device<float>(n > 0 ? n : 1, q);
  float *d_out = sycl::malloc_device<float>(n > 0 ? n : 1, q);
  if (!d_inp || !d_out) {
    std::fprintf(stderr, "device allocation failed\n");
    std::fclose(fout);
    return 1;
  }
  std::vector<float> host_out(n);

  if (n > 0) {
    q.memcpy(d_inp, host_in.data(), n * sizeof(float)).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.memset(d_out, 0, n * sizeof(float)).wait();
    }
    gelu_forward(q, k, d_out, d_inp, n);
    if (n > 0) {
      q.memcpy(host_out.data(), d_out, n * sizeof(float)).wait();
      std::fwrite(host_out.data(), sizeof(float), n, fout);
    }
  }

  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::fclose(fout);
  return 0;
}
