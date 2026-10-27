#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <sycl/sycl.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using floatX = float;

namespace {
// sqrtf(2/pi), matching llm.c GELU_SCALING_FACTOR
inline float gelu_scaling_factor() {
  return sycl::sqrt(2.0f / static_cast<float>(M_PI));
}

inline float gelu_scalar(float xi, float scaling) {
  float cube = 0.044715f * xi * xi * xi;
  return 0.5f * xi * (1.0f + sycl::tanh(scaling * (xi + cube)));
}

class GeluKernel1;
class GeluKernel2;
} // namespace

void gelu_forward1(sycl::queue &q, floatX *out, const floatX *inp,
                   std::size_t n) {
  if (n == 0) {
    return;
  }
  const float scaling = gelu_scaling_factor();
  q.parallel_for<GeluKernel1>(sycl::range<1>(n), [=](sycl::id<1> idx) {
     std::size_t i = idx[0];
     float xi = inp[i];
     out[i] = static_cast<floatX>(gelu_scalar(xi, scaling));
   }).wait();
}

void gelu_forward2(sycl::queue &q, floatX *out, const floatX *inp,
                   std::size_t n) {
  if (n == 0) {
    return;
  }
  const float scaling = gelu_scaling_factor();
  constexpr std::size_t kVec = 4;
  const std::size_t num_vec = n / kVec;
  const std::size_t tail = num_vec * kVec;
  if (num_vec > 0) {
    const auto *vec_in =
        reinterpret_cast<const sycl::vec<float, kVec> *>(inp);
    auto *vec_out = reinterpret_cast<sycl::vec<float, kVec> *>(out);
    q.parallel_for<GeluKernel2>(sycl::range<1>(num_vec),
                                [=](sycl::id<1> idx) {
                                  std::size_t v = idx[0];
                                  sycl::vec<float, kVec> vin = vec_in[v];
                                  sycl::vec<float, kVec> vout;
                                  vout[0] = gelu_scalar(vin[0], scaling);
                                  vout[1] = gelu_scalar(vin[1], scaling);
                                  vout[2] = gelu_scalar(vin[2], scaling);
                                  vout[3] = gelu_scalar(vin[3], scaling);
                                  vec_out[v] = vout;
                                }).wait();
  }
  if (tail < n) {
    q.parallel_for(sycl::range<1>(n - tail), [=](sycl::id<1> idx) {
       std::size_t i = tail + idx[0];
       float xi = inp[i];
       out[i] = static_cast<floatX>(gelu_scalar(xi, scaling));
     }).wait();
  }
}

void gelu_forward(sycl::queue &q, int kernel_num, floatX *out,
                  const floatX *inp, int B, int T, int C) {
  const std::size_t n = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
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
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
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
  float *d_out = sycl::malloc_device<float>(n, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if ((n > 0) && (!d_inp || !d_out || !out)) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }

  if (n > 0) {
    q.memcpy(d_inp, inp, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.fill(d_out, 0.0f, n).wait();
    }
    gelu_forward(q, k, d_out, d_inp, B, T, C);
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
