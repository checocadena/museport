#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// sqrtf(2/pi)
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;

inline float gelu_element(float xi) {
  float cube = 0.044715f * xi * xi * xi;
  return 0.5f * xi * (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube)));
}

class GeluForwardKernel1;
class GeluForwardKernel2;

void gelu_forward1(sycl::queue &q, float *out, const float *inp, std::size_t n) {
  if (n == 0) {
    return;
  }
  q.parallel_for<GeluForwardKernel1>(sycl::range<1>(n), [=](sycl::id<1> idx) {
    std::size_t i = idx[0];
    out[i] = gelu_element(inp[i]);
  });
  q.wait();
}

void gelu_forward2(sycl::queue &q, float *out, const float *inp, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr std::size_t kVec = 4; // 128-bit = 4 x fp32, as in x128
  std::size_t nvec = (n + kVec - 1) / kVec;
  q.parallel_for<GeluForwardKernel2>(sycl::range<1>(nvec), [=](sycl::id<1> idx) {
    std::size_t base = idx[0] * kVec;
    sycl::vec<float, 4> vin(0.0f);
    for (std::size_t k = 0; k < kVec; ++k) {
      if (base + k < n) {
        vin[k] = inp[base + k];
      }
    }
    sycl::vec<float, 4> vout(0.0f);
    for (std::size_t k = 0; k < kVec; ++k) {
      vout[k] = gelu_element(vin[k]);
    }
    for (std::size_t k = 0; k < kVec; ++k) {
      if (base + k < n) {
        out[base + k] = vout[k];
      }
    }
  });
  q.wait();
}

void gelu_forward(sycl::queue &q, int kernel_num, float *out, const float *inp,
                  std::size_t n) {
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
  std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                  static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  if (n > 0) {
    if (std::fread(inp, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at inp\n");
      return 2;
    }
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(n ? n : 1, q);
  float *d_out = sycl::malloc_device<float>(n ? n : 1, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  if (!d_inp || !d_out || !out) {
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
      q.memset(d_out, 0, sizeof(float) * n).wait();
    }
    gelu_forward(q, k, d_out, d_inp, n);
    if (n > 0) {
      q.memcpy(out, d_out, sizeof(float) * n).wait();
      std::fwrite(out, sizeof(float), n, fout);
    }
  }

  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(out);
  std::fclose(fout);
  return 0;
}
