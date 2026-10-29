#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

using floatX = float;
constexpr int kVecWidth = 4; // 128-bit / 32-bit = 4 floats
constexpr float kGeluScaling = 0.7978845608028654f; // sqrtf(2/pi)

// ----------------------------------------------------------------------------
// SYCL kernels: one per variant, computed on the device.

class GeluKernel1;
class GeluKernel2;

inline void gelu_forward1(sycl::queue &q, floatX *out, const floatX *inp, int N) {
  constexpr int kBlockSize = 256;
  const int num_items = ((N + kBlockSize - 1) / kBlockSize) * kBlockSize;
  if (num_items <= 0)
    return;
  q.parallel_for<GeluKernel1>(
      sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_items)),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i < static_cast<size_t>(N)) {
          const float xi = inp[i];
          const float cube = 0.044715f * xi * xi * xi;
          out[i] = 0.5f * xi *
                   (1.0f + sycl::tanh(kGeluScaling * (xi + cube)));
        }
      });
  q.wait();
}

inline void gelu_forward2(sycl::queue &q, floatX *out, const floatX *inp, int N) {
  constexpr int kBlockSize = 256;
  const int num_vectors = (N + kVecWidth - 1) / kVecWidth;
  const int num_items =
      ((num_vectors + kBlockSize - 1) / kBlockSize) * kBlockSize;
  if (num_items <= 0)
    return;
  q.parallel_for<GeluKernel2>(
      sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_items)),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> it) {
        const size_t v = it.get_global_id(0);
        const size_t base = v * kVecWidth;
        if (base >= static_cast<size_t>(N))
          return;
        if (base + kVecWidth <= static_cast<size_t>(N)) {
          // 128-bit vector load/store.
          const sycl::vec<float, kVecWidth> vin(
              inp[base], inp[base + 1], inp[base + 2], inp[base + 3]);
          sycl::vec<float, kVecWidth> vout;
          for (int k = 0; k < kVecWidth; ++k) {
            const float xi = vin[k];
            const float cube = 0.044715f * xi * xi * xi;
            vout[k] = 0.5f * xi * (1.0f + sycl::tanh(kGeluScaling * (xi + cube)));
          }
          out[base] = vout[0];
          out[base + 1] = vout[1];
          out[base + 2] = vout[2];
          out[base + 3] = vout[3];
        } else {
          for (size_t i = base; i < static_cast<size_t>(N); ++i) {
            const float xi = inp[i];
            const float cube = 0.044715f * xi * xi * xi;
            out[i] = 0.5f * xi *
                     (1.0f + sycl::tanh(kGeluScaling * (xi + cube)));
          }
        }
      });
  q.wait();
}

inline void gelu_forward(sycl::queue &q, int kernel_num, floatX *out,
                         const floatX *inp, int B, int T, int C) {
  const int N = B * T * C;
  switch (kernel_num) {
  case 1:
    gelu_forward1(q, out, inp, N);
    break;
  case 2:
    gelu_forward2(q, out, inp, N);
    break;
  default:
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// Contract main: <input> <output> B T C; runs every kernel variant and writes
// each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const size_t N = static_cast<size_t>(B) * static_cast<size_t>(T) *
                   static_cast<size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(N * sizeof(float)));
  if (!inp) {
    std::fclose(fin);
    return 2;
  }
  if (std::fread(inp, sizeof(float), N, fin) != N) {
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
  float *out = static_cast<float *>(std::malloc(N * sizeof(float)));
  if (!out) {
    std::fclose(fout);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};
    float *d_inp = sycl::malloc_device<float>(N, q);
    float *d_out = sycl::malloc_device<float>(N, q);
    if (!d_inp || !d_out) {
      std::fprintf(stderr, "device allocation failed\n");
      return 2;
    }
    q.memcpy(d_inp, inp, N * sizeof(float)).wait();

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
      const int k = variants[vi];
      q.memset(d_out, 0, N * sizeof(float)).wait();
      gelu_forward(q, k, d_out, d_inp, B, T, C);
      q.memcpy(out, d_out, N * sizeof(float)).wait();
      std::fwrite(out, sizeof(float), N, fout);
    }
    sycl::free(d_inp, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(inp);
  std::free(out);
  return 0;
}
