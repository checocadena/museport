#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// fp32 build
using floatX = float;

// sqrt(2/pi), matching GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI) in CUDA
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;

class GeluForwardKernel1;
class GeluForwardKernel2;

void gelu_forward1(sycl::queue &q, floatX *out, const floatX *inp,
                   std::size_t N) {
  if (N == 0) {
    return;
  }
  constexpr std::size_t block_size = 256;
  const std::size_t global_size =
      ((N + block_size - 1) / block_size) * block_size;
  q.parallel_for<GeluForwardKernel1>(
       sycl::nd_range<1>{sycl::range<1>{global_size},
                         sycl::range<1>{block_size}},
       [=](sycl::nd_item<1> item) {
         const std::size_t i = item.get_global_id(0);
         if (i < N) {
           const float xi = static_cast<float>(inp[i]);
           const float cube = 0.044715f * xi * xi * xi;
           out[i] = static_cast<floatX>(
               0.5f * xi *
               (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube))));
         }
       })
      .wait();
}

void gelu_forward2(sycl::queue &q, floatX *out, const floatX *inp,
                   std::size_t N) {
  if (N == 0) {
    return;
  }
  // Vectorized variant: each work-item handles 4 consecutive elements
  // (the role of Packed128 / x128 in the CUDA source).
  constexpr std::size_t vec_width = 4;
  constexpr std::size_t block_size = 256;
  const std::size_t n_vec = (N + vec_width - 1) / vec_width;
  const std::size_t global_size =
      ((n_vec + block_size - 1) / block_size) * block_size;
  q.parallel_for<GeluForwardKernel2>(
       sycl::nd_range<1>{sycl::range<1>{global_size},
                         sycl::range<1>{block_size}},
       [=](sycl::nd_item<1> item) {
         const std::size_t base = item.get_global_id(0) * vec_width;
         if (base >= N) {
           return;
         }
         // Full 4-wide group: use a vector type so the compiler emits
         // wide loads/stores, mirroring load128/store128.
         if (base + vec_width <= N) {
           sycl::vec<float, 4> v;
           v[0] = static_cast<float>(inp[base]);
           v[1] = static_cast<float>(inp[base + 1]);
           v[2] = static_cast<float>(inp[base + 2]);
           v[3] = static_cast<float>(inp[base + 3]);
           sycl::vec<float, 4> r;
           for (int k = 0; k < 4; ++k) {
             const float xi = v[k];
             const float cube = 0.044715f * xi * xi * xi;
             r[k] = 0.5f * xi *
                    (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube)));
           }
           out[base] = static_cast<floatX>(r[0]);
           out[base + 1] = static_cast<floatX>(r[1]);
           out[base + 2] = static_cast<floatX>(r[2]);
           out[base + 3] = static_cast<floatX>(r[3]);
         } else {
           // Scalar tail for N not divisible by 4.
           for (std::size_t k = base; k < N; ++k) {
             const float xi = static_cast<float>(inp[k]);
             const float cube = 0.044715f * xi * xi * xi;
             out[k] = static_cast<floatX>(
                 0.5f * xi *
                 (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube))));
           }
         }
       })
      .wait();
}

void gelu_forward(sycl::queue &q, int kernel_num, floatX *out,
                  const floatX *inp, int B, int T, int C) {
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
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

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const std::size_t N =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (N > 0 &&
      std::fread(inp, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::property::queue::in_order{}};

  float *d_inp = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_out = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (N == 0 ? 1 : N)));

  if (N > 0) {
    q.memcpy(d_inp, inp, sizeof(float) * N).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (N > 0) {
      q.memset(d_out, 0, sizeof(float) * N).wait();
    }
    gelu_forward(q, k, d_out, d_inp, B, T, C);
    q.wait();
    if (N > 0) {
      q.memcpy(out, d_out, sizeof(float) * N).wait();
      std::fwrite(out, sizeof(float), N, fout);
    }
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(out);
  return 0;
}
