// Idiomatic SYCL 2020 port of residual_forward (fp32).
// Two kernel variants, each its own SYCL kernel, 1-D launches.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace {

constexpr int kBlockSize = 256;
constexpr int kVecWidth = 4; // 128-bit groups of float

template <typename T>
constexpr T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class ResidualForwardKernel1;
class ResidualForwardKernel2;

void residual_forward1(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, std::size_t n) {
  if (n == 0) return;
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t num_groups =
      static_cast<std::size_t>(ceil_div<std::size_t>(n, local));
  const std::size_t global = num_groups * local;
  q.parallel_for<ResidualForwardKernel1>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
      [=](sycl::nd_item<1> it) {
        const std::size_t idx = it.get_global_id(0);
        if (idx < n) {
          out[idx] = inp1[idx] + inp2[idx];
        }
      });
}

void residual_forward2(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, std::size_t n) {
  if (n == 0) return;
  const std::size_t num_groups =
      static_cast<std::size_t>(ceil_div<std::size_t>(n, kVecWidth));
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t num_blocks =
      static_cast<std::size_t>(ceil_div<std::size_t>(num_groups, local));
  const std::size_t global = num_blocks * local;
  q.parallel_for<ResidualForwardKernel2>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
      [=](sycl::nd_item<1> it) {
        const std::size_t gid = it.get_global_id(0);
        if (gid >= num_groups) return;
        const std::size_t base = gid * kVecWidth;
        if (base + kVecWidth <= n) {
          const auto *a4 =
              reinterpret_cast<const sycl::float4 *>(inp1);
          const auto *b4 =
              reinterpret_cast<const sycl::float4 *>(inp2);
          auto *o4 = reinterpret_cast<sycl::float4 *>(out);
          sycl::float4 av = a4[gid];
          sycl::float4 bv = b4[gid];
          sycl::float4 ov;
          ov[0] = av[0] + bv[0];
          ov[1] = av[1] + bv[1];
          ov[2] = av[2] + bv[2];
          ov[3] = av[3] + bv[3];
          o4[gid] = ov;
        } else {
          // Tail (only when n is not a multiple of 4): scalar fallback.
          for (int k = 0; k < kVecWidth; ++k) {
            const std::size_t idx = base + static_cast<std::size_t>(k);
            if (idx < n) {
              out[idx] = inp1[idx] + inp2[idx];
            }
          }
        }
      });
}

void residual_forward_dispatch(sycl::queue &q, int kernel_num, float *out,
                               const float *inp1, const float *inp2,
                               std::size_t n) {
  switch (kernel_num) {
    case 1:
      residual_forward1(q, out, inp1, inp2, n);
      break;
    case 2:
      residual_forward2(q, out, inp1, inp2, n);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

}  // namespace

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
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!inp1 || !inp2 || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp1, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v,
                sycl::property::queue::in_order{}};

  float *d_inp1 = sycl::malloc_device<float>(n, q);
  float *d_inp2 = sycl::malloc_device<float>(n, q);
  float *d_out = sycl::malloc_device<float>(n, q);
  if (n > 0 && (!d_inp1 || !d_inp2 || !d_out)) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }

  if (n > 0) {
    q.memcpy(d_inp1, inp1, sizeof(float) * n).wait();
    q.memcpy(d_inp2, inp2, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.fill(d_out, 0.0f, n).wait();
    }
    residual_forward_dispatch(q, k, d_out, d_inp1, d_inp2, n);
    q.wait();
    if (n > 0) {
      q.memcpy(out, d_out, sizeof(float) * n).wait();
    }
    std::fwrite(out, sizeof(float), n, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_out, q);
  std::free(inp1);
  std::free(inp2);
  std::free(out);
  return 0;
}
