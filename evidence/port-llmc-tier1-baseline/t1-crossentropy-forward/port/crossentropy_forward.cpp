// Idiomatic SYCL 2020 port of crossentropy_forward (fp32, variant 1).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// ----------------------------------------------------------------------------
// Kernel variant 1: parallel over B*T, losses[i] = -log(probs_bt[ix]).
// Kept as its own SYCL kernel (named function object) computed on device.
struct CrossEntropyForwardKernel1 {
  float *losses;
  const float *probs;
  const int *targets;
  int B, T, V;
  int N;
  void operator()(sycl::nd_item<1> item) const {
    const std::size_t i = item.get_global_linear_id();
    if (i < static_cast<std::size_t>(N)) {
      const int b = static_cast<int>(i / static_cast<std::size_t>(T));
      const int t = static_cast<int>(i % static_cast<std::size_t>(T));
      const float *probs_bt =
          probs + static_cast<std::size_t>(b) * static_cast<std::size_t>(T) *
                      static_cast<std::size_t>(V) +
                  static_cast<std::size_t>(t) * static_cast<std::size_t>(V);
      const int ix = targets[i];
      losses[i] = -sycl::log(probs_bt[ix]);
    }
  }
};

static inline std::size_t ceil_div(std::size_t n, std::size_t d) {
  return (n + d - 1) / d;
}

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0) {
    return;
  }
  constexpr std::size_t block_size = 256;
  const std::size_t grid = ceil_div(N, block_size);
  const sycl::nd_range<1> launch(sycl::range<1>(grid * block_size),
                                 sycl::range<1>(block_size));
  q.parallel_for(
      launch, CrossEntropyForwardKernel1{losses, probs, targets, B, T, V,
                                         static_cast<int>(N)});
}

// kernel version dispatch
void crossentropy_forward(int kernel_num, sycl::queue &q, float *losses,
                          const float *probs, const int *targets, int B, int T,
                          int V) {
  switch (kernel_num) {
  case 1:
    crossentropy_forward1(q, losses, probs, targets, B, T, V);
    break;
  default:
    std::fprintf(stderr, "Invalid kernel number\n");
    std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> B T V; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int V = std::atoi(argv[5]);
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t m =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(V);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *probs = (float *)std::malloc(sizeof(float) * m);
  int *targets = (int *)std::malloc(sizeof(int) * (n == 0 ? 1 : n));
  float *out = (float *)std::malloc(sizeof(float) * (n == 0 ? 1 : n));
  if ((m && !probs) || !targets || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(probs, sizeof(float), m, fin) != m) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  if (std::fread(targets, sizeof(int), n, fin) != n) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_probs = sycl::malloc_device<float>(m == 0 ? 1 : m, q);
  int *d_targets = sycl::malloc_device<int>(n == 0 ? 1 : n, q);
  float *d_out = sycl::malloc_device<float>(n == 0 ? 1 : n, q);
  if ((m && !d_probs) || !d_targets || !d_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  if (m) {
    q.memcpy(d_probs, probs, sizeof(float) * m).wait();
  }
  if (n) {
    q.memcpy(d_targets, targets, sizeof(int) * n).wait();
  }

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (n) {
      q.memset(d_out, 0, sizeof(float) * n).wait();
    }
    crossentropy_forward(k, q, d_out, d_probs, d_targets, B, T, V);
    q.wait();
    if (n) {
      q.memcpy(out, d_out, sizeof(float) * n).wait();
      std::fwrite(out, sizeof(float), n, fout);
    }
  }
  std::fclose(fout);

  sycl::free(d_probs, q);
  sycl::free(d_targets, q);
  sycl::free(d_out, q);
  std::free(probs);
  std::free(targets);
  std::free(out);
  return 0;
}
