/*
Crossentropy forward pass (fp32), idiomatic SYCL 2020 port of llm.c
dev/cuda/crossentropy_forward.cu.

Variant 1: straightforward port, parallel over B*T.
*/

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

// ----------------------------------------------------------------------------
// GPU kernels

// Each kernel variant is its own named SYCL kernel type so it can be
// distinguished on device.
class crossentropy_forward_kernel1_tag;

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V) {
  const int N = B * T;
  if (N <= 0) {
    return;
  }
  q.parallel_for<crossentropy_forward_kernel1_tag>(
      sycl::range<1>(static_cast<std::size_t>(N)),
      [=](sycl::item<1> it) {
        const int i = static_cast<int>(it.get_id(0));
        const int b = i / T;
        const int t = i % T;
        const float *probs_bt = probs + static_cast<std::size_t>(b) * T * V +
                                static_cast<std::size_t>(t) * V;
        const int ix = targets[i];
        losses[i] = -sycl::log(probs_bt[ix]);
      });
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
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int V = std::atoi(argv[5]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *probs =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(B * T * V)));
  if (std::fread(probs, sizeof(float), (std::size_t)(B * T * V), fin) !=
      (std::size_t)(B * T * V)) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  int *targets =
      static_cast<int *>(std::malloc(sizeof(int) * (std::size_t)(B * T)));
  if (std::fread(targets, sizeof(int), (std::size_t)(B * T), fin) !=
      (std::size_t)(B * T)) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_probs = sycl::malloc_device<float>((std::size_t)(B * T * V), q);
  q.memcpy(d_probs, probs, sizeof(float) * (std::size_t)(B * T * V)).wait();
  int *d_targets = sycl::malloc_device<int>((std::size_t)(B * T), q);
  q.memcpy(d_targets, targets, sizeof(int) * (std::size_t)(B * T)).wait();

  float *out =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(B * T)));
  float *d_out = sycl::malloc_device<float>((std::size_t)(B * T), q);

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (std::size_t)(B * T)).wait();
    crossentropy_forward(k, q, d_out, d_probs, d_targets, B, T, V);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * (std::size_t)(B * T)).wait();
    std::fwrite(out, sizeof(float), (std::size_t)(B * T), fout);
  }
  std::fclose(fout);

  sycl::free(d_out, q);
  sycl::free(d_targets, q);
  sycl::free(d_probs, q);
  std::free(out);
  std::free(targets);
  std::free(probs);
  return 0;
}
