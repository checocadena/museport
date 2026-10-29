#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// 1-D ceiling division for launch sizing.
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

class CrossEntropyForwardKernel1;

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V,
                           int block_size) {
  const int N = B * T;
  if (N <= 0) return;
  int local = block_size > 0 ? block_size : 256;
  // Clamp local size to device limit.
  std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if ((std::size_t)local > max_wg) local = static_cast<int>(max_wg);
  const std::size_t n = static_cast<std::size_t>(N);
  const std::size_t l = static_cast<std::size_t>(local);
  const std::size_t num_groups = (n + l - 1u) / l;
  const std::size_t global = num_groups * l;
  q.parallel_for<CrossEntropyForwardKernel1>(
      sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{l}},
      [=](sycl::nd_item<1> it) {
        std::size_t i = it.get_global_id(0);
        if (i < n) {
          int b = static_cast<int>(i / static_cast<std::size_t>(T));
          int t = static_cast<int>(i % static_cast<std::size_t>(T));
          const float *probs_bt =
              probs + static_cast<std::size_t>(b) *
                          static_cast<std::size_t>(T) *
                          static_cast<std::size_t>(V) +
                      static_cast<std::size_t>(t) * static_cast<std::size_t>(V);
          int ix = targets[i];
          losses[i] = -sycl::log(probs_bt[ix]);
        }
      });
}

void crossentropy_forward(sycl::queue &q, int kernel_num, float *losses,
                          const float *probs, const int *targets, int B, int T,
                          int V, int block_size) {
  switch (kernel_num) {
    case 1:
      crossentropy_forward1(q, losses, probs, targets, B, T, V, block_size);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

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
      (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * V));
  if (std::fread(probs, sizeof(float), (std::size_t)(B * T * V), fin) !=
      (std::size_t)(B * T * V)) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  int *targets = (int *)std::malloc(sizeof(int) * (std::size_t)(B * T));
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
  float *d_probs =
      sycl::malloc_device<float>((std::size_t)(B * T * V), q);
  q.memcpy(d_probs, probs, sizeof(float) * (std::size_t)(B * T * V)).wait();
  int *d_targets = sycl::malloc_device<int>((std::size_t)(B * T), q);
  q.memcpy(d_targets, targets, sizeof(int) * (std::size_t)(B * T)).wait();
  float *out = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T));
  float *d_out = sycl::malloc_device<float>((std::size_t)(B * T), q);

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (std::size_t)(B * T)).wait();
    crossentropy_forward(q, k, d_out, d_probs, d_targets, B, T, V, 256);
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
