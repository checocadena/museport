/*
 * Idiomatic SYCL 2020 port of llm.c crossentropy_forward (fp32, kernel 1).
 * Contract: build/candidate <input> <output> B T V reads probs/targets and
 * writes each variant's losses in order as raw float32.
 */
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ----------------------------------------------------------------------------
// SYCL kernels: one named kernel per variant.
// ----------------------------------------------------------------------------

class CrossEntropyFwdK1;

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V,
                           std::size_t local_size) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0) {
    return;
  }
  if (local_size == 0) {
    local_size = 256;
  }
  // Clamp work-group size to the device limit.
  const std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (local_size > max_wg) {
    local_size = max_wg;
  }
  const std::size_t n_groups =
      (N + local_size - 1) / local_size;
  const std::size_t global_size = n_groups * local_size;

  q.submit([&](sycl::handler &h) {
    h.parallel_for<class CrossEntropyFwdK1>(
        sycl::nd_range<1>{sycl::range<1>{global_size},
                         sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> it) {
          const std::size_t i = it.get_global_id(0);
          if (i < N) {
            const std::size_t t = i % static_cast<std::size_t>(T);
            const std::size_t b = i / static_cast<std::size_t>(T);
            const float *probs_bt =
                probs + (b * static_cast<std::size_t>(T) + t) *
                            static_cast<std::size_t>(V);
            const int ix = targets[i];
            losses[i] = -sycl::log(probs_bt[ix]);
          }
        });
  });
  q.wait();
}

void crossentropy_forward(sycl::queue &q, int kernel_num, float *losses,
                          const float *probs, const int *targets, int B, int T,
                          int V, std::size_t block_size) {
  switch (kernel_num) {
    case 1:
      crossentropy_forward1(q, losses, probs, targets, B, T, V, block_size);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// Contract main.
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::srand(0);
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
  const std::size_t n_probs =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(V);
  const std::size_t n_bt =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  float *probs = static_cast<float *>(std::malloc(sizeof(float) * n_probs));
  int *targets = static_cast<int *>(std::malloc(sizeof(int) * n_bt));
  if (n_probs > 0 &&
      std::fread(probs, sizeof(float), n_probs, fin) != n_probs) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  if (n_bt > 0 && std::fread(targets, sizeof(int), n_bt, fin) != n_bt) {
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
  float *d_probs = sycl::malloc_device<float>(n_probs > 0 ? n_probs : 1, q);
  int *d_targets = sycl::malloc_device<int>(n_bt > 0 ? n_bt : 1, q);
  float *d_out = sycl::malloc_device<float>(n_bt > 0 ? n_bt : 1, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n_bt > 0 ? n_bt : 1)));
  if (!d_probs || !d_targets || !d_out || !out) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }

  if (n_probs > 0) {
    q.memcpy(d_probs, probs, sizeof(float) * n_probs).wait();
  }
  if (n_bt > 0) {
    q.memcpy(d_targets, targets, sizeof(int) * n_bt).wait();
  }

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (n_bt > 0) {
      q.memset(d_out, 0, sizeof(float) * n_bt).wait();
    }
    crossentropy_forward(q, k, d_out, d_probs, d_targets, B, T, V, 256);
    if (n_bt > 0) {
      q.memcpy(out, d_out, sizeof(float) * n_bt).wait();
      std::fwrite(out, sizeof(float), n_bt, fout);
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
