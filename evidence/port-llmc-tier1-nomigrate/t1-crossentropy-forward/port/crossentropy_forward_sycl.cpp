#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

// ceil division helper
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// SYCL kernels: one named kernel per variant.

class CrossEntropyForwardKernel1;

void crossentropy_forward1(sycl::queue &q, float *losses,
                           const float *probs, const int *targets,
                           int B, int T, int V, int block_size) {
  const int N = B * T;
  const int global_size = ceil_div(N, block_size) * block_size;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<CrossEntropyForwardKernel1>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global_size)),
                          sycl::range<1>(static_cast<size_t>(block_size))),
        [=](sycl::nd_item<1> item) {
          const int i = static_cast<int>(item.get_global_id(0));
          if (i < N) {
            const int b = i / T;
            const int t = i % T;
            const float *probs_bt = probs + static_cast<size_t>(b) * T * V +
                                    static_cast<size_t>(t) * V;
            const int ix = targets[static_cast<size_t>(b) * T + t];
            losses[static_cast<size_t>(b) * T + t] =
                -sycl::log(probs_bt[ix]);
          }
        });
  }).wait();
}

// kernel version dispatch
void crossentropy_forward(int kernel_num, sycl::queue &q, float *losses,
                          const float *probs, const int *targets,
                          int B, int T, int V, int block_size) {
  switch (kernel_num) {
    case 1:
      crossentropy_forward1(q, losses, probs, targets, B, T, V, block_size);
      break;
    default:
      printf("Invalid kernel number\n");
      exit(1);
  }
}

// ----------------------------------------------------------------------------
// contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = atoi(argv[3]);
  const int T = atoi(argv[4]);
  const int V = atoi(argv[5]);

  FILE *fin = fopen(argv[1], "rb");
  if (!fin) { perror(argv[1]); return 2; }
  float *probs =
      (float *)malloc(sizeof(float) * (size_t)(B) * T * V);
  if (fread(probs, sizeof(float), (size_t)(B) * T * V, fin) !=
      (size_t)(B) * T * V) {
    fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  int *targets = (int *)malloc(sizeof(int) * (size_t)(B) * T);
  if (fread(targets, sizeof(int), (size_t)(B) * T, fin) !=
      (size_t)(B) * T) {
    fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  fclose(fin);

  FILE *fout = fopen(argv[2], "wb");
  if (!fout) { perror(argv[2]); return 2; }

  sycl::queue q{sycl::default_selector_v};

  float *d_probs = sycl::malloc_device<float>((size_t)(B) * T * V, q);
  q.memcpy(d_probs, probs, sizeof(float) * (size_t)(B) * T * V).wait();
  int *d_targets = sycl::malloc_device<int>((size_t)(B) * T, q);
  q.memcpy(d_targets, targets, sizeof(int) * (size_t)(B) * T).wait();

  float *out = (float *)malloc(sizeof(float) * (size_t)(B) * T);
  float *d_out = sycl::malloc_device<float>((size_t)(B) * T, q);

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (size_t)(B) * T).wait();
    crossentropy_forward(k, q, d_out, d_probs, d_targets, B, T, V, 256);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * (size_t)(B) * T).wait();
    fwrite(out, sizeof(float), (size_t)(B) * T, fout);
  }
  fclose(fout);

  sycl::free(d_probs, q);
  sycl::free(d_targets, q);
  sycl::free(d_out, q);
  free(probs);
  free(targets);
  free(out);
  return 0;
}
