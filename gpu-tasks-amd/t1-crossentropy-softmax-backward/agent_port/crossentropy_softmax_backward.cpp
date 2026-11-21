#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

// Kernel tag types: one distinct SYCL kernel per variant.
class CrossEntropySoftmaxBackwardKernel1;

// 1-D parallel kernel: one work-item per (b, t, v) element.
void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                    const float *dlosses, const float *probs,
                                    const int *targets, int B, int T, int V) {
  const long long N = static_cast<long long>(B) * T * V;
  if (N <= 0) {
    return;
  }
  constexpr std::size_t kBlockSize = 256;
  const std::size_t n = static_cast<std::size_t>(N);
  const std::size_t global =
      ((n + kBlockSize - 1) / kBlockSize) * kBlockSize;
  const long long tv = static_cast<long long>(T) * V;
  const long long bt = static_cast<long long>(B) * T;

  (void)bt;
  q.parallel_for<CrossEntropySoftmaxBackwardKernel1>(
       sycl::nd_range<1>(sycl::range<1>(global),
                         sycl::range<1>(kBlockSize)),
       [=](sycl::nd_item<1> it) {
         std::size_t i = it.get_global_id(0);
         if (i < n) {
           long long ii = static_cast<long long>(i);
           long long b = ii / tv;
           long long t = (ii / V) % T;
           long long v = ii % V;
           const float *probs_bt = probs + b * tv + t * V;
           float *dlogits_bt = dlogits + b * tv + t * V;
           float dloss = dlosses[b * T + t];
           int ix = targets[b * T + t];
           float p = probs_bt[v];
           float indicator = (v == static_cast<long long>(ix)) ? 1.0f : 0.0f;
           dlogits_bt[v] += (p - indicator) * dloss;
         }
       });
}

void crossentropy_softmax_backward(sycl::queue &q, int kernel_num,
                                   float *dlogits, const float *dlosses,
                                   const float *probs, const int *targets,
                                   int B, int T, int V) {
  switch (kernel_num) {
  case 1:
    crossentropy_softmax_backward1(q, dlogits, dlosses, probs, targets, B, T,
                                   V);
    break;
  default:
    std::fprintf(stderr, "Invalid kernel number\n");
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
  const std::size_t n_btv =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(V);
  const std::size_t n_bt =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *probs = static_cast<float *>(std::malloc(sizeof(float) * n_btv));
  int *targets = static_cast<int *>(std::malloc(sizeof(int) * n_bt));
  float *dlosses = static_cast<float *>(std::malloc(sizeof(float) * n_bt));
  if (!probs || !targets || !dlosses) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(probs, sizeof(float), n_btv, fin) != n_btv) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  if (std::fread(targets, sizeof(int), n_bt, fin) != n_bt) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  if (std::fread(dlosses, sizeof(float), n_bt, fin) != n_bt) {
    std::fprintf(stderr, "input too short at dlosses\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};
    float *d_probs = sycl::malloc_device<float>(n_btv, q);
    int *d_targets = sycl::malloc_device<int>(n_bt, q);
    float *d_dlosses = sycl::malloc_device<float>(n_bt, q);
    float *d_dlogits = sycl::malloc_device<float>(n_btv, q);
    if (!d_probs || !d_targets || !d_dlosses || !d_dlogits) {
      std::fprintf(stderr, "device malloc failed\n");
      return 2;
    }
    float *dlogits = static_cast<float *>(std::malloc(sizeof(float) * n_btv));
    if (!dlogits) {
      std::fprintf(stderr, "host malloc failed\n");
      return 2;
    }

    q.memcpy(d_probs, probs, sizeof(float) * n_btv).wait();
    q.memcpy(d_targets, targets, sizeof(int) * n_bt).wait();
    q.memcpy(d_dlosses, dlosses, sizeof(float) * n_bt).wait();

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
      const int k = variants[vi];
      q.memset(d_dlogits, 0, sizeof(float) * n_btv).wait();
      crossentropy_softmax_backward(q, k, d_dlogits, d_dlosses, d_probs,
                                    d_targets, B, T, V);
      q.wait();
      q.memcpy(dlogits, d_dlogits, sizeof(float) * n_btv).wait();
      std::fwrite(dlogits, sizeof(float), n_btv, fout);
    }

    std::free(dlogits);
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_dlogits, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(probs);
  std::free(targets);
  std::free(dlosses);
  return 0;
}
