#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// ----------------------------------------------------------------------------
// Kernel variant 1: naive parallel over B*T*V (own SYCL kernel)
class CrossEntropySoftmaxBwdKernel1;

static void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                           const float *dlosses,
                                           const float *probs,
                                           const int *targets, int B, int T,
                                           int V) {
  const size_t N = static_cast<size_t>(B) * static_cast<size_t>(T) *
                   static_cast<size_t>(V);
  const size_t TV = static_cast<size_t>(T) * static_cast<size_t>(V);
  if (N == 0) return;
  q.parallel_for<CrossEntropySoftmaxBwdKernel1>(
      sycl::range<1>(N), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        size_t b = i / TV;
        size_t t = (i / static_cast<size_t>(V)) % static_cast<size_t>(T);
        size_t v = i % static_cast<size_t>(V);
        size_t bt = b * static_cast<size_t>(T) + t;
        const float *probs_bt = probs + b * TV + t * static_cast<size_t>(V);
        float *dlogits_bt = dlogits + b * TV + t * static_cast<size_t>(V);
        float dloss = dlosses[bt];
        int ix = targets[bt];
        float p = probs_bt[v];
        float indicator = (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
      })
      .wait();
}

static void crossentropy_softmax_backward(sycl::queue &q, int kernel_num,
                                          float *dlogits, const float *dlosses,
                                          const float *probs,
                                          const int *targets, int B, int T,
                                          int V) {
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
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  const size_t nBTV =
      static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(V);
  const size_t nBT = static_cast<size_t>(B) * static_cast<size_t>(T);
  float *probs = static_cast<float *>(std::malloc(sizeof(float) * nBTV));
  int *targets = static_cast<int *>(std::malloc(sizeof(int) * nBT));
  float *dlosses = static_cast<float *>(std::malloc(sizeof(float) * nBT));
  if (nBTV > 0) {
    if (std::fread(probs, sizeof(float), nBTV, fin) != nBTV) {
      std::fprintf(stderr, "input too short at probs\n");
      return 2;
    }
  }
  if (nBT > 0) {
    if (std::fread(targets, sizeof(int), nBT, fin) != nBT) {
      std::fprintf(stderr, "input too short at targets\n");
      return 2;
    }
    if (std::fread(dlosses, sizeof(float), nBT, fin) != nBT) {
      std::fprintf(stderr, "input too short at dlosses\n");
      return 2;
    }
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_probs = sycl::malloc_device<float>(nBTV, q);
  int *d_targets = sycl::malloc_device<int>(nBT, q);
  float *d_dlosses = sycl::malloc_device<float>(nBT, q);
  float *d_dlogits = sycl::malloc_device<float>(nBTV, q);
  float *dlogits = static_cast<float *>(std::malloc(sizeof(float) * nBTV));

  if (nBTV > 0)
    q.memcpy(d_probs, probs, sizeof(float) * nBTV).wait();
  if (nBT > 0) {
    q.memcpy(d_targets, targets, sizeof(int) * nBT).wait();
    q.memcpy(d_dlosses, dlosses, sizeof(float) * nBT).wait();
  }

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (nBTV > 0)
      q.memset(d_dlogits, 0, sizeof(float) * nBTV).wait();
    crossentropy_softmax_backward(q, k, d_dlogits, d_dlosses, d_probs,
                                  d_targets, B, T, V);
    q.wait();
    if (nBTV > 0)
      q.memcpy(dlogits, d_dlogits, sizeof(float) * nBTV).wait();
    if (nBTV > 0)
      std::fwrite(dlogits, sizeof(float), nBTV, fout);
  }
  std::fclose(fout);

  sycl::free(d_probs, q);
  sycl::free(d_targets, q);
  sycl::free(d_dlosses, q);
  sycl::free(d_dlogits, q);
  std::free(probs);
  std::free(targets);
  std::free(dlosses);
  std::free(dlogits);
  return 0;
}
