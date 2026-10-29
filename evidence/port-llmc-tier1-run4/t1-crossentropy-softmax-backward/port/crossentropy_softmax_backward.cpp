#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel 1: naive parallel over B*T*V. Each work-item handles one (b,t,v)
// and accumulates (p - indicator) * dloss into dlogits.
struct CrossEntropySoftmaxBackwardKernel1;

void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                    const float *dlosses, const float *probs,
                                    const int *targets, int B, int T, int V) {
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(V);
  const std::size_t tv = static_cast<std::size_t>(T) * static_cast<std::size_t>(V);
  const std::size_t v_dim = static_cast<std::size_t>(V);
  const std::size_t t_dim = static_cast<std::size_t>(T);
  q.parallel_for<CrossEntropySoftmaxBackwardKernel1>(
      sycl::range<1>(n), [=](sycl::id<1> idx) {
        std::size_t i = idx[0];
        std::size_t b = i / tv;
        std::size_t t = (i / v_dim) % t_dim;
        std::size_t v = i % v_dim;
        std::size_t bt = b * t_dim + t;
        float dloss = dlosses[bt];
        int ix = targets[bt];
        float p = probs[i];
        float indicator = (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
        dlogits[i] = (p - indicator) * dloss;
      })
      .wait();
}

void crossentropy_softmax_backward(int kernel_num, sycl::queue &q,
                                   float *dlogits, const float *dlosses,
                                   const float *probs, const int *targets, int B,
                                   int T, int V) {
  switch (kernel_num) {
    case 1:
      crossentropy_softmax_backward1(q, dlogits, dlosses, probs, targets, B, T,
                                     V);
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
  const std::size_t n_btv =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(V);
  const std::size_t n_bt =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  float *probs = static_cast<float *>(std::malloc(sizeof(float) * n_btv));
  int *targets = static_cast<int *>(std::malloc(sizeof(int) * n_bt));
  float *dlosses = static_cast<float *>(std::malloc(sizeof(float) * n_bt));
  float *dlogits = static_cast<float *>(std::malloc(sizeof(float) * n_btv));
  if (!probs || !targets || !dlosses || !dlogits) {
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

  sycl::queue q{sycl::default_selector_v};
  float *d_probs = sycl::malloc_device<float>(n_btv, q);
  int *d_targets = sycl::malloc_device<int>(n_bt, q);
  float *d_dlosses = sycl::malloc_device<float>(n_bt, q);
  float *d_dlogits = sycl::malloc_device<float>(n_btv, q);
  if (!d_probs || !d_targets || !d_dlosses || !d_dlogits) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_probs, probs, sizeof(float) * n_btv).wait();
  q.memcpy(d_targets, targets, sizeof(int) * n_bt).wait();
  q.memcpy(d_dlosses, dlosses, sizeof(float) * n_bt).wait();

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    // dlogits is fully overwritten by the kernel (each element written once),
    // but zero it first to match the CUDA harness which memsets before launch.
    q.fill(d_dlogits, 0.0f, n_btv).wait();
    crossentropy_softmax_backward(k, q, d_dlogits, d_dlosses, d_probs,
                                  d_targets, B, T, V);
    q.memcpy(dlogits, d_dlogits, sizeof(float) * n_btv).wait();
    std::fwrite(dlogits, sizeof(float), n_btv, fout);
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
