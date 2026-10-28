#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel variant 1: naive parallel over B*T*V.
class CrossEntropySoftmaxBackwardKernel1;

void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                    const float *dlosses, const float *probs,
                                    const int *targets, int B, int T, int V) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(V);
  const std::size_t TV = static_cast<std::size_t>(T) * static_cast<std::size_t>(V);
  const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  (void)BT;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<CrossEntropySoftmaxBackwardKernel1>(
        sycl::range<1>(N), [=](sycl::id<1> idx) {
          std::size_t i = idx[0];
          // i < N guaranteed by 1-D range; decode b, t, v.
          std::size_t v = i % static_cast<std::size_t>(V);
          std::size_t t = (i / static_cast<std::size_t>(V)) % static_cast<std::size_t>(T);
          std::size_t b = i / TV;
          const float *probs_bt =
              probs + b * static_cast<std::size_t>(T) * static_cast<std::size_t>(V) +
              t * static_cast<std::size_t>(V);
          float *dlogits_bt =
              dlogits + b * static_cast<std::size_t>(T) * static_cast<std::size_t>(V) +
              t * static_cast<std::size_t>(V);
          float dloss = dlosses[b * static_cast<std::size_t>(T) + t];
          int ix = targets[b * static_cast<std::size_t>(T) + t];
          float p = probs_bt[v];
          float indicator = (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
          dlogits_bt[v] += (p - indicator) * dloss;
        });
  });
  q.wait();
}

void crossentropy_softmax_backward(int kernel_num, sycl::queue &q,
                                   float *dlogits, const float *dlosses,
                                   const float *probs, const int *targets,
                                   int B, int T, int V) {
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
  float *probs = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * V));
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
  float *dlosses = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T));
  if (std::fread(dlosses, sizeof(float), (std::size_t)(B * T), fin) !=
      (std::size_t)(B * T)) {
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

  float *d_probs =
      sycl::malloc_device<float>((std::size_t)(B * T * V), q);
  q.memcpy(d_probs, probs, sizeof(float) * (std::size_t)(B * T * V)).wait();
  int *d_targets = sycl::malloc_device<int>((std::size_t)(B * T), q);
  q.memcpy(d_targets, targets, sizeof(int) * (std::size_t)(B * T)).wait();
  float *d_dlosses = sycl::malloc_device<float>((std::size_t)(B * T), q);
  q.memcpy(d_dlosses, dlosses, sizeof(float) * (std::size_t)(B * T)).wait();
  float *dlogits = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * V));
  float *d_dlogits =
      sycl::malloc_device<float>((std::size_t)(B * T * V), q);

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_dlogits, 0, sizeof(float) * (std::size_t)(B * T * V)).wait();
    crossentropy_softmax_backward(k, q, d_dlogits, d_dlosses, d_probs,
                                  d_targets, B, T, V);
    q.memcpy(dlogits, d_dlogits, sizeof(float) * (std::size_t)(B * T * V))
        .wait();
    std::fwrite(dlogits, sizeof(float), (std::size_t)(B * T * V), fout);
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
