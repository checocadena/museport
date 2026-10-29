#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel variant 1: naive parallel over B*T*V.
class CrossEntropySoftmaxBackwardKernel1;

static void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                           const float *dlosses,
                                           const float *probs,
                                           const int *targets, int B, int T,
                                           int V) {
  const size_t N = static_cast<size_t>(B) * static_cast<size_t>(T) *
                   static_cast<size_t>(V);
  if (N == 0) {
    return;
  }
  const size_t TV = static_cast<size_t>(T) * static_cast<size_t>(V);
  const size_t BT = static_cast<size_t>(B) * static_cast<size_t>(T);
  (void)BT;
  q.parallel_for<CrossEntropySoftmaxBackwardKernel1>(sycl::range<1>(N),
                                                    [=](sycl::item<1> it) {
    const size_t i = it.get_id(0);
    const size_t b = i / TV;
    const size_t t = (i / static_cast<size_t>(V)) % static_cast<size_t>(T);
    const size_t v = i % static_cast<size_t>(V);
    const float *probs_bt =
        probs + b * TV + t * static_cast<size_t>(V);
    float *dlogits_bt =
        dlogits + b * TV + t * static_cast<size_t>(V);
    const float dloss = dlosses[b * static_cast<size_t>(T) + t];
    const int ix = targets[b * static_cast<size_t>(T) + t];
    const float p = probs_bt[v];
    const float indicator =
        (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
    dlogits_bt[v] += (p - indicator) * dloss;
  }).wait();
}

static void crossentropy_softmax_backward(int kernel_num, sycl::queue &q,
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
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(B * T * V)));
  if (std::fread(probs, sizeof(float), (size_t)(B * T * V), fin) !=
      (size_t)(B * T * V)) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  int *targets =
      static_cast<int *>(std::malloc(sizeof(int) * (size_t)(B * T)));
  if (std::fread(targets, sizeof(int), (size_t)(B * T), fin) !=
      (size_t)(B * T)) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  float *dlosses =
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(B * T)));
  if (std::fread(dlosses, sizeof(float), (size_t)(B * T), fin) !=
      (size_t)(B * T)) {
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

  float *d_probs = sycl::malloc_device<float>((size_t)(B * T * V), q);
  int *d_targets = sycl::malloc_device<int>((size_t)(B * T), q);
  float *d_dlosses = sycl::malloc_device<float>((size_t)(B * T), q);
  float *d_dlogits = sycl::malloc_device<float>((size_t)(B * T * V), q);
  float *dlogits =
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(B * T * V)));

  q.memcpy(d_probs, probs, sizeof(float) * (size_t)(B * T * V)).wait();
  q.memcpy(d_targets, targets, sizeof(int) * (size_t)(B * T)).wait();
  q.memcpy(d_dlosses, dlosses, sizeof(float) * (size_t)(B * T)).wait();

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_dlogits, 0, sizeof(float) * (size_t)(B * T * V)).wait();
    crossentropy_softmax_backward(k, q, d_dlogits, d_dlosses, d_probs,
                                  d_targets, B, T, V);
    q.wait();
    q.memcpy(dlogits, d_dlogits, sizeof(float) * (size_t)(B * T * V)).wait();
    std::fwrite(dlogits, sizeof(float), (size_t)(B * T * V), fout);
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
