#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel variant 1: naive parallel over B*T*V.
// Kept as its own SYCL kernel (named function object) computed on the device.
class CrossentropySoftmaxBackwardKernel1;

static inline std::size_t ceil_div(std::size_t n, std::size_t d) {
  return (n + d - 1) / d;
}

void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                    const float *dlosses, const float *probs,
                                    const int *targets, int B, int T, int V) {
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(V);
  if (n == 0) {
    return;
  }
  const std::size_t local = 256;
  const std::size_t n_groups = ceil_div(n, local);
  const std::size_t global = n_groups * local;
  const std::size_t tv = static_cast<std::size_t>(T) * static_cast<std::size_t>(V);
  const std::size_t btv =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(V);
  const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

  q.parallel_for<CrossentropySoftmaxBackwardKernel1>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i < btv) {
           const std::size_t v = i % static_cast<std::size_t>(V);
           const std::size_t bt_idx = i / static_cast<std::size_t>(V);
           const float dloss = dlosses[bt_idx];
           const int ix = targets[bt_idx];
           const float p = probs[i];
           const float indicator = (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
           dlogits[i] += (p - indicator) * dloss;
         }
       });
}

void crossentropy_softmax_backward(sycl::queue &q, int kernel_num, float *dlogits,
                                   const float *dlosses, const float *probs,
                                   const int *targets, int B, int T, int V) {
  switch (kernel_num) {
    case 1:
      crossentropy_softmax_backward1(q, dlogits, dlosses, probs, targets, B, T, V);
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
  float *probs = (float *)std::malloc(sizeof(float) * (size_t)(B * T * V));
  int *targets = (int *)std::malloc(sizeof(int) * (size_t)(B * T));
  float *dlosses = (float *)std::malloc(sizeof(float) * (size_t)(B * T));
  if (!probs || !targets || !dlosses) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(probs, sizeof(float), (size_t)(B * T * V), fin) != (size_t)(B * T * V)) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  if (std::fread(targets, sizeof(int), (size_t)(B * T), fin) != (size_t)(B * T)) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  if (std::fread(dlosses, sizeof(float), (size_t)(B * T), fin) != (size_t)(B * T)) {
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

    float *d_probs =
        sycl::malloc_device<float>((size_t)(B * T * V), q);
    int *d_targets = sycl::malloc_device<int>((size_t)(B * T), q);
    float *d_dlosses = sycl::malloc_device<float>((size_t)(B * T), q);
    float *d_dlogits = sycl::malloc_device<float>((size_t)(B * T * V), q);
    if (!d_probs || !d_targets || !d_dlosses || !d_dlogits) {
      std::fprintf(stderr, "device malloc failed\n");
      return 2;
    }
    q.memcpy(d_probs, probs, sizeof(float) * (size_t)(B * T * V)).wait();
    q.memcpy(d_targets, targets, sizeof(int) * (size_t)(B * T)).wait();
    q.memcpy(d_dlosses, dlosses, sizeof(float) * (size_t)(B * T)).wait();

    float *dlogits = (float *)std::malloc(sizeof(float) * (size_t)(B * T * V));
    if (!dlogits) {
      std::fprintf(stderr, "host malloc failed\n");
      return 2;
    }

    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
      const int k = variants[vi];
      q.memset(d_dlogits, 0, sizeof(float) * (size_t)(B * T * V)).wait();
      crossentropy_softmax_backward(q, k, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V);
      q.wait();
      q.memcpy(dlogits, d_dlogits, sizeof(float) * (size_t)(B * T * V)).wait();
      std::fwrite(dlogits, sizeof(float), (size_t)(B * T * V), fout);
    }

    std::free(dlogits);
    sycl::free(d_probs, q);
    sycl::free(d_targets, q);
    sycl::free(d_dlosses, q);
    sycl::free(d_dlogits, q);
  } catch (sycl::exception const &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(probs);
  std::free(targets);
  std::free(dlosses);
  return 0;
}
