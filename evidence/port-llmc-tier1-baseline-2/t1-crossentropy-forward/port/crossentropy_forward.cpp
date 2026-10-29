#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

// Kernel variant 1: straightforward port, parallel over B*T.
class CrossEntropyForwardKernel1;

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V) {
  const int N = B * T;
  if (N <= 0) {
    return;
  }
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class CrossEntropyForwardKernel1>(
        sycl::range<1>(static_cast<std::size_t>(N)),
        [=](sycl::id<1> idx) {
          const int i = static_cast<int>(idx[0]);
          const int b = i / T;
          const int t = i % T;
          const float *probs_bt =
              probs + static_cast<std::size_t>(b) * static_cast<std::size_t>(T) *
                          static_cast<std::size_t>(V) +
              static_cast<std::size_t>(t) * static_cast<std::size_t>(V);
          const int ix = targets[static_cast<std::size_t>(b) *
                                     static_cast<std::size_t>(T) +
                                 static_cast<std::size_t>(t)];
          losses[static_cast<std::size_t>(b) * static_cast<std::size_t>(T) +
                 static_cast<std::size_t>(t)] =
              -sycl::log(probs_bt[ix]);
        });
  });
  q.wait();
}

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
      static_cast<float *>(std::malloc(sizeof(float) * static_cast<std::size_t>(B) *
                                       static_cast<std::size_t>(T) *
                                       static_cast<std::size_t>(V)));
  if (std::fread(probs, sizeof(float),
                 static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                     static_cast<std::size_t>(V),
                 fin) != static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                          static_cast<std::size_t>(V)) {
    std::fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  int *targets =
      static_cast<int *>(std::malloc(sizeof(int) * static_cast<std::size_t>(B) *
                                     static_cast<std::size_t>(T)));
  if (std::fread(targets, sizeof(int),
                 static_cast<std::size_t>(B) * static_cast<std::size_t>(T),
                 fin) != static_cast<std::size_t>(B) * static_cast<std::size_t>(T)) {
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

  float *d_probs = sycl::malloc_device<float>(
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
          static_cast<std::size_t>(V),
      q);
  int *d_targets = sycl::malloc_device<int>(
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T), q);
  float *d_out = sycl::malloc_device<float>(
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T), q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) *
                                                static_cast<std::size_t>(B) *
                                                static_cast<std::size_t>(T)));

  q.memcpy(d_probs, probs,
           sizeof(float) * static_cast<std::size_t>(B) *
               static_cast<std::size_t>(T) * static_cast<std::size_t>(V))
      .wait();
  q.memcpy(d_targets, targets,
           sizeof(int) * static_cast<std::size_t>(B) *
               static_cast<std::size_t>(T))
      .wait();

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0,
             sizeof(float) * static_cast<std::size_t>(B) *
                 static_cast<std::size_t>(T))
        .wait();
    crossentropy_forward(k, q, d_out, d_probs, d_targets, B, T, V);
    q.memcpy(out, d_out,
             sizeof(float) * static_cast<std::size_t>(B) *
                 static_cast<std::size_t>(T))
        .wait();
    std::fwrite(out, sizeof(float),
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T),
                fout);
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
