#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sycl/sycl.hpp>

class CrossEntropySoftmaxBackwardKernel1;

void crossentropy_softmax_backward1(sycl::queue &q, float *dlogits,
                                    const float *dlosses, const float *probs,
                                    const int *targets, int B, int T, int V) {
  const size_t N = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(V);
  const size_t TV = static_cast<size_t>(T) * static_cast<size_t>(V);
  const size_t BT = static_cast<size_t>(B) * static_cast<size_t>(T);
  (void)BT;
  q.parallel_for<CrossEntropySoftmaxBackwardKernel1>(
       sycl::range<1>(N), [=](sycl::id<1> idx) {
         size_t i = idx[0];
         if (i < N) {
           size_t b = i / TV;
           size_t t = (i / static_cast<size_t>(V)) % static_cast<size_t>(T);
           size_t v = i % static_cast<size_t>(V);
           size_t bt = b * static_cast<size_t>(T) + t;
           float dloss = dlosses[bt];
           int ix = targets[bt];
           float p = probs[i];
           float indicator = (static_cast<int>(v) == ix) ? 1.0f : 0.0f;
           dlogits[i] = (p - indicator) * dloss;
         }
       }).wait();
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
    printf("Invalid kernel number\n");
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = atoi(argv[3]);
  const int T = atoi(argv[4]);
  const int V = atoi(argv[5]);

  FILE *fin = fopen(argv[1], "rb");
  if (!fin) {
    perror(argv[1]);
    return 2;
  }
  size_t n_probs = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(V);
  size_t n_bt = static_cast<size_t>(B) * static_cast<size_t>(T);
  float *probs = (float *)malloc(sizeof(float) * n_probs);
  int *targets = (int *)malloc(sizeof(int) * n_bt);
  float *dlosses = (float *)malloc(sizeof(float) * n_bt);
  if (fread(probs, sizeof(float), n_probs, fin) != n_probs) {
    fprintf(stderr, "input too short at probs\n");
    return 2;
  }
  if (fread(targets, sizeof(int), n_bt, fin) != n_bt) {
    fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  if (fread(dlosses, sizeof(float), n_bt, fin) != n_bt) {
    fprintf(stderr, "input too short at dlosses\n");
    return 2;
  }
  fclose(fin);

  FILE *fout = fopen(argv[2], "wb");
  if (!fout) {
    perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_probs = sycl::malloc_device<float>(n_probs, q);
  int *d_targets = sycl::malloc_device<int>(n_bt, q);
  float *d_dlosses = sycl::malloc_device<float>(n_bt, q);
  float *d_dlogits = sycl::malloc_device<float>(n_probs, q);
  float *dlogits = (float *)malloc(sizeof(float) * n_probs);

  q.memcpy(d_probs, probs, sizeof(float) * n_probs).wait();
  q.memcpy(d_targets, targets, sizeof(int) * n_bt).wait();
  q.memcpy(d_dlosses, dlosses, sizeof(float) * n_bt).wait();

  const int variants[] = {1};
  for (int vi = 0; vi < 1; vi++) {
    const int k = variants[vi];
    q.memset(d_dlogits, 0, sizeof(float) * n_probs).wait();
    crossentropy_softmax_backward(k, q, d_dlogits, d_dlosses, d_probs,
                                  d_targets, B, T, V);
    q.wait();
    q.memcpy(dlogits, d_dlogits, sizeof(float) * n_probs).wait();
    fwrite(dlogits, sizeof(float), n_probs, fout);
  }
  fclose(fout);

  sycl::free(d_probs, q);
  sycl::free(d_targets, q);
  sycl::free(d_dlosses, q);
  sycl::free(d_dlogits, q);
  free(probs);
  free(targets);
  free(dlosses);
  free(dlogits);
  return 0;
}
