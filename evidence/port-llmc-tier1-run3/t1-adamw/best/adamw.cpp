// Idiomatic SYCL 2020 port of adamw_harness.cu (llm.c dev/cuda/adamw.cu, fp32).
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Two-op lerp: fma(weight, end, fma(-weight, start, start)).
inline float adamw_lerp(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamWKernel1;
class AdamWKernel2;

constexpr std::size_t kWorkGroupSize = 256;

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float lr, float beta1,
                     float beta2, float b1_corr, float b2_corr, float eps,
                     float wd) {
  if (n == 0) return;
  const std::size_t nblocks = (n + kWorkGroupSize - 1) / kWorkGroupSize;
  const sycl::range<1> local(kWorkGroupSize);
  const sycl::range<1> global(nblocks * kWorkGroupSize);
  q.parallel_for<AdamWKernel1>(sycl::nd_range<1>(global, local),
                               [=](sycl::nd_item<1> it) {
                                 const std::size_t i = it.get_global_id(0);
                                 if (i >= n) return;
                                 m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
                                 v[i] = beta2 * v[i] +
                                        (1.0f - beta2) * grads[i] * grads[i];
                                 const float m_hat = m[i] / b1_corr;
                                 const float v_hat = v[i] / b2_corr;
                                 params[i] -=
                                     lr * (m_hat / (sycl::sqrt(v_hat) + eps) +
                                           wd * params[i]);
                               });
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float lr, float beta1,
                     float beta2, float b1_corr, float b2_corr, float eps,
                     float wd) {
  if (n == 0) return;
  const std::size_t nblocks = (n + kWorkGroupSize - 1) / kWorkGroupSize;
  const sycl::range<1> local(kWorkGroupSize);
  const sycl::range<1> global(nblocks * kWorkGroupSize);
  q.parallel_for<AdamWKernel2>(sycl::nd_range<1>(global, local),
                               [=](sycl::nd_item<1> it) {
                                 const std::size_t i = it.get_global_id(0);
                                 if (i >= n) return;
                                 const float grad = grads[i];
                                 float mi = m[i];
                                 float vi = v[i];
                                 mi = adamw_lerp(grad, mi, beta1);
                                 m[i] = mi;
                                 vi = adamw_lerp(grad * grad, vi, beta2);
                                 v[i] = vi;
                                 mi /= b1_corr;
                                 vi /= b2_corr;
                                 params[i] -= lr * (mi / (sycl::sqrt(vi) + eps) +
                                                    wd * params[i]);
                               });
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m, float *v, int t, std::size_t n,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  const float b1_corr = 1.0f - powf(beta1, static_cast<float>(t));
  const float b2_corr = 1.0f - powf(beta2, static_cast<float>(t));
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m, v, n, learning_rate, beta1, beta2,
                      b1_corr, b2_corr, eps, weight_decay);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m, v, n, learning_rate, beta1, beta2,
                      b1_corr, b2_corr, eps, weight_decay);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
    return 2;
  }
  const std::size_t N = static_cast<std::size_t>(std::atoi(argv[3]));

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *params = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *grads = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *m = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *v = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!params || !grads || !m || !v) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  auto read_array = [&](float *dst, const char *what) -> bool {
    if (N == 0) return true;
    if (std::fread(dst, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at %s\n", what);
      return false;
    }
    return true;
  };
  if (!read_array(params, "params") || !read_array(grads, "grads") ||
      !read_array(m, "m") || !read_array(v, "v")) {
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_grads = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_params_out = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_m_out = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_v_out = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * (N == 0 ? 1 : N)));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * (N == 0 ? 1 : N)));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * (N == 0 ? 1 : N)));
  if (!d_grads || !d_params_out || !d_m_out || !d_v_out || !params_out ||
      !m_out || !v_out) {
    std::fprintf(stderr, "allocation failed\n");
    return 2;
  }

  if (N > 0) {
    q.memcpy(d_grads, grads, sizeof(float) * N).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (N > 0) {
      q.memcpy(d_params_out, params, sizeof(float) * N).wait();
      q.memcpy(d_m_out, m, sizeof(float) * N).wait();
      q.memcpy(d_v_out, v, sizeof(float) * N).wait();
    }
    adamw(q, k, d_params_out, d_grads, d_m_out, d_v_out, 10, N);
    q.wait();
    if (N > 0) {
      q.memcpy(params_out, d_params_out, sizeof(float) * N).wait();
      std::fwrite(params_out, sizeof(float), N, fout);
      q.memcpy(m_out, d_m_out, sizeof(float) * N).wait();
      std::fwrite(m_out, sizeof(float), N, fout);
      q.memcpy(v_out, d_v_out, sizeof(float) * N).wait();
      std::fwrite(v_out, sizeof(float), N, fout);
    }
  }

  std::fclose(fout);
  sycl::free(d_grads, q);
  sycl::free(d_params_out, q);
  sycl::free(d_m_out, q);
  sycl::free(d_v_out, q);
  std::free(params);
  std::free(grads);
  std::free(m);
  std::free(v);
  std::free(params_out);
  std::free(m_out);
  std::free(v_out);
  return 0;
}
