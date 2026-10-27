#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

class AdamWKernel1;
class AdamWKernel2;

static void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                            float *m, float *v, std::size_t n, float lr,
                            float beta1, float beta2, float b1_corr,
                            float b2_corr, float eps, float wd) {
  if (n == 0) return;
  constexpr std::size_t block_size = 512;
  std::size_t num_blocks = (n + block_size - 1) / block_size;
  sycl::range<1> local(block_size);
  sycl::range<1> global(num_blocks * block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<AdamWKernel1>(
        sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
          std::size_t i = it.get_global_id(0);
          if (i >= n) return;
          float mi = m[i];
          float vi = v[i];
          float g = grads[i];
          float p = params[i];
          mi = beta1 * mi + (1.0f - beta1) * g;
          m[i] = mi;
          vi = beta2 * vi + (1.0f - beta2) * g * g;
          v[i] = vi;
          float m_hat = mi / b1_corr;
          float v_hat = vi / b2_corr;
          p -= lr * (m_hat / (sycl::sqrt(v_hat) + eps) + wd * p);
          params[i] = p;
        });
  });
  q.wait();
}

static void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                            float *m, float *v, std::size_t n, float lr,
                            float beta1, float beta2, float b1_corr,
                            float b2_corr, float eps, float wd) {
  if (n == 0) return;
  constexpr std::size_t block_size = 512;
  std::size_t num_blocks = (n + block_size - 1) / block_size;
  sycl::range<1> local(block_size);
  sycl::range<1> global(num_blocks * block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<AdamWKernel2>(
        sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
          std::size_t i = it.get_global_id(0);
          if (i >= n) return;
          float g = grads[i];
          // lerp(grad, m, beta1) == fma(beta1, m, fma(-beta1, grad, grad))
          float mi = sycl::fma(beta1, m[i], sycl::fma(-beta1, g, g));
          m[i] = mi;
          float g2 = g * g;
          float vi = sycl::fma(beta2, v[i], sycl::fma(-beta2, g2, g2));
          v[i] = vi;
          float m_hat = mi / b1_corr;
          float v_hat = vi / b2_corr;
          float p = params[i];
          p -= lr * (m_hat / (sycl::sqrt(v_hat) + eps) + wd * p);
          params[i] = p;
        });
  });
  q.wait();
}

static void adamw(sycl::queue &q, int kernel_num, float *params,
                  const float *grads, float *m, float *v, int t,
                  std::size_t n, float learning_rate = 1e-3f,
                  float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                  float weight_decay = 0.0f) {
  float b1_corr = 1.0f - std::pow(beta1, (float)t);
  float b2_corr = 1.0f - std::pow(beta2, (float)t);
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
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
    return 2;
  }
  const std::size_t N = (std::size_t)std::atol(argv[3]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *params = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  float *grads = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  float *m = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  float *v = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  if (!params || !grads || !m || !v) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (N > 0) {
    if (std::fread(params, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at params\n");
      return 2;
    }
    if (std::fread(grads, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at grads\n");
      return 2;
    }
    if (std::fread(m, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at m\n");
      return 2;
    }
    if (std::fread(v, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at v\n");
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

  float *d_grads = sycl::malloc_device<float>(N > 0 ? N : 1, q);
  float *d_params_out = sycl::malloc_device<float>(N > 0 ? N : 1, q);
  float *d_m_out = sycl::malloc_device<float>(N > 0 ? N : 1, q);
  float *d_v_out = sycl::malloc_device<float>(N > 0 ? N : 1, q);
  if (!d_grads || !d_params_out || !d_m_out || !d_v_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }

  float *params_out = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  float *m_out = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));
  float *v_out = (float *)std::malloc(sizeof(float) * (N > 0 ? N : 1));

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
