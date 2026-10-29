#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

inline float lerp_fma(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamWKernel1;
class AdamWKernel2;

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m_mem, float *v_mem, std::size_t n,
                     float lr, float beta1, float beta2, float b1_corr,
                     float b2_corr, float eps, float wd) {
  if (n == 0)
    return;
  q.parallel_for<AdamWKernel1>(sycl::range<1>(n),
                               [=](sycl::item<1> it) {
                                 std::size_t i = it.get_id(0);
                                 float g = grads[i];
                                 float m = m_mem[i];
                                 float v = v_mem[i];
                                 m = beta1 * m + (1.0f - beta1) * g;
                                 m_mem[i] = m;
                                 v = beta2 * v + (1.0f - beta2) * g * g;
                                 v_mem[i] = v;
                                 float m_hat = m / b1_corr;
                                 float v_hat = v / b2_corr;
                                 float p = params[i];
                                 p -= lr * (m_hat / (sycl::sqrt(v_hat) + eps) +
                                            wd * p);
                                 params[i] = p;
                               })
      .wait();
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m_mem, float *v_mem, std::size_t n,
                     float lr, float beta1, float beta2, float b1_corr,
                     float b2_corr, float eps, float wd) {
  if (n == 0)
    return;
  q.parallel_for<AdamWKernel2>(sycl::range<1>(n),
                               [=](sycl::item<1> it) {
                                 std::size_t i = it.get_id(0);
                                 float grad = grads[i];
                                 float m = m_mem[i];
                                 float v = v_mem[i];
                                 m = lerp_fma(grad, m, beta1);
                                 m_mem[i] = m;
                                 v = lerp_fma(grad * grad, v, beta2);
                                 v_mem[i] = v;
                                 m /= b1_corr;
                                 v /= b2_corr;
                                 float p = params[i];
                                 p -= lr * (m / (sycl::sqrt(v) + eps) + wd * p);
                                 params[i] = p;
                               })
      .wait();
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m_mem, float *v_mem, int t, std::size_t n,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  float b1_corr = 1.0f - powf(beta1, (float)t);
  float b2_corr = 1.0f - powf(beta2, (float)t);
  switch (kernel_num) {
  case 1:
    adamw_dispatch1(q, params, grads, m_mem, v_mem, n, learning_rate, beta1,
                    beta2, b1_corr, b2_corr, eps, weight_decay);
    break;
  case 2:
    adamw_dispatch2(q, params, grads, m_mem, v_mem, n, learning_rate, beta1,
                    beta2, b1_corr, b2_corr, eps, weight_decay);
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
  std::size_t N = static_cast<std::size_t>(std::atol(argv[3]));

  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> params(N), grads(N), m(N), v(N);
  auto read_buf = [&](std::vector<float> &b, const char *what) {
    if (N > 0 &&
        std::fread(b.data(), sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at %s\n", what);
      std::exit(2);
    }
  };
  read_buf(params, "params");
  read_buf(grads, "grads");
  read_buf(m, "m");
  read_buf(v, "v");
  std::fclose(fin);

  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::property::queue::in_order{}};

  float *d_grads = sycl::malloc_device<float>(N, q);
  float *d_params_out = sycl::malloc_device<float>(N, q);
  float *d_m_out = sycl::malloc_device<float>(N, q);
  float *d_v_out = sycl::malloc_device<float>(N, q);
  if (N > 0 && (!d_grads || !d_params_out || !d_m_out || !d_v_out)) {
    std::fprintf(stderr, "device allocation failed\n");
    return 1;
  }
  std::vector<float> params_out(N), m_out(N), v_out(N);

  if (N > 0)
    q.memcpy(d_grads, grads.data(), sizeof(float) * N).wait();

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    int k = variants[vi];
    if (N > 0) {
      q.memcpy(d_params_out, params.data(), sizeof(float) * N).wait();
      q.memcpy(d_m_out, m.data(), sizeof(float) * N).wait();
      q.memcpy(d_v_out, v.data(), sizeof(float) * N).wait();
    }
    adamw(q, k, d_params_out, d_grads, d_m_out, d_v_out, 10, N);
    q.wait();
    if (N > 0) {
      q.memcpy(params_out.data(), d_params_out, sizeof(float) * N).wait();
      std::fwrite(params_out.data(), sizeof(float), N, fout);
      q.memcpy(m_out.data(), d_m_out, sizeof(float) * N).wait();
      std::fwrite(m_out.data(), sizeof(float), N, fout);
      q.memcpy(v_out.data(), d_v_out, sizeof(float) * N).wait();
      std::fwrite(v_out.data(), sizeof(float), N, fout);
    }
  }

  sycl::free(d_grads, q);
  sycl::free(d_params_out, q);
  sycl::free(d_m_out, q);
  sycl::free(d_v_out, q);
  std::fclose(fout);
  return 0;
}
