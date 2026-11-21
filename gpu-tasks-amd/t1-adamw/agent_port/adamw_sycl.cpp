#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

// Linear interpolation with two flops: weight * end + (1 - weight) * start.
inline float lerp(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamwKernel1Tag;
class AdamwKernel2Tag;

constexpr std::size_t kWorkGroupSize = 256;

static std::size_t ceil_div(std::size_t n, std::size_t d) {
  return (n + d - 1) / d;
}

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m_state, float *v_state, std::size_t n,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (n == 0) return;
  const std::size_t local = kWorkGroupSize;
  const std::size_t global = ceil_div(n, local) * local;
  q.parallel_for<AdamwKernel1Tag>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i >= n) return;
         m_state[i] = beta1 * m_state[i] + (1.0f - beta1) * grads[i];
         v_state[i] =
             beta2 * v_state[i] + (1.0f - beta2) * grads[i] * grads[i];
         const float m_hat = m_state[i] / beta1_correction;
         const float v_hat = v_state[i] / beta2_correction;
         params[i] -= learning_rate *
                      (m_hat / (sycl::sqrt(v_hat) + eps) +
                       weight_decay * params[i]);
       })
      .wait();
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m_state, float *v_state, std::size_t n,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (n == 0) return;
  const std::size_t local = kWorkGroupSize;
  const std::size_t global = ceil_div(n, local) * local;
  q.parallel_for<AdamwKernel2Tag>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i >= n) return;
         const float grad = grads[i];
         float m = m_state[i];
         float v = v_state[i];
         m = lerp(grad, m, beta1);
         m_state[i] = m;
         v = lerp(grad * grad, v, beta2);
         v_state[i] = v;
         m /= beta1_correction;
         v /= beta2_correction;
         params[i] -=
             learning_rate * (m / (sycl::sqrt(v) + eps) + weight_decay * params[i]);
       })
      .wait();
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m_state, float *v_state, int t, std::size_t n,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  const float beta1_correction = 1.0f - std::pow(beta1, t);
  const float beta2_correction = 1.0f - std::pow(beta2, t);
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m_state, v_state, n, learning_rate,
                      beta1, beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m_state, v_state, n, learning_rate,
                      beta1, beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
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
  const std::size_t n = static_cast<std::size_t>(std::atoi(argv[3]));

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *params = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *grads = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *m_init = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *v_init = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!params || !grads || !m_init || !v_init) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(params, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at params\n");
    return 2;
  }
  if (std::fread(grads, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at grads\n");
    return 2;
  }
  if (std::fread(m_init, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at m\n");
    return 2;
  }
  if (std::fread(v_init, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at v\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_grads = sycl::malloc_device<float>(n, q);
  float *d_params_out = sycl::malloc_device<float>(n, q);
  float *d_m_out = sycl::malloc_device<float>(n, q);
  float *d_v_out = sycl::malloc_device<float>(n, q);
  if (!d_grads || !d_params_out || !d_m_out || !d_v_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_grads, grads, sizeof(float) * n).wait();

  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!params_out || !m_out || !v_out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    q.memcpy(d_params_out, params, sizeof(float) * n).wait();
    q.memcpy(d_m_out, m_init, sizeof(float) * n).wait();
    q.memcpy(d_v_out, v_init, sizeof(float) * n).wait();
    adamw(q, k, d_params_out, d_grads, d_m_out, d_v_out, 10, n);
    q.wait();
    q.memcpy(params_out, d_params_out, sizeof(float) * n).wait();
    std::fwrite(params_out, sizeof(float), n, fout);
    q.memcpy(m_out, d_m_out, sizeof(float) * n).wait();
    std::fwrite(m_out, sizeof(float), n, fout);
    q.memcpy(v_out, d_v_out, sizeof(float) * n).wait();
    std::fwrite(v_out, sizeof(float), n, fout);
  }
  std::fclose(fout);

  sycl::free(d_grads, q);
  sycl::free(d_params_out, q);
  sycl::free(d_m_out, q);
  sycl::free(d_v_out, q);
  std::free(params);
  std::free(grads);
  std::free(m_init);
  std::free(v_init);
  std::free(params_out);
  std::free(m_out);
  std::free(v_out);
  return 0;
}
