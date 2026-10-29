#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

// Forward-declared kernel names: one distinct SYCL kernel per variant.
class AdamWKernel1;
class AdamWKernel2;

inline float lerp_device(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float learning_rate,
                     float beta1, float beta2, float beta1_correction,
                     float beta2_correction, float eps, float weight_decay) {
  if (n == 0) return;
  q.parallel_for<AdamWKernel1>(sycl::range<1>(n),
                               [=](sycl::item<1> it) {
                                 std::size_t i = it.get_id(0);
                                 // update the first moment (momentum)
                                 m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
                                 // update the second moment (RMSprop)
                                 v[i] = beta2 * v[i] +
                                        (1.0f - beta2) * grads[i] * grads[i];
                                 float m_hat = m[i] / beta1_correction;
                                 float v_hat = v[i] / beta2_correction;
                                 params[i] -= learning_rate *
                                              (m_hat / (sycl::sqrt(v_hat) + eps) +
                                               weight_decay * params[i]);
                               })
      .wait();
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float learning_rate,
                     float beta1, float beta2, float beta1_correction,
                     float beta2_correction, float eps, float weight_decay) {
  if (n == 0) return;
  q.parallel_for<AdamWKernel2>(sycl::range<1>(n),
                               [=](sycl::item<1> it) {
                                 std::size_t i = it.get_id(0);
                                 float grad = grads[i];
                                 float m_val = m[i];
                                 float v_val = v[i];
                                 // update the first moment (momentum)
                                 m_val = lerp_device(grad, m_val, beta1);
                                 m[i] = m_val;
                                 // update the second moment (RMSprop)
                                 v_val = lerp_device(grad * grad, v_val, beta2);
                                 v[i] = v_val;
                                 m_val /= beta1_correction; // m_hat
                                 v_val /= beta2_correction; // v_hat
                                 params[i] -= learning_rate *
                                              (m_val / (sycl::sqrt(v_val) + eps) +
                                               weight_decay * params[i]);
                               })
      .wait();
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m, float *v, int t, std::size_t n,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  float beta1_correction = 1.0f - powf(beta1, static_cast<float>(t));
  float beta2_correction = 1.0f - powf(beta2, static_cast<float>(t));
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m, v, n, learning_rate, beta1, beta2,
                      beta1_correction, beta2_correction, eps, weight_decay);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m, v, n, learning_rate, beta1, beta2,
                      beta1_correction, beta2_correction, eps, weight_decay);
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
    std::fprintf(stderr, "host allocation failed\n");
    return 2;
  }
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
  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * (N > 0 ? N : 1)));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * (N > 0 ? N : 1)));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * (N > 0 ? N : 1)));
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
