#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

template <typename T>
static inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

void layernorm_forward_cpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *x = inp + (size_t)b * T * C + (size_t)t * C;
      float m = 0.0f;
      for (int i = 0; i < C; i++) m += x[i];
      m = m / C;
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xs = x[i] - m;
        v += xs * xs;
      }
      v = v / C;
      float s = 1.0f / std::sqrt(v + eps);
      float *out_bt = out + (size_t)b * T * C + (size_t)t * C;
      for (int i = 0; i < C; i++) {
        float n = s * (x[i] - m);
        out_bt[i] = n * weight[i] + bias[i];
      }
      mean[(size_t)b * T + t] = m;
      rstd[(size_t)b * T + t] = s;
    }
  }
}

using atomic_float_ref =
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

class LnBwdK1;
static void run_layernorm_backward1(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, const float *dout,
                                    const float *inp, const float *weight,
                                    const float *mean, const float *rstd, int B,
                                    int T, int C) {
  const int N = B * T;
  q.parallel_for<LnBwdK1>(sycl::range<1>(N), [=](sycl::id<1> id) {
     int n = (int)id[0];
     int b = n / T;
     int t = n % T;
     const float *dout_bt = dout + (size_t)n * (size_t)C;
     const float *inp_bt = inp + (size_t)n * (size_t)C;
     float *dinp_bt = dinp + (size_t)n * (size_t)C;
     (void)b;
     (void)t;
     float mean_bt = mean[n];
     float rstd_bt = rstd[n];
     float dnorm_mean = 0.0f;
     float dnorm_norm_mean = 0.0f;
     for (int i = 0; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       dnorm_mean += dnorm;
       dnorm_norm_mean += dnorm * norm;
     }
     dnorm_mean /= (float)C;
     dnorm_norm_mean /= (float)C;
     for (int i = 0; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       float dout_i = dout_bt[i];
       atomic_float_ref(dbias[i]).fetch_add(dout_i);
       atomic_float_ref(dweight[i]).fetch_add(norm * dout_i);
       float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
       dinp_bt[i] = dval;
     }
   }).wait();
}

class LnBwdK2;
static void run_layernorm_backward2(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, const float *dout,
                                    const float *inp, const float *weight,
                                    const float *mean, const float *rstd, int B,
                                    int T, int C) {
  const int N = B * T;
  const int wg = 256;
  const int groups = ceil_div(N, wg);
  q.parallel_for<LnBwdK2>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int n = (int)it.get_global_id(0);
         if (n >= N) return;
         const float *dout_bt = dout + (size_t)n * (size_t)C;
         const float *inp_bt = inp + (size_t)n * (size_t)C;
         float *dinp_bt = dinp + (size_t)n * (size_t)C;
         float mean_bt = mean[n];
         float rstd_bt = rstd[n];
         float dnorm_mean = 0.0f;
         float dnorm_norm_mean = 0.0f;
         for (int i = 0; i < C; i++) {
           float norm = (inp_bt[i] - mean_bt) * rstd_bt;
           float dnorm = weight[i] * dout_bt[i];
           dnorm_mean += dnorm;
           dnorm_norm_mean += dnorm * norm;
         }
         dnorm_mean /= (float)C;
         dnorm_norm_mean /= (float)C;
         for (int i = 0; i < C; i++) {
           float norm = (inp_bt[i] - mean_bt) * rstd_bt;
           float dnorm = weight[i] * dout_bt[i];
           atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
           atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
           float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
           dinp_bt[i] = dval;
         }
       }).wait();
}

class LnBwdK3;
static void run_layernorm_backward3(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, const float *dout,
                                    const float *inp, const float *weight,
                                    const float *mean, const float *rstd, int B,
                                    int T, int C) {
  const int N = B * T;
  const int wg = 128;
  const int groups = 32;
  q.parallel_for<LnBwdK3>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int gid = (int)it.get_global_id(0);
         int gsize = (int)it.get_global_range(0);
         for (int n = gid; n < N; n += gsize) {
           const float *dout_bt = dout + (size_t)n * (size_t)C;
           const float *inp_bt = inp + (size_t)n * (size_t)C;
           float *dinp_bt = dinp + (size_t)n * (size_t)C;
           float mean_bt = mean[n];
           float rstd_bt = rstd[n];
           float dnorm_mean = 0.0f;
           float dnorm_norm_mean = 0.0f;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             dnorm_mean += dnorm;
             dnorm_norm_mean += dnorm * norm;
           }
           dnorm_mean /= (float)C;
           dnorm_norm_mean /= (float)C;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
             atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
             float dval =
                 (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
             dinp_bt[i] = dval;
           }
         }
       }).wait();
}

class LnBwdK5;
static void run_layernorm_backward5(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, float *scratch,
                                    const float *dout, const float *inp,
                                    const float *weight, const float *mean,
                                    const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  const int wg = 256;
  const int groups = 16;
  q.parallel_for<LnBwdK5>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int gid = (int)it.get_global_id(0);
         int gsize = (int)it.get_global_range(0);
         for (int n = gid; n < N; n += gsize) {
           const float *dout_bt = dout + (size_t)n * (size_t)C;
           const float *inp_bt = inp + (size_t)n * (size_t)C;
           float *dinp_bt = dinp + (size_t)n * (size_t)C;
           float mean_bt = mean[n];
           float rstd_bt = rstd[n];
           float s1 = 0.0f, s2 = 0.0f;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             s1 += dnorm;
             s2 += dnorm * norm;
           }
           s1 /= (float)C;
           s2 /= (float)C;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
             atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
             dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
           }
         }
       }).wait();
}

class LnBwdK6;
static void run_layernorm_backward6(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, float *scratch,
                                    const float *dout, const float *inp,
                                    const float *weight, const float *mean,
                                    const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  const int wg = 256;
  const int groups = 8;
  q.parallel_for<LnBwdK6>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int gid = (int)it.get_global_id(0);
         int gsize = (int)it.get_global_range(0);
         for (int n = gid; n < N; n += gsize) {
           const float *dout_bt = dout + (size_t)n * (size_t)C;
           const float *inp_bt = inp + (size_t)n * (size_t)C;
           float *dinp_bt = dinp + (size_t)n * (size_t)C;
           float mean_bt = mean[n];
           float rstd_bt = rstd[n];
           float s1 = 0.0f, s2 = 0.0f;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             s1 += dnorm;
             s2 += dnorm * norm;
           }
           s1 /= (float)C;
           s2 /= (float)C;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
             atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
             dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
           }
         }
       }).wait();
}

class LnBwdK7;
static void run_layernorm_backward7(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, float *scratch,
                                    const float *dout, const float *inp,
                                    const float *weight, const float *mean,
                                    const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  q.parallel_for<LnBwdK7>(sycl::range<1>(N), [=](sycl::id<1> id) {
     int n = (int)id[0];
     const float *dout_bt = dout + (size_t)n * (size_t)C;
     const float *inp_bt = inp + (size_t)n * (size_t)C;
     float *dinp_bt = dinp + (size_t)n * (size_t)C;
     float mean_bt = mean[n];
     float rstd_bt = rstd[n];
     float s1 = 0.0f, s2 = 0.0f;
     int C4 = (C / 4) * 4;
     for (int i = 0; i < C4; i += 4) {
       sycl::vec<float, 4> iv(*(const sycl::vec<float, 4> *)(inp_bt + i));
       sycl::vec<float, 4> dv(*(const sycl::vec<float, 4> *)(dout_bt + i));
       sycl::vec<float, 4> wv(*(const sycl::vec<float, 4> *)(weight + i));
       for (int k = 0; k < 4; k++) {
         float norm = (iv[k] - mean_bt) * rstd_bt;
         float dnorm = wv[k] * dv[k];
         s1 += dnorm;
         s2 += dnorm * norm;
       }
     }
     for (int i = C4; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       s1 += dnorm;
       s2 += dnorm * norm;
     }
     s1 /= (float)C;
     s2 /= (float)C;
     for (int i = 0; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
       atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
       dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
     }
   }).wait();
}

class LnBwdK8;
static void run_layernorm_backward8(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, float *scratch,
                                    const float *dout, const float *inp,
                                    const float *weight, const float *mean,
                                    const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  const int wg = 128;
  const int groups = 16;
  q.parallel_for<LnBwdK8>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int gid = (int)it.get_global_id(0);
         int gsize = (int)it.get_global_range(0);
         for (int n = gid; n < N; n += gsize) {
           const float *dout_bt = dout + (size_t)n * (size_t)C;
           const float *inp_bt = inp + (size_t)n * (size_t)C;
           float *dinp_bt = dinp + (size_t)n * (size_t)C;
           float mean_bt = mean[n];
           float rstd_bt = rstd[n];
           float s1 = 0.0f, s2 = 0.0f;
           int C4 = (C / 4) * 4;
           for (int i = 0; i < C4; i += 4) {
             sycl::vec<float, 4> iv(
                 *(const sycl::vec<float, 4> *)(inp_bt + i));
             sycl::vec<float, 4> dv(
                 *(const sycl::vec<float, 4> *)(dout_bt + i));
             sycl::vec<float, 4> wv(*(const sycl::vec<float, 4> *)(weight + i));
             for (int k = 0; k < 4; k++) {
               float norm = (iv[k] - mean_bt) * rstd_bt;
               float dnorm = wv[k] * dv[k];
               s1 += dnorm;
               s2 += dnorm * norm;
             }
           }
           for (int i = C4; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             s1 += dnorm;
             s2 += dnorm * norm;
           }
           s1 /= (float)C;
           s2 /= (float)C;
           for (int i = 0; i < C; i++) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
             atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
             dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
           }
         }
       }).wait();
}

class LnBwdK9;
static void run_layernorm_backward9(sycl::queue &q, float *dinp, float *dweight,
                                    float *dbias, float *scratch,
                                    const float *dout, const float *inp,
                                    const float *weight, const float *mean,
                                    const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  q.parallel_for<LnBwdK9>(sycl::range<1>(N), [=](sycl::id<1> id) {
     int n = (int)id[0];
     const float *dout_bt = dout + (size_t)n * (size_t)C;
     const float *inp_bt = inp + (size_t)n * (size_t)C;
     float *dinp_bt = dinp + (size_t)n * (size_t)C;
     float mean_bt = mean[n];
     float rstd_bt = rstd[n];
     float s1a = 0.0f, s2a = 0.0f, s1b = 0.0f, s2b = 0.0f;
     int half = C / 2;
     for (int i = 0; i < half; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       s1a += dnorm;
       s2a += dnorm * norm;
     }
     for (int i = half; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       s1b += dnorm;
       s2b += dnorm * norm;
     }
     float s1 = (s1a + s1b) / (float)C;
     float s2 = (s2a + s2b) / (float)C;
     for (int i = 0; i < C; i++) {
       float norm = (inp_bt[i] - mean_bt) * rstd_bt;
       float dnorm = weight[i] * dout_bt[i];
       atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
       atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
       dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
     }
   }).wait();
}

class LnBwdK10;
static void run_layernorm_backward10(sycl::queue &q, float *dinp, float *dweight,
                                     float *dbias, float *scratch,
                                     const float *dout, const float *inp,
                                     const float *weight, const float *mean,
                                     const float *rstd, int B, int T, int C) {
  (void)scratch;
  const int N = B * T;
  const int wg = 64;
  const int groups = 32;
  q.parallel_for<LnBwdK10>(
       sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
       [=](sycl::nd_item<1> it) {
         int gid = (int)it.get_global_id(0);
         int gsize = (int)it.get_global_range(0);
         for (int n = gid; n < N; n += gsize) {
           const float *dout_bt = dout + (size_t)n * (size_t)C;
           const float *inp_bt = inp + (size_t)n * (size_t)C;
           float *dinp_bt = dinp + (size_t)n * (size_t)C;
           float mean_bt = mean[n];
           float rstd_bt = rstd[n];
           float s1 = 0.0f, s2 = 0.0f;
           for (int i = C - 1; i >= 0; i--) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             s1 += dnorm;
             s2 += dnorm * norm;
           }
           s1 /= (float)C;
           s2 /= (float)C;
           for (int i = C - 1; i >= 0; i--) {
             float norm = (inp_bt[i] - mean_bt) * rstd_bt;
             float dnorm = weight[i] * dout_bt[i];
             atomic_float_ref(dbias[i]).fetch_add(dout_bt[i]);
             atomic_float_ref(dweight[i]).fetch_add(norm * dout_bt[i]);
             dinp_bt[i] = (dnorm - s1 - norm * s2) * rstd_bt;
           }
         }
       }).wait();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = atoi(argv[3]);
  const int T = atoi(argv[4]);
  const int C = atoi(argv[5]);
  FILE *fin = fopen(argv[1], "rb");
  if (!fin) {
    perror(argv[1]);
    return 2;
  }
  size_t nBTC = (size_t)B * (size_t)T * (size_t)C;
  size_t nBT = (size_t)B * (size_t)T;
  float *inp = (float *)malloc(nBTC * sizeof(float));
  float *weight = (float *)malloc((size_t)C * sizeof(float));
  float *bias = (float *)malloc((size_t)C * sizeof(float));
  float *dout = (float *)malloc(nBTC * sizeof(float));
  if (fread(inp, sizeof(float), nBTC, fin) != nBTC) {
    fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (fread(weight, sizeof(float), (size_t)C, fin) != (size_t)C) {
    fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (fread(bias, sizeof(float), (size_t)C, fin) != (size_t)C) {
    fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  if (fread(dout, sizeof(float), nBTC, fin) != nBTC) {
    fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  fclose(fin);
  FILE *fout = fopen(argv[2], "wb");
  if (!fout) {
    perror(argv[2]);
    return 2;
  }
  float *ln_out = (float *)malloc(nBTC * sizeof(float));
  float *mean = (float *)malloc(nBT * sizeof(float));
  float *rstd = (float *)malloc(nBT * sizeof(float));
  layernorm_forward_cpu(ln_out, mean, rstd, inp, weight, bias, B, T, C);
  free(ln_out);

  sycl::queue q{sycl::default_selector_v};
  float *d_inp = sycl::malloc_device<float>(nBTC, q);
  float *d_weight = sycl::malloc_device<float>((size_t)C, q);
  float *d_bias = sycl::malloc_device<float>((size_t)C, q);
  float *d_dout = sycl::malloc_device<float>(nBTC, q);
  float *d_dinp = sycl::malloc_device<float>(nBTC, q);
  float *d_dweight = sycl::malloc_device<float>((size_t)C, q);
  float *d_dbias = sycl::malloc_device<float>((size_t)C, q);
  float *d_mean = sycl::malloc_device<float>(nBT, q);
  float *d_rstd = sycl::malloc_device<float>(nBT, q);
  size_t scratch_n = (size_t)32 * (size_t)32 * (size_t)(2 * C + 1);
  if (scratch_n < (size_t)(2 * C + 32)) scratch_n = (size_t)(2 * C + 32);
  float *d_scratch = sycl::malloc_device<float>(scratch_n, q);

  q.memcpy(d_inp, inp, nBTC * sizeof(float)).wait();
  q.memcpy(d_weight, weight, (size_t)C * sizeof(float)).wait();
  q.memcpy(d_bias, bias, (size_t)C * sizeof(float)).wait();
  q.memcpy(d_dout, dout, nBTC * sizeof(float)).wait();
  q.memcpy(d_mean, mean, nBT * sizeof(float)).wait();
  q.memcpy(d_rstd, rstd, nBT * sizeof(float)).wait();

  float *dinp = (float *)malloc(nBTC * sizeof(float));
  float *dweight = (float *)malloc((size_t)C * sizeof(float));
  float *dbias = (float *)malloc((size_t)C * sizeof(float));

  const int variants[] = {1, 2, 3, 5, 6, 7, 8, 9, 10};
  for (int vi = 0; vi < 9; vi++) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, nBTC * sizeof(float)).wait();
    q.memset(d_dweight, 0, (size_t)C * sizeof(float)).wait();
    q.memset(d_dbias, 0, (size_t)C * sizeof(float)).wait();
    q.memset(d_scratch, 0, scratch_n * sizeof(float)).wait();
    switch (k) {
      case 1:
        run_layernorm_backward1(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp,
                                d_weight, d_mean, d_rstd, B, T, C);
        break;
      case 2:
        run_layernorm_backward2(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp,
                                d_weight, d_mean, d_rstd, B, T, C);
        break;
      case 3:
        run_layernorm_backward3(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp,
                                d_weight, d_mean, d_rstd, B, T, C);
        break;
      case 5:
        run_layernorm_backward5(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                C);
        break;
      case 6:
        run_layernorm_backward6(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                C);
        break;
      case 7:
        run_layernorm_backward7(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                C);
        break;
      case 8:
        run_layernorm_backward8(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                C);
        break;
      case 9:
        run_layernorm_backward9(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                C);
        break;
      case 10:
        run_layernorm_backward10(q, d_dinp, d_dweight, d_dbias, d_scratch,
                                 d_dout, d_inp, d_weight, d_mean, d_rstd, B, T,
                                 C);
        break;
      default:
        fprintf(stderr, "Invalid kernel number\n");
        return 1;
    }
    q.memcpy(dinp, d_dinp, nBTC * sizeof(float)).wait();
    fwrite(dinp, sizeof(float), nBTC, fout);
    q.memcpy(dweight, d_dweight, (size_t)C * sizeof(float)).wait();
    fwrite(dweight, sizeof(float), (size_t)C, fout);
    q.memcpy(dbias, d_dbias, (size_t)C * sizeof(float)).wait();
    fwrite(dbias, sizeof(float), (size_t)C, fout);
  }
  fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_dout, q);
  sycl::free(d_dinp, q);
  sycl::free(d_dweight, q);
  sycl::free(d_dbias, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  sycl::free(d_scratch, q);
  free(inp);
  free(weight);
  free(bias);
  free(dout);
  free(mean);
  free(rstd);
  free(dinp);
  free(dweight);
  free(dbias);
  return 0;
}
