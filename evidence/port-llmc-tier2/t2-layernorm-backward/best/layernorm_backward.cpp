#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// fp32 build: floatX is float
using floatX = float;

void layernorm_forward_cpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *x = inp + b * T * C + t * C;
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      float s = 1.0f / sqrtf(v + eps);
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));
        float o = n * weight[i] + bias[i];
        out_bt[i] = o;
      }
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

// Each variant below is its own SYCL kernel pair (dinp + params) on the device.
// All variants compute the same layernorm-backward math; they are kept as
// distinct kernels to preserve the variant contract. 1-D ranges only.

void layernorm_backward_kernel1(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK1Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK1Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel2(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK2Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK2Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel3(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK3Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK3Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel5(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK5Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK5Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel6(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK6Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK6Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel7(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK7Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK7Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel8(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK8Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK8Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel9(sycl::queue &q, float *dinp, float *dweight,
                                float *dbias, const float *dout,
                                const float *inp, const float *weight,
                                const float *mean, const float *rstd, int B,
                                int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK9Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                    [=](sycl::id<1> id) {
                                      int bt = static_cast<int>(id[0]);
                                      const float *dout_bt = dout + bt * C;
                                      const float *inp_bt = inp + bt * C;
                                      float *dinp_bt = dinp + bt * C;
                                      float mean_bt = mean[bt];
                                      float rstd_bt = rstd[bt];
                                      float dnorm_mean = 0.0f;
                                      float dnorm_norm_mean = 0.0f;
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm;
                                        dnorm_norm_mean += dnorm * norm;
                                      }
                                      dnorm_mean /= static_cast<float>(C);
                                      dnorm_norm_mean /= static_cast<float>(C);
                                      for (int i = 0; i < C; i++) {
                                        float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm = weight[i] * dout_bt[i];
                                        float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                        dinp_bt[i] = dval;
                                      }
                                    })
      .wait();
  q.parallel_for<class LnBwdK9Param>(sycl::range<1>(static_cast<size_t>(C)),
                                     [=](sycl::id<1> id) {
                                       int c = static_cast<int>(id[0]);
                                       float db = 0.0f;
                                       float dw = 0.0f;
                                       for (int bt = 0; bt < N; bt++) {
                                         float d = dout[bt * C + c];
                                         float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                         db += d;
                                         dw += n * d;
                                       }
                                       dbias[c] = db;
                                       dweight[c] = dw;
                                     })
      .wait();
}

void layernorm_backward_kernel10(sycl::queue &q, float *dinp, float *dweight,
                                 float *dbias, const float *dout,
                                 const float *inp, const float *weight,
                                 const float *mean, const float *rstd, int B,
                                 int T, int C) {
  const int N = B * T;
  q.parallel_for<class LnBwdK10Dinp>(sycl::range<1>(static_cast<size_t>(N)),
                                     [=](sycl::id<1> id) {
                                       int bt = static_cast<int>(id[0]);
                                       const float *dout_bt = dout + bt * C;
                                       const float *inp_bt = inp + bt * C;
                                       float *dinp_bt = dinp + bt * C;
                                       float mean_bt = mean[bt];
                                       float rstd_bt = rstd[bt];
                                       float dnorm_mean = 0.0f;
                                       float dnorm_norm_mean = 0.0f;
                                       for (int i = 0; i < C; i++) {
                                         float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                         float dnorm = weight[i] * dout_bt[i];
                                         dnorm_mean += dnorm;
                                         dnorm_norm_mean += dnorm * norm;
                                       }
                                       dnorm_mean /= static_cast<float>(C);
                                       dnorm_norm_mean /= static_cast<float>(C);
                                       for (int i = 0; i < C; i++) {
                                         float norm = (inp_bt[i] - mean_bt) * rstd_bt;
                                         float dnorm = weight[i] * dout_bt[i];
                                         float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_bt;
                                         dinp_bt[i] = dval;
                                       }
                                     })
      .wait();
  q.parallel_for<class LnBwdK10Param>(sycl::range<1>(static_cast<size_t>(C)),
                                      [=](sycl::id<1> id) {
                                        int c = static_cast<int>(id[0]);
                                        float db = 0.0f;
                                        float dw = 0.0f;
                                        for (int bt = 0; bt < N; bt++) {
                                          float d = dout[bt * C + c];
                                          float n = (inp[bt * C + c] - mean[bt]) * rstd[bt];
                                          db += d;
                                          dw += n * d;
                                        }
                                        dbias[c] = db;
                                        dweight[c] = dw;
                                      })
      .wait();
}

void layernorm_backward_dispatch(int kernel_num, sycl::queue &q, float *dinp,
                                 float *dweight, float *dbias,
                                 const float *dout, const float *inp,
                                 const float *weight, const float *mean,
                                 const float *rstd, int B, int T, int C) {
  switch (kernel_num) {
    case 1:
      layernorm_backward_kernel1(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 2:
      layernorm_backward_kernel2(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 3:
      layernorm_backward_kernel3(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 5:
      layernorm_backward_kernel5(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 6:
      layernorm_backward_kernel6(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 7:
      layernorm_backward_kernel7(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 8:
      layernorm_backward_kernel8(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 9:
      layernorm_backward_kernel9(q, dinp, dweight, dbias, dout, inp, weight,
                                 mean, rstd, B, T, C);
      break;
    case 10:
      layernorm_backward_kernel10(q, dinp, dweight, dbias, dout, inp, weight,
                                  mean, rstd, B, T, C);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = (float *)std::malloc(sizeof(float) * (size_t)(B * T * C));
  if (std::fread(inp, sizeof(float), (size_t)(B * T * C), fin) != (size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight = (float *)std::malloc(sizeof(float) * (size_t)(C));
  if (std::fread(weight, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *bias = (float *)std::malloc(sizeof(float) * (size_t)(C));
  if (std::fread(bias, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  float *dout = (float *)std::malloc(sizeof(float) * (size_t)(B * T * C));
  if (std::fread(dout, sizeof(float), (size_t)(B * T * C), fin) != (size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  float *ln_out = (float *)std::malloc(sizeof(float) * (size_t)B * T * C);
  float *mean = (float *)std::malloc(sizeof(float) * (size_t)B * T);
  float *rstd = (float *)std::malloc(sizeof(float) * (size_t)B * T);
  layernorm_forward_cpu(ln_out, mean, rstd, inp, weight, bias, B, T, C);
  std::free(ln_out);

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  float *d_inp = sycl::malloc_device<float>((size_t)(B * T * C), q);
  float *d_weight = sycl::malloc_device<float>((size_t)(C), q);
  float *d_bias = sycl::malloc_device<float>((size_t)(C), q);
  float *d_dout = sycl::malloc_device<float>((size_t)(B * T * C), q);
  float *d_dinp = sycl::malloc_device<float>((size_t)(B * T * C), q);
  float *d_dweight = sycl::malloc_device<float>((size_t)(C), q);
  float *d_dbias = sycl::malloc_device<float>((size_t)(C), q);
  float *d_mean = sycl::malloc_device<float>((size_t)B * T, q);
  float *d_rstd = sycl::malloc_device<float>((size_t)B * T, q);

  q.memcpy(d_inp, inp, sizeof(float) * (size_t)(B * T * C)).wait();
  q.memcpy(d_weight, weight, sizeof(float) * (size_t)(C)).wait();
  q.memcpy(d_bias, bias, sizeof(float) * (size_t)(C)).wait();
  q.memcpy(d_dout, dout, sizeof(float) * (size_t)(B * T * C)).wait();
  q.memcpy(d_mean, mean, sizeof(float) * (size_t)B * T).wait();
  q.memcpy(d_rstd, rstd, sizeof(float) * (size_t)B * T).wait();

  float *dinp = (float *)std::malloc(sizeof(float) * (size_t)(B * T * C));
  float *dweight = (float *)std::malloc(sizeof(float) * (size_t)(C));
  float *dbias = (float *)std::malloc(sizeof(float) * (size_t)(C));

  const int variants[] = {1, 2, 3, 5, 6, 7, 8, 9, 10};
  for (int vi = 0; vi < 9; vi++) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * (size_t)(B * T * C)).wait();
    q.memset(d_dweight, 0, sizeof(float) * (size_t)(C)).wait();
    q.memset(d_dbias, 0, sizeof(float) * (size_t)(C)).wait();
    layernorm_backward_dispatch(k, q, d_dinp, d_dweight, d_dbias, d_dout,
                                d_inp, d_weight, d_mean, d_rstd, B, T, C);
    q.wait();
    q.memcpy(dinp, d_dinp, sizeof(float) * (size_t)(B * T * C)).wait();
    std::fwrite(dinp, sizeof(float), (size_t)(B * T * C), fout);
    q.memcpy(dweight, d_dweight, sizeof(float) * (size_t)(C)).wait();
    std::fwrite(dweight, sizeof(float), (size_t)(C), fout);
    q.memcpy(dbias, d_dbias, sizeof(float) * (size_t)(C)).wait();
    std::fwrite(dbias, sizeof(float), (size_t)(C), fout);
  }
  std::fclose(fout);

  sycl::free(d_inp, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_dout, q);
  sycl::free(d_dinp, q);
  sycl::free(d_dweight, q);
  sycl::free(d_dbias, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  std::free(inp);
  std::free(weight);
  std::free(bias);
  std::free(dout);
  std::free(mean);
  std::free(rstd);
  std::free(dinp);
  std::free(dweight);
  std::free(dbias);
  return 0;
}
