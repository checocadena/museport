// Idiomatic SYCL 2020 port of classifier_fused (fp32).
// Five fused softmax/xent/dlogits variants, each its own SYCL kernel.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstring>

template <typename T>
static inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

struct SoftmaxParams {
  float scale;
  float offset;
};

class FusedK1;
class FusedK2;
class FusedK3;
class FusedK4;
class FusedK5;

static void run_k1(sycl::queue &q, float *dlogits, float *losses,
                   const float *logits, const float *dlosses,
                   const int *targets, int B, int T, int V, int P) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = 256;
  // Over-provision sub-groups so every row is covered for any sub-group size.
  const std::size_t wanted = (std::size_t)N * 32;
  const std::size_t global =
      ceil_div(wanted, local) * local;
  q.parallel_for<FusedK1>(sycl::nd_range<1>(global, local),
                          [=](sycl::nd_item<1> it) {
                            auto sg = it.get_sub_group();
                            const std::size_t lid = sg.get_local_id()[0];
                            const std::size_t sg_sz = sg.get_local_range()[0];
                            const std::size_t sg_in_grp = sg.get_group_id()[0];
                            const std::size_t sgs_per_grp =
                                sg.get_group_range()[0];
                            const std::size_t grp = it.get_group(0);
                            int64_t idx =
                                (int64_t)(grp * sgs_per_grp + sg_in_grp);
                            if (idx >= N) {
                              return;
                            }
                            const float *x = logits + idx * (int64_t)P;
                            float maxval =
                                -std::numeric_limits<float>::infinity();
                            float sumval = 0.0f;
                            for (int i = (int)lid; i < V; i += (int)sg_sz) {
                              float v = x[i];
                              float old = maxval;
                              maxval = sycl::fmax(maxval, v);
                              sumval *= sycl::exp(old - maxval);
                              sumval += sycl::exp(v - maxval);
                            }
                            float gmax = sycl::reduce_over_group(
                                sg, maxval, sycl::maximum<float>{});
                            sumval *= sycl::exp(maxval - gmax);
                            float gsum = sycl::reduce_over_group(
                                sg, sumval, sycl::plus<float>{});
                            float scale = 1.0f / gsum;
                            float offset = gmax;
                            int ix = targets[idx];
                            if (lid == 0) {
                              float prob = sycl::exp(x[ix] - offset) * scale;
                              losses[idx] = -sycl::log(prob);
                            }
                            float dloss = dlosses[idx];
                            float *dl = dlogits + idx * (int64_t)P;
                            for (int i = (int)lid; i < V; i += (int)sg_sz) {
                              float prob = sycl::exp(x[i] - offset) * scale;
                              float ind = (i == ix) ? 1.0f : 0.0f;
                              dl[i] = (prob - ind) * dloss;
                            }
                          })
      .wait();
}

static void run_k2(sycl::queue &q, float *dlogits, float *losses,
                   const float *logits, const float *dlosses,
                   const int *targets, int B, int T, int V, int P) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = 256;
  const std::size_t global = (std::size_t)N * local;
  const int chunks = (V + 3) / 4;
  q.parallel_for<FusedK2>(sycl::nd_range<1>(global, local),
                          [=](sycl::nd_item<1> it) {
                            auto grp = it.get_group();
                            const std::size_t lid = it.get_local_id(0);
                            const std::size_t L = it.get_local_range(0);
                            int64_t idx = (int64_t)it.get_group(0);
                            const float *row = logits + idx * (int64_t)P;
                            const sycl::float4 *row4 =
                                reinterpret_cast<const sycl::float4 *>(row);
                            float tmax =
                                -std::numeric_limits<float>::infinity();
                            float tsum = 0.0f;
                            for (int c = chunks + (int)lid - (int)L; c >= 0;
                                 c -= (int)L) {
                              sycl::float4 v4 = row4[c];
                              const float *s =
                                  reinterpret_cast<const float *>(&v4);
                              for (int k = 0; k < 4; ++k) {
                                int e = c * 4 + k;
                                if (e >= V) {
                                  continue;
                                }
                                float v = s[k];
                                float old = tmax;
                                tmax = sycl::fmax(tmax, v);
                                tsum *= sycl::exp(old - tmax);
                                tsum += sycl::exp(v - tmax);
                              }
                            }
                            float bmax = sycl::reduce_over_group(
                                grp, tmax, sycl::maximum<float>{});
                            tsum *= sycl::exp(tmax - bmax);
                            float bsum = sycl::reduce_over_group(
                                grp, tsum, sycl::plus<float>{});
                            float scale = 1.0f / bsum;
                            float offset = bmax;
                            int ix = targets[idx];
                            if (lid == 0) {
                              float prob =
                                  sycl::exp(row[ix] - offset) * scale;
                              losses[idx] = -sycl::log(prob);
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            float dloss = dlosses[idx];
                            float *dl = dlogits + idx * (int64_t)P;
                            for (int c = (int)lid; c < chunks; c += (int)L) {
                              sycl::float4 v4 = row4[c];
                              const float *s =
                                  reinterpret_cast<const float *>(&v4);
                              for (int k = 0; k < 4; ++k) {
                                int e = c * 4 + k;
                                if (e >= V) {
                                  continue;
                                }
                                float prob =
                                    sycl::exp(s[k] - offset) * scale;
                                float ind = (e == ix) ? 1.0f : 0.0f;
                                dl[e] = (prob - ind) * dloss;
                              }
                            }
                          })
      .wait();
}

static void run_k3(sycl::queue &q, float *dlogits, float *losses,
                   const float *logits, const float *dlosses,
                   const int *targets, int B, int T, int V, int P) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = 256;
  const std::size_t global = (std::size_t)N * local;
  q.parallel_for<FusedK3>(sycl::nd_range<1>(global, local),
                          [=](sycl::nd_item<1> it) {
                            auto grp = it.get_group();
                            const std::size_t lid = it.get_local_id(0);
                            const std::size_t L = it.get_local_range(0);
                            int64_t idx = (int64_t)it.get_group(0);
                            const float *row = logits + idx * (int64_t)P;
                            float tmax =
                                -std::numeric_limits<float>::infinity();
                            float tsum = 0.0f;
                            for (int i = V + (int)lid - (int)L; i >= 0;
                                 i -= (int)L) {
                              float v = row[i];
                              float old = tmax;
                              tmax = sycl::fmax(tmax, v);
                              tsum *= sycl::exp(old - tmax);
                              tsum += sycl::exp(v - tmax);
                            }
                            float bmax = sycl::reduce_over_group(
                                grp, tmax, sycl::maximum<float>{});
                            tsum *= sycl::exp(tmax - bmax);
                            float bsum = sycl::reduce_over_group(
                                grp, tsum, sycl::plus<float>{});
                            float scale = 1.0f / bsum;
                            float offset = bmax;
                            int ix = targets[idx];
                            if (lid == 0) {
                              float prob =
                                  sycl::exp(row[ix] - offset) * scale;
                              losses[idx] = -sycl::log(prob);
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            float dloss = dlosses[idx];
                            float *dl = dlogits + idx * (int64_t)P;
                            for (int i = (int)lid; i < V; i += (int)L) {
                              float prob = sycl::exp(row[i] - offset) * scale;
                              float ind = (i == ix) ? 1.0f : 0.0f;
                              dl[i] = (prob - ind) * dloss;
                            }
                          })
      .wait();
}

static void run_k4(sycl::queue &q, float *dlogits, float *losses,
                   const float *logits, const float *dlosses,
                   const int *targets, int B, int T, int V, int P) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = 256;
  const std::size_t global = (std::size_t)N * local;
  const int vec = 4;
  const int chunks = (V + vec - 1) / vec;
  q.parallel_for<FusedK4>(sycl::nd_range<1>(global, local),
                          [=](sycl::nd_item<1> it) {
                            auto grp = it.get_group();
                            const std::size_t lid = it.get_local_id(0);
                            const std::size_t L = it.get_local_range(0);
                            int64_t idx = (int64_t)it.get_group(0);
                            const float *row = logits + idx * (int64_t)P;
                            const sycl::int4 *rowBits =
                                reinterpret_cast<const sycl::int4 *>(row);
                            float tmax =
                                -std::numeric_limits<float>::infinity();
                            float tsum = 0.0f;
                            for (int c = chunks + (int)lid - (int)L; c >= 0;
                                 c -= (int)L) {
                              sycl::int4 bits = rowBits[c];
                              sycl::float4 v4;
                              std::memcpy(&v4, &bits, sizeof(v4));
                              const float *s =
                                  reinterpret_cast<const float *>(&v4);
                              for (int k = 0; k < 4; ++k) {
                                int e = c * 4 + k;
                                if (e >= V) {
                                  continue;
                                }
                                float v = s[k];
                                float old = tmax;
                                tmax = sycl::fmax(tmax, v);
                                tsum *= sycl::exp(old - tmax);
                                tsum += sycl::exp(v - tmax);
                              }
                            }
                            float bmax = sycl::reduce_over_group(
                                grp, tmax, sycl::maximum<float>{});
                            tsum *= sycl::exp(tmax - bmax);
                            float bsum = sycl::reduce_over_group(
                                grp, tsum, sycl::plus<float>{});
                            float scale = 1.0f / bsum;
                            float offset = bmax;
                            int ix = targets[idx];
                            if (lid == 0) {
                              float prob =
                                  sycl::exp(row[ix] - offset) * scale;
                              losses[idx] = -sycl::log(prob);
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            float dloss = dlosses[idx];
                            float *dl = dlogits + idx * (int64_t)P;
                            sycl::int4 *dlBits =
                                reinterpret_cast<sycl::int4 *>(dl);
                            for (int c = (int)lid; c < chunks; c += (int)L) {
                              sycl::int4 bits = rowBits[c];
                              sycl::float4 vin;
                              std::memcpy(&vin, &bits, sizeof(vin));
                              const float *s =
                                  reinterpret_cast<const float *>(&vin);
                              float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                              for (int k = 0; k < 4; ++k) {
                                int e = c * 4 + k;
                                if (e >= V) {
                                  continue;
                                }
                                float prob =
                                    sycl::exp(s[k] - offset) * scale;
                                float ind = (e == ix) ? 1.0f : 0.0f;
                                out[k] = (prob - ind) * dloss;
                              }
                              sycl::float4 vout(out[0], out[1], out[2], out[3]);
                              // Only commit lanes that belong to V; padding
                              // stays zero from the pre-fill.
                              bool full = (c * 4 + 4 <= V);
                              if (full) {
                                sycl::int4 obits;
                                std::memcpy(&obits, &vout, sizeof(obits));
                                dlBits[c] = obits;
                              } else {
                                for (int k = 0; k < 4; ++k) {
                                  int e = c * 4 + k;
                                  if (e < V) {
                                    dl[e] = out[k];
                                  }
                                }
                              }
                            }
                          })
      .wait();
}

static void run_k5(sycl::queue &q, float *dlogits, float *losses,
                   const float *logits, const float *dlosses,
                   const int *targets, int B, int T, int V, int P) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = 256;
  const std::size_t global = (std::size_t)N * local;
  const int vec = 4;
  const int bulk = V / vec;
  const int tail0 = bulk * vec;
  q.parallel_for<FusedK5>(sycl::nd_range<1>(global, local),
                          [=](sycl::nd_item<1> it) {
                            auto grp = it.get_group();
                            const std::size_t lid = it.get_local_id(0);
                            const std::size_t L = it.get_local_range(0);
                            int64_t idx = (int64_t)it.get_group(0);
                            const float *row = logits + idx * (int64_t)P;
                            const sycl::float4 *row4 =
                                reinterpret_cast<const sycl::float4 *>(row);
                            float tmax =
                                -std::numeric_limits<float>::infinity();
                            float tsum = 0.0f;
                            auto step = [&](float v) {
                              float old = tmax;
                              tmax = sycl::fmax(tmax, v);
                              tsum *= sycl::exp(old - tmax);
                              tsum += sycl::exp(v - tmax);
                            };
                            // Tail (unaligned remainder) first, strided.
                            for (int i = tail0 + (int)lid; i < V;
                                 i += (int)L) {
                              step(row[i]);
                            }
                            // Bulk multiples of vec without bounds checks.
                            int nb = bulk;
                            for (int c = (int)lid; c < nb; c += (int)L) {
                              sycl::float4 v4 = row4[c];
                              const float *s =
                                  reinterpret_cast<const float *>(&v4);
                              step(s[0]);
                              step(s[1]);
                              step(s[2]);
                              step(s[3]);
                            }
                            float bmax = sycl::reduce_over_group(
                                grp, tmax, sycl::maximum<float>{});
                            tsum *= sycl::exp(tmax - bmax);
                            float bsum = sycl::reduce_over_group(
                                grp, tsum, sycl::plus<float>{});
                            float scale = 1.0f / bsum;
                            float offset = bmax;
                            int ix = targets[idx];
                            if (lid == 0) {
                              float prob =
                                  sycl::exp(row[ix] - offset) * scale;
                              losses[idx] = -sycl::log(prob);
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            float dloss = dlosses[idx];
                            float *dl = dlogits + idx * (int64_t)P;
                            for (int c = (int)lid; c < nb; c += (int)L) {
                              sycl::float4 v4 = row4[c];
                              const float *s =
                                  reinterpret_cast<const float *>(&v4);
                              for (int k = 0; k < 4; ++k) {
                                int e = c * 4 + k;
                                float prob =
                                    sycl::exp(s[k] - offset) * scale;
                                float ind = (e == ix) ? 1.0f : 0.0f;
                                dl[e] = (prob - ind) * dloss;
                              }
                            }
                            for (int i = tail0 + (int)lid; i < V;
                                 i += (int)L) {
                              float prob = sycl::exp(row[i] - offset) * scale;
                              float ind = (i == ix) ? 1.0f : 0.0f;
                              dl[i] = (prob - ind) * dloss;
                            }
                          })
      .wait();
}

static void fused_classifier(int k, sycl::queue &q, float *dlogits,
                             float *losses, const float *logits,
                             const float *dlosses, const int *targets, int B,
                             int T, int V, int P) {
  switch (k) {
    case 1:
      run_k1(q, dlogits, losses, logits, dlosses, targets, B, T, V, P);
      break;
    case 2:
      run_k2(q, dlogits, losses, logits, dlosses, targets, B, T, V, P);
      break;
    case 3:
      run_k3(q, dlogits, losses, logits, dlosses, targets, B, T, V, P);
      break;
    case 4:
      run_k4(q, dlogits, losses, logits, dlosses, targets, B, T, V, P);
      break;
    case 5:
      run_k5(q, dlogits, losses, logits, dlosses, targets, B, T, V, P);
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
  const int64_t N = (int64_t)B * T;
  const int64_t NV = N * (int64_t)V;
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> logits((std::size_t)NV);
  std::vector<float> dlosses((std::size_t)N);
  std::vector<int> targets((std::size_t)N);
  std::vector<int> outliers((std::size_t)N * 3);
  if (std::fread(logits.data(), sizeof(float), (std::size_t)NV, fin) !=
      (std::size_t)NV) {
    std::fprintf(stderr, "input too short at logits\n");
    return 2;
  }
  if (std::fread(dlosses.data(), sizeof(float), (std::size_t)N, fin) !=
      (std::size_t)N) {
    std::fprintf(stderr, "input too short at dlosses\n");
    return 2;
  }
  if (std::fread(targets.data(), sizeof(int), (std::size_t)N, fin) !=
      (std::size_t)N) {
    std::fprintf(stderr, "input too short at targets\n");
    return 2;
  }
  if (std::fread(outliers.data(), sizeof(int), (std::size_t)N * 3, fin) !=
      (std::size_t)N * 3) {
    std::fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  std::fclose(fin);
  for (int k = 0; k < 3; ++k) {
    for (int64_t j = 0; j < N; ++j) {
      int c = outliers[(std::size_t)j * 3 + (std::size_t)k];
      logits[(std::size_t)j * (std::size_t)V + (std::size_t)c] *= 20.0f;
    }
  }

  const int P = (V + 63) & ~63;
  const int64_t NP = N * (int64_t)P;
  std::vector<float> h_logits_p((std::size_t)NP, 0.0f);
  for (int64_t r = 0; r < N; ++r) {
    std::memcpy(&h_logits_p[(std::size_t)r * (std::size_t)P],
                &logits[(std::size_t)r * (std::size_t)V],
                (std::size_t)V * sizeof(float));
  }

  sycl::queue q;
  float *d_logits_p = sycl::malloc_device<float>((std::size_t)NP, q);
  float *d_dlogits_p = sycl::malloc_device<float>((std::size_t)NP, q);
  float *d_losses = sycl::malloc_device<float>((std::size_t)N, q);
  float *d_dlosses = sycl::malloc_device<float>((std::size_t)N, q);
  int *d_targets = sycl::malloc_device<int>((std::size_t)N, q);
  if (!d_logits_p || !d_dlogits_p || !d_losses || !d_dlosses || !d_targets) {
    std::fprintf(stderr, "device alloc failed\n");
    return 1;
  }
  q.memcpy(d_logits_p, h_logits_p.data(), (std::size_t)NP * sizeof(float))
      .wait();
  q.memcpy(d_dlosses, dlosses.data(), (std::size_t)N * sizeof(float)).wait();
  q.memcpy(d_targets, targets.data(), (std::size_t)N * sizeof(int)).wait();

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  std::vector<float> losses((std::size_t)N);
  std::vector<float> h_dlogits_p((std::size_t)NP);
  std::vector<float> dlogits((std::size_t)NV);
  const int variants[5] = {1, 2, 3, 4, 5};
  for (int vi = 0; vi < 5; ++vi) {
    int k = variants[vi];
    q.fill(d_losses, 0.0f, (std::size_t)N).wait();
    q.fill(d_dlogits_p, 0.0f, (std::size_t)NP).wait();
    fused_classifier(k, q, d_dlogits_p, d_losses, d_logits_p, d_dlosses,
                     d_targets, B, T, V, P);
    q.memcpy(losses.data(), d_losses, (std::size_t)N * sizeof(float)).wait();
    std::fwrite(losses.data(), sizeof(float), (std::size_t)N, fout);
    q.memcpy(h_dlogits_p.data(), d_dlogits_p,
             (std::size_t)NP * sizeof(float))
        .wait();
    for (int64_t r = 0; r < N; ++r) {
      std::memcpy(&dlogits[(std::size_t)r * (std::size_t)V],
                  &h_dlogits_p[(std::size_t)r * (std::size_t)P],
                  (std::size_t)V * sizeof(float));
    }
    std::fwrite(dlogits.data(), sizeof(float), (std::size_t)NV, fout);
  }
  std::fclose(fout);
  sycl::free(d_logits_p, q);
  sycl::free(d_dlogits_p, q);
  sycl::free(d_losses, q);
  sycl::free(d_dlosses, q);
  sycl::free(d_targets, q);
  return 0;
}
