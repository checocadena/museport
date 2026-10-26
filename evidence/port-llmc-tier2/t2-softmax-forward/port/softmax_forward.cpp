// Idiomatic SYCL 2020 port of llm.c softmax_forward (fp32).
// Eight distinct device kernels, one per variant, 1-D launches only.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstdint>

namespace {

constexpr float kNegInf = std::numeric_limits<float>::infinity() * -1.0f;

inline std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

struct SumMax {
  float maxval;
  float sum;
};

struct SumMaxCombine {
  SumMax operator()(const SumMax &a, const SumMax &b) const {
    const bool a_empty = (a.maxval == kNegInf);
    const bool b_empty = (b.maxval == kNegInf);
    if (a_empty && b_empty) return SumMax{kNegInf, 0.0f};
    if (a_empty) return b;
    if (b_empty) return a;
    const bool a_bigger = (a.maxval > b.maxval);
    const SumMax bigger = a_bigger ? a : b;
    const SumMax smaller = a_bigger ? b : a;
    SumMax r;
    r.maxval = bigger.maxval;
    r.sum = bigger.sum + smaller.sum * sycl::exp(smaller.maxval - bigger.maxval);
    return r;
  }
};

// Distinct kernel names so each variant is its own SYCL kernel.
class SoftmaxK1;
class SoftmaxK2;
class SoftmaxK3;
class SoftmaxK4;
class SoftmaxK5;
class SoftmaxK6;
class SoftmaxK7;
class SoftmaxK8;

constexpr int kWorkGroup = 256;
constexpr int kUnroll = 8;

void run_k1(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  const std::size_t G = ceil_div((std::size_t)N, L);
  q.parallel_for<SoftmaxK1>(sycl::nd_range<1>(G * L, L),
                            [=](sycl::nd_item<1> it) {
                              const std::size_t i = it.get_global_id(0);
                              if (i >= (std::size_t)N) return;
                              const float *row_in = inp + i * (std::size_t)C;
                              float *row_out = out + i * (std::size_t)C;
                              float maxv = kNegInf;
                              for (int j = 0; j < C; ++j)
                                maxv = sycl::fmax(maxv, row_in[j]);
                              double sum = 0.0;
                              for (int j = 0; j < C; ++j) {
                                const float e = sycl::exp(row_in[j] - maxv);
                                row_out[j] = e;
                                sum += (double)e;
                              }
                              const float inv = (float)(1.0 / sum);
                              for (int j = 0; j < C; ++j) row_out[j] *= inv;
                            })
      .wait();
}

// Variant 2: one work-group per row, local-memory tree reduction (no sub-groups).
void run_k2(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> loc(sycl::range<1>(L), h);
     h.parallel_for<SoftmaxK2>(
         sycl::nd_range<1>((std::size_t)N * L, L),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           const std::size_t row = g.get_group_id(0);
           const std::size_t lid = it.get_local_id(0);
           const std::size_t Lsz = it.get_local_range(0);
           const float *x = inp + row * (std::size_t)C;
           float *y = out + row * (std::size_t)C;
           float part = kNegInf;
           for (int i = (int)lid; i < C; i += (int)Lsz)
             part = sycl::fmax(part, x[i]);
           loc[lid] = part;
           for (std::size_t s = Lsz / 2; s >= 1; s /= 2) {
             sycl::group_barrier(g);
             if (lid < s) loc[lid] = sycl::fmax(loc[lid], loc[lid + s]);
             if (s == 1) break;
           }
           sycl::group_barrier(g);
           const float offset = loc[0];
           for (int i = (int)lid; i < C; i += (int)Lsz)
             y[i] = sycl::exp(x[i] - offset);
           sycl::group_barrier(g);
           const float *r = y;
           float psum = 0.0f;
           for (int i = (int)lid; i < C; i += (int)Lsz) psum += r[i];
           loc[lid] = psum;
           for (std::size_t s = Lsz / 2; s >= 1; s /= 2) {
             sycl::group_barrier(g);
             if (lid < s) loc[lid] = loc[lid] + loc[lid + s];
             if (s == 1) break;
           }
           sycl::group_barrier(g);
           const float sum = loc[0];
           for (int i = (int)lid; i < C; i += (int)Lsz) y[i] = r[i] / sum;
         });
   })
      .wait();
}

// Variant 3: one work-group per row, whole-group reduction via group algorithm.
void run_k3(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.parallel_for<SoftmaxK3>(
       sycl::nd_range<1>((std::size_t)N * L, L),
       [=](sycl::nd_item<1> it) {
         sycl::group<1> g = it.get_group();
         const std::size_t row = g.get_group_id(0);
         const std::size_t lid = it.get_local_id(0);
         const std::size_t Lsz = it.get_local_range(0);
         const float *x = inp + row * (std::size_t)C;
         float *y = out + row * (std::size_t)C;
         float part = kNegInf;
         for (int i = (int)lid; i < C; i += (int)Lsz)
           part = sycl::fmax(part, x[i]);
         const float offset =
             sycl::reduce_over_group(g, part, sycl::maximum<float>{});
         for (int i = (int)lid; i < C; i += (int)Lsz)
           y[i] = sycl::exp(x[i] - offset);
         sycl::group_barrier(g);
         float psum = 0.0f;
         for (int i = (int)lid; i < C; i += (int)Lsz) psum += y[i];
         const float sum =
             sycl::reduce_over_group(g, psum, sycl::plus<float>{});
         for (int i = (int)lid; i < C; i += (int)Lsz) y[i] /= sum;
       })
      .wait();
}

// Variant 4: one work-group per row, sub-group reduction + local fan-in.
void run_k4(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> loc(sycl::range<1>(L), h);
     h.parallel_for<SoftmaxK4>(
         sycl::nd_range<1>((std::size_t)N * L, L),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           sycl::sub_group sg = it.get_sub_group();
           const std::size_t row = g.get_group_id(0);
           const std::size_t lid = it.get_local_id(0);
           const std::size_t Lsz = it.get_local_range(0);
           const float *x = inp + row * (std::size_t)C;
           float *y = out + row * (std::size_t)C;
           float part = kNegInf;
           for (int i = (int)lid; i < C; i += (int)Lsz)
             part = sycl::fmax(part, x[i]);
           const float sg_max =
               sycl::reduce_over_group(sg, part, sycl::maximum<float>{});
           if (sg.leader()) loc[sg.get_group_linear_id()] = sg_max;
           sycl::group_barrier(g);
           if (lid == 0) {
             float v = loc[0];
             const std::size_t nsg = sg.get_group_linear_range();
             for (std::size_t i = 1; i < nsg; ++i)
               v = sycl::fmax(v, loc[i]);
             loc[0] = v;
           }
           sycl::group_barrier(g);
           const float offset = loc[0];
           for (int i = (int)lid; i < C; i += (int)Lsz)
             y[i] = sycl::exp(x[i] - offset);
           sycl::group_barrier(g);
           float psum = 0.0f;
           for (int i = (int)lid; i < C; i += (int)Lsz) psum += y[i];
           const float sg_sum =
               sycl::reduce_over_group(sg, psum, sycl::plus<float>{});
           if (sg.leader()) loc[sg.get_group_linear_id()] = sg_sum;
           sycl::group_barrier(g);
           if (lid == 0) {
             float v = loc[0];
             const std::size_t nsg = sg.get_group_linear_range();
             for (std::size_t i = 1; i < nsg; ++i) v += loc[i];
             loc[0] = v;
           }
           sycl::group_barrier(g);
           const float sum = loc[0];
           for (int i = (int)lid; i < C; i += (int)Lsz) y[i] /= sum;
         });
   })
      .wait();
}

// Variant 5: online, one work-item per row.
void run_k5(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  const std::size_t G = ceil_div((std::size_t)N, L);
  q.parallel_for<SoftmaxK5>(sycl::nd_range<1>(G * L, L),
                            [=](sycl::nd_item<1> it) {
                              const std::size_t i = it.get_global_id(0);
                              if (i >= (std::size_t)N) return;
                              const float *row_in = inp + i * (std::size_t)C;
                              float *row_out = out + i * (std::size_t)C;
                              float maxv = kNegInf;
                              double sum = 0.0;
                              for (int j = 0; j < C; ++j) {
                                const float cur = row_in[j];
                                if (cur > maxv) {
                                  const float prev = maxv;
                                  maxv = cur;
                                  sum = sum * (double)sycl::exp(prev - maxv) +
                                        (double)sycl::exp(cur - maxv);
                                } else {
                                  sum += (double)sycl::exp(cur - maxv);
                                }
                              }
                              for (int j = 0; j < C; ++j)
                                row_out[j] =
                                    (float)((double)sycl::exp(row_in[j] - maxv) /
                                            sum);
                            })
      .wait();
}

// Variant 6: online, one work-group per row with joint (max,sum) local reduction.
void run_k6(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<SumMax, 1> loc(sycl::range<1>(L), h);
    h.parallel_for<SoftmaxK6>(
       sycl::nd_range<1>((std::size_t)N * L, L),
       [=](sycl::nd_item<1> it) {
         sycl::group<1> g = it.get_group();
         const std::size_t row = g.get_group_id(0);
         const std::size_t lid = it.get_local_id(0);
         const std::size_t Lsz = it.get_local_range(0);
         const float *x = inp + row * (std::size_t)C;
         float *y = out + row * (std::size_t)C;
         SumMaxCombine combine;
         SumMax part{kNegInf, 0.0f};
         for (int i = (int)lid; i < C; i += (int)Lsz)
           part = combine(part, SumMax{x[i], 1.0f});
         loc[lid] = part;
         for (std::size_t s = Lsz / 2; s >= 1; s /= 2) {
           sycl::group_barrier(g);
           if (lid < s) loc[lid] = combine(loc[lid], loc[lid + s]);
           if (s == 1) break;
         }
         sycl::group_barrier(g);
         const SumMax total = loc[0];
         for (int i = (int)lid; i < C; i += (int)Lsz)
           y[i] = sycl::exp(x[i] - total.maxval) / total.sum;
       });
  })
      .wait();
}

// Variant 7: unrolled large-V version of the sub-group + local pattern.
void run_k7(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> loc(sycl::range<1>(2 * L), h);
     h.parallel_for<SoftmaxK7>(
         sycl::nd_range<1>((std::size_t)N * L, L),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           sycl::sub_group sg = it.get_sub_group();
           const std::size_t row = g.get_group_id(0);
           const std::size_t lid = it.get_local_id(0);
           const std::size_t Lsz = it.get_local_range(0);
           const std::size_t nsg = sg.get_group_linear_range();
           const std::size_t sgid = sg.get_group_linear_id();
           const float *x = inp + row * (std::size_t)C;
           float *y = out + row * (std::size_t)C;
           const int step = (int)(Lsz * (std::size_t)kUnroll);
           float maxv = kNegInf;
           for (int base = (int)lid; base < C; base += step) {
#pragma unroll
             for (int u = 0; u < kUnroll; ++u) {
               const int idx = base + u * (int)Lsz;
               if (idx < C) maxv = sycl::fmax(maxv, x[idx]);
             }
           }
           const float sg_max =
               sycl::reduce_over_group(sg, maxv, sycl::maximum<float>{});
           if (sg.leader()) loc[sgid] = sg_max;
           sycl::group_barrier(g);
           if (lid == 0) {
             float v = loc[0];
             for (std::size_t i = 1; i < nsg; ++i)
               v = sycl::fmax(v, loc[i]);
             loc[0] = v;
           }
           sycl::group_barrier(g);
           const float offset = loc[0];
           float psum = 0.0f;
           for (int base = (int)lid; base < C; base += step) {
             float regs[kUnroll];
#pragma unroll
             for (int u = 0; u < kUnroll; ++u) {
               const int idx = base + u * (int)Lsz;
               regs[u] = (idx < C) ? x[idx] : 0.0f;
             }
#pragma unroll
             for (int u = 0; u < kUnroll; ++u) {
               const int idx = base + u * (int)Lsz;
               if (idx < C) {
                 const float e = sycl::exp(regs[u] - offset);
                 y[idx] = e;
                 psum += e;
               }
             }
           }
           const float sg_sum =
               sycl::reduce_over_group(sg, psum, sycl::plus<float>{});
           if (sg.leader()) loc[L + sgid] = sg_sum;
           sycl::group_barrier(g);
           if (lid == 0) {
             float v = loc[L];
             for (std::size_t i = 1; i < nsg; ++i) v += loc[L + i];
             loc[L] = v;
           }
           sycl::group_barrier(g);
           const float sum = loc[L];
           for (int base = (int)lid; base < C; base += step) {
             float regs[kUnroll];
#pragma unroll
             for (int u = 0; u < kUnroll; ++u) {
               const int idx = base + u * (int)Lsz;
               regs[u] = (idx < C) ? y[idx] : 0.0f;
             }
#pragma unroll
             for (int u = 0; u < kUnroll; ++u) {
               const int idx = base + u * (int)Lsz;
               if (idx < C) y[idx] = regs[u] / sum;
             }
           }
         });
   })
      .wait();
}

// Variant 8: online, one sub-group per row (portable sub-group size).
// Sub-group joint reduction is done with local memory so no custom
// reduce_over_group is needed; rows are mapped as group_id*nsg+sgid.
void run_k8(sycl::queue &q, float *out, const float *inp, int N, int C) {
  if (N <= 0 || C <= 0) return;
  const std::size_t L = kWorkGroup;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<SumMax, 1> parts(sycl::range<1>(L), h);
    sycl::local_accessor<int, 1> owners(sycl::range<1>(L), h);
    sycl::local_accessor<SumMax, 1> results(sycl::range<1>(L), h);
    h.parallel_for<SoftmaxK8>(
       sycl::nd_range<1>((std::size_t)N * L, L),
       [=](sycl::nd_item<1> it) {
         sycl::group<1> g = it.get_group();
         sycl::sub_group sg = it.get_sub_group();
         const std::size_t lid = it.get_local_id(0);
         const std::size_t Lsz = it.get_local_range(0);
         const std::size_t nsg = sg.get_group_linear_range();
         const std::size_t sgid = sg.get_group_linear_id();
         const std::size_t row = g.get_group_id(0) * nsg + sgid;
         const std::size_t lane = sg.get_local_linear_id();
         const std::size_t S = sg.get_local_linear_range();
         SumMaxCombine combine;
         SumMax part{kNegInf, 0.0f};
         if (row < (std::size_t)N) {
           const float *x = inp + row * (std::size_t)C;
           float maxv = kNegInf;
           float sumv = 0.0f;
           for (int i = (int)lane; i < C; i += (int)S) {
             const float bigger = sycl::fmax(maxv, x[i]);
             sumv = sumv * sycl::exp(maxv - bigger) + sycl::exp(x[i] - bigger);
             maxv = bigger;
           }
           part = SumMax{maxv, sumv};
         }
         parts[lid] = part;
         owners[lid] = (row < (std::size_t)N) ? (int)sgid : -1;
         sycl::group_barrier(g);
         if (lane == 0 && row < (std::size_t)N + nsg) {
           // One leader per present sub-group folds its members by scanning.
           // Only leaders of rows in range write; guard by checking any owner.
           SumMax total{kNegInf, 0.0f};
           bool present = false;
           for (std::size_t l = 0; l < Lsz; ++l) {
             if (owners[l] == (int)sgid) {
               present = true;
               total = combine(total, parts[l]);
             }
           }
           if (present) results[sgid] = total;
         }
         sycl::group_barrier(g);
         if (row >= (std::size_t)N) return;
         const SumMax total = results[sgid];
         const float *x = inp + row * (std::size_t)C;
         float *y = out + row * (std::size_t)C;
         for (int i = (int)lane; i < C; i += (int)S)
           y[i] = sycl::exp(x[i] - total.maxval) / total.sum;
       });
  })
      .wait();
}

void dispatch(sycl::queue &q, int k, float *out, const float *inp, int N,
              int C) {
  switch (k) {
    case 1: run_k1(q, out, inp, N, C); break;
    case 2: run_k2(q, out, inp, N, C); break;
    case 3: run_k3(q, out, inp, N, C); break;
    case 4: run_k4(q, out, inp, N, C); break;
    case 5: run_k5(q, out, inp, N, C); break;
    case 6: run_k6(q, out, inp, N, C); break;
    case 7: run_k7(q, out, inp, N, C); break;
    case 8: run_k8(q, out, inp, N, C); break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int V = std::atoi(argv[5]);
  const std::size_t N = (std::size_t)B * (std::size_t)T;
  const std::size_t Vs = (std::size_t)V;

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = (float *)std::malloc(sizeof(float) * N * Vs);
  if (!inp) return 2;
  if (std::fread(inp, sizeof(float), N * Vs, fin) != N * Vs) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  int *outliers = (int *)std::malloc(sizeof(int) * N * 3);
  if (!outliers) return 2;
  if (std::fread(outliers, sizeof(int), N * 3, fin) != N * 3) {
    std::fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  std::fclose(fin);

  for (int k = 0; k < 3; ++k)
    for (std::size_t j = 0; j < N; ++j) inp[j * Vs + (std::size_t)outliers[j * 3 + k]] *= 20.0f;

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(N * Vs, q);
  float *d_out = sycl::malloc_device<float>(N * Vs, q);
  if (!d_inp || !d_out) {
    std::fprintf(stderr, "device alloc failed\n");
    return 2;
  }
  q.memcpy(d_inp, inp, sizeof(float) * N * Vs).wait();

  float *out = (float *)std::malloc(sizeof(float) * N * Vs);
  if (!out) return 2;

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  const int variants[] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int vi = 0; vi < 8; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * N * Vs).wait();
    dispatch(q, k, d_out, d_inp, (int)N, V);
    q.memcpy(out, d_out, sizeof(float) * N * Vs).wait();
    std::fwrite(out, sizeof(float), N * Vs, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(outliers);
  std::free(out);
  return 0;
}
