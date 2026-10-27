#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

static constexpr float kNegInf() {
  return -std::numeric_limits<float>::infinity();
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct SumMax {
  float maxval;
  float sum;
};

inline SumMax combine_sum_max(SumMax a, SumMax b, float neg_inf) {
  // identity handling so empty partials (no elements) don't poison the sum
  bool a_empty = (a.sum == 0.0f && a.maxval == neg_inf);
  bool b_empty = (b.sum == 0.0f && b.maxval == neg_inf);
  if (a_empty) return b;
  if (b_empty) return a;
  bool a_bigger = (a.maxval > b.maxval);
  SumMax bigger = a_bigger ? a : b;
  SumMax smaller = a_bigger ? b : a;
  SumMax r;
  r.maxval = bigger.maxval;
  r.sum = bigger.sum + smaller.sum * sycl::exp(smaller.maxval - bigger.maxval);
  return r;
}

// ---------------------------------------------------------------------------
// variant 1: one work-item per row, serial loop over C
void softmax_forward1(sycl::queue &q, float *out, const float *inp, int N,
                      int C) {
  const int L = 256;
  std::size_t G =
      static_cast<std::size_t>(ceil_div(N, L)) * static_cast<std::size_t>(L);
  q.parallel_for<class SoftmaxK1>(sycl::nd_range<1>(G, L),
                                  [=](sycl::nd_item<1> it) {
                                    int i = static_cast<int>(it.get_global_id(0));
                                    if (i >= N) return;
                                    const float *inp_row = inp + (std::size_t)i * (std::size_t)C;
                                    float *out_row = out + (std::size_t)i * (std::size_t)C;
                                    float maxval = kNegInf();
                                    for (int j = 0; j < C; ++j) {
                                      float v = inp_row[j];
                                      if (v > maxval) maxval = v;
                                    }
                                    double sum = 0.0;
                                    for (int j = 0; j < C; ++j) {
                                      float e = sycl::exp(inp_row[j] - maxval);
                                      out_row[j] = e;
                                      sum += static_cast<double>(e);
                                    }
                                    float inv = static_cast<float>(1.0 / sum);
                                    for (int j = 0; j < C; ++j) out_row[j] *= inv;
                                  })
      .wait();
}

// ---------------------------------------------------------------------------
// variant 2: one work-group per row, strided max/exp/sum with local-memory
// tree reductions.
void softmax_forward2(sycl::queue &q, float *out, const float *inp, int N,
                      int C) {
  const int L = 256;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK2>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           float maxval = kNegInf();
           for (int i = tid; i < C; i += L) maxval = sycl::fmax(maxval, x[i]);
           sh[tid] = maxval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = sycl::fmax(sh[tid], sh[tid + s]);
             it.barrier(sycl::access::fence_space::local_space);
           }
           float offset = sh[0];
           float *y = out + (std::size_t)idx * (std::size_t)C;
           for (int i = tid; i < C; i += L) y[i] = sycl::exp(x[i] - offset);
           it.barrier();
           float sumval = 0.0f;
           for (int i = tid; i < C; i += L) sumval += y[i];
           sh[tid] = sumval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] += sh[tid + s];
             it.barrier(sycl::access::fence_space::local_space);
           }
           float sum = sh[0];
           for (int i = tid; i < C; i += L) y[i] /= sum;
         });
   }).wait();
}

// ---------------------------------------------------------------------------
// variant 3: one small work-group (32) per row, local-memory reductions.
void softmax_forward3(sycl::queue &q, float *out, const float *inp, int N,
                      int C) {
  const int L = 32;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK3>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           float maxval = kNegInf();
           for (int i = tid; i < C; i += L) maxval = sycl::fmax(maxval, x[i]);
           sh[tid] = maxval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = sycl::fmax(sh[tid], sh[tid + s]);
             it.barrier(sycl::access::fence_space::local_space);
           }
           float offset = sh[0];
           float *y = out + (std::size_t)idx * (std::size_t)C;
           for (int i = tid; i < C; i += L) y[i] = sycl::exp(x[i] - offset);
           it.barrier();
           float sumval = 0.0f;
           for (int i = tid; i < C; i += L) sumval += y[i];
           sh[tid] = sumval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] += sh[tid + s];
             it.barrier(sycl::access::fence_space::local_space);
           }
           float sum = sh[0];
           for (int i = tid; i < C; i += L) y[i] /= sum;
         });
   }).wait();
}

// ---------------------------------------------------------------------------
// variant 4: one work-group per row; per-work-item strided max, then
// work-group-wide tree reduction in local memory (tolerates any power-of-two
// work-group size).
void softmax_forward4(sycl::queue &q, float *out, const float *inp, int N,
                      int C) {
  const int L = 256;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK4>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           float maxval = kNegInf();
           for (int i = tid; i < C; i += L) maxval = sycl::fmax(maxval, x[i]);
           sh[tid] = maxval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = sycl::fmax(sh[tid], sh[tid + s]);
             it.barrier(sycl::access::fence_space::local_space);
           }
           float offset = sh[0];
           float *y = out + (std::size_t)idx * (std::size_t)C;
           for (int i = tid; i < C; i += L) y[i] = sycl::exp(x[i] - offset);
           it.barrier();
           float sumval = 0.0f;
           for (int i = tid; i < C; i += L) sumval += y[i];
           sh[tid] = sumval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] += sh[tid + s];
             it.barrier(sycl::access::fence_space::local_space);
           }
           float sum = sh[0];
           for (int i = tid; i < C; i += L) y[i] /= sum;
         });
   }).wait();
}

// ---------------------------------------------------------------------------
// variant 5 (online 1): one work-item per row, single-pass online
// max/sum then normalise.
void softmax_forward_online1(sycl::queue &q, float *out, const float *inp,
                             int N, int C) {
  const int L = 256;
  std::size_t G =
      static_cast<std::size_t>(ceil_div(N, L)) * static_cast<std::size_t>(L);
  q.parallel_for<class SoftmaxK5>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int i = static_cast<int>(it.get_global_id(0));
           if (i >= N) return;
           const float *inp_row = inp + (std::size_t)i * (std::size_t)C;
           float *out_row = out + (std::size_t)i * (std::size_t)C;
           float maxval = kNegInf();
           double sum = 0.0;
           for (int j = 0; j < C; ++j) {
             float cur = inp_row[j];
             if (cur > maxval) {
               sum = sum * static_cast<double>(sycl::exp(maxval - cur)) +
                     static_cast<double>(sycl::exp(cur - cur));
               // exp(cur-cur)==1 ; keeps the same math as online reference
               maxval = cur;
             } else {
               sum += static_cast<double>(sycl::exp(cur - maxval));
             }
           }
           float inv = static_cast<float>(1.0 / sum);
           for (int j = 0; j < C; ++j)
             out_row[j] = sycl::exp(inp_row[j] - maxval) * inv;
         })
      .wait();
}

// ---------------------------------------------------------------------------
// variant 6 (online 2): one work-group per row, per-work-item online
// partials combined with the online SumMax reduction in local memory.
void softmax_forward_online2(sycl::queue &q, float *out, const float *inp,
                             int N, int C) {
  const int L = 32;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  const float neg_inf = kNegInf();
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<SumMax, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK6>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           SumMax part{neg_inf, 0.0f};
           for (int i = tid; i < C; i += L) {
             SumMax cur{x[i], 1.0f};
             part = combine_sum_max(part, cur, neg_inf);
           }
           sh[tid] = part;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = combine_sum_max(sh[tid], sh[tid + s], neg_inf);
             it.barrier(sycl::access::fence_space::local_space);
           }
           SumMax total = sh[0];
           float *y = out + (std::size_t)idx * (std::size_t)C;
           float inv = 1.0f / total.sum;
           for (int i = tid; i < C; i += L)
             y[i] = sycl::exp(x[i] - total.maxval) * inv;
         });
   }).wait();
}

// ---------------------------------------------------------------------------
// variant 7: one work-group per row, unrolled (factor 8) passes for large C.
void softmax_forward7(sycl::queue &q, float *out, const float *inp, int N,
                      int C) {
  const int L = 256;
  const int UF = 8;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK7>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           float *y = out + (std::size_t)idx * (std::size_t)C;
           float maxval = kNegInf();
           for (int i = tid; i < C; i += L * UF) {
#pragma unroll
             for (int u = 0; u < UF; ++u) {
               int j = i + u * L;
               if (j < C) maxval = sycl::fmax(maxval, x[j]);
             }
           }
           sh[tid] = maxval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = sycl::fmax(sh[tid], sh[tid + s]);
             it.barrier(sycl::access::fence_space::local_space);
           }
           float offset = sh[0];
           float sumval = 0.0f;
           for (int i = tid; i < C; i += L * UF) {
             float regs[UF];
#pragma unroll
             for (int u = 0; u < UF; ++u) {
               int j = i + u * L;
               regs[u] = (j < C) ? x[j] : 0.0f;
             }
#pragma unroll
             for (int u = 0; u < UF; ++u) {
               int j = i + u * L;
               if (j < C) {
                 float o = sycl::exp(regs[u] - offset);
                 y[j] = o;
                 sumval += o;
               }
             }
           }
           sh[tid] = sumval;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] += sh[tid + s];
             it.barrier(sycl::access::fence_space::local_space);
           }
           float sum = sh[0];
           for (int i = tid; i < C; i += L * UF) {
             float regs[UF];
#pragma unroll
             for (int u = 0; u < UF; ++u) {
               int j = i + u * L;
               regs[u] = (j < C) ? y[j] : 0.0f;
             }
#pragma unroll
             for (int u = 0; u < UF; ++u) {
               int j = i + u * L;
               if (j < C) y[j] = regs[u] / sum;
             }
           }
         });
   }).wait();
}

// ---------------------------------------------------------------------------
// variant 8 (online 8): one work-group per row, fused online max+sum in a
// single strided pass, then SumMax tree reduction, then normalise.
void softmax_forward_online8(sycl::queue &q, float *out, const float *inp,
                             int N, int C) {
  const int L = 32;
  std::size_t G = static_cast<std::size_t>(N) * static_cast<std::size_t>(L);
  const float neg_inf = kNegInf();
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<SumMax, 1> sh(sycl::range<1>(L), h);
     h.parallel_for<class SoftmaxK8>(
         sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
           int idx = static_cast<int>(it.get_group(0));
           int tid = static_cast<int>(it.get_local_id(0));
           const float *x = inp + (std::size_t)idx * (std::size_t)C;
           float maxval = neg_inf;
           float sumval = 0.0f;
           for (int i = tid; i < C; i += L) {
             float v = x[i];
             float bigger = sycl::fmax(maxval, v);
             sumval = sumval * sycl::exp(maxval - bigger) + sycl::exp(v - bigger);
             maxval = bigger;
           }
           // empty-partial guard: threads with no elements stay identity
           SumMax part{maxval, sumval};
           if (sumval == 0.0f && maxval == neg_inf) part = SumMax{neg_inf, 0.0f};
           sh[tid] = part;
           it.barrier(sycl::access::fence_space::local_space);
           for (int s = L / 2; s >= 1; s /= 2) {
             if (tid < s) sh[tid] = combine_sum_max(sh[tid], sh[tid + s], neg_inf);
             it.barrier(sycl::access::fence_space::local_space);
           }
           SumMax total = sh[0];
           float *y = out + (std::size_t)idx * (std::size_t)C;
           float inv = 1.0f / total.sum;
           for (int i = tid; i < C; i += L)
             y[i] = sycl::exp(x[i] - total.maxval) * inv;
         });
   }).wait();
}

void softmax_forward(sycl::queue &q, int kernel_num, float *out,
                     const float *inp, int N, int C) {
  switch (kernel_num) {
    case 1: softmax_forward1(q, out, inp, N, C); break;
    case 2: softmax_forward2(q, out, inp, N, C); break;
    case 3: softmax_forward3(q, out, inp, N, C); break;
    case 4: softmax_forward4(q, out, inp, N, C); break;
    case 5: softmax_forward_online1(q, out, inp, N, C); break;
    case 6: softmax_forward_online2(q, out, inp, N, C); break;
    case 7: softmax_forward7(q, out, inp, N, C); break;
    case 8: softmax_forward_online8(q, out, inp, N, C); break;
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
  std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  std::size_t NV = N * static_cast<std::size_t>(V);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * NV));
  if (!inp) return 2;
  if (std::fread(inp, sizeof(float), NV, fin) != NV) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  int *outliers =
      static_cast<int *>(std::malloc(sizeof(int) * N * 3));
  if (!outliers) return 2;
  if (std::fread(outliers, sizeof(int), N * 3, fin) != N * 3) {
    std::fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  std::fclose(fin);
  for (int k = 0; k < 3; ++k) {
    for (std::size_t j = 0; j < N; ++j) {
      inp[j * (std::size_t)V + (std::size_t)outliers[j * 3 + k]] *= 20.0f;
    }
  }

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(NV, q);
  float *d_out = sycl::malloc_device<float>(NV, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * NV));
  if (!d_inp || !d_out || !out) return 2;
  q.memcpy(d_inp, inp, sizeof(float) * NV).wait();

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  const int variants[] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int vi = 0; vi < 8; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * NV).wait();
    softmax_forward(q, k, d_out, d_inp, static_cast<int>(N), V);
    q.memcpy(out, d_out, sizeof(float) * NV).wait();
    std::fwrite(out, sizeof(float), NV, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(outliers);
  std::free(out);
  return 0;
}
