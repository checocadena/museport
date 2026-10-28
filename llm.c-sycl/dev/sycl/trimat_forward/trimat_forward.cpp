#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstddef>

// Kernel name tags: one distinct SYCL kernel per variant.
class PermuteKer;
class TrimulNaiveKer;
class TrimulRegsKer;
class TrimulTri3Ker;
class TrimulTri4Ker;

static void run_permute(sycl::queue &q, float *qq, float *kk, float *vv,
                        const float *inp, int B, int T, int NH, int HS) {
  std::size_t total = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<PermuteKer>(sycl::range<1>(total), [=](sycl::item<1> it) {
      std::size_t idx = it.get_id(0);
      std::size_t nhts = (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
      std::size_t b = idx / nhts;
      std::size_t rest = idx % nhts;
      std::size_t ths = (std::size_t)T * (std::size_t)HS;
      std::size_t nh = rest / ths;
      rest = rest % ths;
      std::size_t t = rest / (std::size_t)HS;
      std::size_t hs = rest % (std::size_t)HS;
      std::size_t inp_idx = b * (std::size_t)T * 3 * (std::size_t)NH * (std::size_t)HS +
                            t * (std::size_t)3 * (std::size_t)NH * (std::size_t)HS +
                            nh * (std::size_t)HS + hs;
      float qv = inp[inp_idx];
      float kv = inp[inp_idx + (std::size_t)NH * (std::size_t)HS];
      float vv_ = inp[inp_idx + (std::size_t)2 * (std::size_t)NH * (std::size_t)HS];
      qq[idx] = qv;
      kk[idx] = kv;
      vv[idx] = vv_;
    });
  });
}

static void run_gemm(sycl::queue &q, float *preatt, const float *qq, const float *kk,
                     int B, int T, int HS, int NH, float alpha) {
  float beta = 0.0f;
  std::int64_t m = T;
  std::int64_t n = T;
  std::int64_t k = HS;
  std::int64_t lda = HS;
  std::int64_t ldb = HS;
  std::int64_t ldc = T;
  std::int64_t stride_a = (std::int64_t)T * (std::int64_t)HS;
  std::int64_t stride_b = (std::int64_t)T * (std::int64_t)HS;
  std::int64_t stride_c = (std::int64_t)T * (std::int64_t)T;
  std::int64_t batch = (std::int64_t)B * (std::int64_t)NH;
  // Row-major (T,HS) seen as column-major (HS,T): compute K^T @ Q.
  // k is first arg (trans), q is second arg (nontrans).
  oneapi::math::blas::column_major::gemm_batch(
      q, oneapi::math::transpose::trans,
      oneapi::math::transpose::nontrans, m, n, k, alpha, kk, lda,
      stride_a, qq, ldb, stride_b, beta, preatt, ldc, stride_c, batch);
}

static void run_trimul_naive(sycl::queue &q, float *out, const float *inp, int B,
                             int T, int C, int NH) {
  int HS = C / NH;
  int C3 = C * 3;
  float scale = 1.0f / std::sqrt((float)HS);
  assert(T % 128 == 0);
  int ntx = T / 128;
  int nty = T / 128;
  int ntz = B * NH;
  sycl::range<3> global{(std::size_t)ntz, (std::size_t)nty * 16, (std::size_t)ntx * 16};
  sycl::range<3> local{1, 16, 16};
  q.submit([&](sycl::handler &h) {
    h.parallel_for<TrimulNaiveKer>(sycl::nd_range<3>(global, local),
                                   [=](sycl::nd_item<3> it) {
      int bz = (int)it.get_group(0);
      int by = (int)it.get_group(1);
      int bx = (int)it.get_group(2);
      int ty = (int)it.get_local_id(1);
      int tx = (int)it.get_local_id(2);
      if (by > bx)
        return;
      int b = bz / NH;
      int nh = bz % NH;
      const float *qptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 + (std::size_t)nh * (std::size_t)HS;
      const float *kptr = qptr + C;
      float *rptr = out + (std::size_t)(b * NH + nh) * (std::size_t)T * (std::size_t)T;
      int i_base = 128 * bx + 8 * tx;
      int j_base = 128 * by + 8 * ty;
      if (j_base > i_base)
        return;
      for (int io = 0; io < 8; ++io) {
        int i = i_base + io;
        for (int jo = 0; jo < 8; ++jo) {
          int j = j_base + jo;
          float val = 0.0f;
          for (int s = 0; s < HS; ++s)
            val += qptr[(std::size_t)i * (std::size_t)C3 + (std::size_t)s] *
                   kptr[(std::size_t)j * (std::size_t)C3 + (std::size_t)s];
          rptr[(std::size_t)i * (std::size_t)T + (std::size_t)j] = val * scale;
        }
      }
    });
  });
}

static void run_trimul_registers(sycl::queue &q, float *out, const float *inp,
                                 int B, int T, int C, int NH) {
  int HS = C / NH;
  int C3 = C * 3;
  float scale = 1.0f / std::sqrt((float)HS);
  assert(T % 128 == 0);
  int ntx = T / 128;
  int nty = T / 128;
  int ntz = B * NH;
  sycl::range<3> global{(std::size_t)ntz, (std::size_t)nty * 16, (std::size_t)ntx * 16};
  sycl::range<3> local{1, 16, 16};
  q.submit([&](sycl::handler &h) {
    h.parallel_for<TrimulRegsKer>(sycl::nd_range<3>(global, local),
                                  [=](sycl::nd_item<3> it) {
      int bz = (int)it.get_group(0);
      int by = (int)it.get_group(1);
      int bx = (int)it.get_group(2);
      int ty = (int)it.get_local_id(1);
      int tx = (int)it.get_local_id(2);
      if (by > bx)
        return;
      int b = bz / NH;
      int nh = bz % NH;
      const float *qbase = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 + (std::size_t)nh * (std::size_t)HS;
      const float *kbase = qbase + C;
      float *rbase = out + (std::size_t)(b * NH + nh) * (std::size_t)T * (std::size_t)T;
      int i_base = 128 * bx + 8 * tx;
      int j_base = 128 * by + 8 * ty;
      if (j_base > i_base)
        return;
      const float *qptr = qbase + (std::size_t)i_base * (std::size_t)C3;
      const float *kptr = kbase + (std::size_t)j_base * (std::size_t)C3;
      float *pptr = rbase + (std::size_t)i_base * (std::size_t)T + (std::size_t)j_base;
      float vals[8][8] = {};
      for (int hs = 0; hs < HS; ++hs) {
        float lhs[8];
        float rhs[8];
        for (int u = 0; u < 8; ++u) {
          lhs[u] = qptr[(std::size_t)u * (std::size_t)C3 + (std::size_t)hs];
          rhs[u] = kptr[(std::size_t)u * (std::size_t)C3 + (std::size_t)hs];
        }
        for (int i = 0; i < 8; ++i)
          for (int j = 0; j < 8; ++j)
            vals[i][j] += lhs[i] * rhs[j];
      }
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          pptr[(std::size_t)i * (std::size_t)T + (std::size_t)j] = vals[i][j] * scale;
    });
  });
}

static void run_trimul_tri3(sycl::queue &q, float *out, const float *inp, int B,
                            int T, int C, int NH) {
  int HS = C / NH;
  int C3 = C * 3;
  float scale = 1.0f / std::sqrt((float)HS);
  assert(T % 128 == 0);
  int ntx = T / 128;
  int nty = T / 128;
  int ntz = B * NH;
  sycl::range<3> global{(std::size_t)ntz, (std::size_t)nty * 16, (std::size_t)ntx * 16};
  sycl::range<3> local{1, 16, 16};
  q.submit([&](sycl::handler &h) {
    h.parallel_for<TrimulTri3Ker>(sycl::nd_range<3>(global, local),
                                  [=](sycl::nd_item<3> it) {
      int bz = (int)it.get_group(0);
      int by = (int)it.get_group(1);
      int bx = (int)it.get_group(2);
      int ty = (int)it.get_local_id(1);
      int tx = (int)it.get_local_id(2);
      if (by > bx)
        return;
      int b = bz / NH;
      int nh = bz % NH;
      const float *qbase = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 + (std::size_t)nh * (std::size_t)HS;
      const float *kbase = qbase + C;
      float *rbase = out + (std::size_t)(b * NH + nh) * (std::size_t)T * (std::size_t)T;
      int i_base = 128 * bx + 8 * tx;
      int j_base = 128 * by + 8 * ty;
      if (j_base > i_base)
        return;
      const float *qptr = qbase + (std::size_t)i_base * (std::size_t)C3;
      const float *kptr = kbase + (std::size_t)j_base * (std::size_t)C3;
      float *pptr = rbase + (std::size_t)i_base * (std::size_t)T + (std::size_t)j_base;
      float vals[8][8] = {};
      for (int hs = 0; hs < HS; hs += 4) {
        sycl::vec<float, 4> rhs[8];
        for (int u = 0; u < 8; ++u)
          rhs[u] = reinterpret_cast<const sycl::vec<float, 4> *>(
              kptr + (std::size_t)u * (std::size_t)C3 + (std::size_t)hs)[0];
        for (int i = 0; i < 8; ++i) {
          sycl::vec<float, 4> lhs =
              reinterpret_cast<const sycl::vec<float, 4> *>(
                  qptr + (std::size_t)i * (std::size_t)C3 + (std::size_t)hs)[0];
          float lx0 = lhs[0], lx1 = lhs[1], lx2 = lhs[2], lx3 = lhs[3];
          for (int j = 0; j < 8; ++j) {
            vals[i][j] += lx0 * rhs[j][0];
            vals[i][j] += lx1 * rhs[j][1];
            vals[i][j] += lx2 * rhs[j][2];
            vals[i][j] += lx3 * rhs[j][3];
          }
        }
      }
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
          sycl::vec<float, 4> res;
          res[0] = vals[i][j + 0] * scale;
          res[1] = vals[i][j + 1] * scale;
          res[2] = vals[i][j + 2] * scale;
          res[3] = vals[i][j + 3] * scale;
          reinterpret_cast<sycl::vec<float, 4> *>(
              pptr + (std::size_t)i * (std::size_t)T + (std::size_t)j)[0] = res;
        }
      }
    });
  });
}

static void run_trimul_tri4(sycl::queue &q, float *out, const float *inp, int B,
                            int T, int C, int NH) {
  int HS = C / NH;
  int C3 = C * 3;
  float scale = 1.0f / std::sqrt((float)HS);
  assert(T % 128 == 0);
  int ntx = T / 128;
  int nty = T / 128;
  int ntz = B * NH;
  sycl::range<3> global{(std::size_t)ntz, (std::size_t)nty * 16, (std::size_t)ntx * 16};
  sycl::range<3> local{1, 16, 16};
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 2> lhs({128, 32}, h);
    sycl::local_accessor<float, 2> rhs({128, 32}, h);
    h.parallel_for<TrimulTri4Ker>(sycl::nd_range<3>(global, local),
                                  [=](sycl::nd_item<3> it) {
      int bz = (int)it.get_group(0);
      int by = (int)it.get_group(1);
      int bx = (int)it.get_group(2);
      int ty = (int)it.get_local_id(1);
      int tx = (int)it.get_local_id(2);
      // Uniform per work-group: all items take the same path, so the
      // barriers below stay converged.
      if (by > bx)
        return;
      int b = bz / NH;
      int nh = bz % NH;
      const float *qbase = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 + (std::size_t)nh * (std::size_t)HS;
      const float *kbase = qbase + C;
      float *rbase = out + (std::size_t)(b * NH + nh) * (std::size_t)T * (std::size_t)T;
      const float *qblk = qbase + (std::size_t)(128 * bx) * (std::size_t)C3;
      const float *kblk = kbase + (std::size_t)(128 * by) * (std::size_t)C3;
      int i_base = 128 * bx + 8 * tx;
      int j_base = 128 * by + 8 * ty;
      float vals[8][8] = {};
      for (int so = 0; so < HS; so += 32) {
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        for (int y = ty / 2; y < 128; y += 8) {
          int xo = (ty % 2) * 16;
          lhs[y][tx + xo] = qblk[(std::size_t)y * (std::size_t)C3 + (std::size_t)so + (std::size_t)(tx + xo)];
          rhs[y][tx + xo] = kblk[(std::size_t)y * (std::size_t)C3 + (std::size_t)so + (std::size_t)(tx + xo)];
        }
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        for (int si = 0; si < 32; ++si) {
          float r[8];
          int col = (si + tx) % 32;
          for (int u = 0; u < 8; ++u)
            r[u] = rhs[(std::size_t)u + (std::size_t)8 * (std::size_t)ty][col];
          for (int ii = 0; ii < 8; ++ii) {
            float l = lhs[(std::size_t)ii + (std::size_t)8 * (std::size_t)tx][col];
            for (int ji = 0; ji < 8; ++ji)
              vals[ii][ji] += l * r[ji];
          }
        }
      }
      if (j_base > i_base)
        return;
      for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ji += 4) {
          int i = i_base + ii;
          int j = j_base + ji;
          sycl::vec<float, 4> res;
          res[0] = vals[ii][ji + 0] * scale;
          res[1] = vals[ii][ji + 1] * scale;
          res[2] = vals[ii][ji + 2] * scale;
          res[3] = vals[ii][ji + 3] * scale;
          reinterpret_cast<sycl::vec<float, 4> *>(
              rbase + (std::size_t)i * (std::size_t)T + (std::size_t)j)[0] = res;
        }
      }
    });
  });
}

static void trimul_oneblas(sycl::queue &q, float *out, const float *inp, int B,
                           int T, int C, int NH, float *qkvr) {
  int HS = C / NH;
  float *qq = qkvr + (std::size_t)0 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
  float *kk = qkvr + (std::size_t)1 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
  float *vv = qkvr + (std::size_t)2 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
  run_permute(q, qq, kk, vv, inp, B, T, NH, HS);
  float alpha = 1.0f / std::sqrt((float)HS);
  run_gemm(q, out, qq, kk, B, T, HS, NH, alpha);
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C NH\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int NH = std::atoi(argv[6]);

  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::size_t n_inp = (std::size_t)B * (std::size_t)T * 3 * (std::size_t)C;
  std::size_t n_out = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)T;
  float *inp = (float *)std::malloc(sizeof(float) * n_inp);
  if (!inp) {
    std::fclose(fin);
    return 2;
  }
  if (std::fread(inp, sizeof(float), n_inp, fin) != n_inp) {
    std::fprintf(stderr, "input too short at inp\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);
  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  float *out = (float *)std::malloc(sizeof(float) * n_out);
  if (!out)
    return 2;

  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order()}};
  float *d_inp = sycl::malloc_device<float>(n_inp, q);
  float *d_out = sycl::malloc_device<float>(n_out, q);
  float *d_qkvr = sycl::malloc_device<float>((std::size_t)B * (std::size_t)T * 3 * (std::size_t)C, q);
  if (!d_inp || !d_out || !d_qkvr)
    return 2;
  q.memcpy(d_inp, inp, sizeof(float) * n_inp).wait();

  const int variants[] = {0, 1, 2, 3, 4};
  for (int vi = 0; vi < 5; ++vi) {
    int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * n_out).wait();
    switch (k) {
    case 0:
      trimul_oneblas(q, d_out, d_inp, B, T, C, NH, d_qkvr);
      break;
    case 1:
      run_trimul_naive(q, d_out, d_inp, B, T, C, NH);
      break;
    case 2:
      run_trimul_registers(q, d_out, d_inp, B, T, C, NH);
      break;
    case 3:
      run_trimul_tri3(q, d_out, d_inp, B, T, C, NH);
      break;
    case 4:
      run_trimul_tri4(q, d_out, d_inp, B, T, C, NH);
      break;
    default:
      std::printf("Invalid kernel number\n");
      return 1;
    }
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * n_out).wait();
    for (int i = 0; i < B * NH; ++i) {
      for (int t = 0; t < T; ++t) {
        for (int t2 = t + 1; t2 < T; ++t2)
          out[(std::size_t)i * (std::size_t)T * (std::size_t)T + (std::size_t)t * (std::size_t)T + (std::size_t)t2] = 0.0f;
      }
    }
    std::fwrite(out, sizeof(float), n_out, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  sycl::free(d_qkvr, q);
  std::free(inp);
  std::free(out);
  return 0;
}
