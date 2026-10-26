#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cfloat>

// Kernel name tags: one distinct SYCL kernel per variant (0..4).
// Variant 0 uses PermuteKernel + oneMath gemm_batch on device.
class PermuteKernel;
class TrimulNaiveKernel;
class TrimulRegistersKernel;
class TrimulTri3Kernel;
class TrimulTri4Kernel;

static inline sycl::float4 ld_vec4(const float *p) {
    return *reinterpret_cast<const sycl::float4 *>(p);
}
static inline void st_vec4(float *p, sycl::float4 v) {
    *reinterpret_cast<sycl::float4 *>(p) = v;
}

void run_permute(sycl::queue &q, float *qq, float *kk, float *vv,
                 const float *inp, int B, int T, int NH, int HS) {
    std::size_t total = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
    if (total == 0) return;
    q.parallel_for<PermuteKernel>(sycl::range<1>(total), [=](sycl::id<1> id) {
        std::size_t idx = id[0];
        std::size_t nhts = (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
        std::size_t ths = (std::size_t)T * (std::size_t)HS;
        std::size_t b = idx / nhts;
        std::size_t rest = idx % nhts;
        std::size_t nh = rest / ths;
        rest = rest % ths;
        std::size_t t = rest / (std::size_t)HS;
        std::size_t hs = rest % (std::size_t)HS;
        std::size_t inp_idx = b * (std::size_t)T * 3 * (std::size_t)NH * (std::size_t)HS +
                              t * 3 * (std::size_t)NH * (std::size_t)HS +
                              (std::size_t)nh * (std::size_t)HS + hs;
        float qv = inp[inp_idx];
        float kv = inp[inp_idx + (std::size_t)NH * (std::size_t)HS];
        float vv_ = inp[inp_idx + 2 * (std::size_t)NH * (std::size_t)HS];
        qq[idx] = qv;
        kk[idx] = kv;
        vv[idx] = vv_;
    });
    q.wait();
}

void run_cublas_variant(sycl::queue &q, float *preatt, const float *inp,
                        float *qkvr, int B, int T, int C, int NH) {
    int HS = C / NH;
    float *qq = qkvr + (std::size_t)0 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
    float *kk = qkvr + (std::size_t)1 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
    float *vv = qkvr + (std::size_t)2 * (std::size_t)B * (std::size_t)T * (std::size_t)C;
    run_permute(q, qq, kk, vv, inp, B, T, NH, HS);
    // Row-major equivalent of the cuBLAS call:
    // preatt[bh] (T x T) = alpha * Q[bh] (T x HS) * K[bh]^T (HS x T)
    float alpha = 1.0f / std::sqrt((float)HS);
    float beta = 0.0f;
    std::int64_t m = T, n = T, k = HS;
    std::int64_t lda = HS, ldb = HS, ldc = T;
    std::int64_t stride_a = (std::int64_t)T * HS;
    std::int64_t stride_b = (std::int64_t)T * HS;
    std::int64_t stride_c = (std::int64_t)T * T;
    std::int64_t batch = (std::int64_t)B * NH;
    if (batch == 0) return;
    oneapi::math::blas::row_major::gemm_batch(
        q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
        m, n, k, alpha, qq, lda, stride_a, kk, ldb, stride_b, beta, preatt,
        ldc, stride_c, batch);
    q.wait();
}

void run_naive(sycl::queue &q, float *out, const float *inp, int B, int T,
               int C, int NH) {
    assert(T % 128 == 0);
    int C3 = C * 3;
    int HS = C / NH;
    float scale = 1.0f / std::sqrt((float)HS);
    std::size_t bh_count = (std::size_t)B * (std::size_t)NH;
    std::size_t tiles = (std::size_t)T / 128;
    sycl::range<3> local{1, 16, 16};
    sycl::range<3> global{bh_count * 1, tiles * 16, tiles * 16};
    q.parallel_for<TrimulNaiveKernel>(
        sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
            std::size_t bh = it.get_group(0);
            std::size_t j_tile = it.get_group(1);
            std::size_t i_tile = it.get_group(2);
            if (j_tile > i_tile) return;
            int tx = (int)it.get_local_id(2);
            int ty = (int)it.get_local_id(1);
            int i_base = (int)(128 * i_tile + 8 * (std::size_t)tx);
            int j_base = (int)(128 * j_tile + 8 * (std::size_t)ty);
            if (j_base > i_base) return;
            int b = (int)(bh / (std::size_t)NH);
            int nh = (int)(bh % (std::size_t)NH);
            const float *qptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                (std::size_t)nh * (std::size_t)HS;
            const float *kptr = qptr + C;
            float *rptr = out + bh * (std::size_t)T * (std::size_t)T;
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
    q.wait();
}

void run_registers(sycl::queue &q, float *out, const float *inp, int B, int T,
                   int C, int NH) {
    assert(T % 128 == 0);
    int C3 = C * 3;
    int HS = C / NH;
    float scale = 1.0f / std::sqrt((float)HS);
    std::size_t bh_count = (std::size_t)B * (std::size_t)NH;
    std::size_t tiles = (std::size_t)T / 128;
    sycl::range<3> local{1, 16, 16};
    sycl::range<3> global{bh_count * 1, tiles * 16, tiles * 16};
    q.parallel_for<TrimulRegistersKernel>(
        sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
            std::size_t bh = it.get_group(0);
            std::size_t j_tile = it.get_group(1);
            std::size_t i_tile = it.get_group(2);
            if (j_tile > i_tile) return;
            int tx = (int)it.get_local_id(2);
            int ty = (int)it.get_local_id(1);
            int i_base = (int)(128 * i_tile + 8 * (std::size_t)tx);
            int j_base = (int)(128 * j_tile + 8 * (std::size_t)ty);
            if (j_base > i_base) return;
            int b = (int)(bh / (std::size_t)NH);
            int nh = (int)(bh % (std::size_t)NH);
            const float *qptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                (std::size_t)nh * (std::size_t)HS + (std::size_t)i_base * (std::size_t)C3;
            const float *kptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                (std::size_t)nh * (std::size_t)HS + (std::size_t)C +
                                (std::size_t)j_base * (std::size_t)C3;
            float *pptr = out + bh * (std::size_t)T * (std::size_t)T +
                          (std::size_t)i_base * (std::size_t)T + (std::size_t)j_base;
            float vals[8][8] = {};
            for (int hs = 0; hs < HS; ++hs) {
                float lhs[8];
                float rhs[8];
                for (int u = 0; u < 8; ++u) {
                    lhs[u] = qptr[(std::size_t)u * (std::size_t)C3 + hs];
                    rhs[u] = kptr[(std::size_t)u * (std::size_t)C3 + hs];
                }
                for (int i = 0; i < 8; ++i)
                    for (int j = 0; j < 8; ++j) vals[i][j] += lhs[i] * rhs[j];
            }
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    pptr[(std::size_t)i * (std::size_t)T + (std::size_t)j] = vals[i][j] * scale;
        });
    q.wait();
}

void run_tri3(sycl::queue &q, float *out, const float *inp, int B, int T,
              int C, int NH) {
    assert(T % 128 == 0);
    int C3 = C * 3;
    int HS = C / NH;
    float scale = 1.0f / std::sqrt((float)HS);
    std::size_t bh_count = (std::size_t)B * (std::size_t)NH;
    std::size_t tiles = (std::size_t)T / 128;
    sycl::range<3> local{1, 16, 16};
    sycl::range<3> global{bh_count * 1, tiles * 16, tiles * 16};
    q.parallel_for<TrimulTri3Kernel>(
        sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
            std::size_t bh = it.get_group(0);
            std::size_t j_tile = it.get_group(1);
            std::size_t i_tile = it.get_group(2);
            if (j_tile > i_tile) return;
            int tx = (int)it.get_local_id(2);
            int ty = (int)it.get_local_id(1);
            int i_base = (int)(128 * i_tile + 8 * (std::size_t)tx);
            int j_base = (int)(128 * j_tile + 8 * (std::size_t)ty);
            if (j_base > i_base) return;
            int b = (int)(bh / (std::size_t)NH);
            int nh = (int)(bh % (std::size_t)NH);
            const float *qptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                (std::size_t)nh * (std::size_t)HS + (std::size_t)i_base * (std::size_t)C3;
            const float *kptr = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                (std::size_t)nh * (std::size_t)HS + (std::size_t)C +
                                (std::size_t)j_base * (std::size_t)C3;
            float *pptr = out + bh * (std::size_t)T * (std::size_t)T +
                          (std::size_t)i_base * (std::size_t)T + (std::size_t)j_base;
            float vals[8][8] = {};
            for (int hs = 0; hs < HS; hs += 4) {
                sycl::float4 rhs[8];
                for (int u = 0; u < 8; ++u)
                    rhs[u] = ld_vec4(kptr + (std::size_t)u * (std::size_t)C3 + hs);
                for (int i = 0; i < 8; ++i) {
                    sycl::float4 lhs = ld_vec4(qptr + (std::size_t)i * (std::size_t)C3 + hs);
                    float lx[4] = {lhs.x(), lhs.y(), lhs.z(), lhs.w()};
                    for (int j = 0; j < 8; ++j) {
                        float rx[4] = {rhs[j].x(), rhs[j].y(), rhs[j].z(), rhs[j].w()};
                        vals[i][j] += lx[0] * rx[0] + lx[1] * rx[1] + lx[2] * rx[2] + lx[3] * rx[3];
                    }
                }
            }
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; j += 4) {
                    sycl::float4 r;
                    r.x() = vals[i][j + 0] * scale;
                    r.y() = vals[i][j + 1] * scale;
                    r.z() = vals[i][j + 2] * scale;
                    r.w() = vals[i][j + 3] * scale;
                    st_vec4(pptr + (std::size_t)i * (std::size_t)T + j, r);
                }
            }
        });
    q.wait();
}

void run_tri4(sycl::queue &q, float *out, const float *inp, int B, int T,
              int C, int NH) {
    assert(T % 128 == 0);
    int C3 = C * 3;
    int HS = C / NH;
    float scale = 1.0f / std::sqrt((float)HS);
    std::size_t bh_count = (std::size_t)B * (std::size_t)NH;
    std::size_t tiles = (std::size_t)T / 128;
    sycl::range<3> local{1, 16, 16};
    sycl::range<3> global{bh_count * 1, tiles * 16, tiles * 16};
    sycl::nd_range<3> ndr(global, local);
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lhs_s(sycl::range<1>(128 * 32), h);
        sycl::local_accessor<float, 1> rhs_s(sycl::range<1>(128 * 32), h);
        h.parallel_for<TrimulTri4Kernel>(ndr, [=](sycl::nd_item<3> it) {
            std::size_t bh = it.get_group(0);
            std::size_t j_tile = it.get_group(1);
            std::size_t i_tile = it.get_group(2);
            // Whole-group early exit keeps barriers converged.
            if (j_tile > i_tile) return;
            int tx = (int)it.get_local_id(2);
            int ty = (int)it.get_local_id(1);
            int i_base = (int)(128 * i_tile + 8 * (std::size_t)tx);
            int j_base = (int)(128 * j_tile + 8 * (std::size_t)ty);
            int b = (int)(bh / (std::size_t)NH);
            int nh = (int)(bh % (std::size_t)NH);
            const float *qbase = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                 (std::size_t)nh * (std::size_t)HS +
                                 (std::size_t)(128 * i_tile) * (std::size_t)C3;
            const float *kbase = inp + (std::size_t)b * (std::size_t)T * (std::size_t)C3 +
                                 (std::size_t)nh * (std::size_t)HS + (std::size_t)C +
                                 (std::size_t)(128 * j_tile) * (std::size_t)C3;
            float *pbase = out + bh * (std::size_t)T * (std::size_t)T;
            float vals[8][8] = {};
            for (int so = 0; so < HS; so += 32) {
                it.barrier(sycl::access::fence_space::local_space);
                for (int y = ty / 2; y < 128; y += 8) {
                    int xo = (ty % 2) * 16;
                    int col = tx + xo;
                    lhs_s[(std::size_t)y * 32 + (std::size_t)col] =
                        qbase[(std::size_t)y * (std::size_t)C3 + (std::size_t)so + (std::size_t)col];
                    rhs_s[(std::size_t)y * 32 + (std::size_t)col] =
                        kbase[(std::size_t)y * (std::size_t)C3 + (std::size_t)so + (std::size_t)col];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int si = 0; si < 32; ++si) {
                    int col = (si + tx) % 32;
                    float rhs[8];
                    for (int u = 0; u < 8; ++u)
                        rhs[u] = rhs_s[(std::size_t)(u + 8 * ty) * 32 + (std::size_t)col];
                    for (int ii = 0; ii < 8; ++ii) {
                        float lhs = lhs_s[(std::size_t)(ii + 8 * tx) * 32 + (std::size_t)col];
                        for (int ji = 0; ji < 8; ++ji) vals[ii][ji] += lhs * rhs[ji];
                    }
                }
            }
            if (j_base > i_base) return;
            for (int ii = 0; ii < 8; ++ii) {
                for (int ji = 0; ji < 8; ji += 4) {
                    int i = i_base + ii;
                    int j = j_base + ji;
                    sycl::float4 r;
                    r.x() = vals[ii][ji + 0] * scale;
                    r.y() = vals[ii][ji + 1] * scale;
                    r.z() = vals[ii][ji + 2] * scale;
                    r.w() = vals[ii][ji + 3] * scale;
                    st_vec4(pbase + (std::size_t)i * (std::size_t)T + (std::size_t)j, r);
                }
            }
        });
    });
    q.wait();
}

void trimul_gpu(sycl::queue &q, int kernel_num, float *out, const float *inp,
                float *qkvr, int B, int T, int C, int NH) {
    switch (kernel_num) {
        case 0: run_cublas_variant(q, out, inp, qkvr, B, T, C, NH); break;
        case 1: run_naive(q, out, inp, B, T, C, NH); break;
        case 2: run_registers(q, out, inp, B, T, C, NH); break;
        case 3: run_tri3(q, out, inp, B, T, C, NH); break;
        case 4: run_tri4(q, out, inp, B, T, C, NH); break;
        default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
    }
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

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror(argv[1]); return 2; }
    std::size_t inp_count = (std::size_t)B * (std::size_t)T * 3 * (std::size_t)C;
    float *inp = (float *)std::malloc(sizeof(float) * inp_count);
    if (std::fread(inp, sizeof(float), inp_count, fin) != inp_count) {
        std::fprintf(stderr, "input too short at inp\n");
        return 2;
    }
    std::fclose(fin);
    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) { std::perror(argv[2]); return 2; }

    sycl::queue q{sycl::property::queue::in_order{}};
    float *d_inp = sycl::malloc_device<float>(inp_count, q);
    std::size_t out_count = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)T;
    float *d_out = sycl::malloc_device<float>(out_count, q);
    float *d_qkvr = sycl::malloc_device<float>((std::size_t)B * (std::size_t)T * 3 * (std::size_t)C, q);
    q.memcpy(d_inp, inp, sizeof(float) * inp_count).wait();

    float *out = (float *)std::malloc(sizeof(float) * out_count);
    const int variants[] = {0, 1, 2, 3, 4};
    for (int vi = 0; vi < 5; ++vi) {
        int k = variants[vi];
        q.memset(d_out, 0, sizeof(float) * out_count).wait();
        trimul_gpu(q, k, d_out, d_inp, d_qkvr, B, T, C, NH);
        q.memcpy(out, d_out, sizeof(float) * out_count).wait();
        for (int i = 0; i < B * NH; ++i)
            for (int t = 0; t < T; ++t)
                for (int t2 = t + 1; t2 < T; ++t2)
                    out[(std::size_t)i * (std::size_t)T * (std::size_t)T + (std::size_t)t * (std::size_t)T + (std::size_t)t2] = 0.0f;
        std::fwrite(out, sizeof(float), out_count, fout);
    }
    std::fclose(fout);
    sycl::free(d_inp, q);
    sycl::free(d_out, q);
    sycl::free(d_qkvr, q);
    std::free(inp);
    std::free(out);
    return 0;
}
