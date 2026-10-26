#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

template <typename T>
static T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

static constexpr float kEps = 1e-5f;
static constexpr int kWorkGroupSize = 256;

static void layernorm_forward1(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    const int grid = ceil_div(N, local);
    const std::size_t global = static_cast<std::size_t>(grid) * static_cast<std::size_t>(local);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
                       const int idx = static_cast<int>(item.get_global_id(0));
                       if (idx >= N) {
                           return;
                       }
                       const float *x = inp + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
                       float m = 0.0f;
                       for (int i = 0; i < C; ++i) {
                           m += x[i];
                       }
                       m = m / static_cast<float>(C);
                       float v = 0.0f;
                       for (int i = 0; i < C; ++i) {
                           float d = x[i] - m;
                           v += d * d;
                       }
                       v = v / static_cast<float>(C);
                       float s = 1.0f / sycl::sqrt(v + kEps);
                       float *o = out + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
                       for (int i = 0; i < C; ++i) {
                           float n = s * (x[i] - m);
                           o[i] = n * weight[i] + bias[i];
                       }
                       mean[idx] = m;
                       rstd[idx] = s;
                   })
        .wait();
}

static void layernorm_forward2(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    {
        const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) {
                           sycl::group<1> grp = item.get_group();
                           const int row = static_cast<int>(grp.get_group_id(0));
                           const int lid = static_cast<int>(item.get_local_id(0));
                           const int width = static_cast<int>(item.get_local_range(0));
                           const float *x = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                           float partial = 0.0f;
                           for (int i = lid; i < C; i += width) {
                               partial += x[i];
                           }
                           float total = sycl::reduce_over_group(grp, partial, sycl::plus<float>{});
                           if (lid == 0) {
                               mean[row] = total / static_cast<float>(C);
                           }
                       })
            .wait();
    }
    {
        const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) {
                           sycl::group<1> grp = item.get_group();
                           const int row = static_cast<int>(grp.get_group_id(0));
                           const int lid = static_cast<int>(item.get_local_id(0));
                           const int width = static_cast<int>(item.get_local_range(0));
                           const float *x = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                           float m = mean[row];
                           float partial = 0.0f;
                           for (int i = lid; i < C; i += width) {
                               float d = x[i] - m;
                               partial += d * d;
                           }
                           float total = sycl::reduce_over_group(grp, partial, sycl::plus<float>{});
                           if (lid == 0) {
                               rstd[row] = 1.0f / sycl::sqrt(total / static_cast<float>(C) + kEps);
                           }
                       })
            .wait();
    }
    {
        const long total_elems = static_cast<long>(N) * static_cast<long>(C);
        const int l2 = kWorkGroupSize;
        const int g2 = ceil_div<long>(total_elems, l2);
        const std::size_t global = static_cast<std::size_t>(g2) * static_cast<std::size_t>(l2);
        const int cc = C;
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(l2)),
                       [=](sycl::nd_item<1> item) {
                           long idx = static_cast<long>(item.get_global_id(0));
                           if (idx >= total_elems) {
                               return;
                           }
                           int bt = static_cast<int>(idx / cc);
                           int c = static_cast<int>(idx % cc);
                           float m = mean[bt];
                           float s = rstd[bt];
                           float xi = inp[idx];
                           float n = s * (xi - m);
                           out[idx] = n * weight[c] + bias[c];
                       })
            .wait();
    }
}

static void layernorm_forward3(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
                       sycl::group<1> grp = item.get_group();
                       sycl::sub_group sg = item.get_sub_group();
                       const std::size_t subgroups_per_group = sg.get_group_linear_range();
                       const int row = static_cast<int>(grp.get_group_id(0) * subgroups_per_group +
                                                        sg.get_group_linear_id());
                       if (row >= N) {
                           return;
                       }
                       const std::size_t lane = sg.get_local_linear_id();
                       const std::size_t width = sg.get_local_linear_range();
                       const float *x = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       float sum = 0.0f;
                       for (std::size_t i = lane; i < static_cast<std::size_t>(C); i += width) {
                           sum += x[i];
                       }
                       sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
                       float m = sum / static_cast<float>(C);
                       if (lane == 0) {
                           mean[row] = m;
                       }
                       float sq = 0.0f;
                       for (std::size_t i = lane; i < static_cast<std::size_t>(C); i += width) {
                           float d = x[i] - m;
                           sq += d * d;
                       }
                       sq = sycl::reduce_over_group(sg, sq, sycl::plus<float>{});
                       float s = sycl::rsqrt(sq / static_cast<float>(C) + kEps);
                       if (lane == 0) {
                           rstd[row] = s;
                       }
                       float *o = out + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       for (std::size_t c = lane; c < static_cast<std::size_t>(C); c += width) {
                           float n = s * (x[c] - m);
                           o[c] = n * weight[c] + bias[c];
                       }
                   })
        .wait();
}

static void layernorm_forward4(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
                       sycl::group<1> grp = item.get_group();
                       sycl::sub_group sg = item.get_sub_group();
                       const std::size_t subgroups_per_group = sg.get_group_linear_range();
                       const int row = static_cast<int>(grp.get_group_id(0) * subgroups_per_group +
                                                        sg.get_group_linear_id());
                       if (row >= N) {
                           return;
                       }
                       const std::size_t lane = sg.get_local_linear_id();
                       const std::size_t width = sg.get_local_linear_range();
                       const float *x = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       float sum = 0.0f;
                       float sum2 = 0.0f;
                       for (std::size_t i = lane; i < static_cast<std::size_t>(C); i += width) {
                           float xi = x[i];
                           sum += xi;
                           sum2 += xi * xi;
                       }
                       sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
                       sum2 = sycl::reduce_over_group(sg, sum2, sycl::plus<float>{});
                       sum /= static_cast<float>(C);
                       sum2 /= static_cast<float>(C);
                       float m = sum;
                       float var = sum2 - sum * sum;
                       float s = sycl::rsqrt(var + kEps);
                       if (lane == 0) {
                           mean[row] = m;
                       }
                       if (lane == 0) {
                           rstd[row] = s;
                       }
                       float *o = out + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       for (std::size_t c = lane; c < static_cast<std::size_t>(C); c += width) {
                           float n = s * (x[c] - m);
                           o[c] = n * weight[c] + bias[c];
                       }
                   })
        .wait();
}

static void layernorm_forward5(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
                       sycl::group<1> grp = item.get_group();
                       const int row = static_cast<int>(grp.get_group_id(0));
                       const int lid = static_cast<int>(item.get_local_id(0));
                       const int width = static_cast<int>(item.get_local_range(0));
                       const float *x = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       float thread_sum = 0.0f;
                       float thread_sum2 = 0.0f;
                       for (int i = lid; i < C; i += width) {
                           float xi = x[i];
                           thread_sum += xi;
                           thread_sum2 += xi * xi;
                       }
                       float block_sum = sycl::reduce_over_group(grp, thread_sum, sycl::plus<float>{});
                       float block_sum2 = sycl::reduce_over_group(grp, thread_sum2, sycl::plus<float>{});
                       block_sum /= static_cast<float>(C);
                       block_sum2 /= static_cast<float>(C);
                       float m = block_sum;
                       float var = block_sum2 - m * m;
                       float s = sycl::rsqrt(var + kEps);
                       if (lid == 0) {
                           mean[row] = m;
                       }
                       if (lid == 0) {
                           rstd[row] = s;
                       }
                       float *o = out + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                       for (int i = lid; i < C; i += width) {
                           float n = s * (x[i] - m);
                           o[i] = n * weight[i] + bias[i];
                       }
                   })
        .wait();
}

static void layernorm_forward6(sycl::queue &q, float *out, float *mean,
                               float *rstd, const float *inp,
                               const float *weight, const float *bias, int N,
                               int C) {
    const int local = kWorkGroupSize;
    const std::size_t global = static_cast<std::size_t>(N) * static_cast<std::size_t>(local);
    const std::size_t c_sz = static_cast<std::size_t>(C);
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> s_weight(sycl::range<1>(c_sz), h);
        sycl::local_accessor<float, 1> s_bias(sycl::range<1>(c_sz), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
            [=](sycl::nd_item<1> item) {
                sycl::group<1> grp = item.get_group();
                const int row = static_cast<int>(grp.get_group_id(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int width = static_cast<int>(item.get_local_range(0));
                for (int i = lid; i < C; i += width) {
                    s_weight[static_cast<std::size_t>(i)] = weight[static_cast<std::size_t>(i)];
                    s_bias[static_cast<std::size_t>(i)] = bias[static_cast<std::size_t>(i)];
                }
                sycl::group_barrier(grp, sycl::memory_scope::work_group);
                const float *xp = inp + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                float partial = 0.0f;
                for (int i = lid; i < C; i += width) {
                    partial += xp[i];
                }
                float total = sycl::reduce_over_group(grp, partial, sycl::plus<float>{});
                float m = total / static_cast<float>(C);
                float vpart = 0.0f;
                for (int i = lid; i < C; i += width) {
                    float d = xp[i] - m;
                    vpart += d * d;
                }
                float vtotal = sycl::reduce_over_group(grp, vpart, sycl::plus<float>{});
                float s = sycl::rsqrt(vtotal / static_cast<float>(C) + kEps);
                float *op = out + static_cast<std::size_t>(row) * static_cast<std::size_t>(C);
                for (int i = lid; i < C; i += width) {
                    float n = s * (xp[i] - m);
                    op[i] = n * s_weight[static_cast<std::size_t>(i)] +
                            s_bias[static_cast<std::size_t>(i)];
                }
                if (lid == 0) {
                    mean[row] = m;
                    rstd[row] = s;
                }
            });
    }).wait();
}

static void layernorm_forward_dispatch(sycl::queue &q, int kernel_num, float *out,
                                       float *mean, float *rstd,
                                       const float *inp, const float *weight,
                                       const float *bias, int B, int T, int C) {
    const int N = B * T;
    switch (kernel_num) {
        case 1:
            layernorm_forward1(q, out, mean, rstd, inp, weight, bias, N, C);
            break;
        case 2:
            layernorm_forward2(q, out, mean, rstd, inp, weight, bias, N, C);
            break;
        case 3:
            layernorm_forward3(q, out, mean, rstd, inp, weight, bias, N, C);
            break;
        case 4:
            layernorm_forward4(q, out, mean, rstd, inp, weight, bias, N, C);
            break;
        case 5:
            layernorm_forward5(q, out, mean, rstd, inp, weight, bias, N, C);
            break;
        case 6:
            layernorm_forward6(q, out, mean, rstd, inp, weight, bias, N, C);
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
    const std::size_t n_elem = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                               static_cast<std::size_t>(C);
    const std::size_t n_row = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::perror(argv[1]);
        return 2;
    }
    float *inp = static_cast<float *>(std::malloc(n_elem * sizeof(float)));
    float *weight = static_cast<float *>(std::malloc(static_cast<std::size_t>(C) * sizeof(float)));
    float *bias = static_cast<float *>(std::malloc(static_cast<std::size_t>(C) * sizeof(float)));
    if (!inp || !weight || !bias) {
        return 2;
    }
    if (std::fread(inp, sizeof(float), n_elem, fin) != n_elem) {
        std::fprintf(stderr, "input too short at inp\n");
        return 2;
    }
    if (std::fread(weight, sizeof(float), static_cast<std::size_t>(C), fin) !=
        static_cast<std::size_t>(C)) {
        std::fprintf(stderr, "input too short at weight\n");
        return 2;
    }
    if (std::fread(bias, sizeof(float), static_cast<std::size_t>(C), fin) !=
        static_cast<std::size_t>(C)) {
        std::fprintf(stderr, "input too short at bias\n");
        return 2;
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::perror(argv[2]);
        return 2;
    }

    sycl::queue q{sycl::default_selector_v};
    float *d_inp = sycl::malloc_device<float>(n_elem, q);
    float *d_weight = sycl::malloc_device<float>(static_cast<std::size_t>(C), q);
    float *d_bias = sycl::malloc_device<float>(static_cast<std::size_t>(C), q);
    float *d_out = sycl::malloc_device<float>(n_elem, q);
    float *d_mean = sycl::malloc_device<float>(n_row, q);
    float *d_rstd = sycl::malloc_device<float>(n_row, q);
    float *out = static_cast<float *>(std::malloc(n_elem * sizeof(float)));
    float *mean = static_cast<float *>(std::malloc(n_row * sizeof(float)));
    float *rstd = static_cast<float *>(std::malloc(n_row * sizeof(float)));
    if (!d_inp || !d_weight || !d_bias || !d_out || !d_mean || !d_rstd || !out || !mean || !rstd) {
        return 2;
    }
    q.memcpy(d_inp, inp, n_elem * sizeof(float)).wait();
    q.memcpy(d_weight, weight, static_cast<std::size_t>(C) * sizeof(float)).wait();
    q.memcpy(d_bias, bias, static_cast<std::size_t>(C) * sizeof(float)).wait();

    const int variants[] = {1, 2, 3, 4, 5, 6};
    for (int vi = 0; vi < 6; ++vi) {
        const int k = variants[vi];
        q.memset(d_out, 0, n_elem * sizeof(float)).wait();
        q.memset(d_mean, 0, n_row * sizeof(float)).wait();
        q.memset(d_rstd, 0, n_row * sizeof(float)).wait();
        layernorm_forward_dispatch(q, k, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C);
        q.memcpy(out, d_out, n_elem * sizeof(float)).wait();
        std::fwrite(out, sizeof(float), n_elem, fout);
        q.memcpy(mean, d_mean, n_row * sizeof(float)).wait();
        std::fwrite(mean, sizeof(float), n_row, fout);
        q.memcpy(rstd, d_rstd, n_row * sizeof(float)).wait();
        std::fwrite(rstd, sizeof(float), n_row, fout);
    }
    std::fclose(fout);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);
    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    std::free(inp);
    std::free(weight);
    std::free(bias);
    std::free(out);
    std::free(mean);
    std::free(rstd);
    return 0;
}
