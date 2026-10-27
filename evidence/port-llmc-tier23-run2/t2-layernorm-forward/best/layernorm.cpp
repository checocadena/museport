#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <functional>

template <typename T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// ---- Kernel 1: one work-item per row, serial loop over C ----
class LayerNormK1;
class LayerNormK2Mean;
class LayerNormK2Rstd;
class LayerNormK2Norm;
class LayerNormK3;
class LayerNormK4;
class LayerNormK5;
class LayerNormK6;

void layernorm_forward1(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C) {
    if (N <= 0) return;
    q.parallel_for<LayerNormK1>(sycl::range<1>(static_cast<size_t>(N)),
                                [=](sycl::id<1> id) {
        int idx = static_cast<int>(id[0]);
        constexpr float eps = 1e-5f;
        const float *x = inp + static_cast<size_t>(idx) * static_cast<size_t>(C);
        float m = 0.0f;
        for (int i = 0; i < C; ++i) m += x[i];
        m /= static_cast<float>(C);
        float v = 0.0f;
        for (int i = 0; i < C; ++i) {
            float d = x[i] - m;
            v += d * d;
        }
        v /= static_cast<float>(C);
        float s = 1.0f / sycl::sqrt(v + eps);
        float *o = out + static_cast<size_t>(idx) * static_cast<size_t>(C);
        for (int i = 0; i < C; ++i) {
            float n = s * (x[i] - m);
            o[i] = n * weight[i] + bias[i];
        }
        mean[idx] = m;
        rstd[idx] = s;
    }).wait();
}

void layernorm_forward2(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C, size_t local_size) {
    if (N <= 0) return;
    {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scratch(local_size, h);
            h.parallel_for<LayerNormK2Mean>(
                sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
                [=](sycl::nd_item<1> it) {
                    int row = static_cast<int>(it.get_group(0));
                    size_t lid = it.get_local_id(0);
                    size_t ls = it.get_local_range(0);
                    const float *x = inp + static_cast<size_t>(row) * static_cast<size_t>(C);
                    float sum = 0.0f;
                    for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls))
                        sum += x[i];
                    scratch[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = ls / 2; stride >= 1; stride /= 2) {
                        if (lid < stride) scratch[lid] += scratch[lid + stride];
                        it.barrier(sycl::access::fence_space::local_space);
                        if (stride == 1) break;
                    }
                    if (lid == 0) mean[row] = scratch[0] / static_cast<float>(C);
                });
        }).wait();
    }
    {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> scratch(local_size, h);
            h.parallel_for<LayerNormK2Rstd>(
                sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
                [=](sycl::nd_item<1> it) {
                    int row = static_cast<int>(it.get_group(0));
                    size_t lid = it.get_local_id(0);
                    size_t ls = it.get_local_range(0);
                    const float *x = inp + static_cast<size_t>(row) * static_cast<size_t>(C);
                    float m = mean[row];
                    float sum = 0.0f;
                    for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
                        float d = x[i] - m;
                        sum += d * d;
                    }
                    scratch[lid] = sum;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = ls / 2; stride >= 1; stride /= 2) {
                        if (lid < stride) scratch[lid] += scratch[lid + stride];
                        it.barrier(sycl::access::fence_space::local_space);
                        if (stride == 1) break;
                    }
                    if (lid == 0)
                        rstd[row] = 1.0f / sycl::sqrt(scratch[0] / static_cast<float>(C) + 1e-5f);
                });
        }).wait();
    }
    {
        size_t total = static_cast<size_t>(N) * static_cast<size_t>(C);
        if (total == 0) return;
        q.parallel_for<LayerNormK2Norm>(sycl::range<1>(total), [=](sycl::id<1> id) {
            size_t flat = id[0];
            int bt = static_cast<int>(flat / static_cast<size_t>(C));
            int c = static_cast<int>(flat % static_cast<size_t>(C));
            float m = mean[bt];
            float s = rstd[bt];
            float o = s * (inp[flat] - m) * weight[c] + bias[c];
            // expand to match reference ordering: s*(x-m) then *w + b
            float n = s * (inp[flat] - m);
            out[flat] = n * weight[c] + bias[c];
            (void)o;
        }).wait();
    }
}

// k3: one sub-group per row, two-pass mean then variance
void layernorm_forward3(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C, size_t local_size) {
    if (N <= 0) return;
    // One work-group per row-slot oversubscription: N groups guarantees coverage
    // for any sub-group size without host-side queries.
    q.parallel_for<LayerNormK3>(
         sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
         [=](sycl::nd_item<1> it) {
        sycl::sub_group sg = it.get_sub_group();
        size_t sgs_in_group = sg.get_group_range().size();
        int idx = static_cast<int>(it.get_group(0)) * static_cast<int>(sgs_in_group) +
                  static_cast<int>(sg.get_group_id()[0]);
        if (idx >= N) return;
        const float *x = inp + static_cast<size_t>(idx) * static_cast<size_t>(C);
        size_t lane = sg.get_local_id()[0];
        size_t sg_size = sg.get_local_range()[0];
        float sum = 0.0f;
        for (int i = static_cast<int>(lane); i < C; i += static_cast<int>(sg_size))
            sum += x[i];
        sum = sycl::reduce_over_group(sg, sum, std::plus<float>{});
        float m = sum / static_cast<float>(C);
        if (lane == 0) mean[idx] = m;
        float sq = 0.0f;
        for (int i = static_cast<int>(lane); i < C; i += static_cast<int>(sg_size)) {
            float d = x[i] - m;
            sq += d * d;
        }
        sq = sycl::reduce_over_group(sg, sq, std::plus<float>{});
        float s = sycl::rsqrt(sq / static_cast<float>(C) + 1e-5f);
        if (lane == 0) rstd[idx] = s;
        float *o = out + static_cast<size_t>(idx) * static_cast<size_t>(C);
        for (int c = static_cast<int>(lane); c < C; c += static_cast<int>(sg_size)) {
            float n = s * (x[c] - m);
            o[c] = n * weight[c] + bias[c];
        }
    }).wait();
}

// k4: one sub-group per row, single-pass mean(x^2)-mean(x)^2
void layernorm_forward4(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C, size_t local_size) {
    if (N <= 0) return;
    q.parallel_for<LayerNormK4>(
         sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
         [=](sycl::nd_item<1> it) {
        sycl::sub_group sg = it.get_sub_group();
        size_t sgs_in_group = sg.get_group_range().size();
        int idx = static_cast<int>(it.get_group(0)) * static_cast<int>(sgs_in_group) +
                  static_cast<int>(sg.get_group_id()[0]);
        if (idx >= N) return;
        const float *x = inp + static_cast<size_t>(idx) * static_cast<size_t>(C);
        size_t lane = sg.get_local_id()[0];
        size_t sg_size = sg.get_local_range()[0];
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int i = static_cast<int>(lane); i < C; i += static_cast<int>(sg_size)) {
            float xi = x[i];
            sum += xi;
            sum2 += xi * xi;
        }
        sum = sycl::reduce_over_group(sg, sum, std::plus<float>{});
        sum2 = sycl::reduce_over_group(sg, sum2, std::plus<float>{});
        sum /= static_cast<float>(C);
        sum2 /= static_cast<float>(C);
        float m = sum;
        float var = sum2 - m * m;
        float s = sycl::rsqrt(var + 1e-5f);
        if (lane == 0) mean[idx] = m;
        if (lane == 0) rstd[idx] = s;
        float *o = out + static_cast<size_t>(idx) * static_cast<size_t>(C);
        for (int c = static_cast<int>(lane); c < C; c += static_cast<int>(sg_size)) {
            float n = s * (x[c] - m);
            o[c] = n * weight[c] + bias[c];
        }
    }).wait();
}

// k5: one work-group per row, single-pass, group-wide reduction
void layernorm_forward5(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C, size_t local_size) {
    if (N <= 0) return;
    q.parallel_for<LayerNormK5>(
         sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
         [=](sycl::nd_item<1> it) {
        sycl::group<1> g = it.get_group();
        int idx = static_cast<int>(it.get_group(0));
        const float *x = inp + static_cast<size_t>(idx) * static_cast<size_t>(C);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        float thread_sum = 0.0f;
        float thread_sum2 = 0.0f;
        for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
            float xi = x[i];
            thread_sum += xi;
            thread_sum2 += xi * xi;
        }
        float block_sum = sycl::reduce_over_group(g, thread_sum, std::plus<float>{});
        float block_sum2 = sycl::reduce_over_group(g, thread_sum2, std::plus<float>{});
        block_sum /= static_cast<float>(C);
        block_sum2 /= static_cast<float>(C);
        float m = block_sum;
        float var = block_sum2 - m * m;
        float s = sycl::rsqrt(var + 1e-5f);
        if (lid == 0) mean[idx] = m;
        if (lid == 0) rstd[idx] = s;
        float *o = out + static_cast<size_t>(idx) * static_cast<size_t>(C);
        for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
            float n = s * (x[i] - m);
            o[i] = n * weight[i] + bias[i];
        }
    }).wait();
}

// k6: one work-group per row with weight/bias staged in local memory,
// vector-friendly two-pass algorithm
void layernorm_forward6(sycl::queue &q, float *out, float *mean, float *rstd,
                        const float *inp, const float *weight, const float *bias,
                        int N, int C, size_t local_size) {
    if (N <= 0) return;
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> w_local(static_cast<size_t>(C), h);
        sycl::local_accessor<float, 1> b_local(static_cast<size_t>(C), h);
        h.parallel_for<LayerNormK6>(
            sycl::nd_range<1>(static_cast<size_t>(N) * local_size, local_size),
            [=](sycl::nd_item<1> it) {
            sycl::group<1> g = it.get_group();
            int idx = static_cast<int>(it.get_group(0));
            size_t lid = it.get_local_id(0);
            size_t ls = it.get_local_range(0);
            for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
                w_local[i] = weight[i];
                b_local[i] = bias[i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            const float *x = inp + static_cast<size_t>(idx) * static_cast<size_t>(C);
            float sum = 0.0f;
            for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls))
                sum += x[i];
            sum = sycl::reduce_over_group(g, sum, std::plus<float>{});
            float m = sum / static_cast<float>(C);
            float v = 0.0f;
            for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
                float d = x[i] - m;
                v += d * d;
            }
            v = sycl::reduce_over_group(g, v, std::plus<float>{});
            v /= static_cast<float>(C);
            float s = sycl::rsqrt(v + 1e-5f);
            float *o = out + static_cast<size_t>(idx) * static_cast<size_t>(C);
            for (int i = static_cast<int>(lid); i < C; i += static_cast<int>(ls)) {
                float n = s * (x[i] - m);
                o[i] = n * w_local[i] + b_local[i];
            }
            if (lid == 0) mean[idx] = m;
            if (lid == 0) rstd[idx] = s;
        });
    }).wait();
}

void layernorm_forward(sycl::queue &q, int kernel_num, float *out, float *mean, float *rstd,
                       const float *inp, const float *weight, const float *bias,
                       int B, int T, int C, size_t block_size) {
    int N = B * T;
    switch (kernel_num) {
        case 1: layernorm_forward1(q, out, mean, rstd, inp, weight, bias, N, C); break;
        case 2: layernorm_forward2(q, out, mean, rstd, inp, weight, bias, N, C, block_size); break;
        case 3: layernorm_forward3(q, out, mean, rstd, inp, weight, bias, N, C, block_size); break;
        case 4: layernorm_forward4(q, out, mean, rstd, inp, weight, bias, N, C, block_size); break;
        case 5: layernorm_forward5(q, out, mean, rstd, inp, weight, bias, N, C, block_size); break;
        case 6: layernorm_forward6(q, out, mean, rstd, inp, weight, bias, N, C, block_size); break;
        default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    int B = std::atoi(argv[3]);
    int T = std::atoi(argv[4]);
    int C = std::atoi(argv[5]);
    size_t n_inp = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(C);
    size_t n_row = static_cast<size_t>(B) * static_cast<size_t>(T);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror(argv[1]); return 2; }
    float *inp = (float *)std::malloc(sizeof(float) * n_inp);
    float *weight = (float *)std::malloc(sizeof(float) * static_cast<size_t>(C));
    float *bias = (float *)std::malloc(sizeof(float) * static_cast<size_t>(C));
    if (n_inp > 0 && std::fread(inp, sizeof(float), n_inp, fin) != n_inp) {
        std::fprintf(stderr, "input too short at inp\n"); return 2;
    }
    if (static_cast<size_t>(C) > 0 && std::fread(weight, sizeof(float), static_cast<size_t>(C), fin) != static_cast<size_t>(C)) {
        std::fprintf(stderr, "input too short at weight\n"); return 2;
    }
    if (static_cast<size_t>(C) > 0 && std::fread(bias, sizeof(float), static_cast<size_t>(C), fin) != static_cast<size_t>(C)) {
        std::fprintf(stderr, "input too short at bias\n"); return 2;
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) { std::perror(argv[2]); return 2; }

    sycl::queue q{sycl::default_selector_v};
    float *d_inp = sycl::malloc_device<float>(n_inp ? n_inp : 1, q);
    float *d_weight = sycl::malloc_device<float>(C ? static_cast<size_t>(C) : 1, q);
    float *d_bias = sycl::malloc_device<float>(C ? static_cast<size_t>(C) : 1, q);
    float *d_out = sycl::malloc_device<float>(n_inp ? n_inp : 1, q);
    float *d_mean = sycl::malloc_device<float>(n_row ? n_row : 1, q);
    float *d_rstd = sycl::malloc_device<float>(n_row ? n_row : 1, q);
    float *out = (float *)std::malloc(sizeof(float) * (n_inp ? n_inp : 1));
    float *mean = (float *)std::malloc(sizeof(float) * (n_row ? n_row : 1));
    float *rstd = (float *)std::malloc(sizeof(float) * (n_row ? n_row : 1));

    if (n_inp > 0) q.memcpy(d_inp, inp, sizeof(float) * n_inp).wait();
    if (C > 0) q.memcpy(d_weight, weight, sizeof(float) * static_cast<size_t>(C)).wait();
    if (C > 0) q.memcpy(d_bias, bias, sizeof(float) * static_cast<size_t>(C)).wait();

    const size_t block_size = 256;
    const int variants[] = {1, 2, 3, 4, 5, 6};
    for (int vi = 0; vi < 6; ++vi) {
        int k = variants[vi];
        if (n_inp > 0) q.fill(d_out, 0.0f, n_inp).wait();
        if (n_row > 0) q.fill(d_mean, 0.0f, n_row).wait();
        if (n_row > 0) q.fill(d_rstd, 0.0f, n_row).wait();
        layernorm_forward(q, k, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        q.wait();
        if (n_inp > 0) q.memcpy(out, d_out, sizeof(float) * n_inp).wait();
        std::fwrite(out, sizeof(float), n_inp, fout);
        if (n_row > 0) q.memcpy(mean, d_mean, sizeof(float) * n_row).wait();
        std::fwrite(mean, sizeof(float), n_row, fout);
        if (n_row > 0) q.memcpy(rstd, d_rstd, sizeof(float) * n_row).wait();
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
