// Idiomatic SYCL 2020 port of llm.c train_gpt2_fp32.cu (forward pass only)
// for the museport GPT-2 stage-1 harness contract:
//   build/candidate <debug_state.bin> <output> build/gpt2_124M.bin
// writes B*T*V logits (target-free pass) then mean loss (pass with targets).
#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <vector>

static inline void die(const char *msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(EXIT_FAILURE);
}

static inline FILE *open_check(const char *path, const char *mode) {
    FILE *f = std::fopen(path, mode);
    if (!f) { std::fprintf(stderr, "failed to open %s\n", path); std::exit(EXIT_FAILURE); }
    return f;
}
static inline void read_check(void *ptr, std::size_t sz, std::size_t n, FILE *f) {
    if (std::fread(ptr, sz, n, f) != n) { std::fprintf(stderr, "read error\n"); std::exit(EXIT_FAILURE); }
}

// ---------------------------------------------------------------- model config
struct GPT2Config {
    int max_seq_len = 1024;
    int vocab_size = 50257;
    int padded_vocab_size = 50304;
    int num_layers = 12;
    int num_heads = 12;
    int channels = 768;
};

struct ParamTensors {
    float *wte = nullptr;      // (Vp, C)
    float *wpe = nullptr;      // (maxT, C)
    float *ln1w = nullptr;     // (L, C)
    float *ln1b = nullptr;
    float *qkvw = nullptr;     // (L, 3C, C)
    float *qkvb = nullptr;     // (L, 3C)
    float *attprojw = nullptr; // (L, C, C)
    float *attprojb = nullptr;
    float *ln2w = nullptr;
    float *ln2b = nullptr;
    float *fcw = nullptr;      // (L, 4C, C)
    float *fcb = nullptr;
    float *fcprojw = nullptr;  // (L, C, 4C)
    float *fcprojb = nullptr;
    float *lnfw = nullptr;     // (C)
    float *lnfb = nullptr;
};

struct GPT2 {
    GPT2Config config;
    ParamTensors params;
    float *params_memory = nullptr;
    std::size_t num_parameters = 0;
    float mean_loss = -1.0f;
};

// ---------------------------------------------------------------- SYCL helpers
// Row-major matmul: out[M,N] = inp[M,K] * weight[N,K]^T (+ bias[N]).
// Implemented with oneMath column-major GEMM viewing row-major buffers transposed.
static void matmul_forward_onemath(sycl::queue &q, float *out, const float *inp,
                                   const float *weight, const float *bias,
                                   int M, int K, int N) {
    // C_col(N,M) = A_col^T(N,K) * B_col(K,M), A=weight(KxN col), B=inp(KxM col)
    const float alpha = 1.0f, beta = 0.0f;
    std::int64_t m = N, n = M, k = K;
    std::int64_t lda = K, ldb = K, ldc = N;
    oneapi::math::blas::column_major::gemm(
        q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
        m, n, k, alpha, weight, lda, inp, ldb, beta, out, ldc);
    if (bias != nullptr) {
        std::size_t total = (std::size_t)M * (std::size_t)N;
        q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            std::size_t i = id[0];
            out[i] += bias[i % (std::size_t)N];
        }).wait();
    } else {
        q.wait();
    }
}

static void encoder_forward_sycl(sycl::queue &q, float *out, const int *inp,
                                 const float *wte, const float *wpe, int B, int T, int C) {
    std::size_t total = (std::size_t)B * (std::size_t)T * (std::size_t)C;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        std::size_t idx = id[0];
        std::size_t c = idx % (std::size_t)C;
        std::size_t bt = idx / (std::size_t)C;
        std::size_t t = bt % (std::size_t)T;
        int ix = inp[bt];
        out[idx] = wte[(std::size_t)ix * (std::size_t)C + c] + wpe[t * (std::size_t)C + c];
    }).wait();
}

static void layernorm_forward_sycl(sycl::queue &q, float *out, const float *inp,
                                   const float *weight, const float *bias,
                                   int B, int T, int C) {
    std::size_t rows = (std::size_t)B * (std::size_t)T;
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
        std::size_t r = id[0];
        const float *x = inp + r * (std::size_t)C;
        float *o = out + r * (std::size_t)C;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += x[c];
        float m = sum / (float)C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) { float d = x[c] - m; var += d * d; }
        float s = 1.0f / sycl::sqrt(var / (float)C + 1e-5f);
        for (int c = 0; c < C; ++c) {
            float nrm = s * (x[c] - m);
            o[c] = nrm * weight[c] + bias[c];
        }
    }).wait();
}

static void residual_forward_sycl(sycl::queue &q, float *out, const float *a, const float *b, std::size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        std::size_t i = id[0];
        out[i] = a[i] + b[i];
    }).wait();
}

static void gelu_forward_sycl(sycl::queue &q, float *out, const float *inp, std::size_t n) {
    constexpr float scaling = 0.7978845608028654f; // sqrt(2/pi)
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        std::size_t i = id[0];
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + sycl::tanh(scaling * (xi + cube)));
    }).wait();
}

static void permute_sycl(sycl::queue &q, float *qp, float *kp, float *vp, const float *inp,
                         int B, int T, int NH, int HS) {
    std::size_t total = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
    int NHl = NH, Tl = T, HSl = HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        std::size_t flat = id[0];
        std::size_t d_ = flat % (std::size_t)HSl; flat /= (std::size_t)HSl;
        std::size_t n = flat % (std::size_t)Tl; flat /= (std::size_t)Tl;
        std::size_t nh = flat % (std::size_t)NHl; flat /= (std::size_t)NHl;
        std::size_t b = flat;
        // inp layout (B,T,3,NH,HS): base + k*NH*HS
        std::size_t base = (((b * (std::size_t)Tl + n) * 3) * (std::size_t)NHl + nh) * (std::size_t)HSl + d_;
        std::size_t dst = (((b * (std::size_t)NHl + nh) * (std::size_t)Tl + n) * (std::size_t)HSl + d_);
        qp[dst] = inp[base + 0 * (std::size_t)NHl * (std::size_t)HSl];
        kp[dst] = inp[base + 1 * (std::size_t)NHl * (std::size_t)HSl];
        vp[dst] = inp[base + 2 * (std::size_t)NHl * (std::size_t)HSl];
    }).wait();
}

static void unpermute_sycl(sycl::queue &q, const float *inp, float *out, int B, int T, int NH, int HS) {
    std::size_t total = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)HS;
    int NHl = NH, Tl = T, HSl = HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        std::size_t flat = id[0];
        std::size_t d_ = flat % (std::size_t)HSl; flat /= (std::size_t)HSl;
        std::size_t n = flat % (std::size_t)Tl; flat /= (std::size_t)Tl;
        std::size_t nh = flat % (std::size_t)NHl; flat /= (std::size_t)NHl;
        std::size_t b = flat;
        std::size_t in_idx = (((b * (std::size_t)NHl + nh) * (std::size_t)Tl + n) * (std::size_t)HSl + d_);
        std::size_t out_idx = (((b * (std::size_t)Tl + n) * (std::size_t)NHl + nh) * (std::size_t)HSl + d_);
        out[out_idx] = inp[in_idx];
    }).wait();
}

// Causal softmax over the lower-triangular part, one work-item per query row.
// out/preatt shape (rows, T); scale = 1/sqrt(HS) fused inside exp, as in llm.c.
static void softmax_forward_sycl(sycl::queue &q, float *out, const float *preatt,
                                 float scale, int rows, int T) {
    q.parallel_for(sycl::range<1>((std::size_t)rows), [=](sycl::id<1> id) {
        std::size_t idx = id[0];
        int own = (int)(idx % (std::size_t)T);
        const float *x = preatt + idx * (std::size_t)T;
        float *o = out + idx * (std::size_t)T;
        float maxv = -FLT_MAX;
        for (int i = 0; i <= own; ++i) maxv = sycl::fmax(maxv, x[i]);
        float sum = 0.0f;
        for (int i = 0; i <= own; ++i) sum += sycl::exp(scale * (x[i] - maxv));
        float norm = 1.0f / sum;
        for (int i = 0; i <= own; ++i) o[i] = sycl::exp(scale * (x[i] - maxv)) * norm;
        for (int i = own + 1; i < T; ++i) o[i] = 0.0f;
    }).wait();
}

static void attention_forward_sycl(sycl::queue &q, float *out, float *qkvr, float *att,
                                   float *scratch, int B, int T, int C, int NH) {
    int HS = C / NH;
    float *qp = qkvr + 0 * (std::size_t)B * T * C;
    float *kp = qkvr + 1 * (std::size_t)B * T * C;
    float *vp = qkvr + 2 * (std::size_t)B * T * C;
    permute_sycl(q, qp, kp, vp, scratch, B, T, NH, HS);

    const float alpha = 1.0f, beta = 0.0f;
    float *preatt = scratch; // (B*NH, T, T)
    // preatt^T = K @ Q^T  (column-major batched gemm, as in llm.c)
    oneapi::math::blas::column_major::gemm_batch(
        q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
        T, T, HS, alpha, kp, HS, (std::int64_t)T * HS, qp, HS, (std::int64_t)T * HS,
        beta, preatt, T, (std::int64_t)T * T, (std::int64_t)B * NH);
    q.wait();

    float scale = 1.0f / sycl::sqrt((float)HS);
    softmax_forward_sycl(q, att, preatt, scale, B * NH * T, T);

    float *vaccum = scratch;
    // vaccum^T(=HSxT) = V(HSxT) @ att(TxT)
    oneapi::math::blas::column_major::gemm_batch(
        q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
        HS, T, T, alpha, vp, HS, (std::int64_t)T * HS, att, T, (std::int64_t)T * T,
        beta, vaccum, HS, (std::int64_t)T * HS, (std::int64_t)B * NH);
    q.wait();

    unpermute_sycl(q, vaccum, out, B, T, NH, HS);
}

// Cross-entropy forward: losses[bt] = -log softmax(logits[bt,:V])[target].
// Logits use stride P (padded vocab). Precise sycl::exp/log.
static void classifier_loss_sycl(sycl::queue &q, float *losses, const float *logits,
                                 const int *targets, int B, int T, int V, int P) {
    std::size_t rows = (std::size_t)B * (std::size_t)T;
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
        std::size_t idx = id[0];
        const float *row = logits + idx * (std::size_t)P;
        int ix = targets[idx];
        float maxv = -FLT_MAX;
        for (int i = 0; i < V; ++i) maxv = sycl::fmax(maxv, row[i]);
        float sum = 0.0f;
        for (int i = 0; i < V; ++i) sum += sycl::exp(row[i] - maxv);
        float prob = sycl::exp(row[ix] - maxv) / sum;
        losses[idx] = -sycl::log(prob);
    }).wait();
}

// ---------------------------------------------------------------- checkpoint
static void fill_param_sizes(std::size_t *s, const GPT2Config &c) {
    int Vp = c.padded_vocab_size, C = c.channels, maxT = c.max_seq_len, L = c.num_layers;
    s[0] = (std::size_t)Vp * C;
    s[1] = (std::size_t)maxT * C;
    s[2] = (std::size_t)L * C;
    s[3] = (std::size_t)L * C;
    s[4] = (std::size_t)L * 3 * C * C;
    s[5] = (std::size_t)L * 3 * C;
    s[6] = (std::size_t)L * C * C;
    s[7] = (std::size_t)L * C;
    s[8] = (std::size_t)L * C;
    s[9] = (std::size_t)L * C;
    s[10] = (std::size_t)L * 4 * C * C;
    s[11] = (std::size_t)L * 4 * C;
    s[12] = (std::size_t)L * C * 4 * C;
    s[13] = (std::size_t)L * C;
    s[14] = (std::size_t)C;
    s[15] = (std::size_t)C;
}

static void gpt2_build_from_checkpoint(GPT2 &model, sycl::queue &q, const char *path) {
    FILE *f = open_check(path, "rb");
    int header[256];
    read_check(header, sizeof(int), 256, f);
    if (header[0] != 20240326) die("bad magic");
    if (header[1] != 3) die("bad version");
    model.config.max_seq_len = header[2];
    model.config.vocab_size = header[3];
    model.config.num_layers = header[4];
    model.config.num_heads = header[5];
    model.config.channels = header[6];
    model.config.padded_vocab_size = header[7];
    std::size_t sizes[16];
    fill_param_sizes(sizes, model.config);
    std::size_t n = 0;
    for (int i = 0; i < 16; ++i) n += sizes[i];
    model.num_parameters = n;
    model.params_memory = sycl::malloc_device<float>(n, q);
    if (!model.params_memory) die("oom params");
    float *cpu = (float *)std::malloc(n * sizeof(float));
    if (!cpu) die("oom cpu params");
    read_check(cpu, sizeof(float), n, f);
    std::fclose(f);
    q.memcpy(model.params_memory, cpu, n * sizeof(float)).wait();
    std::free(cpu);
    float *it = model.params_memory;
    model.params.wte = it; it += sizes[0];
    model.params.wpe = it; it += sizes[1];
    model.params.ln1w = it; it += sizes[2];
    model.params.ln1b = it; it += sizes[3];
    model.params.qkvw = it; it += sizes[4];
    model.params.qkvb = it; it += sizes[5];
    model.params.attprojw = it; it += sizes[6];
    model.params.attprojb = it; it += sizes[7];
    model.params.ln2w = it; it += sizes[8];
    model.params.ln2b = it; it += sizes[9];
    model.params.fcw = it; it += sizes[10];
    model.params.fcb = it; it += sizes[11];
    model.params.fcprojw = it; it += sizes[12];
    model.params.fcprojb = it; it += sizes[13];
    model.params.lnfw = it; it += sizes[14];
    model.params.lnfb = it;
    model.mean_loss = -1.0f;
}

// ---------------------------------------------------------------- forward
struct ForwardBufs {
    float *resid = nullptr;   // B*T*C current residual
    float *tmp = nullptr;     // B*T*C (residual2)
    float *ln = nullptr;      // B*T*C
    float *scratch = nullptr; // B*T*Vp (qkv scratch + attention preatt/vaccum + logits)
    float *qkvr = nullptr;    // B*T*3C
    float *att = nullptr;     // B*NH*T*T
    float *atty = nullptr;    // B*T*C
    float *attproj = nullptr; // B*T*C
    float *fch = nullptr;     // B*T*4C
    float *fchg = nullptr;    // B*T*4C
    float *fcproj = nullptr;  // B*T*C
    float *losses = nullptr;  // B*T
    int *d_inputs = nullptr;
    int *d_targets = nullptr;
};

static void forward_alloc(sycl::queue &q, ForwardBufs &b, int B, int T, const GPT2Config &c) {
    std::size_t BTC = (std::size_t)B * T * c.channels;
    std::size_t BT3C = (std::size_t)B * T * 3 * c.channels;
    std::size_t BT4C = (std::size_t)B * T * 4 * c.channels;
    std::size_t BT = (std::size_t)B * T;
    std::size_t BTVp = BT * (std::size_t)c.padded_vocab_size;
    std::size_t ATT = (std::size_t)B * c.num_heads * T * T;
    b.resid = sycl::malloc_device<float>(BTC, q);
    b.tmp = sycl::malloc_device<float>(BTC, q);
    b.ln = sycl::malloc_device<float>(BTC, q);
    b.scratch = sycl::malloc_device<float>(BTVp, q);
    b.qkvr = sycl::malloc_device<float>(BT3C, q);
    b.att = sycl::malloc_device<float>(ATT, q);
    b.atty = sycl::malloc_device<float>(BTC, q);
    b.attproj = sycl::malloc_device<float>(BTC, q);
    b.fch = sycl::malloc_device<float>(BT4C, q);
    b.fchg = sycl::malloc_device<float>(BT4C, q);
    b.fcproj = sycl::malloc_device<float>(BTC, q);
    b.losses = sycl::malloc_device<float>(BT, q);
    b.d_inputs = sycl::malloc_device<int>(BT, q);
    b.d_targets = sycl::malloc_device<int>(BT, q);
    if (!b.resid || !b.tmp || !b.ln || !b.scratch || !b.qkvr || !b.att || !b.atty ||
        !b.attproj || !b.fch || !b.fchg || !b.fcproj || !b.losses || !b.d_inputs || !b.d_targets)
        die("oom activations");
}

static void forward_free(sycl::queue &q, ForwardBufs &b) {
    sycl::free(b.resid, q); sycl::free(b.tmp, q); sycl::free(b.ln, q);
    sycl::free(b.scratch, q); sycl::free(b.qkvr, q); sycl::free(b.att, q);
    sycl::free(b.atty, q); sycl::free(b.attproj, q); sycl::free(b.fch, q);
    sycl::free(b.fchg, q); sycl::free(b.fcproj, q); sycl::free(b.losses, q);
    sycl::free(b.d_inputs, q); sycl::free(b.d_targets, q);
}

// Runs forward; if targets_host==nullptr, mean_loss=-1 and scratch holds logits.
// Else computes mean loss (scratch still holds logits; losses separate).
static void gpt2_forward(sycl::queue &q, GPT2 &model, ForwardBufs &b,
                         const int *inputs_host, const int *targets_host, int B, int T) {
    int C = model.config.channels, L = model.config.num_layers;
    int NH = model.config.num_heads;
    int V = model.config.vocab_size, Vp = model.config.padded_vocab_size;
    std::size_t BT = (std::size_t)B * T;
    std::size_t BTC = BT * (std::size_t)C;
    std::size_t BT4C = BT * (std::size_t)4 * (std::size_t)C;

    q.memcpy(b.d_inputs, inputs_host, BT * sizeof(int)).wait();
    if (targets_host) q.memcpy(b.d_targets, targets_host, BT * sizeof(int)).wait();

    encoder_forward_sycl(q, b.resid, b.d_inputs, model.params.wte, model.params.wpe, B, T, C);

    for (int l = 0; l < L; ++l) {
        const float *ln1w = model.params.ln1w + (std::size_t)l * C;
        const float *ln1b = model.params.ln1b + (std::size_t)l * C;
        const float *qkvw = model.params.qkvw + (std::size_t)l * 3 * C * C;
        const float *qkvb = model.params.qkvb + (std::size_t)l * 3 * C;
        const float *apw = model.params.attprojw + (std::size_t)l * C * C;
        const float *apb = model.params.attprojb + (std::size_t)l * C;
        const float *ln2w = model.params.ln2w + (std::size_t)l * C;
        const float *ln2b = model.params.ln2b + (std::size_t)l * C;
        const float *fcw = model.params.fcw + (std::size_t)l * 4 * C * C;
        const float *fcb = model.params.fcb + (std::size_t)l * 4 * C;
        const float *fpw = model.params.fcprojw + (std::size_t)l * C * 4 * C;
        const float *fpb = model.params.fcprojb + (std::size_t)l * C;

        layernorm_forward_sycl(q, b.ln, b.resid, ln1w, ln1b, B, T, C);
        matmul_forward_onemath(q, b.scratch, b.ln, qkvw, qkvb, B * T, C, 3 * C);
        attention_forward_sycl(q, b.atty, b.qkvr, b.att, b.scratch, B, T, C, NH);
        matmul_forward_onemath(q, b.attproj, b.atty, apw, apb, B * T, C, C);
        residual_forward_sycl(q, b.tmp, b.resid, b.attproj, BTC);
        layernorm_forward_sycl(q, b.ln, b.tmp, ln2w, ln2b, B, T, C);
        matmul_forward_onemath(q, b.fch, b.ln, fcw, fcb, B * T, C, 4 * C);
        gelu_forward_sycl(q, b.fchg, b.fch, BT4C);
        matmul_forward_onemath(q, b.fcproj, b.fchg, fpw, fpb, B * T, 4 * C, C);
        residual_forward_sycl(q, b.resid, b.tmp, b.fcproj, BTC);
    }

    layernorm_forward_sycl(q, b.ln, b.resid, model.params.lnfw, model.params.lnfb, B, T, C);
    matmul_forward_onemath(q, b.scratch, b.ln, model.params.wte, nullptr, B * T, C, Vp);

    if (targets_host) {
        classifier_loss_sycl(q, b.losses, b.scratch, b.d_targets, B, T, V, Vp);
        std::vector<float> lh(BT);
        q.memcpy(lh.data(), b.losses, BT * sizeof(float)).wait();
        float mean_loss = 0.0f;
        for (std::size_t i = 0; i < BT; ++i) mean_loss += lh[i];
        mean_loss /= (float)BT;
        model.mean_loss = mean_loss;
    } else {
        model.mean_loss = -1.0f;
    }
}

// ---------------------------------------------------------------- contract main
int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    sycl::queue q{sycl::property::queue::in_order{}};

    GPT2 model;
    gpt2_build_from_checkpoint(model, q, argv[3]);
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;

    FILE *s = open_check(argv[1], "rb");
    int sh[256];
    read_check(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    std::vector<int> x((std::size_t)B * T), y((std::size_t)B * T);
    read_check(x.data(), sizeof(int), (std::size_t)B * T, s);
    read_check(y.data(), sizeof(int), (std::size_t)B * T, s);
    std::fclose(s);

    ForwardBufs b;
    forward_alloc(q, b, B, T, model.config);

    std::size_t BT = (std::size_t)B * T;
    std::vector<float> logits(BT * (std::size_t)Vp);

    gpt2_forward(q, model, b, x.data(), nullptr, B, T);
    q.memcpy(logits.data(), b.scratch, BT * (std::size_t)Vp * sizeof(float)).wait();

    gpt2_forward(q, model, b, x.data(), y.data(), B, T);

    FILE *o = open_check(argv[2], "wb");
    for (std::size_t bt = 0; bt < BT; ++bt) {
        if (std::fwrite(logits.data() + bt * (std::size_t)Vp, sizeof(float), (std::size_t)V, o) != (std::size_t)V) die("write");
    }
    float loss = model.mean_loss;
    if (std::fwrite(&loss, sizeof(float), 1, o) != 1) die("write loss");
    std::fclose(o);

    forward_free(q, b);
    sycl::free(model.params_memory, q);
    return 0;
}
