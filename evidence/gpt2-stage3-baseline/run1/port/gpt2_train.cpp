// Idiomatic SYCL 2020 port of llm.c train_gpt2_fp32.cu (fp32 GPT-2 training).
// Contract: build/candidate <debug_state.bin> <output> build/gpt2_124M.bin
// runs 10 AdamW steps on the debug batch and writes 10 float32 mean losses.
#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <cstdint>

// ---- host file helpers ----
static inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) { fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line); exit(EXIT_FAILURE); }
    return fp;
}
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)
static inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) { fprintf(stderr, "Error: File read error at %s:%d\n", file, line); exit(EXIT_FAILURE); }
}
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)
static inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) { fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line); exit(EXIT_FAILURE); }
}
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)
static inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) { fprintf(stderr, "Error: malloc failed at %s:%d\n", file, line); exit(EXIT_FAILURE); }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ---- SYCL queue (in-order so kernels + oneMath are sequenced) ----
static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

using atomic_f = sycl::atomic_ref<float, sycl::memory_order::relaxed,
    sycl::memory_scope::device, sycl::access::address_space::global_space>;

// ----------------------------------------------------------------------------
// elementwise / row kernels (all 1-D launches)
// ----------------------------------------------------------------------------

void encoder_forward(float *out, const int *inp, const float *wte, const float *wpe,
                     int B, int T, int C) {
    size_t N = (size_t)B * T * C;
    if (N == 0) return;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t c = idx % (size_t)C;
        size_t bt = idx / (size_t)C;
        size_t b = bt / (size_t)T;
        size_t t = bt % (size_t)T;
        int ix = inp[b * (size_t)T + t];
        out[idx] = wte[(size_t)ix * C + c] + wpe[t * (size_t)C + c];
    }).wait();
}

void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *inp,
                      int B, int T, int C) {
    size_t N = (size_t)B * T * C;
    if (N == 0) return;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t c = idx % (size_t)C;
        size_t bt = idx / (size_t)C;
        size_t b = bt / (size_t)T;
        size_t t = bt % (size_t)T;
        int ix = inp[b * (size_t)T + t];
        float v = dout[b * (size_t)T * C + t * (size_t)C + c];
        atomic_f(dwte[(size_t)ix * C + c]).fetch_add(v);
        atomic_f(dwpe[t * (size_t)C + c]).fetch_add(v);
    }).wait();
}

void layernorm_forward(float *out, float *mean, float *rstd,
                       float *inp, float *weight, float *bias,
                       int B, int T, int C) {
    size_t N = (size_t)B * T;
    if (N == 0) return;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t idx = id[0];
        const float *x = inp + idx * (size_t)C;
        float s = 0.0f;
        for (int i = 0; i < C; i++) s += x[i];
        float m = s / (float)C;
        if (mean != nullptr) mean[idx] = m;
        float v = 0.0f;
        for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
        float rs = sycl::rsqrt(v / (float)C + 1e-5f);
        if (rstd != nullptr) rstd[idx] = rs;
        float *o = out + idx * (size_t)C;
        for (int c = 0; c < C; c++) {
            float n = rs * (x[c] - m);
            o[c] = n * weight[c] + bias[c];
        }
    }).wait();
}

void permute_forward(float *qq, float *kk, float *vv, const float *inp,
                     int B, int N, int NH, int d) {
    size_t total = (size_t)B * NH * N * d;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t inp_idx = b * (size_t)N * 3 * NH * d + n * (size_t)3 * NH * d + nh * (size_t)d + dd;
        qq[idx] = inp[inp_idx];
        kk[idx] = inp[inp_idx + (size_t)NH * d];
        vv[idx] = inp[inp_idx + (size_t)2 * NH * d];
    }).wait();
}

void permute_backward(float *dinp, const float *dq, const float *dk, const float *dv,
                      int B, int N, int NH, int d) {
    size_t total = (size_t)B * NH * N * d;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t inp_idx = b * (size_t)N * 3 * NH * d + n * (size_t)3 * NH * d + nh * (size_t)d + dd;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + (size_t)NH * d] = dk[idx];
        dinp[inp_idx + (size_t)2 * NH * d] = dv[idx];
    }).wait();
}

void unpermute_forward(float *inp, float *out, int B, int N, int NH, int d) {
    size_t total = (size_t)B * NH * N * d;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t other_idx = b * (size_t)NH * N * d + n * (size_t)NH * d + nh * (size_t)d + dd;
        out[other_idx] = inp[idx];
    }).wait();
}

void unpermute_backward(float *dinp, const float *dout, int B, int N, int NH, int d) {
    size_t total = (size_t)B * NH * N * d;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / ((size_t)NH * N * d);
        size_t rest = idx % ((size_t)NH * N * d);
        size_t nh = rest / ((size_t)N * d);
        rest = rest % ((size_t)N * d);
        size_t n = rest / (size_t)d;
        size_t dd = rest % (size_t)d;
        size_t other_idx = b * (size_t)NH * N * d + n * (size_t)NH * d + nh * (size_t)d + dd;
        dinp[idx] = dout[other_idx];
    }).wait();
}

void softmax_forward(float *out, float inv_temperature, const float *inp, int N, int T) {
    size_t rows = (size_t)N * T;
    if (rows == 0) return;
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t own_pos = idx % (size_t)T;
        const float *x = inp + idx * (size_t)T;
        float *o = out + idx * (size_t)T;
        float maxval = -FLT_MAX;
        for (size_t i = 0; i <= own_pos; i++) maxval = sycl::fmax(maxval, x[i]);
        float sum = 0.0f;
        for (size_t i = 0; i <= own_pos; i++) sum += sycl::exp(inv_temperature * (x[i] - maxval));
        float norm = 1.0f / sum;
        for (size_t i = 0; i <= own_pos; i++) o[i] = sycl::exp(inv_temperature * (x[i] - maxval)) * norm;
        for (size_t i = own_pos + 1; i < (size_t)T; i++) o[i] = 0.0f;
    }).wait();
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
    if (N <= 0) return;
    q.parallel_for(sycl::range<1>((size_t)N), [=](sycl::id<1> id) {
        size_t i = id[0];
        out[i] = inp1[i] + inp2[i];
    }).wait();
}

static constexpr float GELU_SCALE = 0.7978845608028654f; // sqrt(2/pi)

void gelu_forward(float *out, const float *inp, int N) {
    if (N <= 0) return;
    q.parallel_for(sycl::range<1>((size_t)N), [=](sycl::id<1> id) {
        size_t i = id[0];
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + sycl::tanh(GELU_SCALE * (xi + cube)));
    }).wait();
}

void gelu_backward(float *dinp, const float *inp, const float *dout, const int N) {
    if (N <= 0) return;
    q.parallel_for(sycl::range<1>((size_t)N), [=](sycl::id<1> id) {
        size_t i = id[0];
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALE * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float cosh_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (cosh_out * cosh_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALE * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }).wait();
}

void matmul_backward_bias(float *dbias, const float *dout, int B, int T, int OC) {
    if (OC <= 0) return;
    size_t rows = (size_t)B * T;
    q.parallel_for(sycl::range<1>((size_t)OC), [=](sycl::id<1> id) {
        size_t j = id[0];
        float s = 0.0f;
        for (size_t r = 0; r < rows; r++) s += dout[r * (size_t)OC + j];
        dbias[j] += s;
    }).wait();
}

void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        const float *dout, const float *inp, const float *weight,
                        const float *mean, const float *rstd,
                        int B, int T, int C) {
    size_t N = (size_t)B * T;
    if (N == 0) return;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t idx = id[0];
        size_t b = idx / (size_t)T;
        size_t t = idx % (size_t)T;
        const float *dout_bt = dout + (b * (size_t)T + t) * (size_t)C;
        const float *inp_bt = inp + (b * (size_t)T + t) * (size_t)C;
        float *dinp_bt = dinp + (b * (size_t)T + t) * (size_t)C;
        float mean_bt = mean[b * (size_t)T + t];
        float rstd_bt = rstd[b * (size_t)T + t];
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean /= (float)C;
        dnorm_norm_mean /= (float)C;
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            atomic_f(dbias[i]).fetch_add(dout_bt[i]);
            atomic_f(dweight[i]).fetch_add(norm_bti * dout_bt[i]);
            float dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
            dval *= rstd_bt;
            dinp_bt[i] += dval;
        }
    }).wait();
}

void softmax_bwd_rows(float *dpreatt, const float *datt, const float *att,
                      size_t rows, int T, float scale) {
    if (rows == 0) return;
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
        size_t r = id[0];
        size_t bh = r / (size_t)T;
        size_t t = r % (size_t)T;
        const float *att_row = att + (bh * (size_t)T + t) * (size_t)T;
        const float *datt_row = datt + (bh * (size_t)T + t) * (size_t)T;
        float *dpre_row = dpreatt + (bh * (size_t)T + t) * (size_t)T;
        float local_sum = 0.0f;
        for (int t2 = 0; t2 <= (int)t; t2++) local_sum += att_row[t2] * datt_row[t2];
        for (int t3 = 0; t3 <= (int)t; t3++) dpre_row[t3] = scale * att_row[t3] * (datt_row[t3] - local_sum);
        for (int t3 = (int)t + 1; t3 < T; t3++) dpre_row[t3] = 0.0f;
    }).wait();
}

void adamw_update(float *params_memory, float *grads_memory, float *m_memory, float *v_memory,
                  size_t num_parameters, float learning_rate, float beta1, float beta2,
                  float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    if (num_parameters == 0) return;
    q.parallel_for(sycl::range<1>(num_parameters), [=](sycl::id<1> id) {
        size_t i = id[0];
        float grad = grads_memory[i];
        // same two-fma lerp as the CUDA reference:
        // m = lerp(grad, m, beta1), v = lerp(grad*grad, v, beta2)
        float m = sycl::fma(beta1, m_memory[i], sycl::fma(-beta1, grad, grad));
        m_memory[i] = m;
        float v = sycl::fma(beta2, v_memory[i], sycl::fma(-beta2, grad * grad, grad * grad));
        v_memory[i] = v;
        float m_hat = m / beta1_correction;
        float v_hat = v / beta2_correction;
        params_memory[i] -= learning_rate * (m_hat / (sycl::sqrt(v_hat) + eps) + weight_decay * params_memory[i]);
    }).wait();
}

void fused_classifier(float *logits, float *losses,
                      const float *dlosses, const int *targets,
                      int B, int T, int V, int P) {
    size_t N = (size_t)B * T;
    if (N == 0) return;
    float inv_bt = 1.0f / ((float)B * (float)T);
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t idx = id[0];
        int ix = targets[idx];
        float *logits_row = logits + idx * (size_t)P;
        float thread_max = -INFINITY;
        for (int i = 0; i < V; i++) thread_max = sycl::fmax(thread_max, logits_row[i]);
        float s = 0.0f;
        for (int i = 0; i < V; i++) s += sycl::exp(logits_row[i] - thread_max);
        float scale = 1.0f / s;
        float prob = sycl::exp(logits_row[ix] - thread_max) * scale;
        losses[idx] = -sycl::log(prob);
        float dloss = (dlosses != nullptr) ? dlosses[idx] : inv_bt;
        for (int i = 0; i < V; i++) {
            float p = sycl::exp(logits_row[i] - thread_max) * scale;
            float ind = (i == ix) ? 1.0f : 0.0f;
            logits_row[i] = (p - ind) * dloss;
        }
    }).wait();
}

// ----------------------------------------------------------------------------
// matmul via oneMath (column_major, mirroring cuBLAS calls 1:1)
// ----------------------------------------------------------------------------
namespace colb = oneapi::math::blas::column_major;

void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    int64_t N = (int64_t)B * T;
    if (N == 0 || OC == 0 || C == 0) return;
    float alpha = 1.0f, beta = 0.0f;
    // out^T (OC x N) = weight (OC x C) * inp^T (C x N)
    colb::gemm(q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
               (int64_t)OC, N, (int64_t)C,
               alpha, weight, (int64_t)C, inp, (int64_t)C, beta, out, (int64_t)OC);
    q.wait();
    if (bias != nullptr) {
        size_t total = (size_t)N * (size_t)OC;
        q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            size_t i = id[0];
            out[i] += bias[i % (size_t)OC];
        }).wait();
    }
}

void matmul_backward(float *dinp, float *dweight, float *dbias,
                     float *dout, float *inp, float *weight,
                     int B, int T, int C, int OC) {
    int64_t N = (int64_t)B * T;
    if (N == 0) return;
    float one = 1.0f, zero = 0.0f;
    colb::gemm(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
               (int64_t)C, N, (int64_t)OC,
               one, weight, (int64_t)C, dout, (int64_t)OC, zero, dinp, (int64_t)C);
    q.wait();
    colb::gemm(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
               (int64_t)C, (int64_t)OC, N,
               one, inp, (int64_t)C, dout, (int64_t)OC, one, dweight, (int64_t)C);
    q.wait();
    if (dbias != nullptr) {
        matmul_backward_bias(dbias, dout, B, T, OC);
    }
}

void attention_forward(float *out, float *qkvr, float *att, float *inp,
                       int B, int T, int C, int NH) {
    int HS = C / NH;
    float *qq = qkvr + 0 * (size_t)B * T * C;
    float *kk = qkvr + 1 * (size_t)B * T * C;
    float *vv = qkvr + 2 * (size_t)B * T * C;
    permute_forward(qq, kk, vv, inp, B, T, NH, HS);
    float alpha = 1.0f, beta = 0.0f;
    float *preatt = inp;
    int64_t batch = (int64_t)B * NH;
    colb::gemm_batch(q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                     (int64_t)T, (int64_t)T, (int64_t)HS,
                     alpha, kk, (int64_t)HS, (int64_t)T * HS,
                     qq, (int64_t)HS, (int64_t)T * HS,
                     beta, preatt, (int64_t)T, (int64_t)T * T, batch);
    q.wait();
    float scale = 1.0f / sqrtf((float)HS);
    softmax_forward(att, scale, preatt, B * NH, T);
    float *vaccum = inp;
    colb::gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                     (int64_t)HS, (int64_t)T, (int64_t)T,
                     alpha, vv, (int64_t)HS, (int64_t)T * HS,
                     att, (int64_t)T, (int64_t)T * T,
                     beta, vaccum, (int64_t)HS, (int64_t)T * HS, batch);
    q.wait();
    unpermute_forward(vaccum, out, B, T, NH, HS);
}

void attention_backward(float *dinp, float *dqkvr, float *dpreatt, float *datt, float *scratch,
                        const float *dout, const float *qkvr, const float *att,
                        int B, int T, int C, int NH) {
    int HS = C / NH;
    const float *qq = qkvr + 0 * (size_t)B * T * C;
    const float *kk = qkvr + 1 * (size_t)B * T * C;
    const float *vv = qkvr + 2 * (size_t)B * T * C;
    float *dq = dqkvr + 0 * (size_t)B * T * C;
    float *dk = dqkvr + 1 * (size_t)B * T * C;
    float *dv = dqkvr + 2 * (size_t)B * T * C;
    unpermute_backward(scratch, dout, B, T, NH, HS);
    float one = 1.0f, zero = 0.0f;
    int64_t batch = (int64_t)B * NH;
    colb::gemm_batch(q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                     (int64_t)T, (int64_t)T, (int64_t)HS,
                     one, vv, (int64_t)HS, (int64_t)T * HS,
                     scratch, (int64_t)HS, (int64_t)T * HS,
                     zero, datt, (int64_t)T, (int64_t)T * T, batch);
    q.wait();
    colb::gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                     (int64_t)HS, (int64_t)T, (int64_t)T,
                     one, scratch, (int64_t)HS, (int64_t)T * HS,
                     att, (int64_t)T, (int64_t)T * T,
                     zero, dv, (int64_t)HS, (int64_t)T * HS, batch);
    q.wait();
    float scale = 1.0f / sqrtf((float)HS);
    softmax_bwd_rows(dpreatt, datt, att, (size_t)B * NH * T, T, scale);
    colb::gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                     (int64_t)HS, (int64_t)T, (int64_t)T,
                     one, kk, (int64_t)HS, (int64_t)T * HS,
                     dpreatt, (int64_t)T, (int64_t)T * T,
                     zero, dq, (int64_t)HS, (int64_t)T * HS, batch);
    q.wait();
    colb::gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                     (int64_t)HS, (int64_t)T, (int64_t)T,
                     one, qq, (int64_t)HS, (int64_t)T * HS,
                     dpreatt, (int64_t)T, (int64_t)T * T,
                     zero, dk, (int64_t)HS, (int64_t)T * HS, batch);
    q.wait();
    permute_backward(dinp, dq, dk, dv, B, T, NH, HS);
}

void fused_classifier3(float *logits, float *losses, const float *dlosses,
                       const int *targets, int B, int T, int V, int P) {
    fused_classifier(logits, losses, dlosses, targets, B, T, V, P);
}

// ----------------------------------------------------------------------------
// GPT-2 model definition
// ----------------------------------------------------------------------------
typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb;
    float *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb;
} ParameterTensors;

void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size, C = config.channels, maxT = config.max_seq_len, L = config.num_layers;
    param_sizes[0] = (size_t)Vp * C;
    param_sizes[1] = (size_t)maxT * C;
    param_sizes[2] = (size_t)L * C;
    param_sizes[3] = (size_t)L * C;
    param_sizes[4] = (size_t)L * (3 * C) * C;
    param_sizes[5] = (size_t)L * (3 * C);
    param_sizes[6] = (size_t)L * C * C;
    param_sizes[7] = (size_t)L * C;
    param_sizes[8] = (size_t)L * C;
    param_sizes[9] = (size_t)L * C;
    param_sizes[10] = (size_t)L * (4 * C) * C;
    param_sizes[11] = (size_t)L * (4 * C);
    param_sizes[12] = (size_t)L * C * (4 * C);
    param_sizes[13] = (size_t)L * C;
    param_sizes[14] = (size_t)C;
    param_sizes[15] = (size_t)C;
}

float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes, int on_device) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) num_parameters += param_sizes[i];
    float *params_memory;
    if (on_device) {
        params_memory = sycl::malloc_device<float>(num_parameters, q);
        if (!params_memory) { fprintf(stderr, "sycl malloc_device failed\n"); exit(EXIT_FAILURE); }
    } else {
        params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
    }
    float **ptrs[] = { &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb };
    float *it = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) { *(ptrs[i]) = it; it += param_sizes[i]; }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float *encoded, *ln1, *ln1_mean, *ln1_rstd, *atty, *att, *attproj, *residual2;
    float *ln2, *ln2_mean, *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual3;
    float *lnf, *lnf_mean, *lnf_rstd, *losses, *qkvr, *output;
} ActivationTensors;

void fill_in_activation_sizes(size_t *act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size, L = config.num_layers, NH = config.num_heads, C = config.channels;
    act_sizes[0] = (size_t)B * T * C;
    act_sizes[1] = L * (size_t)B * T * C;
    act_sizes[2] = L * (size_t)B * T;
    act_sizes[3] = L * (size_t)B * T;
    act_sizes[4] = L * (size_t)B * T * C;
    act_sizes[5] = L * (size_t)B * NH * T * T;
    act_sizes[6] = L * (size_t)B * T * C;
    act_sizes[7] = L * (size_t)B * T * C;
    act_sizes[8] = L * (size_t)B * T * C;
    act_sizes[9] = L * (size_t)B * T;
    act_sizes[10] = L * (size_t)B * T;
    act_sizes[11] = L * (size_t)B * T * 4 * C;
    act_sizes[12] = L * (size_t)B * T * 4 * C;
    act_sizes[13] = L * (size_t)B * T * C;
    act_sizes[14] = L * (size_t)B * T * C;
    act_sizes[15] = (size_t)B * T * C;
    act_sizes[16] = (size_t)B * T;
    act_sizes[17] = (size_t)B * T;
    act_sizes[18] = (size_t)B * T;
    act_sizes[19] = L * (size_t)B * T * 3 * C;
    size_t m1 = 3 * C, m2 = NH * (size_t)T;
    size_t m = m1 > m2 ? m1 : m2; m = m > Vp ? m : Vp;
    act_sizes[20] = (size_t)B * T * m;
}

#define NUM_BACKWARD_TENSORS 3
typedef struct { float *bt4c, *preatt, *residual3; } GradActTensors;

void fill_in_grad_act_sizes(size_t *act_sizes, int B, int T, GPT2Config config) {
    size_t NH = config.num_heads, C = config.channels;
    act_sizes[0] = (size_t)B * T * 4 * C;
    act_sizes[1] = (size_t)B * NH * T * T;
    act_sizes[2] = (size_t)B * T * C;
}

float *malloc_and_point(float **targets[], const size_t *act_sizes, int n) {
    size_t num = 0;
    for (int i = 0; i < n; i++) num += act_sizes[i];
    float *mem = sycl::malloc_device<float>(num, q);
    if (!mem) { fprintf(stderr, "sycl malloc_device failed\n"); exit(EXIT_FAILURE); }
    float *it = mem;
    for (int i = 0; i < n; i++) { *(targets[i]) = it; it += act_sizes[i]; }
    return mem;
}

float *malloc_and_point_activations(ActivationTensors *acts, const size_t *act_sizes) {
    float **ptrs[] = { &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float *malloc_and_point_backward(GradActTensors *acts, const size_t *act_sizes) {
    float **ptrs[] = { &acts->bt4c, &acts->preatt, &acts->residual3 };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    ParameterTensors grads;
    float *grads_memory;
    float *m_memory;
    float *v_memory;
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float *grads_acts_memory;
    int batch_size, seq_len;
    int *inputs, *targets;
    float mean_loss;
    float *cpu_losses;
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) { fprintf(stderr, "Bad version in model file\n"); exit(EXIT_FAILURE); }
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];
    fill_in_parameter_sizes(model->param_sizes, model->config);
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) num_parameters += model->param_sizes[i];
    model->num_parameters = num_parameters;
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);
    float *cpu = (float *)mallocCheck(num_parameters * sizeof(float));
    freadCheck(cpu, sizeof(float), num_parameters, model_file);
    q.memcpy(model->params_memory, cpu, num_parameters * sizeof(float)).wait();
    free(cpu);
    fcloseCheck(model_file);
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
    if (model->params_memory == NULL) { printf("Error: model was not initialized properly.\n"); exit(EXIT_FAILURE); }
    int V = model->config.vocab_size, Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers, NH = model->config.num_heads, C = model->config.channels;
    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) assert(0 <= targets[i] && targets[i] < V);
    }
    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) num_activations += model->act_sizes[i];
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20);
        model->inputs = sycl::malloc_device<int>((size_t)B * T, q);
        model->targets = sycl::malloc_device<int>((size_t)B * T, q);
        model->cpu_losses = sycl::malloc_host<float>((size_t)B * T, q);
    } else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }
    q.memcpy(model->inputs, inputs, (size_t)B * T * sizeof(int)).wait();
    if (targets != NULL) q.memcpy(model->targets, targets, (size_t)B * T * sizeof(int)).wait();
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float *residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
    for (int l = 0; l < L; l++) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;
        float *l_ln1w = params.ln1w + (size_t)l * C;
        float *l_ln1b = params.ln1b + (size_t)l * C;
        float *l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float *l_qkvb = params.qkvb + (size_t)l * 3 * C;
        float *l_attprojw = params.attprojw + (size_t)l * C * C;
        float *l_attprojb = params.attprojb + (size_t)l * C;
        float *l_ln2w = params.ln2w + (size_t)l * C;
        float *l_ln2b = params.ln2b + (size_t)l * C;
        float *l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float *l_fcb = params.fcb + (size_t)l * 4 * C;
        float *l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + (size_t)l * C;
        float *l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float *l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float *l_atty = acts.atty + (size_t)l * B * T * C;
        float *l_att = acts.att + (size_t)l * B * NH * T * T;
        float *l_attproj = acts.attproj + (size_t)l * B * T * C;
        float *l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float *l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float *l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + (size_t)l * B * T * C;
        float *l_residual3 = acts.residual3 + (size_t)l * B * T * C;
        float *scratch = acts.output;
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }
    residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
    if (targets != NULL) {
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        q.memcpy(model->cpu_losses, acts.losses, (size_t)B * T * sizeof(float)).wait();
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++) mean_loss += model->cpu_losses[i];
        mean_loss /= (float)(B * T);
        model->mean_loss = mean_loss;
    } else {
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL) q.memset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float)).wait();
    if (model->grads_memory != NULL) q.memset(model->grads_memory, 0, model->num_parameters * sizeof(float)).wait();
}

void gpt2_backward(GPT2 *model) {
    if (model->mean_loss == -1.0f) { printf("Error: must forward with targets before backward\n"); exit(EXIT_FAILURE); }
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPT2Config cfg = model->config;
        cfg.num_layers = 1;
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) model->num_grad_acts += bw_act_sizes[i];
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        gpt2_zero_grad(model);
    }
    int B = model->batch_size, T = model->seq_len;
    int Vp = model->config.padded_vocab_size, L = model->config.num_layers;
    int NH = model->config.num_heads, C = model->config.channels;
    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    float *residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    float *dresidual = grads_acts.residual3;
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);
    for (int l = L - 1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;
        float *l_ln1w = params.ln1w + (size_t)l * C;
        float *l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float *l_attprojw = params.attprojw + (size_t)l * C * C;
        float *l_ln2w = params.ln2w + (size_t)l * C;
        float *l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float *l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float *dl_ln1w = grads.ln1w + (size_t)l * C;
        float *dl_ln1b = grads.ln1b + (size_t)l * C;
        float *dl_qkvw = grads.qkvw + (size_t)l * 3 * C * C;
        float *dl_qkvb = grads.qkvb + (size_t)l * 3 * C;
        float *dl_attprojw = grads.attprojw + (size_t)l * C * C;
        float *dl_attprojb = grads.attprojb + (size_t)l * C;
        float *dl_ln2w = grads.ln2w + (size_t)l * C;
        float *dl_ln2b = grads.ln2b + (size_t)l * C;
        float *dl_fcw = grads.fcw + (size_t)l * 4 * C * C;
        float *dl_fcb = grads.fcb + (size_t)l * 4 * C;
        float *dl_fcprojw = grads.fcprojw + (size_t)l * C * 4 * C;
        float *dl_fcprojb = grads.fcprojb + (size_t)l * C;
        float *l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float *l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float *l_atty = acts.atty + (size_t)l * B * T * C;
        float *l_att = acts.att + (size_t)l * B * NH * T * T;
        float *l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float *l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float *l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float *dl_btc = acts.lnf;
        float *dl_bt4c = grads_acts.bt4c;
        float *dl_preatt = grads_acts.preatt;
        float *scratch = acts.output;
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B * T * 4 * C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        float *buffer_a = l_atty;
        float *buffer_b = l_fch;
        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    if (model->m_memory == NULL) {
        model->m_memory = sycl::malloc_device<float>(model->num_parameters, q);
        model->v_memory = sycl::malloc_device<float>(model->num_parameters, q);
        q.memset(model->m_memory, 0, model->num_parameters * sizeof(float)).wait();
        q.memset(model->v_memory, 0, model->num_parameters * sizeof(float)).wait();
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_update(model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                 model->num_parameters, learning_rate, beta1, beta2,
                 beta1_correction, beta2_correction, eps, weight_decay);
}

void gpt2_free(GPT2 *model) {
    sycl::free(model->params_memory, q);
    sycl::free(model->grads_memory, q);
    sycl::free(model->m_memory, q);
    sycl::free(model->v_memory, q);
    sycl::free(model->acts_memory, q);
    sycl::free(model->grads_acts_memory, q);
    sycl::free(model->inputs, q);
    sycl::free(model->targets, q);
    sycl::free(model->cpu_losses, q);
}

// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int sh[256];
    FILE *s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int *x = (int *)mallocCheck((size_t)B * T * sizeof(int));
    int *y = (int *)mallocCheck((size_t)B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    float losses[10];
    for (int step = 0; step < 10; step++) {
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        losses[step] = model.mean_loss;
    }
    FILE *o = fopenCheck(argv[2], "wb");
    fwrite(losses, sizeof(float), 10, o);
    fcloseCheck(o);
    free(x);
    free(y);
    gpt2_free(&model);
    return 0;
}
