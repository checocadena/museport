/*
Inference for Llama-2 Transformer model in pure SYCL 2020.
Port of llama2.cu (CUDA) to idiomatic SYCL for Intel oneAPI DPC++.
*/
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <algorithm>

// ----------------------------------------------------------------------------
// Device kernels (1-D work, fp16 storage, fp32 accumulation)

static inline int divUp(int a, int b) { return (a - 1) / b + 1; }

void kernel_add(sycl::queue &q, sycl::half *dest, sycl::half *src, int size) {
  if (size <= 0) return;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
                 [=](sycl::id<1> i) {
                   float v = static_cast<float>(dest[i]) + static_cast<float>(src[i]);
                   dest[i] = sycl::half(v);
                 })
      .wait();
}

void kernel_f32_to_f16(sycl::queue &q, sycl::half *out, const float *in, int n) {
  if (n <= 0) return;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
    out[i] = sycl::half(in[i]);
  }).wait();
}

void kernel_f16_to_f32(sycl::queue &q, float *out, const sycl::half *in, int n) {
  if (n <= 0) return;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
    out[i] = static_cast<float>(in[i]);
  }).wait();
}

void kernel_silu_mul(sycl::queue &q, sycl::half *dest, sycl::half *src, int size) {
  if (size <= 0) return;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> i) {
    float v = static_cast<float>(dest[i]);
    v *= 1.0f / (1.0f + sycl::exp(-v));
    v *= static_cast<float>(src[i]);
    dest[i] = sycl::half(v);
  }).wait();
}

// RMSNorm with a single work-group + group reduction. Correctly accumulates
// sum of squares in fp32 (for dim <= 1024 this matches the CUDA original).
void kernel_rmsnorm(sycl::queue &q, sycl::half *o, sycl::half *x,
                    sycl::half *weight, int size) {
  constexpr int WG = 256;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> scratch(sycl::range<1>(1), h);
     h.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(WG), sycl::range<1>(WG)),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           int lid = static_cast<int>(it.get_local_id(0));
           int ls = static_cast<int>(it.get_local_range(0));
           float part = 0.0f;
           for (int i = lid; i < size; i += ls) {
             float v = static_cast<float>(x[i]);
             part += v * v;
           }
           float sum = sycl::reduce_over_group(g, part, sycl::plus<float>{});
           if (lid == 0) {
             sum /= static_cast<float>(size);
             sum += 1e-5f;
             sum = 1.0f / sycl::sqrt(sum);
             scratch[0] = sum;
           }
           sycl::group_barrier(g);
           float inv = scratch[0];
           for (int i = lid; i < size; i += ls) {
             float v = static_cast<float>(x[i]);
             v *= inv * static_cast<float>(weight[i]);
             o[i] = sycl::half(v);
           }
         });
   }).wait();
}

// MatVec: one work-group per output row, group reduction over the row.
void kernel_matvec(sycl::queue &q, sycl::half *out, sycl::half *in,
                   sycl::half *w, int n, int d) {
  if (d <= 0 || n <= 0) return;
  constexpr int WG = 64;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(d) * WG),
                           sycl::range<1>(WG)),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           int row = static_cast<int>(it.get_group(0));
           int lid = static_cast<int>(it.get_local_id(0));
           int ls = static_cast<int>(it.get_local_range(0));
           float part = 0.0f;
           const sycl::half *wrow = w + static_cast<size_t>(row) * static_cast<size_t>(n);
           for (int j = lid; j < n; j += ls)
             part += static_cast<float>(wrow[j]) * static_cast<float>(in[j]);
           float sum = sycl::reduce_over_group(g, part, sycl::plus<float>{});
           if (lid == 0) out[row] = sycl::half(sum);
         });
   }).wait();
}

// RoPE rotation: 1-D range over (heads * head_size/2) pairs.
void kernel_rope(sycl::queue &q, sycl::half *sq, sycl::half *sk,
                 sycl::half *f_real, sycl::half *f_imag, int num_heads,
                 int head_size) {
  int pairs = num_heads * (head_size / 2);
  if (pairs <= 0) return;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(pairs)), [=](sycl::id<1> id) {
     int p = static_cast<int>(id[0]);
     int h = p / (head_size / 2);
     int k = p % (head_size / 2);
     sycl::half *qq = sq + static_cast<size_t>(h) * static_cast<size_t>(head_size);
     sycl::half *kk = sk + static_cast<size_t>(h) * static_cast<size_t>(head_size);
     int i = k * 2;
     float q0 = static_cast<float>(qq[i]);
     float q1 = static_cast<float>(qq[i + 1]);
     float k0 = static_cast<float>(kk[i]);
     float k1 = static_cast<float>(kk[i + 1]);
     float fcr = static_cast<float>(f_real[k]);
     float fci = static_cast<float>(f_imag[k]);
     qq[i] = sycl::half(q0 * fcr - q1 * fci);
     qq[i + 1] = sycl::half(q0 * fci + q1 * fcr);
     kk[i] = sycl::half(k0 * fcr - k1 * fci);
     kk[i + 1] = sycl::half(k0 * fci + k1 * fcr);
   }).wait();
}

// Multi-head attention: one work-group per head, local scores + group reductions.
void kernel_attention(sycl::queue &q, sycl::half *output, const sycl::half *sq,
                      const sycl::half *key_cache, const sycl::half *value_cache,
                      int num_heads, int head_size, int loff, int seq_len) {
  if (num_heads <= 0 || seq_len <= 0) return;
  int dim = head_size * num_heads;
  constexpr int WG = 128;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> att(sycl::range<1>(static_cast<size_t>(seq_len)), h);
     sycl::local_accessor<float, 1> red(sycl::range<1>(1), h);
     h.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_heads) * WG),
                           sycl::range<1>(WG)),
         [=](sycl::nd_item<1> it) {
           sycl::group<1> g = it.get_group();
           int hd = static_cast<int>(it.get_group(0));
           int lid = static_cast<int>(it.get_local_id(0));
           int ls = static_cast<int>(it.get_local_range(0));
           const sycl::half *qq = sq + static_cast<size_t>(hd) * static_cast<size_t>(head_size);
           float inv_sqrt = 1.0f / sycl::sqrt(static_cast<float>(head_size));
           for (int t = lid; t < seq_len; t += ls) {
             const sycl::half *k =
                 key_cache + static_cast<size_t>(loff) +
                 static_cast<size_t>(t) * static_cast<size_t>(dim) +
                 static_cast<size_t>(hd) * static_cast<size_t>(head_size);
             float s = 0.0f;
             for (int i = 0; i < head_size; i++)
               s += static_cast<float>(qq[i]) * static_cast<float>(k[i]);
             att[t] = s * inv_sqrt;
           }
           sycl::group_barrier(g);
           // max for numerical stability
           float local_max = -1e30f;
           for (int i = lid; i < seq_len; i += ls)
             if (att[i] > local_max) local_max = att[i];
           float mx = sycl::reduce_over_group(g, local_max, sycl::maximum<float>{});
           if (lid == 0) red[0] = mx;
           sycl::group_barrier(g);
           mx = red[0];
           // exp + sum
           float psum = 0.0f;
           for (int i = lid; i < seq_len; i += ls) {
             float e = sycl::exp(att[i] - mx);
             att[i] = e;
             psum += e;
           }
           float tot = sycl::reduce_over_group(g, psum, sycl::plus<float>{});
           if (lid == 0) red[0] = tot;
           sycl::group_barrier(g);
           tot = red[0];
           for (int i = lid; i < seq_len; i += ls) att[i] /= tot;
           sycl::group_barrier(g);
           // weighted sum of values
           for (int i = lid; i < head_size; i += ls) {
             float v = 0.0f;
             for (int t = 0; t < seq_len; t++)
               v += att[t] *
                    static_cast<float>(value_cache[static_cast<size_t>(loff) +
                                                   static_cast<size_t>(t) * static_cast<size_t>(dim) +
                                                   static_cast<size_t>(hd) * static_cast<size_t>(head_size) +
                                                   static_cast<size_t>(i)]);
             output[static_cast<size_t>(hd) * static_cast<size_t>(head_size) + static_cast<size_t>(i)] =
                 sycl::half(v);
           }
         });
   }).wait();
}

// ----------------------------------------------------------------------------
// Transformer structs and memory management

typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
} Config;

typedef struct {
  sycl::half *token_embedding_table;
  sycl::half *rms_att_weight;
  sycl::half *rms_ffn_weight;
  sycl::half *wq;
  sycl::half *wk;
  sycl::half *wv;
  sycl::half *wo;
  sycl::half *w1;
  sycl::half *w2;
  sycl::half *w3;
  sycl::half *rms_final_weight;
  sycl::half *freq_cis_real;
  sycl::half *freq_cis_imag;
  sycl::half *wcls;
} TransformerWeights;

typedef struct {
  sycl::half *x;
  sycl::half *xb;
  sycl::half *xb2;
  sycl::half *hb;
  sycl::half *hb2;
  sycl::half *q;
  sycl::half *k;
  sycl::half *v;
  sycl::half *logits_gpu;
  float *logits_temp; // device-side float scratch
  float *logits;      // host logits
  sycl::half *key_cache;
  sycl::half *value_cache;
} RunState;

static void malloc_run_state(sycl::queue &q, RunState *s, Config *p) {
  s->x = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->xb = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->xb2 = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->hb = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->hidden_dim), q);
  s->hb2 = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->hidden_dim), q);
  s->q = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->k = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->v = sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  s->logits_gpu =
      sycl::malloc_device<sycl::half>(static_cast<size_t>(p->vocab_size), q);
  s->key_cache = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->seq_len) *
          static_cast<size_t>(p->dim),
      q);
  s->value_cache = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->seq_len) *
          static_cast<size_t>(p->dim),
      q);
  s->logits_temp =
      sycl::malloc_device<float>(static_cast<size_t>(p->vocab_size), q);
  s->logits = (float *)malloc(static_cast<size_t>(p->vocab_size) * sizeof(float));
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
      !s->v || !s->logits || !s->key_cache || !s->value_cache ||
      !s->logits_gpu || !s->logits_temp) {
    printf("malloc failed!\n");
    exit(1);
  }
}

static void free_run_state(sycl::queue &q, RunState *s) {
  sycl::free(s->x, q);
  sycl::free(s->xb, q);
  sycl::free(s->xb2, q);
  sycl::free(s->hb, q);
  sycl::free(s->hb2, q);
  sycl::free(s->q, q);
  sycl::free(s->k, q);
  sycl::free(s->v, q);
  sycl::free(s->logits_gpu, q);
  sycl::free(s->logits_temp, q);
  free(s->logits);
  sycl::free(s->key_cache, q);
  sycl::free(s->value_cache, q);
}

static void malloc_weights(sycl::queue &q, TransformerWeights *w, Config *p,
                           int shared_weights) {
  w->token_embedding_table = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->vocab_size) * static_cast<size_t>(p->dim), q);
  w->rms_att_weight = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim), q);
  w->rms_ffn_weight = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim), q);
  w->wq = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim) * static_cast<size_t>(p->dim), q);
  w->wk = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim) * static_cast<size_t>(p->dim), q);
  w->wv = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim) * static_cast<size_t>(p->dim), q);
  w->wo = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim) * static_cast<size_t>(p->dim), q);
  w->w1 = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->hidden_dim) *
          static_cast<size_t>(p->dim),
      q);
  w->w2 = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->dim) *
          static_cast<size_t>(p->hidden_dim),
      q);
  w->w3 = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->n_layers) * static_cast<size_t>(p->hidden_dim) *
          static_cast<size_t>(p->dim),
      q);
  w->rms_final_weight =
      sycl::malloc_device<sycl::half>(static_cast<size_t>(p->dim), q);
  int head_size = p->dim / p->n_heads;
  w->freq_cis_real = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->seq_len) * static_cast<size_t>(head_size) / 2, q);
  w->freq_cis_imag = sycl::malloc_device<sycl::half>(
      static_cast<size_t>(p->seq_len) * static_cast<size_t>(head_size) / 2, q);
  if (shared_weights)
    w->wcls = w->token_embedding_table;
  else
    w->wcls = sycl::malloc_device<sycl::half>(
        static_cast<size_t>(p->vocab_size) * static_cast<size_t>(p->dim), q);
  if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight ||
      !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
      !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag ||
      !w->wcls) {
    printf("malloc failed!\n");
    exit(1);
  }
}

static void free_weights(sycl::queue &q, TransformerWeights *w, int shared_weights) {
  sycl::free(w->token_embedding_table, q);
  sycl::free(w->rms_att_weight, q);
  sycl::free(w->rms_ffn_weight, q);
  sycl::free(w->wq, q);
  sycl::free(w->wk, q);
  sycl::free(w->wv, q);
  sycl::free(w->wo, q);
  sycl::free(w->w1, q);
  sycl::free(w->w2, q);
  sycl::free(w->w3, q);
  sycl::free(w->rms_final_weight, q);
  sycl::free(w->freq_cis_real, q);
  sycl::free(w->freq_cis_imag, q);
  if (!shared_weights) sycl::free(w->wcls, q);
}

static int uploadWeight(sycl::queue &q, sycl::half *w, int elements, FILE *f,
                        float *scratchCpu, float *scratchGpu) {
  int count = static_cast<int>(fread(scratchCpu, sizeof(float), static_cast<size_t>(elements), f));
  if (count != elements) return 1;
  q.memcpy(scratchGpu, scratchCpu, sizeof(float) * static_cast<size_t>(elements)).wait();
  kernel_f32_to_f16(q, w, scratchGpu, elements);
  return 0;
}

static int checkpoint_init_weights(sycl::queue &q, TransformerWeights *w,
                                   Config *p, FILE *f, int shared_weights) {
  size_t scratch_size =
      static_cast<size_t>(p->n_layers) *
      static_cast<size_t>(std::max(p->dim, p->hidden_dim)) * static_cast<size_t>(p->dim);
  scratch_size =
      std::max(static_cast<size_t>(p->vocab_size) * static_cast<size_t>(p->dim), scratch_size);
  float *scratchCpu = (float *)malloc(scratch_size * sizeof(float));
  float *scratchGpu = sycl::malloc_device<float>(scratch_size, q);
  if (!scratchCpu || !scratchGpu) {
    printf("malloc failed!\n");
    exit(1);
  }
  int rc = 0;
  rc |= uploadWeight(q, w->token_embedding_table, p->vocab_size * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->rms_att_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->wq, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->wk, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->wv, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->wo, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->rms_ffn_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->w1, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->w2, p->n_layers * p->hidden_dim * p->dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->w3, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->rms_final_weight, p->dim, f, scratchCpu, scratchGpu);
  int head_size = p->dim / p->n_heads;
  rc |= uploadWeight(q, w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu);
  rc |= uploadWeight(q, w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu);
  if (!shared_weights)
    rc |= uploadWeight(q, w->wcls, p->vocab_size * p->dim, f, scratchCpu, scratchGpu);
  sycl::free(scratchGpu, q);
  free(scratchCpu);
  return rc;
}

// ----------------------------------------------------------------------------
// neural net blocks

static void accum(sycl::queue &q, sycl::half *a, sycl::half *b, int size) {
  kernel_add(q, a, b, size);
}

static void rmsnorm(sycl::queue &q, sycl::half *o, sycl::half *x,
                    sycl::half *weight, int size) {
  kernel_rmsnorm(q, o, x, weight, size);
}

static void softmax_host(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++)
    if (x[i] > max_val) max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul(sycl::queue &q, sycl::half *xout, sycl::half *x,
                   sycl::half *w, int n, int d) {
  kernel_matvec(q, xout, x, w, n, d);
}

static void rope(sycl::queue &q, sycl::half *qq, sycl::half *k,
                 sycl::half *f_real, sycl::half *f_imag, int num_heads,
                 int head_size) {
  kernel_rope(q, qq, k, f_real, f_imag, num_heads, head_size);
}

static void attention(sycl::queue &q, sycl::half *output, sycl::half *qq,
                      sycl::half *key_cache, sycl::half *value_cache,
                      int num_heads, int head_size, int loff, int seq_len) {
  kernel_attention(q, output, qq, key_cache, value_cache, num_heads,
                   head_size, loff, seq_len);
}

static void siluMul(sycl::queue &q, sycl::half *hb, sycl::half *hb2, int size) {
  kernel_silu_mul(q, hb, hb2, size);
}

static void transformer(sycl::queue &q, int token, int pos, Config *p,
                        RunState *s, TransformerWeights *w) {
  sycl::half *x = s->x;
  int dim = p->dim;
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  sycl::half *content_row = &(w->token_embedding_table[static_cast<size_t>(token) * static_cast<size_t>(dim)]);
  q.memcpy(x, content_row, sizeof(sycl::half) * static_cast<size_t>(dim)).wait();

  sycl::half *freq_cis_real_row =
      w->freq_cis_real + static_cast<size_t>(pos) * static_cast<size_t>(head_size) / 2;
  sycl::half *freq_cis_imag_row =
      w->freq_cis_imag + static_cast<size_t>(pos) * static_cast<size_t>(head_size) / 2;

  for (int l = 0; l < p->n_layers; l++) {
    rmsnorm(q, s->xb, x, w->rms_att_weight + static_cast<size_t>(l) * static_cast<size_t>(dim), dim);

    matmul(q, s->q, s->xb, w->wq + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(dim), dim, dim);
    matmul(q, s->k, s->xb, w->wk + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(dim), dim, dim);
    matmul(q, s->v, s->xb, w->wv + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(dim), dim, dim);

    rope(q, s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);

    int loff = l * p->seq_len * dim;
    sycl::half *key_cache_row =
        s->key_cache + static_cast<size_t>(loff) + static_cast<size_t>(pos) * static_cast<size_t>(dim);
    sycl::half *value_cache_row =
        s->value_cache + static_cast<size_t>(loff) + static_cast<size_t>(pos) * static_cast<size_t>(dim);
    q.memcpy(key_cache_row, s->k, sizeof(sycl::half) * static_cast<size_t>(dim)).wait();
    q.memcpy(value_cache_row, s->v, sizeof(sycl::half) * static_cast<size_t>(dim)).wait();

    attention(q, s->xb, s->q, s->key_cache, s->value_cache, p->n_heads,
              head_size, loff, pos + 1);

    matmul(q, s->xb2, s->xb, w->wo + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(dim), dim, dim);

    accum(q, x, s->xb2, dim);

    rmsnorm(q, s->xb, x, w->rms_ffn_weight + static_cast<size_t>(l) * static_cast<size_t>(dim), dim);

    matmul(q, s->hb, s->xb, w->w1 + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(hidden_dim),
           dim, hidden_dim);
    matmul(q, s->hb2, s->xb, w->w3 + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(hidden_dim),
           dim, hidden_dim);

    siluMul(q, s->hb, s->hb2, hidden_dim);

    matmul(q, s->xb, s->hb, w->w2 + static_cast<size_t>(l) * static_cast<size_t>(dim) * static_cast<size_t>(hidden_dim),
           hidden_dim, dim);

    accum(q, x, s->xb, dim);
  }

  rmsnorm(q, x, x, w->rms_final_weight, dim);

  matmul(q, s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

  kernel_f16_to_f32(q, s->logits_temp, s->logits_gpu, p->vocab_size);
  q.memcpy(s->logits, s->logits_temp,
           sizeof(float) * static_cast<size_t>(p->vocab_size)).wait();
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer

static int str_lookup(char *str, char **vocab, int vocab_size) {
  for (int i = 0; i < vocab_size; i++) {
    if (strcmp(str, vocab[i]) == 0) return i;
  }
  return -1;
}

static void bpe_encode(char *text, char **vocab, float *vocab_scores,
                       int vocab_size, unsigned int max_token_length,
                       int *tokens, int *n_tokens) {
  char *str_buffer = (char *)malloc((max_token_length * 2 + 1) * sizeof(char));
  *n_tokens = 0;
  for (char *c = text; *c != '\0'; c++) {
    sprintf(str_buffer, "%c", *c);
    int id = str_lookup(str_buffer, vocab, vocab_size);
    if (id == -1) {
      printf("not good\n");
      exit(1);
    }
    tokens[*n_tokens] = id;
    (*n_tokens)++;
  }
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    for (int i = 0; i < (*n_tokens - 1); i++) {
      sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, vocab, vocab_size);
      if (id != -1 && vocab_scores[id] > best_score) {
        best_score = vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_idx == -1) break;
    tokens[best_idx] = best_id;
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
    (*n_tokens)--;
  }
  free(str_buffer);
}

// ----------------------------------------------------------------------------
// utilities

static long time_in_ms() {
  struct timespec time;
  timespec_get(&time, TIME_UTC);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static unsigned long long rng_seed;
static unsigned int random_u32() {
  rng_seed ^= rng_seed >> 12;
  rng_seed ^= rng_seed << 25;
  rng_seed ^= rng_seed >> 27;
  return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32() { return (random_u32() >> 8) / 16777216.0f; }

static int sample(float *probabilities, int n) {
  float r = random_f32();
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (r < cdf) return i;
  }
  return n - 1;
}

static int argmax(float *v, int n) {
  int max_i = 0;
  float max_p = v[0];
  for (int i = 1; i < n; i++) {
    if (v[i] > max_p) {
      max_i = i;
      max_p = v[i];
    }
  }
  return max_i;
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  char *checkpoint = NULL;
  float temperature = 0.9f;
  int steps = 256;
  char *prompt = NULL;

  if (argc < 2) {
    printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
    return 1;
  }
  if (argc >= 2) checkpoint = argv[1];
  if (argc >= 3) temperature = static_cast<float>(atof(argv[2]));
  if (argc >= 4) steps = atoi(argv[3]);
  if (argc >= 5) prompt = argv[4];

  rng_seed = (unsigned int)time(NULL);

  sycl::queue q{sycl::property::queue::in_order{}};

  Config config;
  TransformerWeights weights;
  int shared_weights;
  {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
      printf("Couldn't open file %s\n", checkpoint);
      return 1;
    }
    if (fread(&config, sizeof(Config), 1, file) != 1) return 1;

    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\n\n",
           config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads,
           config.n_layers, config.seq_len, config.vocab_size);

    shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    malloc_weights(q, &weights, &config, shared_weights);
    if (checkpoint_init_weights(q, &weights, &config, file, shared_weights)) return 1;
    fclose(file);
  }
  if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;

  char **vocab = (char **)malloc(static_cast<size_t>(config.vocab_size) * sizeof(char *));
  float *vocab_scores = (float *)malloc(static_cast<size_t>(config.vocab_size) * sizeof(float));
  unsigned int max_token_length;
  {
    FILE *file = fopen("tokenizer.bin", "rb");
    if (!file) {
      printf("couldn't load tokenizer.bin\n");
      return 1;
    }
    if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
      printf("failed read\n");
      return 1;
    }
    int len;
    for (int i = 0; i < config.vocab_size; i++) {
      if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) {
        printf("failed read\n");
        return 1;
      }
      if (fread(&len, sizeof(int), 1, file) != 1) {
        printf("failed read\n");
        return 1;
      }
      vocab[i] = (char *)malloc(static_cast<size_t>(len) + 1);
      if (fread(vocab[i], static_cast<size_t>(len), 1, file) != 1) {
        printf("failed read\n");
        return 1;
      }
      vocab[i][len] = '\0';
    }
    fclose(file);
  }

  RunState state;
  malloc_run_state(q, &state, &config);

  int *prompt_tokens = NULL;
  int num_prompt_tokens = 0;
  if (prompt != NULL) {
    prompt_tokens = (int *)malloc(static_cast<size_t>(config.seq_len) * sizeof(int));
    bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length,
               prompt_tokens, &num_prompt_tokens);
  }

  long start = 0;
  int next;
  int token = 1;
  int pos = 0;
  printf("<s>\n");
  while (pos < steps) {
    transformer(q, token, pos, &config, &state, &weights);

    if (pos < num_prompt_tokens) {
      next = prompt_tokens[pos];
    } else {
      if (temperature == 0.0f) {
        next = argmax(state.logits, config.vocab_size);
      } else {
        for (int qq = 0; qq < config.vocab_size; qq++) state.logits[qq] /= temperature;
        softmax_host(state.logits, config.vocab_size);
        next = sample(state.logits, config.vocab_size);
      }
    }

    char *token_str =
        (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];
    printf("%s", token_str);
    fflush(stdout);

    if (next == 2) break;

    token = next;
    pos++;
    if (start == 0) start = time_in_ms();
  }

  long end = time_in_ms();
  double time = (end - start) / 1000.0;
  int timed_tokens = pos - 1;
  printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time,
         timed_tokens, time);

  free_run_state(q, &state);
  free_weights(q, &weights, shared_weights);
  for (int i = 0; i < config.vocab_size; i++) free(vocab[i]);
  free(vocab);
  free(vocab_scores);
  if (prompt_tokens != NULL) free(prompt_tokens);
  return 0;
}
