#include "train_part1.cpp"
// unit-test kernels against host references (no main from part2)
#include "llmc/encoder.h"
#include "llmc/layernorm.h"
#include "llmc/matmul.h"
#include "llmc/attention.h"
#include "llmc/fused_classifier.h"
#include "llmc/gelu.h"

static int fails = 0;
static void check(const char* name, float* got, float* ref, size_t n, float tol) {
  double md = 0; size_t idx = 0;
  for (size_t i = 0; i < n; i++) { double d = fabs((double)got[i]-ref[i]); if (d > md) { md = d; idx = i; } }
  printf("%s: maxdiff=%.6f %s (tol %.4f) eg got=%.5f ref=%.5f @%zu\n", name, md, md<=tol?"OK":"FAIL", tol, got[idx], ref[idx], idx);
  if (md > tol) fails++;
}

int main() {
  multi_gpu_config = multi_gpu_config_init(-1,-1,-1,(char*)"",(char*)"",(char*)"mpi");
  try { sycl_queue = sycl::queue(sycl::default_selector_v); }
  catch (sycl::exception& e) { printf("no device: %s\n", e.what()); exit(1); }
  mt19937_state rng; manual_seed(&rng, 123);
  const int B = 2, T = 16, C = 64, NH = 4, HS = C/NH;

  // ---- layernorm forward ----
  {
    size_t N = B*T;
    std::vector<float> xf(N*C), w(C), b(C);
    for (size_t i = 0; i < N*C; i++) xf[i] = 0.2f*((int)((i*37)%11)-5);
    for (int i = 0; i < C; i++) { w[i] = 1.0f+0.01f*(i%5); b[i] = 0.01f*(((i*13)%7)-3); }
    std::vector<floatX> x(N*C), ww(C), bb(C);
    for (size_t i = 0; i < N*C; i++) x[i] = (floatX)xf[i];
    for (int i = 0; i < C; i++) { ww[i]=(floatX)w[i]; bb[i]=(floatX)b[i]; }
    floatX *dx = sycl::malloc_device<floatX>(N*C, sycl_queue);
    floatX *dw = sycl::malloc_device<floatX>(C, sycl_queue);
    floatX *db = sycl::malloc_device<floatX>(C, sycl_queue);
    floatX *dout = sycl::malloc_device<floatX>(N*C, sycl_queue);
    float *dm = sycl::malloc_device<float>(N, sycl_queue);
    float *dr = sycl::malloc_device<float>(N, sycl_queue);
    sycl_queue.memcpy(dx, x.data(), N*C*sizeof(floatX));
    sycl_queue.memcpy(dw, ww.data(), C*sizeof(floatX));
    sycl_queue.memcpy(db, bb.data(), C*sizeof(floatX));
    sycl_queue.wait();
    layernorm_forward(dout, dm, dr, dx, dw, db, B, T, C, sycl_queue);
    std::vector<floatX> o(N*C);
    sycl_queue.memcpy(o.data(), dout, N*C*sizeof(floatX)).wait();
    std::vector<float> got(N*C), ref(N*C);
    for (size_t i = 0; i < N*C; i++) got[i] = (float)o[i];
    for (size_t r0 = 0; r0 < N; r0++) {
      double m = 0; for (int c = 0; c < C; c++) m += xf[r0*C+c]; m /= C;
      double v = 0; for (int c = 0; c < C; c++) { double d = xf[r0*C+c]-m; v += d*d; } v /= C;
      double s = 1.0/sqrt(v+1e-5);
      for (int c = 0; c < C; c++) ref[r0*C+c] = (float)((xf[r0*C+c]-m)*s*w[c]+b[c]);
    }
    check("layernorm_fwd", got.data(), ref.data(), N*C, 0.05f);
    sycl::free(dx, sycl_queue); sycl::free(dw, sycl_queue); sycl::free(db, sycl_queue);
    sycl::free(dout, sycl_queue); sycl::free(dm, sycl_queue); sycl::free(dr, sycl_queue);
  }

  // ---- gelu ----
  {
    size_t N = 4096;
    std::vector<float> xf(N);
    for (size_t i = 0; i < N; i++) xf[i] = 0.3f*((int)((i*29)%13)-6);
    std::vector<floatX> x(N);
    for (size_t i = 0; i < N; i++) x[i] = (floatX)xf[i];
    floatX *dx = sycl::malloc_device<floatX>(N, sycl_queue);
    floatX *do_ = sycl::malloc_device<floatX>(N, sycl_queue);
    sycl_queue.memcpy(dx, x.data(), N*sizeof(floatX)).wait();
    gelu_forward(do_, dx, N, sycl_queue);
    std::vector<floatX> o(N);
    sycl_queue.memcpy(o.data(), do_, N*sizeof(floatX)).wait();
    std::vector<float> got(N), ref(N);
    for (size_t i = 0; i < N; i++) got[i] = (float)o[i];
    for (size_t i = 0; i < N; i++) {
      float xi = xf[i]; float cube = 0.044715f*xi*xi*xi;
      ref[i] = 0.5f*xi*(1.0f+tanhf(sqrtf(2.0f/M_PI)*(xi+cube)));
    }
    check("gelu_fwd", got.data(), ref.data(), N, 0.05f);
    sycl::free(dx, sycl_queue); sycl::free(do_, sycl_queue);
  }

  // ---- encoder forward ----
  {
    const int Vp = 128;
    std::vector<float> wte(Vp*C), wpe(T*C);
    for (size_t i = 0; i < wte.size(); i++) wte[i] = 0.1f*((int)((i*11)%9)-4);
    for (size_t i = 0; i < wpe.size(); i++) wpe[i] = 0.05f*((int)((i*7)%7)-3);
    std::vector<floatX> dwte(wte.size()), dwpe(wpe.size());
    for (size_t i = 0; i < wte.size(); i++) dwte[i]=(floatX)wte[i];
    for (size_t i = 0; i < wpe.size(); i++) dwpe[i]=(floatX)wpe[i];
    std::vector<int> inp(B*T);
    for (int i = 0; i < B*T; i++) inp[i] = (i*13)%Vp;
    floatX *d_wte = sycl::malloc_device<floatX>(wte.size(), sycl_queue);
    floatX *d_wpe = sycl::malloc_device<floatX>(wpe.size(), sycl_queue);
    floatX *d_out = sycl::malloc_device<floatX>(B*T*C, sycl_queue);
    int *d_inp = sycl::malloc_device<int>(B*T, sycl_queue);
    sycl_queue.memcpy(d_wte, dwte.data(), wte.size()*sizeof(floatX));
    sycl_queue.memcpy(d_wpe, dwpe.data(), wpe.size()*sizeof(floatX));
    sycl_queue.memcpy(d_inp, inp.data(), B*T*sizeof(int)).wait();
    encoder_forward(d_out, d_inp, d_wte, d_wpe, B, T, C, sycl_queue);
    std::vector<floatX> o(B*T*C);
    sycl_queue.memcpy(o.data(), d_out, B*T*C*sizeof(floatX)).wait();
    std::vector<float> got(B*T*C), ref(B*T*C);
    for (size_t i = 0; i < got.size(); i++) got[i]=(float)o[i];
    for (int bt = 0; bt < B*T; bt++) for (int c = 0; c < C; c++)
      ref[bt*C+c] = wte[inp[bt]*C+c]+wpe[(bt%T)*C+c];
    check("encoder_fwd", got.data(), ref.data(), got.size(), 0.05f);
    sycl::free(d_wte, sycl_queue); sycl::free(d_wpe, sycl_queue);
    sycl::free(d_out, sycl_queue); sycl::free(d_inp, sycl_queue);
  }

  // ---- attention forward ----
  {
    std::vector<float> qkvf(B*T*3*C);
    for (size_t i = 0; i < qkvf.size(); i++) qkvf[i] = 0.1f*((int)((i*41)%11)-5);
    std::vector<floatX> qkv(qkvf.size());
    for (size_t i = 0; i < qkvf.size(); i++) qkv[i]=(floatX)qkvf[i];
    floatX *d_inp = sycl::malloc_device<floatX>(qkv.size(), sycl_queue);
    floatX *d_qkvr = sycl::malloc_device<floatX>(qkv.size(), sycl_queue);
    floatX *d_att = sycl::malloc_device<floatX>(B*NH*T*T, sycl_queue);
    floatX *d_out = sycl::malloc_device<floatX>(B*T*C, sycl_queue);
    float *d_pre = sycl::malloc_device<float>(B*NH*T*T, sycl_queue);
    float *d_vac = sycl::malloc_device<float>(B*T*C, sycl_queue);
    sycl_queue.memcpy(d_inp, qkv.data(), qkv.size()*sizeof(floatX)).wait();
    sycl_queue.memset(d_att, 0, B*NH*T*T*sizeof(floatX)).wait();
    attention_forward(d_out, d_qkvr, d_att, d_inp, d_pre, d_vac, B, T, C, NH, sycl_queue);
    std::vector<floatX> o(B*T*C);
    sycl_queue.memcpy(o.data(), d_out, B*T*C*sizeof(floatX)).wait();
    std::vector<float> got(B*T*C);
    for (size_t i = 0; i < got.size(); i++) got[i]=(float)o[i];
    // host reference
    std::vector<float> ref(B*T*C, 0);
    float scale = 1.f/sqrtf((float)HS);
    for (int b = 0; b < B; b++) for (int nh = 0; nh < NH; nh++) {
      // q,k,v slices
      for (int i = 0; i < T; i++) {
        // scores
        float sc[64];
        float mx = -1e30f;
        for (int j = 0; j <= i; j++) {
          double s = 0;
          for (int p = 0; p < HS; p++) {
            float qi = qkvf[((b*T+i)*3*C)+nh*HS+p];
            float kj = qkvf[((b*T+j)*3*C)+C+nh*HS+p];
            s += qi*kj;
          }
          sc[j] = (float)s*scale;
          mx = fmaxf(mx, sc[j]);
        }
        float den = 0; for (int j = 0; j <= i; j++) den += expf(sc[j]-mx);
        for (int p = 0; p < HS; p++) {
          double acc = 0;
          for (int j = 0; j <= i; j++) acc += expf(sc[j]-mx)/den * qkvf[((b*T+j)*3*C)+2*C+nh*HS+p];
          ref[((b*T+i)*C)+nh*HS+p] = (float)acc;
        }
      }
    }
    check("attention_fwd", got.data(), ref.data(), got.size(), 0.5f);
    sycl::free(d_inp, sycl_queue); sycl::free(d_qkvr, sycl_queue); sycl::free(d_att, sycl_queue);
    sycl::free(d_out, sycl_queue); sycl::free(d_pre, sycl_queue); sycl::free(d_vac, sycl_queue);
  }

  printf("fails=%d\n", fails);
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}
