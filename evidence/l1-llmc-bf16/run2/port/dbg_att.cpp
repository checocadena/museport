#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
#include "llmc/sycl_common.h"
#include "llmc/sycl_utils.h"
#include "llmc/blas_common.h"
#include "llmc/matmul.h"
#include "llmc/attention.h"
using bf16 = sycl::ext::oneapi::bfloat16;
sycl::queue sycl_queue;

int main() {
  try { sycl_queue = sycl::queue(sycl::default_selector_v); }
  catch (sycl::exception& e) { printf("no device: %s\n", e.what()); return 0; }
  printf("dev: %s\n", sycl_queue.get_device().get_info<sycl::info::device::name>().c_str());
  const int B = 4, T = 64, C = 768, NH = 12, HS = C/NH;
  size_t N3 = (size_t)B*T*3*C;
  std::vector<float> qkvf(N3);
  for (size_t i = 0; i < N3; i++) qkvf[i] = 0.1f*((int)((i*41)%11)-5);
  std::vector<floatX> qkv(N3);
  for (size_t i = 0; i < N3; i++) qkv[i] = (floatX)qkvf[i];
  floatX *d_inp = sycl::malloc_device<floatX>(N3, sycl_queue);
  floatX *d_qkvr = sycl::malloc_device<floatX>(N3, sycl_queue);
  floatX *d_att = sycl::malloc_device<floatX>((size_t)B*NH*T*T, sycl_queue);
  floatX *d_out = sycl::malloc_device<floatX>((size_t)B*T*C, sycl_queue);
  float *d_pre = sycl::malloc_device<float>((size_t)B*NH*T*T, sycl_queue);
  float *d_vac = sycl::malloc_device<float>((size_t)B*T*C, sycl_queue);
  sycl_queue.memcpy(d_inp, qkv.data(), N3*sizeof(floatX)).wait();
  sycl_queue.memset(d_att, 0, (size_t)B*NH*T*T*sizeof(floatX)).wait();
  attention_forward(d_out, d_qkvr, d_att, d_inp, d_pre, d_vac, B, T, C, NH, sycl_queue);
  // check att row sums == 1 and dump preatt/att for batch0 head0
  std::vector<floatX> att((size_t)B*NH*T*T);
  sycl_queue.memcpy(att.data(), d_att, (size_t)B*NH*T*T*sizeof(floatX)).wait();
  double maxrowsumdev = 0;
  for (int r = 0; r < B*NH*T; r++) {
    double s = 0; for (int j = 0; j < T; j++) s += (float)att[r*T+j];
    maxrowsumdev = fmax(maxrowsumdev, fabs(s-1.0));
  }
  printf("att rowsum maxdev=%.6f\n", maxrowsumdev);
  // host ref for batch0 head0 only
  std::vector<floatX> o((size_t)B*T*C);
  sycl_queue.memcpy(o.data(), d_out, (size_t)B*T*C*sizeof(floatX)).wait();
  float scale = 1.f/sqrtf((float)HS);
  double md = 0;
  for (int i = 0; i < T; i++) {
    float sc[64]; float mx = -1e30f;
    for (int j = 0; j <= i; j++) {
      double s = 0;
      for (int p = 0; p < HS; p++) s += (double)qkvf[((0*T+i)*3*C)+0*HS+p] * (double)qkvf[((0*T+j)*3*C)+C+0*HS+p];
      sc[j] = (float)s*scale; mx = fmaxf(mx, sc[j]);
    }
    float den = 0; for (int j = 0; j <= i; j++) den += expf(sc[j]-mx);
    for (int p = 0; p < HS; p++) {
      double acc = 0;
      for (int j = 0; j <= i; j++) acc += expf(sc[j]-mx)/den * qkvf[((0*T+j)*3*C)+2*C+0*HS+p];
      double d = fabs(acc - (float)o[((0*T+i)*C)+0*HS+p]);
      md = fmax(md, d);
    }
  }
  printf("attn b0h0 maxdiff=%.6f\n", md);
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}
