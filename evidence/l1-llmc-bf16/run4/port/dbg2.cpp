#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>
#include "llmc/sycl_common.h"
#include "llmc/sycl_utils.h"
#include "llmc/onemath_common.h"
#include "llmc/encoder.h"
#include "llmc/layernorm.h"
#include "llmc/matmul.h"
#include "llmc/attention.h"
int main(){
  FILE* f = fopen("gpt2_124M_bf16.bin","rb");
  int header[256]; fread(header,sizeof(int),256,f);
  int maxT=header[2], V=header[3], L=header[4], NH=header[5], C=header[6], Vp=header[7];
  size_t wte_n=(size_t)Vp*C, wpe_n=(size_t)maxT*C, ln1_n=(size_t)L*C;
  size_t qkvw_n=(size_t)L*3*C*C, qkvb_n=(size_t)L*3*C;
  std::vector<floatX> wte(wte_n), wpe(wpe_n), ln1w(ln1_n), ln1b(ln1_n), qkvw(qkvw_n), qkvb(qkvb_n);
  fread(wte.data(),sizeof(floatX),wte_n,f); fread(wpe.data(),sizeof(floatX),wpe_n,f);
  fread(ln1w.data(),sizeof(floatX),ln1_n,f); fread(ln1b.data(),sizeof(floatX),ln1_n,f);
  fread(qkvw.data(),sizeof(floatX),qkvw_n,f); fread(qkvb.data(),sizeof(floatX),qkvb_n,f);
  fclose(f);
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s); fclose(s);
  floatX *d_wte=sycl_malloc_device<floatX>(wte_n), *d_wpe=sycl_malloc_device<floatX>(wpe_n);
  floatX *d_ln1w=sycl_malloc_device<floatX>(ln1_n), *d_ln1b=sycl_malloc_device<floatX>(ln1_n);
  floatX *d_qkvw=sycl_malloc_device<floatX>(qkvw_n), *d_qkvb=sycl_malloc_device<floatX>(qkvb_n);
  int *d_inp=sycl_malloc_device<int>(B*T);
  floatX *d_enc=sycl_malloc_device<floatX>((size_t)B*T*C), *d_ln1=sycl_malloc_device<floatX>((size_t)B*T*C);
  float *d_mean=sycl_malloc_device<float>(B*T), *d_rstd=sycl_malloc_device<float>(B*T);
  floatX *d_qkv=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_att=sycl_malloc_device<floatX>((size_t)B*NH*T*T);
  floatX *d_qkvr=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_atty=sycl_malloc_device<floatX>((size_t)B*T*C);
  sycl_memcpy_h2d(d_wte,wte.data(),wte_n*sizeof(floatX)); sycl_memcpy_h2d(d_wpe,wpe.data(),wpe_n*sizeof(floatX));
  sycl_memcpy_h2d(d_ln1w,ln1w.data(),ln1_n*sizeof(floatX)); sycl_memcpy_h2d(d_ln1b,ln1b.data(),ln1_n*sizeof(floatX));
  sycl_memcpy_h2d(d_qkvw,qkvw.data(),qkvw_n*sizeof(floatX)); sycl_memcpy_h2d(d_qkvb,qkvb.data(),qkvb_n*sizeof(floatX));
  sycl_memcpy_h2d(d_inp,x.data(),B*T*sizeof(int));
  encoder_forward(d_enc, d_inp, d_wte, d_wpe, B, T, C);
  layernorm_forward(d_ln1, d_mean, d_rstd, d_enc, d_ln1w, d_ln1b, B, T, C);
  // qkv matmul
  matmul_forward_cublaslt(d_qkv, d_ln1, d_qkvw, d_qkvb, B, T, C, 3*C);
  std::vector<floatX> hqkv(B*T*3*C);
  sycl_memcpy_d2h(hqkv.data(), d_qkv, hqkv.size()*sizeof(floatX));
  // host reference for bt=0
  std::vector<floatX> hln(B*T*C); sycl_memcpy_d2h(hln.data(), d_ln1, hln.size()*sizeof(floatX));
  printf("qkv[0..7] dev: "); for(int i=0;i<8;i++) printf("%f ",(float)hqkv[i]); printf("\n");
  printf("qkv[0..7] ref: ");
  for(int oc=0;oc<8;oc++){ double r=(float)qkvb[oc]; for(int k=0;k<C;k++) r+=(float)hln[k]*(float)qkvw[oc*C+k]; printf("%f ",(float)(floatX)r); }
  printf("\n");
  // attention
  sycl_memset_sync(d_att, 0, (size_t)B*NH*T*T*sizeof(floatX));
  attention_forward(d_atty, d_qkvr, d_att, d_qkv, B, T, C, NH);
  std::vector<floatX> hatty(B*T*C);
  sycl_memcpy_d2h(hatty.data(), d_atty, hatty.size()*sizeof(floatX));
  printf("atty[0..7] dev: "); for(int i=0;i<8;i++) printf("%f ",(float)hatty[i]); printf("\n");
  return 0;
}
