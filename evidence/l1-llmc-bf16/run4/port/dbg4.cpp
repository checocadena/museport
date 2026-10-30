#include <cstdio>
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
#include "llmc/gelu.h"
int main(){
  FILE* f = fopen("gpt2_124M_bf16.bin","rb");
  int header[256]; fread(header,sizeof(int),256,f);
  int maxT=header[2], V=header[3], L=header[4], NH=header[5], C=header[6], Vp=header[7];
  size_t wte_n=(size_t)Vp*C, wpe_n=(size_t)maxT*C, ln1_n=(size_t)L*C;
  size_t qkvw_n=(size_t)L*3*C*C, qkvb_n=(size_t)L*3*C;
  size_t ap_n=(size_t)L*C*C, apb_n=(size_t)L*C, fc_n=(size_t)L*4*C*C, fcb_n=(size_t)L*4*C, fcp_n=(size_t)L*C*4*C, fcpb_n=(size_t)L*C;
  std::vector<floatX> wte(wte_n), wpe(wpe_n), ln1w(ln1_n), ln1b(ln1_n), qkvw(qkvw_n), qkvb(qkvb_n), apw(ap_n), apb(apb_n);
  std::vector<floatX> ln2w(ln1_n), ln2b(ln1_n), fcw(fc_n), fcb(fcb_n), fcpw(fcp_n), fcpb(fcpb_n);
  fread(wte.data(),sizeof(floatX),wte_n,f); fread(wpe.data(),sizeof(floatX),wpe_n,f);
  fread(ln1w.data(),sizeof(floatX),ln1_n,f); fread(ln1b.data(),sizeof(floatX),ln1_n,f);
  fread(qkvw.data(),sizeof(floatX),qkvw_n,f); fread(qkvb.data(),sizeof(floatX),qkvb_n,f);
  fread(apw.data(),sizeof(floatX),ap_n,f); fread(apb.data(),sizeof(floatX),apb_n,f);
  fread(ln2w.data(),sizeof(floatX),ln1_n,f); fread(ln2b.data(),sizeof(floatX),ln1_n,f);
  fread(fcw.data(),sizeof(floatX),fc_n,f); fread(fcb.data(),sizeof(floatX),fcb_n,f);
  fread(fcpw.data(),sizeof(floatX),fcp_n,f); fread(fcpb.data(),sizeof(floatX),fcpb_n,f);
  fclose(f);
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s); fclose(s);
  auto devx=[&](std::vector<floatX>& h){ floatX* d=sycl_malloc_device<floatX>(h.size()); sycl_memcpy_h2d(d,h.data(),h.size()*sizeof(floatX)); return d; };
  floatX *d_wte=devx(wte), *d_wpe=devx(wpe), *d_ln1w=devx(ln1w), *d_ln1b=devx(ln1b);
  floatX *d_qkvw=devx(qkvw), *d_qkvb=devx(qkvb), *d_apw=devx(apw), *d_apb=devx(apb);
  floatX *d_ln2w=devx(ln2w), *d_ln2b=devx(ln2b), *d_fcw=devx(fcw), *d_fcb=devx(fcb), *d_fcpw=devx(fcpw), *d_fcpb=devx(fcpb);
  int *d_inp=sycl_malloc_device<int>(B*T); sycl_memcpy_h2d(d_inp,x.data(),B*T*sizeof(int));
  floatX *d_enc=sycl_malloc_device<floatX>((size_t)B*T*C), *d_ln1=sycl_malloc_device<floatX>((size_t)B*T*C);
  float *d_mean=sycl_malloc_device<float>(B*T), *d_rstd=sycl_malloc_device<float>(B*T);
  floatX *d_qkv=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_att=sycl_malloc_device<floatX>((size_t)B*NH*T*T);
  floatX *d_qkvr=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_atty=sycl_malloc_device<floatX>((size_t)B*T*C);
  floatX *d_scratch=sycl_malloc_device<floatX>((size_t)B*T*4*C);
  floatX *d_res2=sycl_malloc_device<floatX>((size_t)B*T*C), *d_ln2=sycl_malloc_device<floatX>((size_t)B*T*C);
  float *d_m2=sycl_malloc_device<float>(B*T), *d_r2=sycl_malloc_device<float>(B*T);
  floatX *d_fch=sycl_malloc_device<floatX>((size_t)B*T*4*C), *d_fchg=sycl_malloc_device<floatX>((size_t)B*T*4*C);
  floatX *d_res3=sycl_malloc_device<floatX>((size_t)B*T*C);
  encoder_forward(d_enc, d_inp, d_wte, d_wpe, B, T, C);
  layernorm_forward(d_ln1, d_mean, d_rstd, d_enc, d_ln1w, d_ln1b, B, T, C);
  matmul_forward_cublaslt(d_qkv, d_ln1, d_qkvw, d_qkvb, B, T, C, 3*C);
  sycl_memset_sync(d_att, 0, (size_t)B*NH*T*T*sizeof(floatX));
  attention_forward(d_atty, d_qkvr, d_att, d_qkv, B, T, C, NH);
  // attproj
  matmul_forward_cublaslt(d_scratch, d_atty, d_apw, d_apb, B, T, C, C);
  std::vector<floatX> hs(B*T*C); sycl_memcpy_d2h(hs.data(), d_scratch, hs.size()*sizeof(floatX));
  printf("attproj[0..7]: "); for(int i=0;i<8;i++) printf("%f ",(float)hs[i]); printf("\n");
  // residual+ln2
  fused_residual_forward5(d_res2, d_ln2, d_m2, d_r2, d_enc, d_scratch, d_ln2w, d_ln2b, B*T, C);
  std::vector<floatX> hl2(B*T*C); sycl_memcpy_d2h(hl2.data(), d_ln2, hl2.size()*sizeof(floatX));
  printf("ln2[0..7]: "); for(int i=0;i<8;i++) printf("%f ",(float)hl2[i]); printf("\n");
  // fc (no fuse): matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B,T,C,4C, l_fch, gelu_fusion=0)
  // with gelu_fusion=0 and pre_gelu != null: writes pre_gelu then gelu_forward(out,...)
  matmul_forward_cublaslt(d_fchg, d_ln2, d_fcw, d_fcb, B, T, C, 4*C, d_fch, 0);
  std::vector<floatX> hf(B*T*4*C); sycl_memcpy_d2h(hf.data(), d_fchg, hf.size()*sizeof(floatX));
  printf("fch_gelu[0..7]: "); for(int i=0;i<8;i++) printf("%f ",(float)hf[i]); printf("\n");
  std::vector<floatX> hpre(B*T*4*C); sycl_memcpy_d2h(hpre.data(), d_fch, hpre.size()*sizeof(floatX));
  printf("fch_pre[0..7]: "); for(int i=0;i<8;i++) printf("%f ",(float)hpre[i]); printf("\n");
  // host ref for fc row 0
  std::vector<floatX> hh(B*T*C); sycl_memcpy_d2h(hh.data(), d_ln2, hh.size()*sizeof(floatX));
  printf("fch_pre-ref[0..7]: "); for(int oc=0;oc<8;oc++){ double r=(float)fcb[oc]; for(int k=0;k<C;k++) r+=(float)hh[k]*(float)fcw[oc*C+k]; printf("%f ",(float)(floatX)r); } printf("\n");
  return 0;
}
