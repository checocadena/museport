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
  std::vector<floatX> ln2w(ln1_n), ln2b(ln1_n), fcw(fc_n), fcb(fcb_n), fcpw(fcp_n), fcpb(fcpb_n), lnfw(C), lnfb(C);
  fread(wte.data(),sizeof(floatX),wte_n,f); fread(wpe.data(),sizeof(floatX),wpe_n,f);
  fread(ln1w.data(),sizeof(floatX),ln1_n,f); fread(ln1b.data(),sizeof(floatX),ln1_n,f);
  fread(qkvw.data(),sizeof(floatX),qkvw_n,f); fread(qkvb.data(),sizeof(floatX),qkvb_n,f);
  fread(apw.data(),sizeof(floatX),ap_n,f); fread(apb.data(),sizeof(floatX),apb_n,f);
  fread(ln2w.data(),sizeof(floatX),ln1_n,f); fread(ln2b.data(),sizeof(floatX),ln1_n,f);
  fread(fcw.data(),sizeof(floatX),fc_n,f); fread(fcb.data(),sizeof(floatX),fcb_n,f);
  fread(fcpw.data(),sizeof(floatX),fcp_n,f); fread(fcpb.data(),sizeof(floatX),fcpb_n,f);
  fread(lnfw.data(),sizeof(floatX),C,f); fread(lnfb.data(),sizeof(floatX),C,f);
  fclose(f);
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s);
  float* exp_logits=(float*)malloc(B*T*V*sizeof(float)); fread(exp_logits,sizeof(float),B*T*V,s); fclose(s);
  auto devx=[&](std::vector<floatX>& h){ floatX* d=sycl_malloc_device<floatX>(h.size()); sycl_memcpy_h2d(d,h.data(),h.size()*sizeof(floatX)); return d; };
  floatX *d_wte=devx(wte), *d_wpe=devx(wpe);
  floatX *d_ln1w=devx(ln1w), *d_ln1b=devx(ln1b), *d_qkvw=devx(qkvw), *d_qkvb=devx(qkvb);
  floatX *d_apw=devx(apw), *d_apb=devx(apb), *d_ln2w=devx(ln2w), *d_ln2b=devx(ln2b);
  floatX *d_fcw=devx(fcw), *d_fcb=devx(fcb), *d_fcpw=devx(fcpw), *d_fcpb=devx(fcpb);
  floatX *d_lnfw=devx(lnfw), *d_lnfb=devx(lnfb);
  int *d_inp=sycl_malloc_device<int>(B*T); sycl_memcpy_h2d(d_inp,x.data(),B*T*sizeof(int));
  size_t BTC=B*T*C, BT4=B*T*4*C;
  floatX *d_enc=sycl_malloc_device<floatX>(BTC);
  floatX *d_ln1=sycl_malloc_device<floatX>((size_t)L*BTC), *d_qkvr=sycl_malloc_device<floatX>((size_t)L*BTC*3);
  floatX *d_atty=sycl_malloc_device<floatX>((size_t)L*BTC), *d_att=sycl_malloc_device<floatX>((size_t)L*B*NH*T*T);
  floatX *d_res2=sycl_malloc_device<floatX>((size_t)L*BTC), *d_ln2=sycl_malloc_device<floatX>((size_t)L*BTC);
  float *d_m1=sycl_malloc_device<float>((size_t)L*B*T), *d_r1=sycl_malloc_device<float>((size_t)L*B*T);
  float *d_m2=sycl_malloc_device<float>((size_t)L*B*T), *d_r2=sycl_malloc_device<float>((size_t)L*B*T);
  floatX *d_fch=sycl_malloc_device<floatX>((size_t)L*BT4), *d_fchg=sycl_malloc_device<floatX>((size_t)L*BT4);
  floatX *d_res3=sycl_malloc_device<floatX>((size_t)L*BTC);
  floatX *d_lnf=sycl_malloc_device<floatX>(BTC);
  float *d_mf=sycl_malloc_device<float>(B*T), *d_rf=sycl_malloc_device<float>(B*T);
  floatX *d_out=sycl_malloc_device<floatX>((size_t)B*T*Vp);
  floatX *d_scratch=sycl_malloc_device<floatX>((size_t)B*T*4*C);
  encoder_forward(d_enc, d_inp, d_wte, d_wpe, B, T, C);
  layernorm_forward(d_ln1, d_m1, d_r1, d_enc, d_ln1w, d_ln1b, B, T, C);
  for(int l=0;l<L;l++){
    floatX* residual = l==0 ? d_enc : d_res3+(l-1)*BTC;
    floatX* l_ln1 = d_ln1+l*BTC;
    floatX* l_qkvr = d_qkvr+l*BTC*3;
    floatX* l_atty = d_atty+l*BTC;
    floatX* l_res2 = d_res2+l*BTC;
    floatX* l_ln2 = d_ln2+l*BTC;
    floatX* l_fch = d_fch+l*BT4;
    floatX* l_fchg = d_fchg+l*BT4;
    floatX* l_res3 = d_res3+l*BTC;
    floatX* l_att = d_att+l*(size_t)B*NH*T*T;
    matmul_forward_cublaslt(d_scratch, l_ln1, d_qkvw+l*3*C*C, d_qkvb+l*3*C, B, T, C, 3*C);
    attention_forward(l_atty, l_qkvr, l_att, d_scratch, B, T, C, NH);
    matmul_forward_cublaslt(d_scratch, l_atty, d_apw+l*C*C, d_apb+l*C, B, T, C, C);
    fused_residual_forward5(l_res2, l_ln2, d_m2+l*B*T, d_r2+l*B*T, residual, d_scratch, d_ln2w+l*C, d_ln2b+l*C, B*T, C);
    matmul_forward_cublaslt(l_fchg, l_ln2, d_fcw+l*4*C*C, d_fcb+l*4*C, B, T, C, 4*C, l_fch, 0);
    matmul_forward_cublaslt(d_scratch, l_fchg, d_fcpw+l*C*4*C, d_fcpb+l*C, B, T, 4*C, C);
    if(l+1!=L) fused_residual_forward5(l_res3, d_ln1+(l+1)*BTC, d_m1+(l+1)*B*T, d_r1+(l+1)*B*T, l_res2, d_scratch, d_ln1w+(l+1)*C, d_ln1b+(l+1)*C, B*T, C);
    else fused_residual_forward5(l_res3, d_lnf, d_mf, d_rf, l_res2, d_scratch, d_lnfw, d_lnfb, B*T, C);
    // report res3 row 0 norm-ish
    std::vector<floatX> hr(C); sycl_memcpy_d2h(hr.data(), l_res3, C*sizeof(floatX));
    double s2=0; for(int i=0;i<C;i++) s2+=(float)hr[i]*(float)hr[i];
    printf("l=%d res3[0..3]: %f %f %f %f rms=%f\n", l, (float)hr[0],(float)hr[1],(float)hr[2],(float)hr[3], sqrt(s2/C));
  }
  matmul_forward_cublaslt(d_out, d_lnf, d_wte, NULL, B, T, C, Vp);
  std::vector<floatX> ho(B*T*Vp); sycl_memcpy_d2h(ho.data(), d_out, ho.size()*sizeof(floatX));
  printf("logit[0..3] dev: "); for(int i=0;i<4;i++) printf("%f ",(float)ho[i]); printf("\n");
  printf("logit[0..3] exp: "); for(int i=0;i<4;i++) printf("%f ",exp_logits[i]); printf("\n");
  double md=0; for(int bt=0;bt<B*T;bt++) for(int v=0;v<V;v++){ double d=std::fabs(exp_logits[bt*V+v]-(float)ho[bt*Vp+v]); if(d>md)md=d; }
  printf("logit maxdiff=%f\n",md);
  return 0;
}
