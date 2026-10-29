// Debug: load checkpoint, run encoder+ln1+qkv forward pieces on host vs device
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>
#include "llmc/sycl_common.h"
#include "llmc/sycl_utils.h"
#include "llmc/onemath_common.h"
#include "llmc/encoder.h"
#include "llmc/layernorm.h"
int main(){
  // read model header
  FILE* f = fopen("gpt2_124M_bf16.bin","rb");
  int header[256]; fread(header,sizeof(int),256,f);
  int maxT=header[2], V=header[3], L=header[4], NH=header[5], C=header[6], Vp=header[7];
  printf("maxT=%d V=%d L=%d NH=%d C=%d Vp=%d\n",maxT,V,L,NH,C,Vp);
  size_t wte_n=(size_t)Vp*C, wpe_n=(size_t)maxT*C;
  size_t ln1_n=(size_t)L*C;
  std::vector<floatX> wte(wte_n), wpe(wpe_n), ln1w(ln1_n), ln1b(ln1_n);
  fread(wte.data(),sizeof(floatX),wte_n,f);
  fread(wpe.data(),sizeof(floatX),wpe_n,f);
  fread(ln1w.data(),sizeof(floatX),ln1_n,f);
  fread(ln1b.data(),sizeof(floatX),ln1_n,f);
  fclose(f);
  printf("wte[0]=%f wte[1]=%f wpe[0]=%f ln1w[0]=%f ln1b[0]=%f\n",(float)wte[0],(float)wte[1],(float)wpe[0],(float)ln1w[0],(float)ln1b[0]);
  // read debug state
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  printf("B=%d T=%d\n",B,T);
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s);
  fclose(s);
  printf("x[0..7]: "); for(int i=0;i<8;i++) printf("%d ",x[i]); printf("\n");
  // device resident weights
  floatX *d_wte=sycl_malloc_device<floatX>(wte_n), *d_wpe=sycl_malloc_device<floatX>(wpe_n);
  floatX *d_ln1w=sycl_malloc_device<floatX>(ln1_n), *d_ln1b=sycl_malloc_device<floatX>(ln1_n);
  int *d_inp=sycl_malloc_device<int>(B*T);
  floatX *d_enc=sycl_malloc_device<floatX>((size_t)B*T*C), *d_ln1=sycl_malloc_device<floatX>((size_t)B*T*C);
  float *d_mean=sycl_malloc_device<float>(B*T), *d_rstd=sycl_malloc_device<float>(B*T);
  sycl_memcpy_h2d(d_wte,wte.data(),wte_n*sizeof(floatX));
  sycl_memcpy_h2d(d_wpe,wpe.data(),wpe_n*sizeof(floatX));
  sycl_memcpy_h2d(d_ln1w,ln1w.data(),ln1_n*sizeof(floatX));
  sycl_memcpy_h2d(d_ln1b,ln1b.data(),ln1_n*sizeof(floatX));
  sycl_memcpy_h2d(d_inp,x.data(),B*T*sizeof(int));
  encoder_forward(d_enc, d_inp, d_wte, d_wpe, B, T, C);
  std::vector<floatX> henc(B*T*C);
  sycl_memcpy_d2h(henc.data(), d_enc, henc.size()*sizeof(floatX));
  // host reference for position 0
  int ix = x[0];
  printf("enc[0..7] dev: "); for(int i=0;i<8;i++) printf("%f ",(float)henc[i]); printf("\n");
  printf("enc[0..7] ref: "); for(int i=0;i<8;i++) printf("%f ",(float)wte[ix*C+i]+(float)wpe[i]); printf("\n");
  // layernorm
  layernorm_forward(d_ln1, d_mean, d_rstd, d_enc, d_ln1w, d_ln1b, B, T, C);
  std::vector<floatX> hln(B*T*C);
  std::vector<float> hm(B*T), hr(B*T);
  sycl_memcpy_d2h(hln.data(), d_ln1, hln.size()*sizeof(floatX));
  sycl_memcpy_d2h(hm.data(), d_mean, hm.size()*sizeof(float));
  sycl_memcpy_d2h(hr.data(), d_rstd, hr.size()*sizeof(float));
  printf("ln1[0..7] dev: "); for(int i=0;i<8;i++) printf("%f ",(float)hln[i]); printf("\n");
  // host ref layernorm row 0
  double m=0; for(int i=0;i<C;i++) m+=(float)henc[i]; m/=C;
  double v=0; for(int i=0;i<C;i++){double d=(float)henc[i]-m; v+=d*d;} v/=C;
  double rs=1.0/sqrt(v+1e-5);
  printf("mean dev=%f ref=%f rstd dev=%f ref=%f\n", hm[0], m, hr[0], rs);
  printf("ln1[0..7] ref: "); for(int i=0;i<8;i++){ double n=rs*((float)henc[i]-m); printf("%f ",(float)(n*(float)ln1w[i]+(float)ln1b[i])); } printf("\n");
  return 0;
}
