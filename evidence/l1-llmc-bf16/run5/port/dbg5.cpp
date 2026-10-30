// layer-0 forward step by step with REAL weights, check each stage for blowup/nan
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include "llmc/encoder.hpp"
#include "llmc/layernorm.hpp"
#include "llmc/matmul.hpp"
#include "llmc/attention.hpp"
static void chk(const char* n, floatX* p, size_t m){ double mx=0; int nn=0; for(size_t i=0;i<m;i++){float v=(float)p[i]; if(!isfinite(v))nn++; else mx=fmax(mx,fabs(v));} printf("%s: maxabs=%g nans=%d\n",n,mx,nn); }
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t Vp=h[7],C=h[6],maxT=h[2]; int L=h[4],NH=h[5];
  size_t n_wte=Vp*C, n_wpe=maxT*C, n_ln=(size_t)L*C;
  floatX* wte=(floatX*)sycl::malloc_shared(n_wte*sizeof(floatX),sycl_queue); fread(wte,sizeof(floatX),n_wte,f);
  floatX* wpe=(floatX*)sycl::malloc_shared(n_wpe*sizeof(floatX),sycl_queue); fread(wpe,sizeof(floatX),n_wpe,f);
  floatX* ln1w=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(ln1w,sizeof(floatX),n_ln,f);
  floatX* ln1b=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(ln1b,sizeof(floatX),n_ln,f);
  floatX* qkvw=(floatX*)sycl::malloc_shared(((size_t)L*3*C*C)*sizeof(floatX),sycl_queue); fread(qkvw,sizeof(floatX),(size_t)L*3*C*C,f);
  floatX* qkvb=(floatX*)sycl::malloc_shared(((size_t)L*3*C)*sizeof(floatX),sycl_queue); fread(qkvb,sizeof(floatX),(size_t)L*3*C,f);
  floatX* apw=(floatX*)sycl::malloc_shared(((size_t)L*C*C)*sizeof(floatX),sycl_queue); fread(apw,sizeof(floatX),(size_t)L*C*C,f);
  floatX* apb=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(apb,sizeof(floatX),n_ln,f);
  fclose(f);
  // debug inputs
  FILE* sf=fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,sf);
  int B=sh[2],T=sh[3];
  int* x=(int*)malloc(B*T*sizeof(int)); int* y=(int*)malloc(B*T*sizeof(int));
  fread(x,sizeof(int),B*T,sf); fread(y,sizeof(int),B*T,sf); fclose(sf);
  printf("B=%d T=%d C=%zu\n",B,T,C);
  floatX* enc=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  int* xin=(int*)sycl::malloc_shared((size_t)B*T*sizeof(int),sycl_queue);
  memcpy(xin,x,(size_t)B*T*sizeof(int));
  encoder_forward(enc,xin,wte,wpe,B,T,C,sycl_queue); chk("enc",enc,(size_t)B*T*C);
  floatX* ln1=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  float* m1=(float*)sycl::malloc_shared((size_t)B*T*sizeof(float),sycl_queue);
  float* r1=(float*)sycl::malloc_shared((size_t)B*T*sizeof(float),sycl_queue);
  layernorm_forward(ln1,m1,r1,enc,ln1w,ln1b,B,T,C,sycl_queue); chk("ln1",ln1,(size_t)B*T*C);
  size_t BT=B*T, OC=3*C;
  floatX* qkv=(floatX*)sycl::malloc_shared(BT*OC*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared(BT*OC*sizeof(float),sycl_queue);
  matmul_forward_onemath(qkv,ln1,qkvw,qkvb,B,T,C,OC,sycl_queue,tmp); chk("qkv",qkv,BT*OC);
  floatX* qkvr=(floatX*)sycl::malloc_shared(BT*OC*sizeof(floatX),sycl_queue);
  floatX* att=(floatX*)sycl::malloc_shared((size_t)B*NH*T*T*sizeof(floatX),sycl_queue);
  memset(att,0,(size_t)B*NH*T*T*sizeof(floatX));
  floatX* atty=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  attention_forward(atty,qkvr,att,qkv,B,T,C,NH,sycl_queue,tmp); chk("atty",atty,BT*C);
  return 0;
}
