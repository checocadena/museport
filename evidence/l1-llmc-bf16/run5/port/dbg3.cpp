// full forward determinism + logits magnitude check with real checkpoint, tiny path
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
#include "llmc/fused_classifier.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  // load model header + wte/wpe/ln only? simpler: replicate gpt2_forward first ops manually is heavy.
  // Instead check classifier gemm path with real wte (Vp x C):
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  printf("V=%d Vp=%d C=%d L=%d\n",h[3],h[7],h[6],h[4]);
  size_t Vp=h[7],C=h[6];
  floatX* wte=(floatX*)sycl::malloc_shared(Vp*C*sizeof(floatX),sycl_queue);
  fread(wte, sizeof(floatX), Vp*C, f);
  fclose(f);
  // fake lnf: random bf16
  srand(2);
  int B=1,T=4;
  floatX* lnf=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  for(int i=0;i<B*T*(int)C;i++) lnf[i]=(floatX)((rand()/float(RAND_MAX)-0.5f)*1.0f);
  floatX* logits=(floatX*)sycl::malloc_shared((size_t)B*T*Vp*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared((size_t)B*T*Vp*sizeof(float),sycl_queue);
  matmul_forward_onemath(logits,lnf,(floatX*)wte,nullptr,B,T,C,Vp,sycl_queue,tmp);
  double mx=0; int nn=0;
  for(int i=0;i<B*T*(int)Vp;i++){ float v=(float)logits[i]; if(!isfinite(v)) nn++; mx=fmax(mx,fabs(v)); }
  printf("logits maxabs=%g nans=%d (expect maxabs ~ 1e2, nans 0)\n",mx,nn);
  for(int i=0;i<5;i++) printf("%.3f ",(float)logits[i]); printf("\n");
  return 0;
}
