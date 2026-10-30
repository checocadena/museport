// check qkv matmul with real weights: out(BT x 3C) = ln1(BT x C) @ qkvw^T, bias add; also qkv bias path
#define TESTING_REDUCED 1
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include "llmc/matmul.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t Vp=h[7],C=h[6],maxT=h[2]; int L=h[4];
  // skip wte(Vp*C), wpe(maxT*C), ln1w(L*C), ln1b(L*C)
  auto skip=[&](size_t n){ fseek(f, (long)(n*sizeof(floatX)), SEEK_CUR); };
  skip(Vp*C); skip(maxT*C); skip((size_t)L*C); skip((size_t)L*C);
  size_t qkvw_n = (size_t)L*3*C*C;
  floatX* qkvw=(floatX*)sycl::malloc_shared(qkvw_n*sizeof(floatX),sycl_queue);
  fread(qkvw,sizeof(floatX),qkvw_n,f);
  size_t qkvb_n=(size_t)L*3*C;
  floatX* qkvb=(floatX*)sycl::malloc_shared(qkvb_n*sizeof(floatX),sycl_queue);
  fread(qkvb,sizeof(floatX),qkvb_n,f);
  fclose(f);
  // check magnitudes
  double mx=0; for(size_t i=0;i<qkvw_n;i++) mx=fmax(mx,fabs((float)qkvw[i]));
  printf("qkvw maxabs=%g\n",mx);
  mx=0; for(size_t i=0;i<qkvb_n;i++) mx=fmax(mx,fabs((float)qkvb[i]));
  printf("qkvb maxabs=%g\n",mx);
  int B=4,T=64,BT=B*T,OC=3*(int)C;
  floatX* ln1=(floatX*)sycl::malloc_shared((size_t)BT*C*sizeof(floatX),sycl_queue);
  // layernorm output magnitude ~ N(0,1)-ish scaled; use random normal-ish
  srand(3);
  for(int i=0;i<BT*(int)C;i++){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f), u2=(rand()+1.0f)/(RAND_MAX+2.0f); ln1[i]=(floatX)(sqrtf(-2*logf(u1))*cosf(6.2831f*u2)*0.5f); }
  floatX* out=(floatX*)sycl::malloc_shared((size_t)BT*OC*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared((size_t)BT*OC*sizeof(float),sycl_queue);
  matmul_forward_onemath(out,ln1,qkvw,qkvb,B,T,C,OC,sycl_queue,tmp);
  double mxo=0; int nn=0; for(int i=0;i<BT*OC;i++){ float v=(float)out[i]; if(!isfinite(v)) nn++; mxo=fmax(mxo,fabs(v)); }
  printf("qkv out maxabs=%g nans=%d\n",mxo,nn);
  for(int i=0;i<5;i++) printf("%.4f ",(float)out[i]); printf("\n");
  // reference first element fp32
  double ref=(float)qkvb[0]; for(int c=0;c<(int)C;c++) ref+=(float)ln1[c]*(float)qkvw[c];
  printf("ref[0]=%g got=%g\n",ref,(float)out[0]);
  return 0;
}
