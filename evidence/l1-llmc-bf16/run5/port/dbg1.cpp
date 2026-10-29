#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <oneapi/math/blas.hpp>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include "llmc/encoder.hpp"
#include "llmc/layernorm.hpp"
#include "llmc/matmul.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  // tiny encoder+ln+matmul sanity with bf16 random data vs fp32 reference
  srand(0);
  int B=2,T=8,C=64,Vp=128;
  floatX* wte=(floatX*)sycl::malloc_shared((size_t)Vp*C*sizeof(floatX),sycl_queue);
  floatX* wpe=(floatX*)sycl::malloc_shared((size_t)64*C*sizeof(floatX),sycl_queue);
  floatX* enc=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  int* inp=(int*)sycl::malloc_shared((size_t)B*T*sizeof(int),sycl_queue);
  for(int i=0;i<Vp*C;i++) wte[i]=(floatX)((rand()/float(RAND_MAX)-0.5f)*0.1f);
  for(int i=0;i<64*C;i++) wpe[i]=(floatX)((rand()/float(RAND_MAX)-0.5f)*0.1f);
  for(int i=0;i<B*T;i++) inp[i]=rand()%Vp;
  encoder_forward(enc,inp,wte,wpe,B,T,C,sycl_queue);
  // reference
  double maxd=0; for(int b=0;b<B;b++)for(int t=0;t<T;t++)for(int c=0;c<C;c++){ float ref=(float)wte[inp[b*T+t]*C+c]+(float)wpe[t*C+c]; float got=(float)enc[(b*T+t)*C+c]; maxd=fmax(maxd,fabs(ref-got)); }
  printf("encoder maxd=%g (expect ~0.01 bf16 rounding)\n",maxd);
  // layernorm
  floatX* w=(floatX*)sycl::malloc_shared((size_t)C*sizeof(floatX),sycl_queue);
  floatX* b=(floatX*)sycl::malloc_shared((size_t)C*sizeof(floatX),sycl_queue);
  floatX* out=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  float* mean=(float*)sycl::malloc_shared((size_t)B*T*sizeof(float),sycl_queue);
  float* rstd=(float*)sycl::malloc_shared((size_t)B*T*sizeof(float),sycl_queue);
  for(int i=0;i<C;i++){w[i]=(floatX)1.0f;b[i]=(floatX)0.0f;}
  layernorm_forward(out,mean,rstd,enc,w,b,B,T,C,sycl_queue);
  // reference mean/rstd
  double maxdm=0; for(int bt=0;bt<B*T;bt++){ double m=0; for(int c=0;c<C;c++) m+=(float)enc[bt*C+c]; m/=C; maxdm=fmax(maxdm,fabs(m-mean[bt])); }
  printf("ln mean maxd=%g\n",maxdm);
  // check normalized: mean(out row) ~ 0
  double m2=0; for(int c=0;c<C;c++) m2+=(float)out[c]; printf("out row0 mean=%g (expect ~0)\n",m2/C);
  // matmul: out2 = out @ w2
  int OC=128;
  floatX* w2=(floatX*)sycl::malloc_shared((size_t)OC*C*sizeof(floatX),sycl_queue);
  floatX* o2=(floatX*)sycl::malloc_shared((size_t)B*T*OC*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared((size_t)B*T*OC*sizeof(float),sycl_queue);
  for(int i=0;i<OC*C;i++) w2[i]=(floatX)((rand()/float(RAND_MAX)-0.5f)*0.1f);
  matmul_forward_onemath(o2,out,w2,nullptr,B,T,C,OC,sycl_queue,tmp);
  double maxd2=0; for(int bt=0;bt<2;bt++)for(int oc=0;oc<4;oc++){ double ref=0; for(int c=0;c<C;c++) ref+=(float)out[bt*C+c]*(float)w2[oc*C+c]; double got=(float)o2[bt*OC+oc]; maxd2=fmax(maxd2,fabs(ref-got)); }
  printf("matmul maxd=%g\n",maxd2);
  return 0;
}
