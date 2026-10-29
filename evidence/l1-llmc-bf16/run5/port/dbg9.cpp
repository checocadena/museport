// compare per-layer res3 of my forward vs a fp32 reference forward (same weights) to find diverging layer
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include "llmc/encoder.hpp"
#include "llmc/layernorm.hpp"
#include "llmc/matmul.hpp"
#include "llmc/attention.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t V=h[3],Vp=h[7],C=h[6],maxT=h[2]; int L=h[4],NH=h[5];
  size_t n_wte=Vp*C, n_wpe=maxT*C, n_ln=(size_t)L*C;
  std::vector<float> W;
  auto rd=[&](size_t n, std::vector<floatX>& s){ s.resize(n); fread(s.data(),sizeof(floatX),n,f); for(auto &v:s) W.push_back((float)v); };
  std::vector<floatX> wte,wpe,ln1w,ln1b,qkvw,qkvb,apw,apb,ln2w,ln2b,fcw,fcb,fpw,fpb,lnfw,lnfb;
  // read raw bf16 + keep fp32 copy with offsets
  std::vector<std::vector<floatX>> P(16);
  size_t ns[16]={n_wte,n_wpe,n_ln,n_ln,(size_t)L*3*C*C,(size_t)L*3*C,(size_t)L*C*C,n_ln,n_ln,n_ln,(size_t)L*4*C*C,(size_t)L*4*C,(size_t)L*C*4*C,n_ln,C,C};
  std::vector<size_t> off(17,0);
  for(int i=0;i<16;i++){ P[i].resize(ns[i]); fread(P[i].data(),sizeof(floatX),ns[i],f); off[i+1]=off[i]+ns[i]; }
  fclose(f);
  auto F=[&](int t,size_t i)->float{ return (float)P[t][i]; };
  FILE* sf=fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,sf);
  int B=sh[2],T=sh[3];
  int* x=(int*)malloc(B*T*sizeof(int)); int* yy=(int*)malloc(B*T*sizeof(int));
  fread(x,sizeof(int),B*T,sf); fread(yy,sizeof(int),B*T,sf); fclose(sf);
  size_t BT=B*T;
  // ---- fp32 reference forward (row-major, naive) ----
  std::vector<float> enc(BT*C), ln1(BT*C);
  for(int b=0;b<B;b++)for(int t=0;t<T;t++)for(size_t c=0;c<C;c++) enc[(b*T+t)*C+c]=F(0,(size_t)x[b*T+t]*C+c)+F(1,(size_t)t*C+c);
  auto layernorm_ref=[](std::vector<float>& out,std::vector<float>& mean,std::vector<float>& rstd,const std::vector<float>& inp,const std::vector<floatX>& w,const std::vector<floatX>& b_,size_t woff,size_t N,size_t Cc){
    out.resize(N*Cc); mean.resize(N); rstd.resize(N);
    for(size_t n=0;n<N;n++){ double m=0; for(size_t c=0;c<Cc;c++) m+=inp[n*Cc+c]; m/=Cc; double v=0; for(size_t c=0;c<Cc;c++){double d=inp[n*Cc+c]-m; v+=d*d;} v/=Cc; float s=1.0f/sqrtf(v+1e-5f); mean[n]=m; rstd[n]=s;
      for(size_t c=0;c<Cc;c++){ float nn2=s*(inp[n*Cc+c]-m); out[n*Cc+c]=nn2*(float)w[woff+c]+(float)b_[woff+c]; } }
  };
  std::vector<float> m,r;
  // first ln
  {
    std::vector<float> o,mm,rr; layernorm_ref(o,mm,rr,enc,P[2],P[3],0,BT,C);
    // my sycl ln1 layer0? compare
    double maxd=0; for(size_t i=0;i<BT*C;i++){ // compute my kernel result
    }
    printf("ref ln1[0..4]: %g %g %g %g\n",o[0],o[1],o[2],o[3]);
  }
  return 0;
}
