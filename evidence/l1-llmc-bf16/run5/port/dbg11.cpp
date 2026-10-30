// attention vs fp32 reference with real qkv (layer 0)
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include <vector>
#include "llmc/attention.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t Vp=h[7],C=h[6],maxT=h[2]; int L=h[4],NH=h[5],HS=C/NH;
  fseek(f,(long)((Vp*C+maxT*C+2*(size_t)L*C)*sizeof(floatX)),SEEK_CUR);
  std::vector<floatX> qkvw((size_t)L*3*C*C); fread(qkvw.data(),sizeof(floatX),qkvw.size(),f);
  fclose(f);
  FILE* sf=fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,sf);
  int B=sh[2],T=sh[3];
  int* x=(int*)malloc(B*T*sizeof(int)); int* yy=(int*)malloc(B*T*sizeof(int));
  fread(x,sizeof(int),B*T,sf); fread(yy,sizeof(int),B*T,sf); fclose(sf);
  // need ln1: recompute quickly via encoder+ln? use dbg7 path: cheat - load ln1 from a file? simpler: run encoder+ln+qkv here
  // ... skip: directly test batched gemm correctness vs naive fp32:
  srand(5);
  int m=T,n=T,k=HS;
  // build fake Q,K row-major (T x HS)
  floatX* Q=(floatX*)sycl::malloc_shared((size_t)2*m*k*sizeof(floatX),sycl_queue);
  floatX* K=(floatX*)sycl::malloc_shared((size_t)2*m*k*sizeof(floatX),sycl_queue);
  floatX* O=(floatX*)sycl::malloc_shared((size_t)2*m*n*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared((size_t)2*m*n*sizeof(float),sycl_queue);
  for(int i=0;i<2*m*k;i++){Q[i]=(floatX)((rand()/float(RAND_MAX)-0.5f));K[i]=(floatX)((rand()/float(RAND_MAX)-0.5f));} // note: two rand calls per i, like before
  matmul_batched_onemath(O,Q,K,m,n,k,false,true,(size_t)m*n,(size_t)m*k,(size_t)m*k,2,false,tmp,sycl_queue);
  double maxd=0; for(int b=0;b<2;b++)for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double ref=0; for(int c=0;c<k;c++) ref+=(float)Q[b*m*k+i*k+c]*(float)K[b*m*k+j*k+c]; double got=(float)O[b*m*n+i*n+j]; maxd=fmax(maxd,fabs(ref-got)); }
  printf("batched F1 maxd=%g (expect <0.05)\n",maxd);
  // F2: vaccum(TxHS)=Att(TxT)*V(TxHS): m=T,n=HS,k=T N,N
  {
    floatX* A=(floatX*)sycl::malloc_shared((size_t)2*T*T*sizeof(floatX),sycl_queue);
    floatX* Vv=(floatX*)sycl::malloc_shared((size_t)2*T*HS*sizeof(floatX),sycl_queue);
    floatX* Oo=(floatX*)sycl::malloc_shared((size_t)2*T*HS*sizeof(floatX),sycl_queue);
    float* t2=(float*)sycl::malloc_shared((size_t)2*T*HS*sizeof(float),sycl_queue);
    for(int i=0;i<2*T*T;i++)A[i]=(floatX)((rand()/float(RAND_MAX))); for(int i=0;i<2*T*HS;i++)Vv[i]=(floatX)((rand()/float(RAND_MAX)-0.5f));
    matmul_batched_onemath(Oo,A,Vv,T,HS,T,false,false,(size_t)T*HS,(size_t)T*T,(size_t)T*HS,2,false,t2,sycl_queue);
    double md=0; for(int b=0;b<2;b++)for(int i=0;i<T;i++)for(int j=0;j<HS;j++){ double ref=0; for(int c=0;c<T;c++) ref+=(float)A[b*T*T+i*T+c]*(float)Vv[b*T*HS+c*HS+j]; md=fmax(md,fabs(ref-(float)Oo[b*T*HS+i*HS+j])); }
    printf("batched F2 maxd=%g\n",md);
  }
  return 0;
}
