#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include "llmc/sycl_common.h"
sycl::queue sycl_queue;
#include "llmc/attention.hpp"
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  srand(1);
  int B=1,T=8,C=64,NH=4,HS=16;
  floatX* inp=(floatX*)sycl::malloc_shared((size_t)B*T*3*C*sizeof(floatX),sycl_queue);
  floatX* qkvr=(floatX*)sycl::malloc_shared((size_t)B*T*3*C*sizeof(floatX),sycl_queue);
  floatX* att=(floatX*)sycl::malloc_shared((size_t)B*NH*T*T*sizeof(floatX),sycl_queue);
  floatX* out=(floatX*)sycl::malloc_shared((size_t)B*T*C*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared((size_t)(B*T*C*4+1024)*sizeof(float),sycl_queue);
  for(int i=0;i<B*T*3*C;i++) inp[i]=(floatX)((rand()/float(RAND_MAX)-0.5f)*0.5f);
  memset(att,0,(size_t)B*NH*T*T*sizeof(floatX));
  attention_forward(out,qkvr,att,inp,B,T,C,NH,sycl_queue,tmp);
  // check attention rows sum to 1
  double maxd=0; for(int b=0;b<B;b++)for(int h=0;h<NH;h++)for(int t=0;t<T;t++){ double s=0; for(int t2=0;t2<T;t2++) s+=(float)att[((b*NH+h)*T+t)*T+t2]; maxd=fmax(maxd,fabs(s-1.0)); }
  printf("att rowsum maxd=%g (expect small)\n",maxd);
  // reference softmax for row0
  // check for NaN
  int nans=0; for(int i=0;i<B*NH*T*T;i++){ float v=(float)att[i]; if(!isfinite(v)) nans++; }
  printf("nans=%d\n",nans);
  // print first row
  for(int t2=0;t2<T;t2++) printf("%.4f ",(float)att[t2]); printf("\n");
  return 0;
}
