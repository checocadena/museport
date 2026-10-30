// check layer-0 pieces: attproj out, ln2, fc, fcproj magnitudes; compare fcproj weight stats
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
static void chk(const char* n, floatX* p, size_t m){ double mx=0,ss=0; int nn=0; for(size_t i=0;i<m;i++){float v=(float)p[i]; if(!isfinite(v))nn++; else {mx=fmax(mx,fabs(v)); ss+=v*v;}} printf("%s: maxabs=%g rms=%g nans=%d\n",n,mx,sqrt(ss/m),nn); fflush(stdout); }
static void chkf(const char* n, float* p, size_t m){ double mx=0; int nn=0; for(size_t i=0;i<m;i++){float v=p[i]; if(!isfinite(v))nn++; else mx=fmax(mx,fabs(v));} printf("%s: maxabs=%g nans=%d\n",n,mx,nn); fflush(stdout); }
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
  floatX* ln2w=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(ln2w,sizeof(floatX),n_ln,f);
  floatX* ln2b=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(ln2b,sizeof(floatX),n_ln,f);
  floatX* fcw=(floatX*)sycl::malloc_shared(((size_t)L*4*C*C)*sizeof(floatX),sycl_queue); fread(fcw,sizeof(floatX),(size_t)L*4*C*C,f);
  floatX* fcb=(floatX*)sycl::malloc_shared(((size_t)L*4*C)*sizeof(floatX),sycl_queue); fread(fcb,sizeof(floatX),(size_t)L*4*C,f);
  floatX* fpw=(floatX*)sycl::malloc_shared(((size_t)L*C*4*C)*sizeof(floatX),sycl_queue); fread(fpw,sizeof(floatX),(size_t)L*C*4*C,f);
  floatX* fpb=(floatX*)sycl::malloc_shared(n_ln*sizeof(floatX),sycl_queue); fread(fpb,sizeof(floatX),n_ln,f);
  fclose(f);
  chk("apw0",apw,(size_t)C*C); chk("ln2w0",ln2w,C); chk("fcw0",fcw,(size_t)4*C*C); chk("fpw0",fpw,(size_t)C*4*C);
  FILE* sf=fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,sf);
  int B=sh[2],T=sh[3];
  int* x=(int*)malloc(B*T*sizeof(int)); int* yy=(int*)malloc(B*T*sizeof(int));
  fread(x,sizeof(int),B*T,sf); fread(yy,sizeof(int),B*T,sf); fclose(sf);
  size_t BT=B*T;
  int* xin=(int*)sycl::malloc_shared(BT*sizeof(int),sycl_queue); memcpy(xin,x,BT*sizeof(int));
  floatX* enc=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  floatX* ln1=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  float* m1=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  float* r1=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  encoder_forward(enc,xin,wte,wpe,B,T,C,sycl_queue);
  layernorm_forward(ln1,m1,r1,enc,ln1w,ln1b,B,T,C,sycl_queue);
  chk("enc",enc,BT*C); chk("ln1",ln1,BT*C);
  floatX* qkv=(floatX*)sycl::malloc_shared(BT*3*C*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared(BT*Vp*sizeof(float),sycl_queue);
  matmul_forward_onemath(qkv,ln1,qkvw,qkvb,B,T,C,3*C,sycl_queue,tmp); chk("qkv",qkv,BT*3*C);
  floatX* qkvr=(floatX*)sycl::malloc_shared(BT*3*C*sizeof(floatX),sycl_queue);
  floatX* att=(floatX*)sycl::malloc_shared((size_t)B*NH*T*T*sizeof(floatX),sycl_queue);
  memset(att,0,(size_t)B*NH*T*T*sizeof(floatX));
  floatX* atty=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  attention_forward(atty,qkvr,att,qkv,B,T,C,NH,sycl_queue,tmp); chk("atty",atty,BT*C);
  floatX* proj=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  matmul_forward_onemath(proj,atty,apw,apb,B,T,C,C,sycl_queue,tmp); chk("proj",proj,BT*C);
  floatX* res2=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  floatX* ln2=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  float* m2=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  float* r2=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  fused_residual_forward5(res2,ln2,m2,r2,enc,proj,ln2w,ln2b,BT,C,sycl_queue);
  chk("res2",res2,BT*C); chk("ln2",ln2,BT*C); chkf("m2",m2,BT); chkf("r2",r2,BT);
  floatX* fch=(floatX*)sycl::malloc_shared(BT*4*C*sizeof(floatX),sycl_queue);
  floatX* fchg=(floatX*)sycl::malloc_shared(BT*4*C*sizeof(floatX),sycl_queue);
  matmul_forward_onemath(fchg,ln2,fcw,fcb,B,T,C,4*C,sycl_queue,tmp,fch,0); chk("fchg",fchg,BT*4*C);
  floatX* fcp=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  matmul_forward_onemath(fcp,fchg,fpw,fpb,B,T,4*C,C,sycl_queue,tmp); chk("fcp",fcp,BT*C);
  return 0;
}
