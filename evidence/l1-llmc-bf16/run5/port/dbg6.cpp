// full 12-layer forward with REAL weights, check per-layer + final logits vs expected
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
static void chk(const char* n, floatX* p, size_t m){ double mx=0; int nn=0; for(size_t i=0;i<m;i++){float v=(float)p[i]; if(!isfinite(v))nn++; else mx=fmax(mx,fabs(v));} printf("%s: maxabs=%g nans=%d\n",n,mx,nn); fflush(stdout); }
int main(){
  sycl_queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t V=h[3],Vp=h[7],C=h[6],maxT=h[2]; int L=h[4],NH=h[5];
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
  floatX* lnfw=(floatX*)sycl::malloc_shared(C*sizeof(floatX),sycl_queue); fread(lnfw,sizeof(floatX),C,f);
  floatX* lnfb=(floatX*)sycl::malloc_shared(C*sizeof(floatX),sycl_queue); fread(lnfb,sizeof(floatX),C,f);
  fclose(f);
  FILE* sf=fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,sf);
  int B=sh[2],T=sh[3];
  int* x=(int*)malloc(B*T*sizeof(int)); int* yy=(int*)malloc(B*T*sizeof(int));
  fread(x,sizeof(int),B*T,sf); fread(yy,sizeof(int),B*T,sf);
  float* explog=(float*)malloc((size_t)B*T*V*sizeof(float)); float exploss; fread(explog,sizeof(float),(size_t)B*T*V,sf); fread(&exploss,sizeof(float),1,sf); fclose(sf);
  printf("B=%d T=%d C=%zu L=%d\n",B,T,C,L); fflush(stdout);
  size_t BT=B*T;
  int* xin=(int*)sycl::malloc_shared(BT*sizeof(int),sycl_queue); memcpy(xin,x,BT*sizeof(int));
  floatX* enc=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  floatX* ln1=(floatX*)sycl::malloc_shared((size_t)L*BT*C*sizeof(floatX),sycl_queue);
  float* m1=(float*)sycl::malloc_shared((size_t)L*BT*sizeof(float),sycl_queue);
  float* r1=(float*)sycl::malloc_shared((size_t)L*BT*sizeof(float),sycl_queue);
  floatX* res2=(floatX*)sycl::malloc_shared((size_t)L*BT*C*sizeof(floatX),sycl_queue);
  floatX* ln2=(floatX*)sycl::malloc_shared((size_t)L*BT*C*sizeof(floatX),sycl_queue);
  float* m2=(float*)sycl::malloc_shared((size_t)L*BT*sizeof(float),sycl_queue);
  float* r2=(float*)sycl::malloc_shared((size_t)L*BT*sizeof(float),sycl_queue);
  floatX* atty=(floatX*)sycl::malloc_shared((size_t)L*BT*C*sizeof(floatX),sycl_queue);
  floatX* res3=(floatX*)sycl::malloc_shared((size_t)L*BT*C*sizeof(floatX),sycl_queue);
  floatX* qkvr=(floatX*)sycl::malloc_shared((size_t)L*BT*3*C*sizeof(floatX),sycl_queue);
  floatX* att=(floatX*)sycl::malloc_shared((size_t)L*B*NH*T*T*sizeof(floatX),sycl_queue);
  memset(att,0,(size_t)L*B*NH*T*T*sizeof(floatX));
  floatX* fch=(floatX*)sycl::malloc_shared((size_t)L*BT*4*C*sizeof(floatX),sycl_queue);
  floatX* fchg=(floatX*)sycl::malloc_shared((size_t)L*BT*4*C*sizeof(floatX),sycl_queue);
  floatX* lnf=(floatX*)sycl::malloc_shared(BT*C*sizeof(floatX),sycl_queue);
  float* mf=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  float* rf=(float*)sycl::malloc_shared(BT*sizeof(float),sycl_queue);
  floatX* scratch=(floatX*)sycl::malloc_shared(BT*3*C*sizeof(floatX),sycl_queue);
  float* tmp=(float*)sycl::malloc_shared(BT*Vp*sizeof(float),sycl_queue);
  floatX* logits=(floatX*)sycl::malloc_shared(BT*Vp*sizeof(floatX),sycl_queue);
  encoder_forward(enc,xin,wte,wpe,B,T,C,sycl_queue);
  layernorm_forward(ln1,m1,r1,enc,ln1w,ln1b,B,T,C,sycl_queue);
  for(int l=0;l<L;l++){
    floatX* residual = l==0?enc:res3+(l-1)*BT*C;
    floatX* l_qkvw=qkvw+l*3*C*C, *l_qkvb=qkvb+l*3*C, *l_apw=apw+l*C*C, *l_apb=apb+l*C;
    floatX* l_ln2w=ln2w+l*C, *l_ln2b=ln2b+l*C, *l_fcw=fcw+l*4*C*C, *l_fcb=fcb+l*4*C, *l_fpw=fpw+l*C*4*C, *l_fpb=fpb+l*C;
    floatX* l_ln1=ln1+l*BT*C, *l_qkvr=qkvr+l*BT*3*C, *l_atty=atty+l*BT*C, *l_res2=res2+l*BT*C;
    floatX* l_ln2=ln2+l*BT*C, *l_fch=fch+l*BT*4*C, *l_fchg=fchg+l*BT*4*C, *l_res3=res3+l*BT*C;
    floatX* l_att=att+l*B*NH*T*T;
    matmul_forward_onemath(scratch,l_ln1,l_qkvw,l_qkvb,B,T,C,3*C,sycl_queue,tmp);
    attention_forward(l_atty,l_qkvr,l_att,scratch,B,T,C,NH,sycl_queue,tmp);
    matmul_forward_onemath(scratch,l_atty,l_apw,l_apb,B,T,C,C,sycl_queue,tmp);
    fused_residual_forward5(l_res2,l_ln2,m2+l*BT,r2+l*BT,residual,scratch,l_ln2w,l_ln2b,BT,C,sycl_queue);
    matmul_forward_onemath(l_fchg,l_ln2,l_fcw,l_fcb,B,T,C,4*C,sycl_queue,tmp,l_fch,0);
    matmul_forward_onemath(scratch,l_fchg,l_fpw,l_fpb,B,T,4*C,C,sycl_queue,tmp);
    if(l+1!=L){ fused_residual_forward5(l_res3,ln1+(l+1)*BT*C,m1+(l+1)*BT,r1+(l+1)*BT,l_res2,scratch,ln1w+(l+1)*C,ln1b+(l+1)*C,BT,C,sycl_queue); }
    else { fused_residual_forward5(l_res3,lnf,mf,rf,l_res2,scratch,lnfw,lnfb,BT,C,sycl_queue); }
    printf("layer %d done: ",l); chk("res3",l_res3,BT*C);
  }
  matmul_forward_onemath(logits,lnf,wte,nullptr,B,T,C,Vp,sycl_queue,tmp);
  chk("logits",logits,BT*Vp);
  double maxd=0; for(int bt=0;bt<BT;bt++)for(size_t v=0;v<V;v++){ float got=(float)logits[bt*Vp+v]; double d=fabs(got-explog[bt*V+v]); if(d>maxd)maxd=d; }
  printf("logit maxdiff vs pytorch=%g (tol 25)\n",maxd);
  return 0;
}
