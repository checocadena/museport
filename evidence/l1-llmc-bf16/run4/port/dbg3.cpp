#include <cstdio>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>
#include "llmc/sycl_common.h"
#include "llmc/sycl_utils.h"
#include "llmc/onemath_common.h"
#include "llmc/encoder.h"
#include "llmc/layernorm.h"
#include "llmc/matmul.h"
#include "llmc/attention.h"
int main(){
  FILE* f = fopen("gpt2_124M_bf16.bin","rb");
  int header[256]; fread(header,sizeof(int),256,f);
  int maxT=header[2], V=header[3], L=header[4], NH=header[5], C=header[6], Vp=header[7];
  size_t wte_n=(size_t)Vp*C, wpe_n=(size_t)maxT*C, ln1_n=(size_t)L*C;
  size_t qkvw_n=(size_t)L*3*C*C, qkvb_n=(size_t)L*3*C;
  size_t ap_n=(size_t)L*C*C, apb_n=(size_t)L*C;
  std::vector<floatX> wte(wte_n), wpe(wpe_n), ln1w(ln1_n), ln1b(ln1_n), qkvw(qkvw_n), qkvb(qkvb_n), apw(ap_n), apb(apb_n);
  std::vector<floatX> ln2w(ln1_n), ln2b(ln1_n);
  fread(wte.data(),sizeof(floatX),wte_n,f); fread(wpe.data(),sizeof(floatX),wpe_n,f);
  fread(ln1w.data(),sizeof(floatX),ln1_n,f); fread(ln1b.data(),sizeof(floatX),ln1_n,f);
  fread(qkvw.data(),sizeof(floatX),qkvw_n,f); fread(qkvb.data(),sizeof(floatX),qkvb_n,f);
  fread(apw.data(),sizeof(floatX),ap_n,f); fread(apb.data(),sizeof(floatX),apb_n,f);
  fread(ln2w.data(),sizeof(floatX),ln1_n,f); fread(ln2b.data(),sizeof(floatX),ln1_n,f);
  fclose(f);
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s); fclose(s);
  int HS = C/NH;
  // ---- host fp32 reference for layer 0 attention ----
  // encoder + ln1 on host (fp32)
  std::vector<float> enc(B*T*C), ln(B*T*C);
  for(int bt=0;bt<B*T;bt++){ int ix=x[bt], t=bt%T; for(int c=0;c<C;c++) enc[bt*C+c]=(float)wte[ix*C+c]+(float)wpe[t*C+c]; }
  for(int bt=0;bt<B*T;bt++){ double m=0; for(int c=0;c<C;c++) m+=enc[bt*C+c]; m/=C;
    double v=0; for(int c=0;c<C;c++){double d=enc[bt*C+c]-m; v+=d*d;} v/=C; double rs=1.0/sqrt(v+1e-5);
    for(int c=0;c<C;c++){ double n=rs*(enc[bt*C+c]-m); ln[bt*C+c]=(float)(n*(float)ln1w[c]+(float)ln1b[c]); } }
  // qkv
  std::vector<float> qkv(B*T*3*C);
  for(int bt=0;bt<B*T;bt++) for(int oc=0;oc<3*C;oc++){ double r=(float)qkvb[oc]; for(int k=0;k<C;k++) r+=(double)ln[bt*C+k]*(float)qkvw[oc*C+k]; qkv[bt*3*C+oc]=(float)r; }
  // permute to heads
  int BNH=B*NH;
  std::vector<float> q(BNH*T*HS), k(BNH*T*HS), v(BNH*T*HS);
  for(int b=0;b<B;b++) for(int nh=0;nh<NH;nh++) for(int n=0;n<T;n++) for(int d=0;d<HS;d++){
    int idx=((b*NH+nh)*T+n)*HS+d;
    q[idx]=qkv[(b*T+n)*3*C+nh*HS+d]; k[idx]=qkv[(b*T+n)*3*C+C+nh*HS+d]; v[idx]=qkv[(b*T+n)*3*C+2*C+nh*HS+d];
  }
  float scale = 1.f/sqrtf((float)HS);
  std::vector<float> att(BNH*T*T, 0.f);
  for(int h=0;h<BNH;h++) for(int a=0;a<T;a++){ double mx=-1e30; for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=scale; if(s2>mx)mx=s2; }
    double sum=0; for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=scale; sum+=exp(s2-mx); }
    for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=scale; att[(h*T+a)*T+b]=exp(s2-mx)/sum; } }
  std::vector<float> y(BNH*T*HS,0.f);
  for(int h=0;h<BNH;h++) for(int a=0;a<T;a++) for(int d=0;d<HS;d++){ double r=0; for(int b=0;b<=a;b++) r+=att[(h*T+a)*T+b]*v[(h*T+b)*HS+d]; y[(h*T+a)*HS+d]=r; }
  std::vector<float> atty(B*T*C);
  for(int b=0;b<B;b++) for(int n=0;n<T;n++) for(int nh=0;nh<NH;nh++) for(int d=0;d<HS;d++) atty[(b*T+n)*C+nh*HS+d]=y[((b*NH+nh)*T+n)*HS+d];
  printf("atty-ref[0..7]: "); for(int i=0;i<8;i++) printf("%f ",atty[i]); printf("\n");
  // device
  floatX *d_wte=sycl_malloc_device<floatX>(wte_n), *d_wpe=sycl_malloc_device<floatX>(wpe_n);
  floatX *d_ln1w=sycl_malloc_device<floatX>(ln1_n), *d_ln1b=sycl_malloc_device<floatX>(ln1_n);
  floatX *d_qkvw=sycl_malloc_device<floatX>(qkvw_n), *d_qkvb=sycl_malloc_device<floatX>(qkvb_n);
  int *d_inp=sycl_malloc_device<int>(B*T);
  floatX *d_enc=sycl_malloc_device<floatX>((size_t)B*T*C), *d_ln1=sycl_malloc_device<floatX>((size_t)B*T*C);
  float *d_mean=sycl_malloc_device<float>(B*T), *d_rstd=sycl_malloc_device<float>(B*T);
  floatX *d_qkv=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_att=sycl_malloc_device<floatX>((size_t)B*NH*T*T);
  floatX *d_qkvr=sycl_malloc_device<floatX>((size_t)B*T*3*C), *d_atty=sycl_malloc_device<floatX>((size_t)B*T*C);
  sycl_memcpy_h2d(d_wte,wte.data(),wte_n*sizeof(floatX)); sycl_memcpy_h2d(d_wpe,wpe.data(),wpe_n*sizeof(floatX));
  sycl_memcpy_h2d(d_ln1w,ln1w.data(),ln1_n*sizeof(floatX)); sycl_memcpy_h2d(d_ln1b,ln1b.data(),ln1_n*sizeof(floatX));
  sycl_memcpy_h2d(d_qkvw,qkvw.data(),qkvw_n*sizeof(floatX)); sycl_memcpy_h2d(d_qkvb,qkvb.data(),qkvb_n*sizeof(floatX));
  sycl_memcpy_h2d(d_inp,x.data(),B*T*sizeof(int));
  encoder_forward(d_enc, d_inp, d_wte, d_wpe, B, T, C);
  layernorm_forward(d_ln1, d_mean, d_rstd, d_enc, d_ln1w, d_ln1b, B, T, C);
  matmul_forward_cublaslt(d_qkv, d_ln1, d_qkvw, d_qkvb, B, T, C, 3*C);
  sycl_memset_sync(d_att, 0, (size_t)B*NH*T*T*sizeof(floatX));
  attention_forward(d_atty, d_qkvr, d_att, d_qkv, B, T, C, NH);
  std::vector<floatX> hatty(B*T*C);
  sycl_memcpy_d2h(hatty.data(), d_atty, hatty.size()*sizeof(floatX));
  printf("atty-dev[0..7]: "); for(int i=0;i<8;i++) printf("%f ",(float)hatty[i]); printf("\n");
  double md=0; for(size_t i=0;i<hatty.size();i++) md=std::fmax(md,std::fabs(atty[i]-(float)hatty[i]));
  printf("maxdiff=%f\n",md);
  return 0;
}
