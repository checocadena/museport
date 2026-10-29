#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
int main(){
  // fp32 host reference of full forward, print res3 rms per layer + first logits
  FILE* f = fopen("gpt2_124M_bf16.bin","rb");
  int header[256]; fread(header,sizeof(int),256,f);
  int maxT=header[2], V=header[3], L=header[4], NH=header[5], C=header[6], Vp=header[7];
  size_t wte_n=(size_t)Vp*C, wpe_n=(size_t)maxT*C, ln1_n=(size_t)L*C;
  size_t qkvw_n=(size_t)L*3*C*C, qkvb_n=(size_t)L*3*C;
  size_t ap_n=(size_t)L*C*C, apb_n=(size_t)L*C, fc_n=(size_t)L*4*C*C, fcb_n=(size_t)L*4*C, fcp_n=(size_t)L*C*4*C, fcpb_n=(size_t)L*C;
  std::vector<uint16_t> wte(wte_n), wpe(wpe_n), ln1w(ln1_n), ln1b(ln1_n), qkvw(qkvw_n), qkvb(qkvb_n), apw(ap_n), apb(apb_n);
  std::vector<uint16_t> ln2w(ln1_n), ln2b(ln1_n), fcw(fc_n), fcb(fcb_n), fcpw(fcp_n), fcpb(fcpb_n), lnfw(C), lnfb(C);
  fread(wte.data(),sizeof(uint16_t),wte_n,f); fread(wpe.data(),sizeof(uint16_t),wpe_n,f);
  fread(ln1w.data(),sizeof(uint16_t),ln1_n,f); fread(ln1b.data(),sizeof(uint16_t),ln1_n,f);
  fread(qkvw.data(),sizeof(uint16_t),qkvw_n,f); fread(qkvb.data(),sizeof(uint16_t),qkvb_n,f);
  fread(apw.data(),sizeof(uint16_t),ap_n,f); fread(apb.data(),sizeof(uint16_t),apb_n,f);
  fread(ln2w.data(),sizeof(uint16_t),ln1_n,f); fread(ln2b.data(),sizeof(uint16_t),ln1_n,f);
  fread(fcw.data(),sizeof(uint16_t),fc_n,f); fread(fcb.data(),sizeof(uint16_t),fcb_n,f);
  fread(fcpw.data(),sizeof(uint16_t),fcp_n,f); fread(fcpb.data(),sizeof(uint16_t),fcpb_n,f);
  fread(lnfw.data(),sizeof(uint16_t),C,f); fread(lnfb.data(),sizeof(uint16_t),C,f);
  fclose(f);
  auto bf16_to_f = [](uint16_t h){ uint32_t b = ((uint32_t)h)<<16; float f; memcpy(&f,&b,4); return f; };
  auto F=[&](std::vector<uint16_t>& h){ std::vector<float> o(h.size()); for(size_t i=0;i<h.size();i++) o[i]=bf16_to_f(h[i]); return o; };
  std::vector<float> WTE=F(wte), WPE=F(wpe), LN1W=F(ln1w), LN1B=F(ln1b), QKVW=F(qkvw), QKVB=F(qkvb), APW=F(apw), APB=F(apb);
  std::vector<float> LN2W=F(ln2w), LN2B=F(ln2b), FCW=F(fcw), FCB=F(fcb), FCPW=F(fcpw), FCPB=F(fcpb), LNFW=F(lnfw), LNFB=F(lnfb);
  FILE* s = fopen("gpt2_124M_debug_state.bin","rb");
  int sh[256]; fread(sh,sizeof(int),256,s);
  int B=sh[2], T=sh[3];
  std::vector<int> x(B*T); fread(x.data(),sizeof(int),B*T,s);
  std::vector<int> y(B*T); fread(y.data(),sizeof(int),B*T,s);
  std::vector<float> expL(B*T*V); fread(expL.data(),sizeof(float),B*T*V,s); fclose(s);
  int HS=C/NH, BNH=B*NH;
  auto ln_fwd=[&](std::vector<float>& out, std::vector<float>& mean, std::vector<float>& rstd, std::vector<float>& inp, float* w, float* b, int rows){
    for(int r=0;r<rows;r++){ double m=0; for(int c=0;c<C;c++) m+=inp[r*C+c]; m/=C;
      double v=0; for(int c=0;c<C;c++){double d=inp[r*C+c]-m; v+=d*d;} v/=C; double rs=1.0/sqrt(v+1e-5);
      mean[r]=m; rstd[r]=rs; for(int c=0;c<C;c++){ double n=rs*(inp[r*C+c]-m); out[r*C+c]=n*w[c]+b[c]; } }
  };
  auto mm=[&](std::vector<float>& out, std::vector<float>& inp, std::vector<float>& W, std::vector<float>& BIAS, int rows, int kin, int OC){
    for(int r=0;r<rows;r++) for(int oc=0;oc<OC;oc++){ double acc=(BIAS.empty()?0:BIAS[oc]); for(int k=0;k<kin;k++) acc+=(double)inp[r*kin+k]*W[oc*kin+k]; out[r*OC+oc]=acc; }
  };
  auto gelu=[&](std::vector<float>& io){ for(auto &v:io){ float xi=v; float cube=0.044715f*xi*xi*xi; v=0.5f*xi*(1.0f+tanhf(0.79788456f*(xi+cube))); } };
  std::vector<float> enc(B*T*C);
  for(int bt=0;bt<B*T;bt++){ int ix=x[bt], t=bt%T; for(int c=0;c<C;c++) enc[bt*C+c]=WTE[ix*C+c]+WPE[t*C+c]; }
  std::vector<float> ln1(B*T*C), m1(B*T), r1(B*T);
  ln_fwd(ln1,m1,r1,enc,LN1W.data(),LN1B.data(),B*T);
  std::vector<float> residual = enc;
  std::vector<float> l_ln1 = ln1;
  for(int l=0;l<L;l++){
    // att
    std::vector<float> qkv(B*T*3*C);
    { std::vector<float> Wl(QKVW.begin()+l*3*C*C, QKVW.begin()+(l+1)*3*C*C), Bl(QKVB.begin()+l*3*C, QKVB.begin()+(l+1)*3*C);
      mm(qkv, l_ln1, Wl, Bl, B*T, C, 3*C); }
    std::vector<float> q(BNH*T*HS), k(BNH*T*HS), v(BNH*T*HS);
    for(int b=0;b<B;b++) for(int nh=0;nh<NH;nh++) for(int n=0;n<T;n++) for(int d=0;d<HS;d++){
      int idx=((b*NH+nh)*T+n)*HS+d;
      q[idx]=qkv[(b*T+n)*3*C+nh*HS+d]; k[idx]=qkv[(b*T+n)*3*C+C+nh*HS+d]; v[idx]=qkv[(b*T+n)*3*C+2*C+nh*HS+d]; }
    float sc=1.f/sqrtf((float)HS);
    std::vector<float> att(BNH*T*T,0.f);
    for(int h=0;h<BNH;h++) for(int a=0;a<T;a++){ double mx=-1e30; for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=sc; if(s2>mx)mx=s2; }
      double sum=0; for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=sc; sum+=exp(s2-mx); }
      for(int b=0;b<=a;b++){ double s2=0; for(int d=0;d<HS;d++) s2+=k[(h*T+b)*HS+d]*q[(h*T+a)*HS+d]; s2*=sc; att[(h*T+a)*T+b]=exp(s2-mx)/sum; } }
    std::vector<float> atty(B*T*C);
    for(int b=0;b<B;b++) for(int n=0;n<T;n++) for(int nh=0;nh<NH;nh++) for(int d=0;d<HS;d++){ double r=0; for(int t2=0;t2<=n;t2++) r+=att[((b*NH+nh)*T+n)*T+t2]*v[((b*NH+nh)*T+t2)*HS+d]; atty[(b*T+n)*C+nh*HS+d]=r; }
    std::vector<float> proj(B*T*C);
    { std::vector<float> Wl(APW.begin()+l*C*C, APW.begin()+(l+1)*C*C), Bl(APB.begin()+l*C, APB.begin()+(l+1)*C); mm(proj, atty, Wl, Bl, B*T, C, C); }
    std::vector<float> res2(B*T*C), ln2(B*T*C), m2(B*T), r2(B*T);
    for(size_t i=0;i<res2.size();i++) res2[i]=residual[i]+proj[i];
    ln_fwd(ln2,m2,r2,res2,LN2W.data()+l*C,LN2B.data()+l*C,B*T);
    std::vector<float> fch(B*T*4*C);
    { std::vector<float> Wl(FCW.begin()+l*4*C*C, FCW.begin()+(l+1)*4*C*C), Bl(FCB.begin()+l*4*C, FCB.begin()+(l+1)*4*C); mm(fch, ln2, Wl, Bl, B*T, C, 4*C); }
    gelu(fch);
    std::vector<float> fcproj(B*T*C);
    { std::vector<float> Wl(FCPW.begin()+l*C*4*C, FCPW.begin()+(l+1)*C*4*C), Bl(FCPB.begin()+l*C, FCPB.begin()+(l+1)*C); mm(fcproj, fch, Wl, Bl, B*T, 4*C, C); }
    std::vector<float> res3(B*T*C);
    for(size_t i=0;i<res3.size();i++) res3[i]=res2[i]+fcproj[i];
    double s2=0; for(int i=0;i<C;i++) s2+=res3[i]*res3[i];
    printf("l=%d res3 rms=%f res3[0..3]=%f %f %f %f\n", l, sqrt(s2/C), res3[0],res3[1],res3[2],res3[3]);
    residual = res3;
    if(l+1!=L){ std::vector<float> nln1(B*T*C), nm(B*T), nr(B*T); ln_fwd(nln1,nm,nr,res3,LN1W.data()+(l+1)*C,LN1B.data()+(l+1)*C,B*T); l_ln1=nln1; }
    else { std::vector<float> lnf(B*T*C), mf(B*T), rf(B*T); ln_fwd(lnf,mf,rf,res3,LNFW.data(),LNFB.data(),B*T);
      std::vector<float> out(B*T*Vp, 0.f); mm(out, lnf, WTE, *(std::vector<float>*)nullptr, B*T, C, Vp);
      printf("logit-ref[0..3]: %f %f %f %f\n", out[0], out[1], out[2], out[3]);
      double md=0; for(int bt=0;bt<B*T;bt++) for(int v2=0;v2<V;v2++){ double d=fabs(expL[bt*V+v2]-out[bt*Vp+v2]); if(d>md)md=d; }
      printf("fp32-ref maxdiff=%f\n", md);
    }
  }
  return 0;
}
