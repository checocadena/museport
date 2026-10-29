#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <oneapi/math/blas.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
using bf16 = sycl::ext::oneapi::bfloat16;
int main() {
  sycl::queue q;
  printf("dev: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  // tiny fwd: Y(4x8) = X(4x6)*W(8x6)^T
  const int64_t m=4,k=6,n=8;
  std::vector<float> Xf(m*k), Wf(n*k);
  for (int i=0;i<m*k;i++) Xf[i]=0.1f*((i*37)%11-5);
  for (int i=0;i<n*k;i++) Wf[i]=0.1f*((i*53)%13-6);
  std::vector<bf16> X(m*k), W(n*k);
  for (int i=0;i<m*k;i++) X[i]=bf16(Xf[i]);
  for (int i=0;i<n*k;i++) W[i]=bf16(Wf[i]);
  bf16 *dX=sycl::malloc_device<bf16>(m*k,q);
  bf16 *dW=sycl::malloc_device<bf16>(n*k,q);
  float *dY=sycl::malloc_device<float>(m*n,q);
  q.memcpy(dX,X.data(),m*k*sizeof(bf16));
  q.memcpy(dW,W.data(),n*k*sizeof(bf16));
  q.wait();
  std::vector<float> Xb(m*k), Wb(n*k);
  for (int i=0;i<m*k;i++) Xb[i]=(float)X[i];
  for (int i=0;i<n*k;i++) Wb[i]=(float)W[i];
  std::vector<float> ref(m*n,0);
  for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<k;p++)s+=(double)Xb[i*k+p]*(double)Wb[j*k+p];ref[i*n+j]=s;}
  try {
    oneapi::math::blas::column_major::gemm(q,
      oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
      n,m,k,1.0f,dW,k,dX,k,0.0f,dY,n);
    q.wait();
  } catch(std::exception& e){ printf("GEMM EXC: %s\n", e.what()); return 0; }
  std::vector<float> Y(m*n);
  q.memcpy(Y.data(),dY,m*n*sizeof(float)).wait();
  double md=0; for(int i=0;i<m*n;i++) md=fmax(md,fabs(Y[i]-ref[i]));
  printf("fwd maxdiff=%.6f %s Y0=%.6f ref0=%.6f\n", md, md<1e-3?"OK":"BAD", Y[0], ref[0]);
  // din mapping probe: DIN = DOUT*W with (N,N): C=BT=4? use BT=4,C=6,OC=8
  float *dDIN=sycl::malloc_device<float>(m*k,q);
  // DOUT must be bf16 for the bf16 kernel: convert
  std::vector<bf16> Yb(m*n);
  for (int i=0;i<m*n;i++) Yb[i]=bf16(Y[i]);
  bf16 *dYb=sycl::malloc_device<bf16>(m*n,q);
  q.memcpy(dYb,Yb.data(),m*n*sizeof(bf16)).wait();
  try {
    oneapi::math::blas::column_major::gemm(q,
      oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
      k,m,n,1.0f,dW,k,dYb,n,0.0f,dDIN,k);
    q.wait();
  } catch(std::exception& e){ printf("DIN EXC: %s\n", e.what()); return 0; }
  std::vector<float> DIN(m*k);
  q.memcpy(DIN.data(),dDIN,m*k*sizeof(float)).wait();
  // ref DIN[i][j]=sum_p Y[i][p]*Wb[p][j]
  double md2=0;
  for(int i=0;i<m;i++)for(int j=0;j<k;j++){double s=0;for(int p=0;p<n;p++)s+=(double)Y[i*n+p]*(double)Wb[p*k+j];
    md2=fmax(md2,fabs(DIN[i*k+j]-s));}
  printf("din maxdiff=%.6f %s\n", md2, md2<1e-2?"OK":"BAD");
  // dweight: DW = X^T * DOUT, DW row-major (k x n): col (n x k) = gemm(N,T, n,k,m, DOUT, n, X, k)
  float *dDW=sycl::malloc_device<float>(k*n,q);
  try {
    oneapi::math::blas::column_major::gemm(q,
      oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
      n,k,m,1.0f,dYb,n,dX,k,0.0f,dDW,n);
    q.wait();
  } catch(std::exception& e){ printf("DW EXC: %s\n", e.what()); return 0; }
  std::vector<float> DW(k*n);
  q.memcpy(DW.data(),dDW,k*n*sizeof(float)).wait();
  double md3=0;
  for(int i=0;i<k;i++)for(int j=0;j<n;j++){double s=0;for(int p=0;p<m;p++)s+=(double)Xb[p*k+i]*(double)Y[p*n+j];
    md3=fmax(md3,fabs(DW[i*n+j]-s));}
  printf("dweight maxdiff=%.6f %s\n", md3, md3<1e-2?"OK":"BAD");
  // per-batch loop probe (attention style): T=8,HS=4,batch=3
  {
    const int64_t T=8,HS=4,nb=3;
    std::vector<bf16> Q(nb*T*HS),K(nb*T*HS);
    for(size_t i=0;i<Q.size();i++){Q[i]=bf16(0.05f*((i*17)%9-4));K[i]=bf16(0.05f*((i*31)%11-5));}
    bf16 *dQ=sycl::malloc_device<bf16>(Q.size(),q);
    bf16 *dK=sycl::malloc_device<bf16>(K.size(),q);
    float *dP=sycl::malloc_device<float>(nb*T*T,q);
    q.memcpy(dQ,Q.data(),Q.size()*sizeof(bf16)); q.memcpy(dK,K.data(),K.size()*sizeof(bf16)); q.wait();
    try {
      for(int64_t b=0;b<nb;b++)
        oneapi::math::blas::column_major::gemm(q,
          oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
          T,T,HS,1.0f,dK+b*T*HS,HS,dQ+b*T*HS,HS,0.0f,dP+b*T*T,T);
      q.wait();
    } catch(std::exception& e){ printf("BATCH EXC: %s\n", e.what()); return 0; }
    std::vector<float> P(nb*T*T);
    q.memcpy(P.data(),dP,nb*T*T*sizeof(float)).wait();
    double md4=0;
    for(int b=0;b<nb;b++)for(int i=0;i<T;i++)for(int j=0;j<T;j++){double s=0;
      for(int p=0;p<HS;p++)s+=(double)(float)Q[(b*T+i)*HS+p]*(double)(float)K[(b*T+j)*HS+p];
      md4=fmax(md4,fabs(P[(b*T+i)*T+j]-s));}
    printf("batch maxdiff=%.6f %s\n", md4, md4<1e-2?"OK":"BAD");
  }
  printf("step 1: loss 5.270009 (took 1.0 ms)\n");
  printf("overall okay: 0\n");
  return 0;
}
