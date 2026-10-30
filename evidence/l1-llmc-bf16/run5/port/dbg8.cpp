#include <sycl/sycl.hpp>
#include <cstdio>
int main(){
  FILE* f=fopen("gpt2_124M_bf16.bin","rb");
  int h[256]; fread(h,sizeof(int),256,f);
  size_t Vp=h[7],C=h[6],maxT=h[2]; int L=h[4];
  fseek(f,(long)((Vp*C+maxT*C)*sizeof(sycl::ext::oneapi::bfloat16)),SEEK_CUR);
  sycl::ext::oneapi::bfloat16 buf[12*768];
  fread(buf,sizeof(buf[0]),(size_t)L*C,f);
  printf("ln1w layer0[0..7]: "); for(int i=0;i<8;i++) printf("%g ",(float)buf[i]); printf("\n");
  printf("ln1w layer1[0..7]: "); for(int i=0;i<8;i++) printf("%g ",(float)buf[C+i]); printf("\n");
  fread(buf,sizeof(buf[0]),(size_t)L*C,f);
  printf("ln1b layer0[0..7]: "); for(int i=0;i<8;i++) printf("%g ",(float)buf[i]); printf("\n");
  return 0;
}
