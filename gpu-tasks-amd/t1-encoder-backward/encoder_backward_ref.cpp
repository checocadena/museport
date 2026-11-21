// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/encoder_backward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void encoder_backward_cpu(float* dwte, float* dwpe,
                            float* dout, int* inp,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int V = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    int* inp = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(inp, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* dwte = (float*)calloc((size_t)(V*C), sizeof(float));
    float* dwpe = (float*)calloc((size_t)(T*C), sizeof(float));
    encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);
    for (int r = 0; r < 2; r++) {  // one copy per variant
        fwrite(dwte, sizeof(float), (size_t)(V*C), fout);
        fwrite(dwpe, sizeof(float), (size_t)(T*C), fout);
    }
    fclose(fout);
    return 0;
}
