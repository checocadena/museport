// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/encoder_forward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void encoder_forward_cpu(float* out,
                   const int* inp, const float* wte, const float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
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
    int* inp = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(inp, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* wte = (float*)malloc(sizeof(float) * (size_t)(V*C));
    if (fread(wte, sizeof(float), (size_t)(V*C), fin) != (size_t)(V*C)) {
        fprintf(stderr, "input too short at wte\n"); return 2;
    }
    float* wpe = (float*)malloc(sizeof(float) * (size_t)(T*C));
    if (fread(wpe, sizeof(float), (size_t)(T*C), fin) != (size_t)(T*C)) {
        fprintf(stderr, "input too short at wpe\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* out = (float*)calloc((size_t)(B*T*C), sizeof(float));
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);
    for (int r = 0; r < 3; r++) {  // one copy per variant
        fwrite(out, sizeof(float), (size_t)(B*T*C), fout);
    }
    fclose(fout);
    return 0;
}
