// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/residual_forward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp1 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp1, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp1\n"); return 2;
    }
    float* inp2 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp2, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp2\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* out = (float*)calloc((size_t)(B*T*C), sizeof(float));
    residual_forward_cpu(out, inp1, inp2, B * T * C);
    for (int r = 0; r < 2; r++) {  // one copy per variant
        fwrite(out, sizeof(float), (size_t)(B*T*C), fout);
    }
    fclose(fout);
    return 0;
}
