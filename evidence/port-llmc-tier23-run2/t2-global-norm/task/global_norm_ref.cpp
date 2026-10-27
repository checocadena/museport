// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/global_norm.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

float global_norm_cpu(const float* data, size_t count) {
    // accumulate in double so we have an accurate numerical reference
    double acc = 0.0;
    for(size_t i = 0; i < count; ++i) {
        acc  += (double)data[i] * (double)data[i];
    }
    return (float)acc;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
        return 2;
    }
    const int N = atoi(argv[3]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(inp, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    for (size_t i = 0; i < (size_t)N; ++i) { inp[i] *= 1e-3; }
    float* out = (float*)calloc((size_t)(1), sizeof(float));
    out[0] = global_norm_cpu(inp, (size_t)N);
    for (int r = 0; r < 4; r++) {  // one copy per variant
        fwrite(out, sizeof(float), (size_t)(1), fout);
    }
    fclose(fout);
    return 0;
}
