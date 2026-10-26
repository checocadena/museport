// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/matmul_backward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> B T C OC\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int OC = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*OC));
    if (fread(dout, sizeof(float), (size_t)(B*T*OC), fin) != (size_t)(B*T*OC)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* weight = (float*)malloc(sizeof(float) * (size_t)(OC*C));
    if (fread(weight, sizeof(float), (size_t)(OC*C), fin) != (size_t)(OC*C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* ones = (float*)malloc(sizeof(float) * (size_t)(OC));
    if (fread(ones, sizeof(float), (size_t)(OC), fin) != (size_t)(OC)) {
        fprintf(stderr, "input too short at ones\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* dinp = (float*)calloc((size_t)(B*T*C), sizeof(float));
    float* dweight = (float*)calloc((size_t)(OC*C), sizeof(float));
    float* dbias = (float*)calloc((size_t)(OC), sizeof(float));
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    for (int r = 0; r < 1; r++) {  // one copy per variant
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
        fwrite(dweight, sizeof(float), (size_t)(OC*C), fout);
        fwrite(dbias, sizeof(float), (size_t)(OC), fout);
    }
    fclose(fout);
    return 0;
}
