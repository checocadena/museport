// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/fused_residual_forward.cu.
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

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
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
    float* weight = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(weight, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* bias = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(bias, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at bias\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* residual = (float*)calloc((size_t)(B*T*C), sizeof(float));
    float* normed = (float*)calloc((size_t)(B*T*C), sizeof(float));
    float* mean = (float*)calloc((size_t)(B*T), sizeof(float));
    float* rstd = (float*)calloc((size_t)(B*T), sizeof(float));
    residual_forward_cpu(residual, inp1, inp2, B*T*C); layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);
    for (int r = 0; r < 6; r++) {  // one copy per variant
        fwrite(residual, sizeof(float), (size_t)(B*T*C), fout);
        fwrite(normed, sizeof(float), (size_t)(B*T*C), fout);
        fwrite(mean, sizeof(float), (size_t)(B*T), fout);
        fwrite(rstd, sizeof(float), (size_t)(B*T), fout);
    }
    fclose(fout);
    return 0;
}
