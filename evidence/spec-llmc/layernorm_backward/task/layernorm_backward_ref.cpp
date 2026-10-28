// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/layernorm_backward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
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
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* weight = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(weight, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float *mean = (float*)malloc((size_t)B * T * sizeof(float)); float *rstd = (float*)malloc((size_t)B * T * sizeof(float)); for (int b = 0; b < B; ++b) { for (int t = 0; t < T; ++t) { const float *x = inp + (size_t)b * T * C + (size_t)t * C; float m = 0.0f; for (int i = 0; i < C; ++i) { m += x[i]; } m /= C; float v = 0.0f; for (int i = 0; i < C; ++i) { float d = x[i] - m; v += d * d; } v /= C; float s = 1.0f / sqrtf(v + 1e-5f); mean[(size_t)b * T + t] = m; rstd[(size_t)b * T + t] = s; } }
    float* dinp = (float*)calloc((size_t)(B*T*C), sizeof(float));
    float* dweight = (float*)calloc((size_t)(C), sizeof(float));
    float* dbias = (float*)calloc((size_t)(C), sizeof(float));
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    for (int r = 0; r < 9; r++) {  // one copy per variant
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
        fwrite(dweight, sizeof(float), (size_t)(C), fout);
        fwrite(dbias, sizeof(float), (size_t)(C), fout);
    }
    fclose(fout);
    return 0;
}
