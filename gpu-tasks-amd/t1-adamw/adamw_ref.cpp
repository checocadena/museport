// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/adamw.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // adapted from: train_gpt2.c

    for (int i = 0; i < num_parameters; i++) {
        float param = params_memory[i];
        float grad = grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
        return 2;
    }
    const int N = atoi(argv[3]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* params = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(params, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at params\n"); return 2;
    }
    float* grads = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(grads, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at grads\n"); return 2;
    }
    float* m = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(m, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at m\n"); return 2;
    }
    float* v = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(v, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at v\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* params_out = (float*)calloc((size_t)(N), sizeof(float));
    memcpy(params_out, params, sizeof(float) * (size_t)(N));
    float* m_out = (float*)calloc((size_t)(N), sizeof(float));
    memcpy(m_out, m, sizeof(float) * (size_t)(N));
    float* v_out = (float*)calloc((size_t)(N), sizeof(float));
    memcpy(v_out, v, sizeof(float) * (size_t)(N));
    adamw_cpu(params_out, grads, m_out, v_out, 10, N);
    for (int r = 0; r < 2; r++) {  // one copy per variant
        fwrite(params_out, sizeof(float), (size_t)(N), fout);
        fwrite(m_out, sizeof(float), (size_t)(N), fout);
        fwrite(v_out, sizeof(float), (size_t)(N), fout);
    }
    fclose(fout);
    return 0;
}
