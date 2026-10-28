// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/classifier_fused.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int64_t i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}

void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int64_t bt = 0; bt < B * T; bt++) {
        // loss = -log(probs[target])
        const float* probs_bt = probs + bt * V;
        int ix = targets[bt];
        losses[bt] = -logf(probs_bt[ix]);
    }
}

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int64_t bt = 0; bt < B * T; bt++) {
        float* dlogits_bt = dlogits + bt * V;
        const float* probs_bt = probs + bt * V;
        float dloss = dlosses[bt];
        int ix = targets[bt];
        for (int i = 0; i < V; i++) {
            float p = probs_bt[i];
            float indicator = i == ix ? 1.0f : 0.0f;
            dlogits_bt[i] = (p - indicator) * dloss;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int V = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* logits = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    if (fread(logits, sizeof(float), (size_t)(B*T*V), fin) != (size_t)(B*T*V)) {
        fprintf(stderr, "input too short at logits\n"); return 2;
    }
    float* dlosses = (float*)malloc(sizeof(float) * (size_t)(B*T));
    if (fread(dlosses, sizeof(float), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at dlosses\n"); return 2;
    }
    int* targets = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(targets, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at targets\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* probs = (float*)malloc((size_t)B * T * V * sizeof(float)); for (size_t j = 0; j < (size_t)B * T; ++j) { for (int k = 0; k < 3; ++k) { size_t idx = (j * 1315423911u + (size_t)k * 2654435761u + 12345u) % (size_t)V; logits[j * (size_t)V + idx] *= 20.0f; } }
    float* losses = (float*)calloc((size_t)(B*T), sizeof(float));
    float* dlogits = (float*)calloc((size_t)(B*T*V), sizeof(float));
    softmax_forward_cpu(probs, logits, B * T, V); crossentropy_forward_cpu(losses, probs, targets, B, T, V); crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);
    for (int r = 0; r < 5; r++) {  // one copy per variant
        fwrite(losses, sizeof(float), (size_t)(B*T), fout);
        fwrite(dlogits, sizeof(float), (size_t)(B*T*V), fout);
    }
    fclose(fout);
    return 0;
}
