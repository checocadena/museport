// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/softmax_forward.cu.
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
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
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
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    if (fread(inp, sizeof(float), (size_t)(B*T*V), fin) != (size_t)(B*T*V)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    int* outliers = (int*)malloc(sizeof(int) * (size_t)(B*T*3));
    if (fread(outliers, sizeof(int), (size_t)(B*T*3), fin) != (size_t)(B*T*3)) {
        fprintf(stderr, "input too short at outliers\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    for (int k = 0; k < 3; ++k) { for (int j = 0; j < B * T; ++j) { inp[j * V + outliers[j * 3 + k]] *= 20; } }
    float* out = (float*)calloc((size_t)(B*T*V), sizeof(float));
    softmax_forward_cpu(out, inp, B * T, V);
    for (int r = 0; r < 8; r++) {  // one copy per variant
        fwrite(out, sizeof(float), (size_t)(B*T*V), fout);
    }
    fclose(fout);
    return 0;
}
