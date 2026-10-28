// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/trimat_forward.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void trimul_cpu(float* out, const float* inp,
                int B, int T, int C, int NH) {
    // inp shape: (B, T, 3, NH, HS)
    // out shape: (B, NH, T, T)
    int C3 = C*3;
    int HS = C / NH; // head size
    float scale = 1.0 / sqrtf(HS);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int nh = 0; nh < NH; nh++) {
                // Q[b][nh][t][:] = inp[b][t][0][nh][:] (where : is the slice operator for hs)
                const float* query_t = inp + b * T * C3 + t * C3 + nh * HS;
                // out[b][nh][t][:]
                float* out_bth = out + b * NH * T * T + nh * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                for (int t2 = 0; t2 <= t; t2++) {
                    // K[b][nh][t2][:] = inp[b][t2][1][nh][:]
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + nh * HS + C; // +C because it's key

                    // Q[b][nh][t][:] dot K[b][nh][t2][:]
                    float val = 0.0f;
                    for (int i = 0; i < HS; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;

                     // out[b][nh][t][t2] = val
                    out_bth[t2] = val;
                }
                for(int t2 = t + 1; t2 < T; ++t2) {
                    // causal mask, using NAN to supress warnings -> it could be -inf
                    // but it doesn't matter because in validate_result we ignore infinities/NANs
                    out_bth[t2] = NAN;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> B T C NH\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int NH = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*3*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*3*C), fin) != (size_t)(B*T*3*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* out = (float*)calloc((size_t)(B*NH*T*T), sizeof(float));
    trimul_cpu(out, inp, B, T, C, NH);
    for (int r = 0; r < 5; r++) {  // one copy per variant
        fwrite(out, sizeof(float), (size_t)(B*NH*T*T), fout);
    }
    fclose(fout);
    return 0;
}
