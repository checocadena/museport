// Hidden reference for museport (generated; not shown to the agent).
// CPU functions extracted verbatim from llm.c dev/cuda/permute.cu.
#define _USE_MATH_DEFINES
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float floatX;  // the fp32 build, as in the harness

void permute_cpu(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int total_threads = dim1 * dim2 * dim3 * dim4;

    for (int idx = 0; idx < total_threads; idx++) {
        // Calculate the 4D indices from the linear index
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // Compute the new index for the permuted matrix
        // Transpose from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        out_matrix[permuted_idx] = matrix[idx];
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> dim1 dim2 dim3 dim4\n", argv[0]);
        return 2;
    }
    const int dim1 = atoi(argv[3]);
    const int dim2 = atoi(argv[4]);
    const int dim3 = atoi(argv[5]);
    const int dim4 = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* matrix = (float*)malloc(sizeof(float) * (size_t)(dim1*dim2*dim3*dim4));
    if (fread(matrix, sizeof(float), (size_t)(dim1*dim2*dim3*dim4), fin) != (size_t)(dim1*dim2*dim3*dim4)) {
        fprintf(stderr, "input too short at matrix\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* permuted_matrix = (float*)calloc((size_t)(dim1*dim2*dim3*dim4), sizeof(float));
    permute_cpu(matrix, permuted_matrix, dim1, dim2, dim3, dim4);
    for (int r = 0; r < 1; r++) {  // one copy per variant
        fwrite(permuted_matrix, sizeof(float), (size_t)(dim1*dim2*dim3*dim4), fout);
    }
    fclose(fout);
    return 0;
}
