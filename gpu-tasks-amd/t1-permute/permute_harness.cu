/*
Kernels to demonstrate permute operation.

Compile example:
nvcc -O3 permute.cu -o permute

The goal is to permute a 4D matrix from its original shape (dim1, dim2, dim3, dim4) to a new shape (dim4, dim3, dim1, dim2).

Before permutation, we need to understand how to access elements in a flattened (linear) form of the matrix.

Given:

dim1 = size of the 1st dimension
dim2 = size of the 2nd dimension
dim3 = size of the 3rd dimension
dim4 = size of the 4th dimension

For any element in a 4D matrix at position (i1, i2, i3, i4), where:

i1 is the index in dimension 1
i2 is the index in dimension 2
i3 is the index in dimension 3
i4 is the index in dimension 4

If you find it challenging to calculate the indices i1, i2, i3, and i4, observe the pattern in the index calculations.
Initially, it might take some time to grasp, but with practice, you'll develop a mental model for it.

To calculate the indices, use the following formulas:

i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
i2 = (idx / (dim3 * dim4)) % dim2;
i3 = (idx / dim4) % dim3;
i4 = idx % dim4;

Pattern Explanation:
To find the index for any dimension, divide the thread ID (idx) by the product of all subsequent dimensions.
Then, perform modulo operation with the current dimension.



The linear index in a flattened 1D array is calculated as:
linear_idx = i1 × ( dim2 × dim3 × dim4 ) + i2 × ( dim3 × dim4 ) + i3 × dim4 + i4
This linear index uniquely identifies the position of the element in the 1D array.

To permute the matrix, we need to rearrange the indices according to the new shape.
In this case, we are permuting from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2).

The new dimension post permutation will be as follows:

dim1 becomes the new 3rd dimension.
dim2 becomes the new 4th dimension.
dim3 becomes the new 2nd dimension.
dim4 becomes the new 1st dimension.

permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;

Here's how this works:

i4 * (dim3 * dim1 * dim2): This accounts for how many complete dim3 × dim1 × dim2 blocks fit before the current i4 block.
i3 * (dim1 * dim2): This accounts for the offset within the current i4 block, specifying which i3 block we are in.
i1 * dim2: This accounts for the offset within the current i3 block, specifying which i1 block we are in.
i2: This gives the offset within the current i1 block.

Lastly at the end we store the current value at idx index of the original value to the permuted index in the permuted_matrix.


--------------------------------------------------------------------------------------------------------------------------------------------------------

Similarly we can follow the above approach to permute matrices of any dimensions.

*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "common.h"

// CPU function to permute a 4D matrix
// museport: CPU reference permute_cpu() removed (it is the hidden check)

// CUDA kernel to permute a 4D matrix
__global__ void permute_kernel(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure index is within bounds
    if (idx < dim1 * dim2 * dim3 * dim4) {
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

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char** argv) {
    setup_main();
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> D1 D2 D3 D4\n", argv[0]);
        return 2;
    }
    const int D1 = atoi(argv[3]);
    const int D2 = atoi(argv[4]);
    const int D3 = atoi(argv[5]);
    const int D4 = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* matrix = (float*)malloc(sizeof(float) * (size_t)(D1*D2*D3*D4));
    if (fread(matrix, sizeof(float), (size_t)(D1*D2*D3*D4), fin) != (size_t)(D1*D2*D3*D4)) {
        fprintf(stderr, "input too short at matrix\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_matrix;
    cudaCheck(cudaMalloc(&d_matrix, sizeof(float) * (size_t)(D1*D2*D3*D4)));
    cudaCheck(cudaMemcpy(d_matrix, matrix, sizeof(float) * (size_t)(D1*D2*D3*D4), cudaMemcpyHostToDevice));
    float* out = (float*)malloc(sizeof(float) * (size_t)(D1*D2*D3*D4));
    float* d_out;
    cudaCheck(cudaMalloc(&d_out, sizeof(float) * (size_t)(D1*D2*D3*D4)));
    const int variants[] = {1};
    for (int vi = 0; vi < 1; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_out, 0, sizeof(float) * (size_t)(D1*D2*D3*D4)));
        permute_kernel<<<(D1*D2*D3*D4 + 255) / 256, 256>>>(d_matrix, d_out, D1, D2, D3, D4);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out, d_out, sizeof(float) * (size_t)(D1*D2*D3*D4), cudaMemcpyDeviceToHost));
        fwrite(out, sizeof(float), (size_t)(D1*D2*D3*D4), fout);
    }
    fclose(fout);
    return 0;
}
