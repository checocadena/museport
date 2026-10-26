<!-- museport scan of karpathy/llm.c at f1e2ace (MIT), 2026-10-26; the root path is shortened. -->
# CUDA inventory: `karpathy/llm.c @ f1e2ace`

38 CUDA file(s), 18165 lines, 148 kernel(s) (tier 1: 19, tier 2: 6, tier 3: 13).

Tier 1: plain kernels. Tier 2: CUDA-only constructs that need a SYCL design. Tier 3: NVIDIA libraries.

## NVIDIA libraries

| Library | Uses | Files | oneAPI | Note |
|---|---|---|---|---|
| cuBLAS | 244 | 11 | oneMath BLAS |  |
| NCCL | 88 | 3 | oneCCL |  |
| cuBLASLt | 73 | 3 | oneMath BLAS | no direct equivalent: gemm plus a hand-written epilogue (bias, GELU) |
| cuDNN | 24 | 2 | oneDNN |  |

## CUDA-only constructs

| Construct | Uses | Files | Needs | SYCL |
|---|---|---|---|---|
| cooperative_groups | 300 | 10 | design | sycl::group / sub_group and group algorithms |
| shared_memory | 56 | 17 | mechanical | sycl::local_accessor |
| atomics | 49 | 5 | mechanical | sycl::atomic_ref |
| streams_events | 42 | 12 | mechanical | sycl::queue (in-order for stream semantics) and sycl::event |
| warp_size_32 | 32 | 4 | design | SYCL sub-group size is not guaranteed to be 32; query it or request one |
| bf16 | 26 | 4 | design | sycl::ext::oneapi::bfloat16 |
| warp_shuffle | 22 | 6 | design | sycl::shift_group_left / permute_group_by_xor / select_from_group |
| unified_memory | 3 | 3 | mechanical | sycl::malloc_shared |

## Files

Tier is set by what a file uses itself. Names in *italics* come from a local header it includes: the header must be ported too, but the file may not use them.

| File | Lines | Tier | Kernels | Libraries | Design constructs |
|---|---|---|---|---|---|
| dev/test/device_file_io.cu | 64 | 1 | 0 | - | *warp_size_32*, *bf16* |
| llmc/gelu.cuh | 66 | 1 | 2 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| profile_gpt2.cu | 74 | 1 | 0 | *cuBLASLt*, *cuBLAS*, *cuDNN*, *NCCL* | *warp_size_32*, *bf16*, *warp_shuffle* |
| llmc/global_norm.cuh | 89 | 1 | 2 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| llmc/adamw.cuh | 98 | 1 | 2 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| dev/cuda/crossentropy_forward.cu | 148 | 1 | 1 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| llmc/fused_classifier.cuh | 149 | 1 | 1 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| dev/cuda/crossentropy_softmax_backward.cu | 165 | 1 | 1 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/residual_forward.cu | 167 | 1 | 2 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/gelu_forward.cu | 174 | 1 | 2 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/permute.cu | 181 | 1 | 1 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/gelu_backward.cu | 195 | 1 | 2 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/encoder_backward.cu | 202 | 1 | 2 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/adamw.cu | 226 | 1 | 2 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| llmc/encoder.cuh | 234 | 1 | 3 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| dev/cuda/encoder_forward.cu | 260 | 1 | 3 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| test_gpt2.cu | 390 | 1 | 0 | *cuBLASLt*, *cuBLAS*, *cuDNN*, *NCCL* | *warp_size_32*, *bf16*, *warp_shuffle* |
| llmc/layernorm.cuh | 505 | 1 | 5 | - | *warp_size_32*, *bf16*, *warp_shuffle* |
| dev/cuda/fused_residual_forward.cu | 695 | 1 | 7 | *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/global_norm.cu | 270 | 2 | 5 | *cuBLASLt*, *cuBLAS* | cooperative_groups, *warp_size_32*, *warp_shuffle*, *bf16* |
| llmc/cuda_utils.cuh | 286 | 2 | 2 | - | warp_shuffle, bf16, *warp_size_32* |
| dev/cuda/layernorm_forward.cu | 629 | 2 | 8 | *cuBLASLt*, *cuBLAS* | cooperative_groups, *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/softmax_forward.cu | 733 | 2 | 8 | *cuBLASLt*, *cuBLAS* | warp_size_32, warp_shuffle, cooperative_groups, *bf16* |
| dev/cuda/classifier_fused.cu | 772 | 2 | 5 | *cuBLASLt*, *cuBLAS* | warp_shuffle, cooperative_groups, bf16, *warp_size_32* |
| dev/cuda/layernorm_backward.cu | 1550 | 2 | 11 | *cuBLASLt*, *cuBLAS* | warp_size_32, cooperative_groups, bf16, *warp_shuffle* |
| dev/cuda/nccl_all_reduce.cu | 198 | 3 | 1 | NCCL, *cuBLASLt*, *cuBLAS* | *warp_size_32*, *warp_shuffle*, *bf16* |
| test_gpt2_fp32.cu | 235 | 3 | 0 | cuBLAS | *warp_size_32*, *cooperative_groups* |
| llmc/attention.cuh | 276 | 3 | 6 | cuBLAS, *cuBLASLt* | *warp_size_32*, *bf16*, *warp_shuffle* |
| dev/cuda/matmul_backward.cu | 282 | 3 | 2 | cuBLAS, *cuBLASLt* | *warp_size_32*, *warp_shuffle*, *bf16* |
| llmc/matmul.cuh | 290 | 3 | 2 | cuBLASLt, cuBLAS | warp_shuffle, *warp_size_32*, *bf16* |
| dev/cuda/matmul_forward.cu | 440 | 3 | 3 | cuBLASLt, cuBLAS | *warp_size_32*, *warp_shuffle*, *bf16* |
| llmc/zero.cuh | 597 | 3 | 0 | NCCL | - |
| dev/cuda/trimat_forward.cu | 650 | 3 | 2 | cuBLAS, *cuBLASLt* | cooperative_groups, *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/matmul_backward_bias.cu | 670 | 3 | 10 | cuBLAS, *cuBLASLt* | warp_size_32, warp_shuffle, cooperative_groups, *bf16* |
| dev/cuda/attention_backward.cu | 1155 | 3 | 13 | cuBLAS, *cuBLASLt* | cooperative_groups, *warp_size_32*, *warp_shuffle*, *bf16* |
| dev/cuda/attention_forward.cu | 1391 | 3 | 15 | cuBLAS, cuDNN, *cuBLASLt* | warp_shuffle, cooperative_groups, bf16, *warp_size_32* |
| train_gpt2_fp32.cu | 1755 | 3 | 17 | cuBLAS | warp_size_32, cooperative_groups |
| train_gpt2.cu | 1904 | 3 | 0 | cuBLASLt, cuBLAS, cuDNN, NCCL | *warp_size_32*, *bf16*, *warp_shuffle* |
