# Headline: port llm.c's GPT-2 fp32 training to SYCL (plan G)

## G1. Survey of `train_gpt2_fp32.cu` (llm.c `f1e2ace`)

- **Size:** 1,754 lines.
- **Test:** `test_gpt2_fp32.cu` (234 lines) `#include`s the training file.
- **Local headers:** `llmc/utils.h`, `llmc/tokenizer.h` and `llmc/dataloader.h`. All three are
  host code (file I/O and a tokenizer), with no CUDA in them.

### Kernels, mapped to museport's ports

| `train_gpt2_fp32.cu` | Ported by Muse from `dev/cuda`? | Evidence |
|---|---|---|
| `encoder_forward_kernel3` | yes | `port-llmc-tier1-run2/t1-encoder-forward` |
| `encoder_backward_kernel` | yes | `t1-encoder-backward` |
| `layernorm_forward_kernel3` | yes | `port-llmc-tier2/t2-layernorm-forward` |
| `layernorm_backward_kernel2` | yes | `port-llmc-reruns-2/t2-layernorm-backward` |
| `residual_forward_kernel` | yes | `t1-residual-forward` |
| `gelu_forward_kernel`, `gelu_backward_kernel` | yes | `t1-gelu-forward`, `t1-gelu-backward` |
| `matmul_forward_kernel4` | yes | `port-llmc-tier3/t3-matmul-forward` |
| `matmul_backward_bias_kernel4` | yes | `t3-matmul-backward-bias` |
| `adamw_kernel2` | yes | `t1-adamw` |
| `fused_classifier_kernel3` | yes | `port-llmc-tier2/t2-classifier-fused` |
| `permute_kernel`, `unpermute_kernel` (and their backward kernels) | no: they belong to `attention_forward.cu` and `attention_backward.cu` (plan C4) | - |
| `softmax_forward_kernel5` | no (attention) | - |
| `softmax_autoregressive_backward_kernel` | no (attention) | - |

12 of the 17 kernels have a Muse port that passed its hidden check. Each kernel version used
here is one of the variants ported. The `dev/cuda` copies can differ from the training file in
small details: for example, `matmul_forward_kernel4` is `__launch_bounds__(16*16)` in
`dev/cuda` and `(16*16, 2)` in the training file. The other 5 are the
attention kernels.

### NVIDIA library calls

- **`cublasSgemm`:** in `matmul_backward`, maps to `oneapi::math::blas::column_major::gemm`.
- **`cublasSgemmStridedBatched`:** 6 calls in `attention_forward` and `attention_backward`,
  map to `gemm_batch`.
- **cuBLAS setup:** `cublasCreate` and `cublasSetMathMode`. The test sets TF32 off.

No cuBLASLt and no cuDNN in the fp32 file.

### Data

From `dev/download_starter_pack.sh` (Hugging Face `karpathy/llmc-starter-pack`, reachable from
this session):

- `gpt2_124M.bin`: the GPT-2 124M weights, about 500 MB.
- `gpt2_124M_debug_state.bin`: the PyTorch reference for one batch (B=4, T=64): the inputs,
  targets, logits, loss and every parameter gradient.
- For training only: `gpt2_tokenizer.bin` and the Tiny Shakespeare token files.

### What llm.c's test checks

`test_gpt2_fp32.cu` builds the model from `gpt2_124M.bin` and turns TF32 off. It runs 10
training steps on the debug batch and checks:

- **Step 0:** logits within 1e-2 of PyTorch's, the loss within 1e-2, and all parameter
  gradients (`check_tensor`).
- **Steps 0-9:** each step's loss within 1e-2 of PyTorch's expected losses.
- **Result:** it prints `overall okay: 1` or `0`.

## The hidden check for the port

llm.c's test program is part of the code being ported, so Muse would see its checks and could
edit them. museport keeps the check hidden, as it does for the kernels:

1. **Harness:** the CUDA harness is `test_gpt2_fp32.cu`'s program with its checks taken out.
   It writes raw float32 output: the step-0 logits, the step-0 loss, the step-0 gradients, and
   the 10 losses.
2. **Hidden reference:** a small C++ program that reads `gpt2_124M_debug_state.bin` and the
   expected losses, and writes the same layout. Verify compares them in array mode, within
   llm.c's own 1e-2.
3. **Proof the check works before any model runs ($0):** llm.c's CPU `train_gpt2.c`, built with
   the same output code, must pass it on this machine.

## G3. Staging the port (one task per stage, each with its own hidden check)

1. **Forward:** step-0 logits and loss. This needs the 12 ported kernels, `matmul_forward`,
   and the attention forward pass, which uses `gemm_batch` and the softmax.
2. **Backward:** step-0 gradients. This adds `matmul_backward` (`gemm`) and attention backward.
3. **Training:** the 10 losses, which adds AdamW.

Each stage starts from the previous stage's passing sources.

**What Muse gets:** the whole training file (about 70,000 characters) and the stage's goal,
plus its own earlier kernel ports as files it may read. It may reuse them, but nothing is
copied in for it.

**Where it runs:**
- On the CPU device here, with oneMath's MKL-CPU backend.
- On an NVIDIA GPU only after plan D2: oneMath with its cuBLAS backend on the GPU host, a
  toolkit change.
- AMD needs oneMath's rocBLAS backend, so not this week.

## Cost estimate

- **Muse:** about $1.50-2.50 per stage, so about $5-7 for all three stages. This assumes about
  20 turns per stage, with about 100k tokens of input per turn, much of it served from cache.
- **Proving the hidden check:** $0 (CPU).
- **GPU:** none until D2.

**Proposed cap: $7** for G3's Muse runs, with $3 per stage. This fits the $10.64 left under the
$20 comparison cap.
