# J3: `llm.c-sycl`, a reassembled repo (plan, Oct 28)

**Goal:** the deliverable "point it at a CUDA repo, get a oneAPI repo" promises. It means one
project that:
- holds every ported piece of llm.c;
- builds with `make` and `icpx -fsycl`;
- passes **llm.c's own test** on a CPU, an NVIDIA GPU and an AMD GPU.

So far each port lives in its own evidence folder with a contract `main` made for checking. J3
puts them back together the way llm.c is laid out.

**Scope: llm.c's fp32 path, plus the kernel workbench.** The bf16 production trainer
(`train_gpt2.cu`: cuBLASLt, cuDNN, ZeRO, multi-GPU) is K2 and is out of scope here.

## What exists, and what is missing

| llm.c file | What we have | Missing for J3 |
|---|---|---|
| `train_gpt2_fp32.cu` (1,754 lines) | Muse's GPT-2 stage-3 port (907 lines): forward, backward, AdamW; passes on CPU, NVIDIA, AMD | llm.c's real `main`: the arguments (`-i -j -b -t -l -v -m -s -g`), the dataloaders, validation loss, sampling with the tokenizer, and timing |
| `test_gpt2_fp32.cu` (234 lines) | its check, reproduced by our stage-3 harness | the test itself, ported: it `#include`s the trainer and prints `overall okay: 1` |
| `llmc/utils.h`, `tokenizer.h`, `dataloader.h`, `rand.h` | plain C, no CUDA | nothing: used unchanged (MIT license kept) |
| `dev/cuda/*.cu` (the kernel workbench) | a correct Muse port of all 19 files we ported (18 in run 2, plus the `softmax_forward` rerun) | a `dev/sycl/` folder with each port and its harness, and a Makefile that builds and checks them |
| `Makefile` | nothing | SYCL targets: `train_gpt2fp32sycl`, `test_gpt2fp32sycl`, `dev/sycl` kernels; `TARGET=cpu|nvidia|amd`; oneMath linking |

## Steps

1. **Assemble, no model ($0).**
   - Copy the 19 kernel ports and their harnesses into `dev/sycl/`, with a provenance line
     each: run, item, commit.
   - Copy llm.c's plain-C headers, `LICENSE` and the data download script.
   - Write the Makefile.
   - **Check:** every `dev/sycl` port builds and passes its hidden check on the CPU.
2. **Muse ports the trainer's `main`** (one agent-loop run):
   - **Task:** the stage-3 port plus llm.c's `train_gpt2_fp32.cu`. Muse adds llm.c's real
     `main` around its model code, keeping the model code it already proved.
   - **Hidden check:** train on tiny-shakespeare for a fixed number of steps with fixed
     settings. The printed train and validation losses must match llm.c's own CUDA
     `train_gpt2_fp32.cu` within 1e-2. That reference is recorded on an NVIDIA GPU, the J1
     way.
   - **Not checked:** sampled text, since sampling can flip on ties.
   - **Data:** `gpt2_tokenizer.bin` and the tiny-shakespeare `.bin` files from llm.c's
     starter pack, checked by sha256.
3. **Muse ports `test_gpt2_fp32.cu`** (one run, small): it `#include`s the new trainer. The
   check is llm.c's own: the program prints `overall okay: 1` and exits 0.
4. **Prove the repo on three targets.**
   - `make test_gpt2fp32sycl && ./test_gpt2fp32sycl` on the CPU (free), on NVIDIA through
     Runpod, and on AMD through DigitalOcean (oneMath built with `gpu_setup`, as for stage 3).
   - Then a 20-step training run on each, with the loss curves compared with llm.c's CUDA. This
     is the start of K1.
5. **Publish:** the repo goes where the author decides (below). The README gets the results
   table, and the report gets a page.

## Cost estimate

| Step | Muse | GPU |
|---|---|---|
| 1. Assemble | $0 | $0 |
| 2. Trainer `main` | $1.00-2.50 (stage 3 cost $1.15; this adds about 400 lines of mostly host code); one retry allowed | $0.01 (recording the reference) |
| 3. Test port | $0.30-0.80 | $0 |
| 4. Three targets | $0 | NVIDIA about $0.05; AMD about $0.30-0.60 (the oneMath build each run) |
| **Total** | **about $1.30-3.30, cap $4** | **about $0.40-0.70, cap $1** |

That fits the comparison cap: about $7.6–7.8 left of $30. It leaves about $4 for J4, K1's
longer runs, C6 or E1, so those would need the cap raised.

## Risks

- **Muse may rewrite the model code** while adding `main`. Mitigation: the goal tells it to keep
  the proven code, and the variant check and hidden check both run.
- **The dataloader shuffles** with llm.c's `rand.h`. The SYCL build must use the same seed and
  code path to match losses step by step. `rand.h` is used unchanged, so it should.
- **llm.c's CUDA trainer on the GPU** needs `-lcublas`; the toolkit's `native_flags` handles
  that.
- **AMD:** the oneMath rocBLAS build from stage 3 is reused (the `MP_` setup).

## Decisions for the author

1. **Where `llm.c-sycl` lives:**
   - (a) a folder `llm.c-sycl/` in this repo (simplest, and covered by the submission);
   - (b) a new public GitHub repo, `checocadena/llm.c-sycl` (clearer as a product). Creating a
     repo is an outward action, so it needs a yes.
2. **The cost cap for J3:** $4 for Muse and $1 for GPU, as estimated, and whether to raise the
   $30 comparison cap for what comes after J3.
