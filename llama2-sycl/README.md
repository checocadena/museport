# llama2-sycl

[ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu) (Llama 2 inference in CUDA, a
port of Karpathy's llama2.c), with its one CUDA file ported to SYCL 2020 by Meta's
`muse-spark-1.3` with [Muse Port](../README.md). The repo is llama2.cu's own at `cb0c6fa`;
only `llama2.cu` is replaced, by Muse's `llama2.cpp`.

This is the second CUDA repo Muse Port took end to end, after llm.c. Nothing in it was written by hand for
this repo except the hidden check's wrapper. Muse planned the migration from a scan, the
check's reference is llama2.cu's own output, recorded on an NVIDIA GPU, and Muse wrote the port
against it. ([evidence](../evidence/j4-llama2/README.md))

**The check:** greedy decoding (temperature 0) with Karpathy's `stories15M` model, 64 steps, two
prompts. The generated story must match llama2.cu's byte for byte.

| Device | Against llama2.cu's recording | Against llama2.cu, live on the same GPU |
|---|---|---|
| CPU (in Muse's loop) | 2 of 2 stories identical | |
| NVIDIA RTX 2000 Ada | 2 of 2 identical | 2 of 2 identical |
| AMD Instinct MI350X | AMD_RESULT | (llama2.cu doesn't run on AMD) |

- **Muse's port:** 737 lines of SYCL 2020. It keeps llama2.cu's fp16 storage (`sycl::half`) and
  turns its CUB reductions into SYCL group algorithms. It uses 0 SYCLomatic helpers.
- **How it was made:** one run, 20 turns, $0.45.
- **Speed:** not tuned or compared beyond one rough number. On the RTX 2000 Ada, one 64-step
  run, including loading the model, took a median 430 ms for the port and 378 ms for llama2.cu
  (3 runs each).

## Quick start

You need a SYCL compiler: `icpx` from Intel oneAPI DPC++ for the CPU, or intel/llvm built
with the CUDA or HIP backend for a GPU.

```bash
cd llama2-sycl
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
make llama2                          # the CPU device; or: make llama2 TARGET=nvidia / TARGET=amd AMD_ARCH=gfx950
./llama2 stories15M.bin 0.0 64 "Once upon a time"
```

`make` alone still builds llama2.c's `run.c`, as upstream. FRESH_CLONE

## What is here

- **`llama2.cpp`:** Muse's port, byte for byte the sources the hidden check graded
  ([PROVENANCE.json](PROVENANCE.json)).
- **`Makefile`:** llama2.cu's Makefile, unchanged, with a `llama2` target for `icpx -fsycl`
  appended. Muse did not write it.
- **Everything else:** llama2.cu's repo as it is upstream (llama2.c's `run.c`, training and
  export scripts, tokenizer). llama2.c's README is
  [`README-llama2.c.md`](README-llama2.c.md).
- `museport llama2-sycl assemble --upstream <llama2.cu checkout>` regenerates every file here
  except this README.

## Limits

- Checked on two prompts, 64 steps each, greedy, with the 15M model only.
- At 128 steps, llama2.c's fp32 `run.c` already differs from llama2.cu's fp16 output, on a
  near-tie token. The check was set to 64 steps before any model ran
  ([evidence](../evidence/j4-llama2/README.md)).

## License

MIT, as upstream ([LICENSE](LICENSE)).
