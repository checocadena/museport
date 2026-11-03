# An AI ported Karpathy's GPT-2 trainer off CUDA, and found a bug in oneMath on the way

*Draft. Every number links to committed evidence in
[checocadena/museport](https://github.com/checocadena/museport).*

[llm.c](https://github.com/karpathy/llm.c) is Andrej Karpathy's GPT-2 trainer in plain C and
CUDA: no PyTorch, no Python, a few thousand lines you can read. It is also NVIDIA-only. For the
Meta Global AI Developer Hackathon I wanted to know whether Meta's model, Muse
(`muse-spark-1.3`), could move real code like this off CUDA, to SYCL, the open standard
behind oneAPI, and **prove** that the result is correct, not just claim it.

The short answer: yes, with caveats I'll be specific about.

## The rule: the model never grades itself

Models are good at writing code that looks right. So nothing in this project counts as done
until a check the model never sees says so.

- For llm.c's trainer, the check is **llm.c's own test**: it compares logits, all 16 gradient
  tensors and 10 training losses against PyTorch, at llm.c's own tolerances.
- For the 22 small kernel files in llm.c's `dev/cuda/`, I build a hidden reference from
  llm.c's own CPU code. Before any model runs, each check is proven: llm.c's original CUDA
  must pass it on a real NVIDIA GPU (or SYCLomatic's raw translation must pass on the CPU).
- Then the ports run on rented GPUs: NVIDIA, against llm.c's CUDA on the same machine, and
  AMD Instinct, against the reference.

Muse works inside an agent loop with a few tools (migrate with SYCLomatic, compile, verify,
scan for leftover helper code). It can't see the hidden check, and it can't change the test.

## What Muse did

**It ported the production trainer.** `train_gpt2.cu` and its headers: 12 files, 4,884 lines
of bf16 mixed precision, cuBLASLt, warp reductions and stochastic rounding. In one run (125
turns, $5.67) the port passed llm.c's test on a CPU, an AMD MI350X and an NVIDIA GPU
([evidence](../evidence/l1-llmc-bf16/README.md)). It was also 36x slower than CUDA. And a
repeat of that run with the same setup failed, so: one success in two tries.

**Then it tuned it, on real GPUs.** I gave Muse a `gpu_test` tool: each call builds its
current code on a rented GPU and returns llm.c's test result, the time per training step next
to llm.c's CUDA on the same GPU, and a kernel profile. Muse profiled, rewrote the slow
kernels and measured each change.

In the last tuning run I banned vendor code: no cuBLAS through interop, no size-specific
fast paths that the test wouldn't exercise. Only portable SYCL and oneMath, the UXL
Foundation's math library. The best version from that run
([evidence](../evidence/l1-llmc-bf16/ptune/README.md)):

| GPU | Port, ms per step | llm.c's CUDA | Ratio |
|---|---|---|---|
| NVIDIA L4 | 199.9 | 142.4 | 1.40x |
| NVIDIA RTX A4000 | 181.6 | 117.7 | 1.54x |
| NVIDIA H100 | 49.5 | 17.1 | 2.90x |
| AMD Instinct MI350X | 63.3 | (no CUDA on AMD) | |

Same code, same test, three vendors' worth of hardware. On mid-range NVIDIA cards it is within
about 1.5x of hand-tuned CUDA. On an H100 the gap is larger, and I'd rather show that than
hide it.

**It ported the kernel experiments.** Every file in llm.c's `dev/cuda/` except the multi-GPU
one now has a Muse port that passes its hidden check, with llm.c's kernel variants kept (one
port still shares one implementation across variants; the README says which). The ones that
call cuBLAS now call oneMath. The two attention files were the last: while building their hidden
checks I found two bugs in llm.c's own kernels (one writes past the end of a row when the
sequence is shorter than the block size; one reuses shared memory across rows without a
barrier). Muse's ports fix both, and pass on NVIDIA against llm.c's CUDA.

## The bug in oneMath

The first GPU runs of the bf16 trainer crashed on NVIDIA with `oneapi::math::unimplemented`.
oneMath's API declares `gemm` with bf16 inputs and fp32 output, and the AMD backend
implements it, but the NVIDIA (cuBLAS) backend just threw.

The fix is two lines, beside the existing fp16 ones. With it, oneMath's own bf16 gemm tests on
NVIDIA go from skipped to passing (323 → 325 passed, nothing else changes). It's submitted
upstream as [uxlfoundation/oneMath#793](https://github.com/uxlfoundation/oneMath/pull/793).

The profile also says where the rest of the gap is: Muse loops over oneMath calls once per
(batch, head), because oneMath has no batched bf16 gemm yet; in 10 training steps the most
used gemm kernel launches 13,824 times. Two open upstream PRs add exactly the missing forms
([#785](https://github.com/uxlfoundation/oneMath/pull/785),
[#770](https://github.com/uxlfoundation/oneMath/pull/770)). Calling NVIDIA's cuBLAS directly
for those two shows what they could buy: 1.13–1.21x CUDA's time instead of 1.36–1.52x on the
same GPUs.

## What didn't work

- The one-run port of the trainer worked once in two tries.
- Every tuning run ended at its turn or budget limit; the results above are the best versions
  inside those runs, re-checked outside them.
- Multi-GPU training (NCCL) isn't ported; the oneAPI path is oneCCL or GPU-aware MPI.
- Kernel ports run fp32 at reduced sizes, so their checks are fast enough for a CPU.

Every mistake, including mine, is written up in
[docs/LESSONS.md](LESSONS.md).

## Why it matters

Moving AI code off one vendor's GPUs is mostly a cost problem: rewriting CUDA by hand is
slow and risky. This project suggests a cheaper loop: plan the port, let a model do it, prove
every step with a check it can't game, and tune on real hardware. The total model spend
for everything above was under $80.

The code, every run's trace, and every number are in
[checocadena/museport](https://github.com/checocadena/museport).

---

*How this was built: the ports are Muse's (`muse-spark-1.3`), driven by an agent loop.
The project around them (the harness, hidden checks, GPU pipeline, oneMath patch and this
write-up) was built with Claude, an AI assistant, working with me.*
