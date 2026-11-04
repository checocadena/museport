# Submission form answers (Nov 4)

Pasted into the hackathon form. Each is under the 3,000-character limit; every number is from the README and evidence.

## Project description

Muse Port uses Meta's Muse model to move real AI code off NVIDIA-only CUDA and onto SYCL, the open standard behind oneAPI, so it runs on NVIDIA GPUs, AMD GPUs and CPUs from one source. Rewriting CUDA by hand is slow and expensive, which keeps AI tied to one vendor's hardware. Muse Port has an AI do the port, and only counts it as done when a check the model never sees agrees.

How it works: Muse Port scans a CUDA repository (kernels, NVIDIA library calls, CUDA-only constructs). Muse writes a migration plan, which is checked against the scan, then ports each piece in an agent loop: migrate, rewrite, compile, verify. Every port is graded by a hidden check built from the project's own reference code and tolerances. Each check is proven before any model runs: the original CUDA on a real NVIDIA GPU, or a known-good translation, must pass it. The ports then run unchanged on rented NVIDIA and AMD GPUs. The agent can also rent GPUs itself through a gpu_test tool (Runpod and DigitalOcean, with a spending cap, every machine deleted afterwards) to measure and tune its own code.

Results, all with committed evidence:
- Karpathy's llm.c production GPT-2 trainer (bf16, 12 CUDA files, 4,884 lines): ported by Muse in one run (125 turns, $5.67). llm.c's own test passes (logits, all 16 gradient tensors, 10 training losses against PyTorch) on an x86 CPU, an NVIDIA GPU and an AMD GPU.
- Muse then tuned it on real GPUs, from 36x CUDA's time to 1.40-1.55x on mid-range NVIDIA cards in portable SYCL + oneMath, with no vendor code (2.90x on an H100); 63 ms per step on an AMD MI350X.
- llm.c's kernel experiments: 21 of 22 CUDA files ported, 20 passing their hidden check, 19 also passing on NVIDIA against llm.c's own CUDA.
- A second repo, llama2.cu (Llama 2 inference), ported in one run for $0.45: its output matches byte for byte on CPU, NVIDIA and AMD.
- The work found a bug in oneMath, the open math library behind oneAPI (bf16 matrix multiply on NVIDIA threw "unimplemented"). The two-line fix is submitted upstream as uxlfoundation/oneMath#793; oneMath's own BLAS tests go from 323 to 325 passing.

Everything is in a report built only from the evidence, failures included. Limits we publish: kernel ports are fp32 at reduced sizes; the one-run trainer port succeeded in 1 of 2 tries; Intel GPUs are untested (no access to Intel's GPU cloud).

Disclosure: before the hackathon I built an open-source toolkit, oneapi-agent-toolkit, with the bare-bones tools for an agent to port code; Muse Port uses it as a pinned dependency. Everything above was built during the hackathon.


## Use of Meta products

Muse Port is built around Meta's Muse Spark model (muse-spark-1.3), called through Meta's API with streaming and a cost cap on every request. Muse does the engineering work itself:

- Planning: Muse reads a whole CUDA repository and the scan, and writes a migration plan in a fixed JSON schema. Every file, kernel, library mapping and risk it names is checked against the scan. For llm.c's 22 kernel files the plan has 23 items, places 22 of 22 files and has 0 grounding issues, for $0.26.
- Porting: Muse drives the agent loop. It decides what to read and what to write, fixes its own compile errors, and removes the migration tool's leftovers. It ported Karpathy's production bf16 GPT-2 trainer (4,884 lines) in one run of 125 turns for $5.67, 21 of llm.c's 22 kernel files, and llama2.cu, Karpathy's CUDA inference for Meta's Llama 2 architecture, in one run for $0.45.
- Writing the tests: Muse wrote the hidden-check specs for 17 of llm.c's kernel files from the CUDA alone; 16 are proven as strict as llm.c's own checks.
- Tuning on real hardware: Muse calls a gpu_test tool that builds its current code on a rented NVIDIA or AMD GPU, runs llm.c's own test, and times training next to llm.c's CUDA, with a kernel profile. Over a 211-turn run it profiled, rewrote its slow kernels and checked each change on a GPU. A second tuning run, with vendor libraries banned, took the trainer from 36x CUDA's time to 1.40-1.55x in portable code on mid-range NVIDIA cards.

What we learned about Muse's API, and fixed: non-streamed calls time out at about 30 s (we use a streaming client); a turn can come back empty when reasoning uses the whole output budget (the loop marks it and continues); usage is sometimes missing from a stream (we charge the worst case so the cost cap holds). All three are documented in the repository.

Every Muse request, its trace and its cost are committed, and the report replays Muse's work turn by turn. The total Muse spend for everything in the project is under $80.


## Why Meta's models are important

Porting CUDA is the kind of work that keeps AI locked to one vendor: it takes expert engineers, it is slow, and a port that looks right can still be wrong. Muse makes the port itself cheap enough to do, and Muse Port's hidden checks make it trustworthy.

- Muse handles real codebases, not snippets. It read and ported a 12-file, 4,884-line bf16 trainer with cuBLASLt calls, warp reductions and stochastic rounding, and kept going for 125 turns until llm.c's own test passed. Its tuning run lasted 211 turns of profiling, rewriting and measuring on real GPUs.
- Muse works as an agent, not autocomplete. It plans, decides what to read, writes code, reads compiler errors and fixes them, and uses tools: the migration tool, the compiler, the verifier and a tool that rents a GPU, runs the test and returns a profile. It found its own mistakes, such as a transpose error in GPT-2's training step, and fixed them.
- Muse's cost makes this practical. A whole repo (llama2.cu) cost $0.45; the production trainer cost $5.67; the entire project, every plan, port, check spec and tuning run, cost under $80 of Muse. That is the difference between porting one demo and porting many projects.
- Muse writes the tests too. It wrote the hidden-check specs for llm.c's kernels from the CUDA alone, and 16 of 17 were proven as strict as llm.c's own checks, which is what lets the approach scale beyond hand-written tests.

The result is a workflow that was not practical before: an agent that ports code, proves it correct, and then tunes it on real hardware from different vendors. That is a step toward AI that is leaner, cheaper and able to run on anybody's machine, and Meta's models are what make each step affordable.
