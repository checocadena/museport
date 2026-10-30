# L1 tuning run: Muse makes its run 5 port faster

**Start:** run 5's port, which passed llm.c's test but ran slowly:

- 2891 ms per step on an NVIDIA RTX A5000, where llm.c's CUDA takes 80.3 ms (36x;
  [run5/gpu/speed-nvidia](../run5/gpu/speed-nvidia)). The goal Muse was given quoted 2850 and
  79 ms, slightly off from the recorded medians; the ratio is the same;
- 1274 ms per step on an AMD MI350X.

**Goal ([goal.txt](goal.txt)):** make it faster while llm.c's test keeps passing. The goal named
the likely causes: memory shared with the host for everything, and one matmul call per attention
head.

**How the run worked:**

- **`--tune` mode:** the loop does not stop when the check passes. It stops only when Muse calls
  `finish` on a turn where compile, verify and zero helpers all hold.
- **What Muse had:** the shell tool, and the CPU device only, so Muse could not time a GPU.

**Result:** finished in **192 turns, $10.39** ([outcome.json](outcome.json)).

## What Muse changed (read from [port/](port))

- **Memory:** every model buffer now lives on the device: 17 `malloc_device` calls, 0
  `malloc_shared`. The two `malloc_host` calls are staging buffers.
  - **A bug it caused, then fixed:** an async copy from a host buffer that was freed while the copy
    was in flight. It showed up as NaN; Muse traced it to that copy and fixed it (turns about
    115-122).
- **Attention:** Muse dropped oneMath for attention and wrote its own SYCL kernels (6
  `parallel_for`), in place of the per-head gemm loop. The other matmuls still use
  oneMath (3 calls in `matmul.hpp`).
- **Waits:** 4 `.wait()` calls left in all.

## Checks

| Check | Result |
|---|---|
| Task folder before and after ([task-before.sha256](task-before.sha256), [task-after.sha256](task-after.sha256)) | unchanged |
| Secrets in the evidence | none |
| [Independent CPU grade](independent-grade): rebuilt outside the loop from Muse's files | **PASS**: `overall okay: 1`, 16/16 gradient tensors, 20 loss lines OK, worst loss 0.0385 vs PyTorch |
| llm.c's test, AMD Instinct MI350X ([gpu/test-amd](gpu/test-amd)) | **PASS**, worst loss 0.0438 vs PyTorch |
| llm.c's test, NVIDIA L4, oneMath patched ([gpu/test-nvidia](gpu/test-nvidia)) | **FAIL by the cross-check**, see below |

**The NVIDIA L4 failure:**

- **llm.c's own test passes:** `overall okay: 1`, worst loss 0.0403 from PyTorch's.
- **The extra check against llm.c's CUDA build on the same GPU fails on one of 12 values:** at
  step 2, the port's loss is 4.0210 and CUDA's is 4.0729. That is a gap of 0.0519 against a
  tolerance of 0.05.
- **Both are within 0.05 of PyTorch's,** on opposite sides of it. By the check's rule this is a
  fail.

## Speed

llm.c's trainer: B=4, T=1024, 30 steps. Times are the median of llm.c's own step timer over steps
6-30, over 3 runs. The setup is described in [../speed/README.md](../speed/README.md).

| Device | Port | ms per step | Against CUDA on the same GPU |
|---|---|---|---|
| NVIDIA RTX 2000 Ada | llm.c's CUDA (native) | 187 | 1x |
| NVIDIA RTX 2000 Ada | run 3's port | 523 (CUDA there: 183) | 2.9x |
| NVIDIA RTX 2000 Ada | **tuned port** | **1058** | **5.7x** |
| NVIDIA RTX A5000 | run 5's port | 2891 (CUDA: 80.3) | 36x |
| AMD MI350X | run 3's port | 867 | |
| AMD MI350X | run 5's port | 1274 | |
| AMD MI350X | **tuned port** | **578** | |

**What the numbers show:**

- **Faster than run 5:** 2.2x on the MI350X (1274 to 578 ms). On NVIDIA, the time against CUDA
  went from 36x to 5.7x; those two numbers come from different GPUs (an A5000 and an RTX 2000
  Ada).
- **The fastest port so far on AMD:** faster than run 3's port too.
- **Not on NVIDIA:** run 3's port, on the same RTX 2000 Ada, is still twice as fast as this one.
- **Not profiled:** Muse tuned with no GPU to time on, and we did not profile its kernels.

**Losses during training:**

- the port's 30 losses are within 0.034 of llm.c's live CUDA run on NVIDIA;
- within 0.037 and 0.035 of the CUDA recording, on NVIDIA and AMD.

## Other notes

- **One file outside the work folder:** Muse wrote one debug program to `/tmp`, outside its work
  folder. The task folder was not touched.
- **NVML:** the port still uses NVIDIA's NVML in `llmc/mfu.h`, as llm.c does. NVIDIA GPU tasks now
  link it in code (`museport.bf16.NVML_FLAGS`).
- **GPU cost:** $0.41 (L4 $0.02, RTX 2000 Ada $0.02, MI350X $0.37).
