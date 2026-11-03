# Muse ports llm.c's `attention_forward.cu` (retry)

The first run of this item was cut off at turn 5 when the connection to Meta's API was refused
([port-llmc-attention](../port-llmc-attention/README.md)). This is the retry: same task, same
hidden check (proven with llm.c's CUDA on an NVIDIA L4,
[harness-checks](../port-llmc-attention/harness-checks/README.md)), `muse-spark-1.3`, 20 turns
max, $1 cap.

## Result

| Item | Variants | Loop result (turns) | Muse cost | Hidden grade (final sources, CPU) | Kernels / variants |
|---|---|---|---|---|---|
| `t3-attention-forward` | 6 | max_iterations (20) | $0.7639 | **PASS**, max abs err 3.58e-07 (atol 1e-3), 0 helpers | 12 / 6 |

([SUMMARY.md](SUMMARY.md), [run.log](run.log), [report](t3-attention-forward/report.md))

- **The port** ([port/attention_forward.cpp](t3-attention-forward/port/attention_forward.cpp))
  keeps llm.c's variants 1-6. llm.c's `cublasSgemmStridedBatched` and
  `cublasGemmStridedBatchedEx` calls become oneMath's
  `oneapi::math::blas::column_major::gemm_batch`.
- **Timeline:** full port at turn 10, one compile fix, verify PASS at turn 14, SYCLomatic's
  leftovers deleted at turn 16. Turns 19-20 edited the port and were not verified inside the
  loop, so its strict rule was not met; the hidden grade after the run checks the final
  sources and passes.

## GPUs

| GPU | vs reference (worst max abs err) | vs llm.c's CUDA (worst max abs err) | atol |
|---|---|---|---|
| NVIDIA L4 | PASS (2.98e-07) | PASS (2.98e-07) | 1e-3 |
| AMD | not run: two DigitalOcean MI350X machines never became ready, and the tool gave up ([log](gpu/logs/amd-not-ready.log)) | | |

Results: [gpu/](gpu/). Each machine's deletion was confirmed.

## Checks

| | |
|---|---|
| Task folder before and after | unchanged ([task-before.sha256](task-before.sha256), [task-after.sha256](task-after.sha256)) |
| Secrets | none (`museport.redact.secret_hits`) |
| Muse | $0.7639 |
| GPU | $0.016 on NVIDIA (the tool's estimate); about $1.50 for the two AMD machines that never became ready |
