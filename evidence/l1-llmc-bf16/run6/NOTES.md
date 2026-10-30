# Run 6: a repeat of run 5's from-scratch port, which did not succeed

**Setup:** the same goal and tools as [run 5](../run5): no SYCLomatic, the shell tool, and the
compacting context. The task was rebuilt fresh, with today's test wrapper. The cap was $10.

**Result:**

- **The loop:** stopped by the USD cap at turn 197 ($10.13).
- **The code at the end:** it compiles with 0 dpct helpers, but the loop's check timed out
  after 600 s.
- **Graded outside the loop, with no time limit** ([independent-grade](independent-grade)):
  llm.c's test ran in 739 s and failed, `overall okay: 0`.
  - The logits are wrong.
  - Most of the 16 gradient tensors are wrong.
  - The training losses stay near 6-7, where PyTorch's go from 5.27 to 0.19.
  - Only the save-and-replay check passes, and that compares the port with itself.
- **Where Muse was:** debugging attention. Its last notes find that `permute_qkv` drops batches.

**What it means:** the from-scratch port of llm.c's bf16 trainer succeeded in 1 of 2 clean
runs with this setup: run 5 in 125 turns for $5.67, and run 6 not at all within $10. Run 5's
result does not reliably repeat, and the "one run" claim is stated as 1 of 2.

Task folder unchanged ([task-before.sha256](task-before.sha256),
[task-after.sha256](task-after.sha256)); no secrets. The test's checkpoint files (1.7 GB)
were left out of [port/](port).
