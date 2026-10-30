# L1 run 2: notes taken during the run

- **A wrapper bug fixed mid-run (about turn 100, 02:58 UTC):** at turn 99 the port printed a
  loss of `-nan`, and the wrapper's number pattern matched only the `-`, so it crashed with a
  Python traceback instead of reporting the NaN. The fix changes that one pattern
  ([before](../task/run_test-before-nan-fix.py), [after](../task/run_test.py)). The verdict is
  the same either way: a NaN loss never matches PyTorch's within 0.05, so it fails. Only the
  message Muse sees changed. J3's wrapper has the same pattern; its runs never printed a NaN.
- **Muse debugs through verify:** it has no tool that runs a program, so from about turn 100 it
  compiles small debug programs (`dbg_probe.cpp`, `dbg_fwd.cpp`) as the candidate and reads
  their printout in the wrapper's failure message. That is how it checked its matmuls (forward,
  input-gradient and weight-gradient differences under 0.001).
- **A gap this shows:** the wrapper trusts what the compiled program prints, and the loop does
  not force that program to be built from `test_gpt2.cpp`; a program that printed PyTorch's
  losses and `overall okay: 1` would pass. J3's check has the same gap. The grade reported for
  L1 therefore does not come from the loop alone: after the run, `test_gpt2.cpp` (with the
  files it includes) is compiled again here, outside the loop, the hidden check runs on that
  build, and the test's source is read to confirm it is a real port of llm.c's test.
