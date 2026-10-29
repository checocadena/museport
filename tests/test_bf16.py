"""L1: the bf16 test wrapper writes llm.c's okay flag, whether the tolerances were kept, and the
10 losses; the tolerance guard refuses a loosened test."""

import json
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from museport.bf16 import GOAL, RUN_TEST, make_proof_task

LLMC_TEST = """
    float logit_accuracy_threshold = 1e-3f;
    float loss_diff_threshold = 1e-5f;
    #if defined(ENABLE_BF16) || defined(ENABLE_F16)
    logit_accuracy_threshold = 25.0f; // 15.0f was too low
    loss_diff_threshold = 0.05f;
    #endif
    float grad_thresholds[NUM_PARAMETER_TENSORS] = {
            5e-1f, 4e-3f, 1e-1f, 4e-2f,
            5e-2f, 3.5e-2f, 2e-2f, 3e-2f,
            5e-2f, 3e-2f, 3e-2f, 3e-2f,
            2e-2f, 1e-2f,1e-1f,2e-2f};
"""


def _run(tmp_path: Path, port_test: str | None) -> list[float]:
    task = tmp_path / "task"
    task.mkdir()
    (task / "run_test.py").write_text(RUN_TEST)
    (task / "test_gpt2.cu").write_text(LLMC_TEST)
    for f in ("gpt2_124M_bf16.bin", "gpt2_124M_debug_state.bin", "tiny_shakespeare_val.bin"):
        (task / f).write_bytes(b"")
    work = tmp_path / "work"
    work.mkdir()
    if port_test is not None:
        (work / "test_gpt2.cpp").write_text(port_test)
    exe = tmp_path / "test"
    exe.write_text("#!/bin/sh\ntest -e dev/data/tinyshakespeare/tiny_shakespeare_val.bin || "
                   "exit 9\nfor i in 1 2 3 4 5 6 7 8 9 10; do echo \"step $i: loss 4.$i (took 1 "
                   "ms)\"; done\necho 'overall okay: 1'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    subprocess.run([sys.executable, str(task / "run_test.py"), str(exe), "x", "o.bin"],
                   cwd=work, check=True, capture_output=True)
    raw = (work / "o.bin").read_bytes()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def test_llmc_test_keeps_its_tolerances(tmp_path: Path) -> None:
    v = _run(tmp_path, None)
    assert v[:2] == [1.0, 1.0] and len(v) == 12


@pytest.mark.parametrize("port, kept", [
    (LLMC_TEST, True),
    (LLMC_TEST.replace("logit_accuracy_threshold = 25.0f;", "const float logit_tol = 25.0f;"),
     True),  # renamed, same value
    (LLMC_TEST.replace("loss_diff_threshold = 0.05f;", "loss_diff_threshold = 0.5f;"), False),
    (LLMC_TEST.replace("25.0f; //", "25.0f; logit_accuracy_threshold = 100.0f; //"), False),
    (LLMC_TEST.replace("5e-1f, 4e-3f", "5e-1f, 4e-2f"), False),
    (LLMC_TEST.replace("// 15.0f was too low", "/* 25.0f */"), True),
])
def test_guard_refuses_a_loosened_test(tmp_path: Path, port: str, kept: bool) -> None:
    assert _run(tmp_path, port)[1] == (1.0 if kept else 0.0)


def test_proof_task_plants_one_bug(tmp_path: Path) -> None:
    llmc = tmp_path / "llmc"
    (llmc / "llmc").mkdir(parents=True)
    (llmc / "dev/data/tinyshakespeare").mkdir(parents=True)
    (llmc / "dev/data/tinyshakespeare/tiny_shakespeare_val.bin").write_bytes(b"")
    (llmc / "llmc" / "gelu.cuh").write_text("float cube = 0.044715f * xi;\n")
    (llmc / "train_gpt2.cu").write_text("// trainer\n")
    (llmc / "test_gpt2.cu").write_text(
        LLMC_TEST + "float expected_losses[10] = {1.0f,2,3,4,5,6,7,8,9,10.5f};\n")
    t = json.loads(make_proof_task(llmc, tmp_path / "p", patch=(
        "llmc/gelu.cuh", "0.044715f", "0.04715f")).read_text())
    assert t["record_native"] and "-DENABLE_BF16" in t["native_flags"]
    assert (tmp_path / "p/llmc/gelu.cuh").read_text() == "float cube = 0.04715f * xi;\n"
    ref = (tmp_path / "p/reference.out").read_bytes()
    assert struct.unpack("<12f", ref)[:3] == (1.0, 1.0, 1.0)
    assert "llm.c's own bf16 tolerances" in GOAL
