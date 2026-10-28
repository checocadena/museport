"""K1 loss-curve tasks: the wrapper reads llm.c's printout, the tasks have what the GPUs need."""

from __future__ import annotations

import json
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

from museport.curves import CURVE_ARGS, RUN_CURVE, load_curve, make_record_task, make_sycl_task

ROOT = Path(__file__).resolve().parents[1]


def _llmc(d: Path) -> Path:
    for f in ("train_gpt2_fp32.cu", "llmc/utils.h", "llmc/tokenizer.h", "llmc/dataloader.h",
              "llmc/rand.h", "dev/data/tinyshakespeare/tiny_shakespeare_train.bin",
              "dev/data/tinyshakespeare/tiny_shakespeare_val.bin"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(f"// {f}\n")
    return d


def _run(tmp_path: Path, printout: str) -> subprocess.CompletedProcess[str]:
    d = tmp_path / "task"
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "run_curve.py").write_text(RUN_CURVE)
    for f in ("gpt2_124M.bin", "gpt2_tokenizer.bin", "data/tiny_shakespeare_train.bin",
              "data/tiny_shakespeare_val.bin"):
        (d / f).write_bytes(b"")
    (d / "args.txt").write_text(CURVE_ARGS)
    exe = tmp_path / "trainer"
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{printout}\nEOF\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return subprocess.run([sys.executable, str(d / "run_curve.py"), str(exe), str(d / "args.txt"),
                           str(tmp_path / "o.bin")], capture_output=True, text=True)


def test_wrapper_reads_llmc_printout_and_prints_the_curve(tmp_path):
    # the real printout of llm.c-sycl's trainer from the fresh-clone run
    # (stopped after 3 steps: a final val line is added, as a full run prints one)
    log = (ROOT / "evidence/j3-fresh-clone/train-first-4min.log").read_text()
    p = _run(tmp_path, log + "\nval loss 4.400000\n")
    assert p.returncode == 0, p.stderr
    curve = json.loads(p.stdout)
    assert curve["val"] == [4.513919, 4.4] and len(curve["train"]) == 3
    assert curve["train"][0] == 4.287294 and 30000 < curve["ms_per_step_median"] < 50000
    raw = (tmp_path / "o.bin").read_bytes()
    assert len(raw) // 4 == 5 and abs(struct.unpack("<f", raw[:4])[0] - 4.513919) < 1e-6


def test_wrapper_fails_on_a_run_with_no_losses(tmp_path):
    p = _run(tmp_path, "nothing printed")
    assert p.returncode != 0 and "expected val and train losses" in p.stderr


def test_record_and_sycl_tasks(tmp_path):
    llmc = _llmc(tmp_path / "llm.c")
    rec = json.loads(make_record_task(llmc, tmp_path / "rec").read_text())
    assert rec["record_native"] is True and rec["benchmark_runs"] == 0
    assert rec["native_flags"] == ["-lcublas"] and {f["path"] for f in rec["fetch"]} == {
        "gpt2_124M.bin", "gpt2_tokenizer.bin"}
    assert re.search(r"-t 256 .*-v 25", (tmp_path / "rec" / "curve_args.txt").read_text())
    (tmp_path / "case0.out").write_bytes(struct.pack("<2f", 4.5, 4.0))
    path = make_sycl_task(llmc, ROOT / "evidence/j3-trainer/run1", tmp_path / "case0.out",
                          tmp_path / "amd", "muse-spark-1.3", "amd")
    task = json.loads(path.read_text())
    assert task["sycl_sources"] == ["agent_port/train_gpt2_fp32.cpp"]
    assert task["gpu_setup"] == "onemath_rocblas_setup.sh" and "record_native" not in task
    ref = (tmp_path / "amd" / "reference.out").read_bytes()
    assert ref == (tmp_path / "case0.out").read_bytes()
    assert task["native_flags"] == [] and task["benchmark_runs"] == 0


def test_curves_come_back_from_the_verify_stdout(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    line = json.dumps({"val": [4.5], "train": [4.2, 4.1], "ms_per_step_median": 12.0})
    (res / "sycl_verify_vs_native.json").write_text(json.dumps({"evidence": {"cases": [
        {"candidate": {"stdout": line + "\n"}, "reference": {"stdout": ""}}]}}))
    assert load_curve(res) == {"sycl_verify_vs_native:candidate": json.loads(line)}
