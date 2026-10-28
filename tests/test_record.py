"""museport record-tasks / diff-task: recording tasks and differential checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from museport.record import differential_task, recording_task


def _task(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "input.bin").write_bytes(b"\x00" * 8)
    (d / "cpu_proof.c").write_text("int main(void) { return 0; }\n")
    (d / "task.json").write_text(json.dumps({
        "name": "t", "cuda_path": "k_harness.cu", "inputs": ["input.bin"],
        "reference_build": ["g++", "{task_dir}/k_ref.cpp", "-o", "build/reference"],
        "reference_cmd": "build/reference {input} {output} 4", "sycl_sources": ["p.cpp"],
        "museport": {"source": "karpathy/llm.c dev/cuda/k.cu"}}))
    return d


def _recording(d: Path, out: bytes, status: str = "ok") -> Path:
    (d / "results" / "recorded").mkdir(parents=True)
    (d / "results" / "recorded" / "case0.out").write_bytes(out)
    rec = [{"case": 0, "input": "input.bin", "file": "recorded/case0.out", "bytes": len(out),
            "sha256": hashlib.sha256(out).hexdigest()}]
    (d / "summary.json").write_text(json.dumps({
        "tool": "gpu_test", "status": status, "summary": "recorded 1 output(s)",
        "evidence": {"recorded": rec, "deterministic": True,
                     "device": {"name": "NVIDIA L4"}}}))
    return d


def test_recording_task_records_without_a_port(tmp_path):
    path = recording_task(_task(tmp_path / "task"), tmp_path / "rec")
    task = json.loads(path.read_text())
    assert task["record_native"] is True and task["reference_build"] == ["true"]
    assert "sycl_sources" not in task and (tmp_path / "rec" / "input.bin").exists()


def test_differential_task_serves_the_recorded_output(tmp_path):
    out = bytes(range(16))
    path = differential_task(_task(tmp_path / "task"), _recording(tmp_path / "r", out),
                             tmp_path / "diff")
    task = json.loads(path.read_text())
    d = tmp_path / "diff"
    assert task["reference_cmd"] == "python3 {task_dir}/golden_ref.py {input} {output}"
    assert task["museport"]["reference"]["device"] == "NVIDIA L4"
    assert not (d / "cpu_proof.c").exists()
    cmd = task["reference_cmd"].replace("{task_dir}", str(d))
    cmd = cmd.replace("{input}", str(d / "input.bin")).replace("{output}", str(tmp_path / "o"))
    subprocess.run([sys.executable, *cmd.split()[1:]], check=True)
    assert (tmp_path / "o").read_bytes() == out
    bad = subprocess.run([sys.executable, str(d / "golden_ref.py"), "other.bin",
                          str(tmp_path / "o2")], capture_output=True, text=True)
    assert bad.returncode != 0 and "no recorded output" in bad.stderr


def test_differential_task_refuses_a_failed_recording(tmp_path):
    with pytest.raises(ValueError, match="not a successful recording"):
        differential_task(_task(tmp_path / "task"), _recording(tmp_path / "r", b"x", "error"),
                          tmp_path / "diff")
