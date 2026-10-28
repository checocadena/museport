"""J4 llama2 tasks: the wrapper cuts the story from the printout, the recording is served by input
order, and the tasks have what the GPUs and the loop need."""

from __future__ import annotations

import json
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from museport.llama2 import (
    CASES,
    GITIGNORE,
    MODEL,
    MODEL_SHA256,
    RECORDED,
    RUN_STORY,
    SYCL_MAKEFILE,
    UPSTREAM_COMMIT,
    make_gpu_task,
    make_loop_task,
    make_record_task,
)


def _repo(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "llama2.cu").write_text("// llama2.cu\n")
    (d / "tokenizer.bin").write_bytes(b"tok")
    return d


def _run(tmp_path: Path, printout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    d = tmp_path / "task"
    d.mkdir()
    (d / "run_story.py").write_text(RUN_STORY)
    (d / MODEL).write_bytes(b"")
    (d / "tokenizer.bin").write_bytes(b"")
    (d / "case.txt").write_text("0.0\n64\nOnce upon a time\n")
    exe = tmp_path / "llama2"
    # The fake program checks it runs next to the model, then prints like llama2.cu.
    exe.write_text(f"#!/bin/sh\ntest -e stories15M.bin || exit 7\ncat <<'EOF'\n{printout}\nEOF\n"
                   f"exit {code}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return subprocess.run([sys.executable, str(d / "run_story.py"), str(exe), str(d / "case.txt"),
                           str(tmp_path / "o.bin")], capture_output=True, text=True)


def test_wrapper_cuts_the_story_and_writes_its_bytes(tmp_path: Path) -> None:
    p = _run(tmp_path, "Model params\n<s>\nOnce upon a time, Lily é.\nachieved tok/s: 812.3")
    assert p.returncode == 0, p.stderr
    story = "Once upon a time, Lily é."
    assert json.loads(p.stdout) == {"story": story}
    raw = (tmp_path / "o.bin").read_bytes()
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert bytes(int(v) for v in values) == story.encode("utf-8")


@pytest.mark.parametrize("printout, code", [
    ("Once upon a time\nachieved tok/s: 1", 0),  # no <s> line
    ("<s>\nOnce upon a time", 0),  # no timing line
    ("<s>\nOnce\nachieved tok/s: 1", 3),  # failed exit
])
def test_wrapper_refuses_a_bad_run(tmp_path: Path, printout: str, code: int) -> None:
    p = _run(tmp_path, printout, code)
    assert p.returncode != 0
    assert not (tmp_path / "o.bin").exists()


def test_recorded_serves_by_input_order(tmp_path: Path) -> None:
    (tmp_path / "recorded").mkdir()
    for i in range(2):
        (tmp_path / "recorded" / f"case{i}.out").write_bytes(bytes([i]) * 4)
    (tmp_path / "task.json").write_text(json.dumps({"inputs": ["case0.txt", "case1.txt"]}))
    (tmp_path / "recorded.py").write_text(RECORDED)
    out = tmp_path / "o.bin"
    subprocess.run([sys.executable, str(tmp_path / "recorded.py"), "/x/case1.txt", str(out)],
                   check=True)
    assert out.read_bytes() == b"\x01" * 4


def test_record_task(tmp_path: Path) -> None:
    task = json.loads(make_record_task(_repo(tmp_path / "repo"), tmp_path / "t").read_text())
    assert task["record_native"] is True and task["benchmark_runs"] == 0
    assert task["tolerance"] == 0.0 and task["rtol"] == 0.0
    assert task["fetch"][0]["path"] == MODEL and task["fetch"][0]["sha256"] == MODEL_SHA256
    assert len(task["inputs"]) == len(CASES)
    assert (tmp_path / "t" / "case1.txt").read_text().splitlines() == list(CASES[1])


def test_loop_task_links_the_model_and_hides_the_recording(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "case0.out").write_bytes(b"x")
    model = tmp_path / MODEL
    model.write_bytes(b"m")
    dest = tmp_path / "t"
    task = json.loads(make_loop_task(_repo(tmp_path / "repo"), rec, model, dest).read_text())
    assert (dest / MODEL).is_symlink()
    assert (dest / "recorded" / "case0.out").read_bytes() == b"x"
    assert "{task_dir}/recorded.py" in task["reference_cmd"]
    assert "{task_dir}/run_story.py" in task["candidate_cmd"]
    assert task["tolerance"] == 0.0 and "sycl::half" in task["goal"]


def test_gpu_task_from_the_committed_port(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "evidence" / "j4-llama2"
    dest = tmp_path / "gpu"
    task = json.loads(make_gpu_task(root, dest, "muse-spark-1.3").read_text())
    assert task["sycl_sources"] == ["agent_port/llama2.cpp"]
    assert (dest / "agent_port" / "llama2.cpp").read_bytes() == (
        root / "run1" / "port" / "llama2.cpp").read_bytes()
    assert task["fetch"][0]["sha256"] == MODEL_SHA256 and not (dest / MODEL).exists()
    assert task["benchmark_runs"] == 3


def test_llama2_sycl_holds_the_graded_port() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "llama2-sycl"
    assert (out / "llama2.cpp").read_bytes() == (
        root / "evidence" / "j4-llama2" / "run1" / "port" / "llama2.cpp").read_bytes()
    assert (out / "Makefile").read_text().endswith(SYCL_MAKEFILE)
    assert (out / ".gitignore").read_text() == GITIGNORE
    assert not (out / "llama2.cu").exists()
    prov = json.loads((out / "PROVENANCE.json").read_text())
    assert prov["upstream_commit"] == UPSTREAM_COMMIT and prov["llama2.cpp"]["helpers"] == 0
