"""J3 trainer task: token files, the losses wrapper, seeding, the recorded reference."""

from __future__ import annotations

import json
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from museport.loop import seed_work
from museport.trainer import (
    HEADER_MAGIC,
    RUN_TRAIN,
    TRAIN_BATCHES,
    B,
    T,
    make_trainer_task,
    record_reference,
    write_tokens,
)

ROOT = Path(__file__).resolve().parents[1]


def _tokens(path: Path, n: int) -> None:
    hdr = [HEADER_MAGIC, 1, n] + [0] * 253
    path.write_bytes(struct.pack("<256i", *hdr) + struct.pack(f"<{n}H", *range(n)))


def test_token_files_are_cut_with_their_header(tmp_path):
    _tokens(tmp_path / "a.bin", 5000)
    write_tokens(tmp_path / "a.bin", tmp_path / "b.bin", 7)
    raw = (tmp_path / "b.bin").read_bytes()
    assert struct.unpack("<3i", raw[:12]) == (HEADER_MAGIC, 1, 7)
    assert struct.unpack("<7H", raw[1024:]) == tuple(range(7))
    with pytest.raises(ValueError, match="fewer than"):
        write_tokens(tmp_path / "a.bin", tmp_path / "c.bin", 6000)


def _data(d: Path) -> Path:
    sh = d / "dev" / "data" / "tinyshakespeare"
    sh.mkdir(parents=True)
    _tokens(sh / "tiny_shakespeare_train.bin", 4000)
    _tokens(sh / "tiny_shakespeare_val.bin", 2000)
    for f in ("gpt2_124M.bin", "gpt2_tokenizer.bin"):
        (d / f).write_bytes(b"x")
    for f in ("train_gpt2_fp32.cu", "train_gpt2.c", "llmc/utils.h", "llmc/tokenizer.h",
              "llmc/dataloader.h", "llmc/rand.h"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(f"// {f}\n")
    return d


def test_task_seeds_the_proven_port_and_hides_the_wrapper(tmp_path):
    llmc = _data(tmp_path / "llm.c")
    path = make_trainer_task(llmc, llmc, ROOT / "evidence" / "gpt2-stage3", tmp_path / "task",
                             record=False)
    task = json.loads(path.read_text())
    assert task["candidate_cmd"].startswith("python3 {task_dir}/run_train.py build/candidate")
    assert task["museport"]["seed"] == "seed" and task["tolerance"] == 1e-2
    head = (tmp_path / "task" / "data" / "tiny_shakespeare_train.bin").read_bytes()[:12]
    assert struct.unpack("<3i", head)[2] == TRAIN_BATCHES * B * T + 1
    work = tmp_path / "work"
    assert seed_work(path, work) == ["gpt2_train_proven.cpp"]
    assert "int main" in (work / "gpt2_train_proven.cpp").read_text()


def _fake_trainer(path: Path, out: str) -> Path:
    path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{out}\nEOF\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _run_wrapper(tmp_path: Path, out: str) -> subprocess.CompletedProcess[str]:
    d = tmp_path / "t"
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "run_train.py").write_text(RUN_TRAIN)
    for f in ("gpt2_124M.bin", "gpt2_tokenizer.bin", "data/tiny_shakespeare_train.bin",
              "data/tiny_shakespeare_val.bin"):
        (d / f).write_bytes(b"")
    (d / "args.txt").write_text("-b 4\n")
    exe = _fake_trainer(tmp_path / "trainer", out)
    return subprocess.run([sys.executable, str(d / "run_train.py"), str(exe), str(d / "args.txt"),
                           str(tmp_path / "o.bin")], capture_output=True, text=True)


def test_wrapper_writes_val_train_val_losses(tmp_path):
    lines = ["val loss 5.5"] + [f"step {i + 1:4d}/10: train loss {4 + i / 10:f} (1 ms)"
                                 for i in range(10)] + ["val loss 4.25", "generating:"]
    p = _run_wrapper(tmp_path, "\n".join(lines))
    assert p.returncode == 0, p.stderr
    vals = struct.unpack("<12f", (tmp_path / "o.bin").read_bytes())
    assert vals[0] == 5.5 and vals[-1] == 4.25 and vals[1] == 4.0
    assert abs(vals[10] - 4.9) < 1e-6
    # llm.c's train_gpt2.c prints "step 0: train loss ..." (0-based, no total): same values
    lines = ["val loss 5.5"] + [f"step {i}: train loss {4 + i / 10:f} (took 1 ms)"
                                 for i in range(11)] + ["val loss 4.25"]
    assert _run_wrapper(tmp_path, "\n".join(lines)).returncode == 0
    bad = _run_wrapper(tmp_path, "val loss 5.5\nstep 1/10: train loss 4.0")
    assert bad.returncode != 0 and "expected 2 val losses" in bad.stderr


def test_the_reference_is_recorded_once(tmp_path):
    d = tmp_path / "task"
    d.mkdir()
    (d / "in.txt").write_text("")
    (d / "ref.py").write_text("import struct, sys\n"
                              "open(sys.argv[2], 'wb').write(struct.pack('<2f', 1.5, 2.5))\n")
    (d / "task.json").write_text(json.dumps({
        "inputs": ["in.txt"], "reference_build": ["true"],
        "reference_cmd": "python3 {task_dir}/ref.py {input} {output}", "museport": {}}))
    assert record_reference(d) == [1.5, 2.5]
    task = json.loads((d / "task.json").read_text())
    assert task["reference_cmd"] == "cp {task_dir}/reference.out {output}"
    assert task["museport"]["reference"]["cmd"].endswith("ref.py {input} {output}")
