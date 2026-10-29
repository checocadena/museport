"""L1 speed wrapper: reads llm.c's bf16 step lines, writes the losses, prints kernel_ms."""

import json
import stat
import struct
import subprocess
import sys
from pathlib import Path

from museport.speed import RUN_SPEED, SPEED_ARGS


def test_wrapper_reads_llmc_step_lines(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "run_speed.py").write_text(RUN_SPEED)
    for f in ("gpt2_124M_bf16.bin", "gpt2_tokenizer.bin", "tiny_shakespeare_train.bin",
              "tiny_shakespeare_val.bin"):
        (task / f).write_bytes(b"")
    (task / "args.txt").write_text(SPEED_ARGS)
    exe = tmp_path / "trainer"
    exe.write_text("#!/bin/sh\ntest -e dev/data/tinyshakespeare/tiny_shakespeare_train.bin || "
                   "exit 9\nfor i in $(seq 1 30); do echo \"step   $i/74 | loss 4.$i (+0.00z)| "
                   "norm 1.0 (+0.00z)| lr 3.00e-04 | 1$i.50 ms | 40.0% bf16 MFU | 30000 "
                   "tok/s\"; done\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    p = subprocess.run([sys.executable, str(task / "run_speed.py"), str(exe),
                        str(task / "args.txt"), "o.bin"], cwd=tmp_path, capture_output=True,
                       text=True, check=True)
    info = json.loads(p.stdout.splitlines()[0])
    assert info["steps"] == 30 and info["losses"][0] == 4.1
    assert "kernel_ms=" in p.stdout and info["ms_per_step_median"] > 100
    assert len((tmp_path / "o.bin").read_bytes()) == 30 * 4
    assert struct.unpack("<f", (tmp_path / "o.bin").read_bytes()[:4])[0] == 4.099999904632568
    assert "-v 0 " not in SPEED_ARGS  # llm.c divides by val_loss_every
