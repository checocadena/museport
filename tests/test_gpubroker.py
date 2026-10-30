"""The gpu_test tool and its broker: the loop's request reaches the broker, which answers with
llm.c's test result and the trainer's ms per step; the budget and the profile are enforced."""

import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import oneapi_agent.loop as tk
import pytest

from museport.gpubroker import GPU_TOOL_SCHEMA, gpu_tool, serve, submit, toolbox_factory
from museport.loop import NoMigrateToolbox
from museport.speed import RUN_SPEED, SPEED_ARGS

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence" / "l1-llmc-bf16"
GPU = EV / "tune" / "gpu"


def _llmc(tmp: Path) -> tuple[Path, Path]:
    llmc = tmp / "llmc"
    (llmc / "llmc").mkdir(parents=True)
    (llmc / "llmc" / "x.cuh").write_text("//\n")
    (llmc / "train_gpt2.cu").write_text("//\n")
    data = llmc / "dev" / "data" / "tinyshakespeare"
    data.mkdir(parents=True)
    for f in ("tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"):
        (data / f).write_bytes(b"")
    rec = tmp / "recorded"
    rec.mkdir()
    (rec / "case0.out").write_bytes(b"\0" * 120)
    return llmc, rec


def _fake_remote(calls: list[str]) -> Any:
    """Answers like remote_test, from the tuned port's real NVIDIA results."""
    def remote(task_file: str, *, into: str, **kw: Any) -> Any:
        key = "speed-nvidia" if "speed" in Path(task_file).parent.name else "test-nvidia"
        calls.append(f"{key} {kw['vendor']} {kw['provider']}")
        shutil.copytree(GPU / key, into)
        return SimpleNamespace(status="ok", summary="x", evidence={"est_cost_usd": 0.02})
    return remote


def _work(tmp: Path) -> Path:
    work = tmp / "work"
    shutil.copytree(EV / "tune" / "port", work)
    return work


def _serve(tmp: Path, queue: Path, remote: Any, **kw: Any) -> threading.Thread:
    llmc, rec = _llmc(tmp)
    args = dict(llmc=llmc, recorded=rec, evidence_dir=EV, run="tune",
                onemath_patch=ROOT / "patches" / "onemath-cublas-bf16-gemm.patch",
                budget_usd=5.0, max_calls=5, model="m", log=lambda m: None, poll_s=0.05,
                remote=remote)
    args.update(kw)
    t = threading.Thread(target=serve, args=(queue,), kwargs=args, daemon=True)
    t.start()
    return t


def test_a_request_gets_the_gpu_numbers(tmp_path: Path) -> None:
    queue, calls = tmp_path / "q", []
    queue.mkdir()
    t = _serve(tmp_path, queue, _fake_remote(calls))
    r = submit(queue, _work(tmp_path), "nvidia", profile=True, timeout_s=60, poll_s=0.05)
    (queue / "STOP").write_text("")
    t.join(5)
    assert sorted(calls) == ["speed-nvidia nvidia runpod", "test-nvidia nvidia runpod"]
    assert "1058.3 ms per step" in r.summary and "187.2 ms" in r.summary
    assert "5.65x llm.c's CUDA time" in r.summary
    assert "llm.c's test on NVIDIA L4: PASS" in r.summary
    assert "speed on NVIDIA RTX 2000 Ada Generation: 1058.3" in r.summary
    assert "same GPU: FAIL" in r.summary  # the L4 cross-check, shown as it was
    assert "$0.04 of $5.00 spent, call 1 of 5" in r.summary
    req = next(queue.glob("req-*"))
    assert (req / "speed-task" / "profile").exists()  # the profile request reached the task
    assert (req / "test-task" / "agent_port" / "test_gpt2.cpp").read_bytes() == (
        EV / "tune" / "port" / "test_gpt2.cpp").read_bytes()


def test_the_budget_is_the_brokers(tmp_path: Path) -> None:
    queue, calls = tmp_path / "q", []
    queue.mkdir()
    t = _serve(tmp_path, queue, _fake_remote(calls), budget_usd=0.5)
    r = submit(queue, _work(tmp_path), "nvidia", timeout_s=30, poll_s=0.05)
    (queue / "STOP").write_text("")
    t.join(5)
    assert calls == [] and r.status == "error" and "GPU budget used" in r.summary


def test_the_tool_is_offered_and_dispatched(tmp_path: Path) -> None:
    with gpu_tool():
        names = [s["name"] for s in tk.tool_schemas(False, True)]
    assert "gpu_test" in names and "run" in names
    assert "gpu_test" not in [s["name"] for s in tk.tool_schemas(False, True)]
    assert GPU_TOOL_SCHEMA["input_schema"]["properties"]["vendor"]["enum"] == ["nvidia", "amd"]
    cfg = SimpleNamespace(baseline=False, shell=True)
    box = toolbox_factory(NoMigrateToolbox, tmp_path / "q")(cfg, tmp_path)
    r = box("gpu_test", {"vendor": "intel"})
    assert r.status == "error" and "vendor" in r.summary


@pytest.mark.skipif(sys.platform != "linux", reason="shell scripts")
def test_profile_prints_the_top_kernels(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "run_speed.py").write_text(RUN_SPEED)
    for f in ("gpt2_124M_bf16.bin", "gpt2_tokenizer.bin", "tiny_shakespeare_train.bin",
              "tiny_shakespeare_val.bin"):
        (task / f).write_bytes(b"")
    (task / "args.txt").write_text(SPEED_ARGS)
    (task / "profile").write_text("1")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "sycl"
    exe.write_text("#!/bin/sh\nfor i in $(seq 1 30); do echo \"step   $i/74 | loss 4.0 (+0.00z)"
                   "| norm 1.0 (+0.00z)| lr 3.00e-04 | 100.0 ms | 40.0% bf16 MFU\"; done\n")
    nsys = bin_dir / "nsys"
    nsys.write_text("#!/bin/sh\n[ \"$1\" = stats ] && printf '%s\\n' 'note' "
                    "'Time (%),Total Time (ns),Instances,Avg (ns),Name' "
                    "'70.0,7000000,120,1,attention_kernel' '30.0,3000000,48,1,gemm'\nexit 0\n")
    for f in (exe, nsys):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    out = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(task / "run_speed.py"), str(exe),
                            str(task / "args.txt"), "o.bin"], cwd=tmp_path, env=env,
                           capture_output=True, text=True, check=True)
        out.append(p.stdout)
    assert "=== profile (sycl, nsys, 10 steps): 2 kernels, 10.0 ms" in out[0]
    assert " 70.0%        7.0 ms  x   120  attention_kernel" in out[0]
    assert "=== profile" not in out[1]  # once per binary
    assert "kernel_ms=100.0" in out[0]


def test_submit_times_out_without_a_broker(tmp_path: Path) -> None:
    t0 = time.time()
    r = submit(tmp_path / "q", _work(tmp_path), timeout_s=0.2, poll_s=0.05)
    assert r.status == "error" and "broker" in r.summary and time.time() - t0 < 5


def test_a_failing_test_shows_its_failed_checks_first() -> None:
    from museport.gpubroker import failing
    out = ("overall okay: 0\nfailed checks:\nTENSOR NOT OK: wpe\ntest output tail:\n"
           + "OK 1 1\n" * 2000 + "overall okay: 0\n")
    shown = failing(out)
    assert shown.startswith("failed checks:\nTENSOR NOT OK: wpe") and len(shown) < 3000
    assert shown.endswith("overall okay: 0")


def test_a_pure_oneapi_broker_refuses_vendor_interop(tmp_path: Path) -> None:
    from museport.gpubroker import VENDOR_INTEROP, forbidden

    queue, calls = tmp_path / "q", []
    queue.mkdir()
    t = _serve(tmp_path, queue, _fake_remote(calls), forbid=VENDOR_INTEROP)
    work = tmp_path / "gwork"
    shutil.copytree(EV / "gtune" / "port", work)  # the final port, with its cuBLAS interop
    r = submit(queue, work, "nvidia", timeout_s=30, poll_s=0.05)
    (queue / "STOP").write_text("")
    t.join(5)
    # the tuned port with cuBLAS interop is refused before any GPU is rented
    assert calls == [] and r.status == "error" and "batched_gemm.hpp" in r.summary
    clean = tmp_path / "clean"
    (clean / "llmc").mkdir(parents=True)
    (clean / "llmc" / "matmul.hpp").write_text("oneapi::math::blas::column_major::gemm(q);\n")
    assert forbidden(clean, VENDOR_INTEROP) == []
