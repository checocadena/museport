"""L1 tuning with GPU feedback: Muse's ``gpu_test`` tool and the broker that runs it.

The loop runs with no credentials (``scripts/clean-env.sh``), so it cannot rent a GPU itself.
Its ``gpu_test`` tool copies the port's current files into a request folder in a queue and waits.
The broker, a separate process started with the GPU providers' keys, picks the request up, runs
the toolkit's ``remote_test`` on two tasks, and writes back a short result. The two tasks are
llm.c's test (correctness on the GPU) and llm.c's trainer for 30 steps (ms per step, against llm.c's
CUDA trainer on the same GPU on NVIDIA). The keys never enter the loop's process, and the
broker enforces its own GPU budget and call limit, whoever writes into the queue.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import oneapi_agent.loop as tk
from oneapi_agent.result import ToolResult

PORT_FILES = ("test_gpt2.cpp", "train_gpt2.cpp", "llmc")
VENDORS = ("nvidia", "amd")
PROVIDER = {"nvidia": ("runpod", False), "amd": ("digitalocean", True)}  # (provider, public_tls)
MAX_USD_PER_TASK = {"nvidia": 0.5, "amd": 1.5}

GPU_TOOL_SCHEMA: dict[str, Any] = {
    "name": "gpu_test",
    "description": (
        "Build your current port (test_gpt2.cpp, train_gpt2.cpp, llmc/) on a real rented GPU "
        "and run it there: llm.c's test (pass/fail and its failing lines), and llm.c's trainer "
        "for 30 steps at B=4, T=1024 (median ms per step over steps 6-30; on NVIDIA also llm.c's "
        "own CUDA trainer on the same GPU, for comparison). profile=true adds a per-kernel time "
        "table for the trainer (nsys on NVIDIA, rocprofv3 on AMD). Takes about 3-8 minutes and "
        "costs real money; the run has a fixed GPU budget, and the result says what is left."),
    "input_schema": {"type": "object", "properties": {
        "vendor": {"type": "string", "enum": list(VENDORS)},
        "profile": {"type": "boolean"}}, "required": [], "additionalProperties": False},
}


# ----------------------------------------------------------------------------- loop side

def submit(queue: Path, work: Path, vendor: str = "nvidia", profile: bool = False, *,
           timeout_s: float = 2400, poll_s: float = 5) -> ToolResult:
    """Write a request with the port's current files into ``queue`` and wait for its result."""
    if vendor not in VENDORS:
        return ToolResult("gpu_test", "error", f"vendor must be one of {list(VENDORS)}")
    missing = [f for f in PORT_FILES if not (work / f).exists()]
    if missing:
        return ToolResult("gpu_test", "error", f"missing in the work dir: {missing}")
    queue.mkdir(parents=True, exist_ok=True)
    req = queue / f"req-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{time.time_ns() % 10**6}"
    tmp = req.with_name(req.name + ".tmp")
    for f in PORT_FILES:
        src = work / f
        if src.is_dir():
            shutil.copytree(src, tmp / "port" / f)
        else:
            (tmp / "port").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tmp / "port" / f)
    (tmp / "request.json").write_text(json.dumps({"vendor": vendor, "profile": bool(profile)}))
    tmp.rename(req)  # the broker only sees complete requests
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (req / "result.json").exists():
            r = json.loads((req / "result.json").read_text())
            return ToolResult("gpu_test", r["status"], r["summary"], r.get("evidence", {}))
        time.sleep(poll_s)
    return ToolResult("gpu_test", "error", f"no result after {timeout_s / 60:.0f} min "
                      "(the GPU broker may be down); carry on without it")


class GpuToolMixin:
    """Adds ``gpu_test`` to a Toolbox; ``queue`` is set by the factory (see ``toolbox_factory``)."""

    queue: Path
    work: Path

    def t_gpu_test(self, a: dict[str, Any]) -> ToolResult:
        return submit(self.queue, self.work, a.get("vendor") or "nvidia", bool(a.get("profile")))


def toolbox_factory(base: type, queue: Path) -> Callable[..., Any]:
    cls = type("GpuToolbox", (GpuToolMixin, base), {})

    def make(cfg: Any, work: Path) -> Any:
        box = cls(cfg, work)
        box.queue = queue
        return box
    return make


@contextmanager
def gpu_tool() -> Iterator[None]:
    """Offers ``gpu_test`` to the model for the length of a run."""
    schemas = tk.tool_schemas
    tk.tool_schemas = lambda *a, **k: [*schemas(*a, **k), GPU_TOOL_SCHEMA]
    try:
        yield
    finally:
        tk.tool_schemas = schemas


# ----------------------------------------------------------------------------- broker side

def _summary(into: Path) -> dict[str, Any] | None:
    found = sorted(into.rglob("summary.json"))
    return json.loads(found[0].read_text()) if found else None


def _results(into: Path, name: str) -> dict[str, Any] | None:
    found = sorted(into.rglob(f"results/{name}.json"))
    return json.loads(found[0].read_text()) if found else None


def _stdout(step: dict[str, Any] | None) -> str:
    if not step:
        return ""
    cases = step.get("evidence", {}).get("cases", [])
    return (cases[0].get("candidate", {}).get("stdout", "") if cases else "") or ""


def _ms(step: dict[str, Any] | None) -> float | None:
    if not step or step.get("status") != "ok":
        return None
    res = step["evidence"]["results"][0]
    rep = res.get("reported") or {}
    return rep.get("median")


def describe(vendor: str, test: tuple[dict[str, Any], Path], speed: tuple[dict[str, Any], Path],
             spent: float, budget: float, calls: int, max_calls: int) -> dict[str, Any]:
    """The tool result Muse sees: short, with the numbers that matter for tuning."""
    lines: list[str] = []
    (tr, tinto), (sr, sinto) = test, speed
    ts, ss = _summary(tinto), _summary(sinto)

    def dev(s: dict[str, Any] | None) -> str:
        return str((s or {}).get("evidence", {}).get("device", {}).get("name", "?"))
    device = dev(ts)
    if ts is None:
        lines.append(f"llm.c's test: did not run: {tr.get('summary', '')[:600]}")
        test_ok = False
    else:
        steps = {s["step"]: s for s in ts["evidence"].get("steps", [])}
        comp = steps.get("sycl_compile", {})
        if comp.get("status") != "ok":
            c = _results(tinto, "sycl_compile") or {}
            diags = json.dumps(c.get("evidence", {}).get("diagnostics", c.get("summary", "")))
            lines.append(f"compile FAILED on the GPU machine: {diags[:2500]}")
            test_ok = False
        else:
            v = _results(tinto, "sycl_verify_vs_reference")
            out = _stdout(v)
            test_ok = (v or {}).get("status") == "ok"
            lines.append(f"llm.c's test on {device}: {(v or {}).get('summary', 'no result')}")
            if not test_ok:
                lines.append(out[-2500:])
            nv = steps.get("sycl_verify_vs_native")
            if nv:
                lines.append(f"  losses vs llm.c's CUDA test on the same GPU: {nv['summary']}")
    if ss is None:
        lines.append(f"speed: did not run: {sr.get('summary', '')[:600]}")
    else:
        ms = _ms(_results(sinto, "sycl_benchmark"))
        native = _ms(_results(sinto, "native_benchmark"))
        v = _results(sinto, "sycl_verify_vs_reference")
        if ms is None:
            lines.append("speed: no timing: " + "; ".join(
                f"{s['step']} {s['status']}: {s['summary'][:300]}"
                for s in ss["evidence"].get("steps", []) if s["status"] != "ok"))
            lines.append(_stdout(v)[-1500:])
        else:
            ratio = f", {ms / native:.2f}x llm.c's CUDA time" if native else ""
            lines.append(f"speed on {dev(ss)}: {ms:.1f} ms per step (median, steps 6-30)"
                         + (f"; llm.c's CUDA trainer: {native:.1f} ms" if native else "")
                         + ratio)
            lines.append(f"  30 training losses vs llm.c's CUDA recording: "
                         f"{(v or {}).get('summary', '?')}")
        for step in ("sycl_verify_vs_reference", "native_verify_vs_reference"):
            prof = re.search(r"=== profile.*", _stdout(_results(sinto, step)), re.S)
            if prof:
                lines.append(prof.group(0)[:3500])
    lines.append(f"GPU budget: ${spent:.2f} of ${budget:.2f} spent, call {calls} of {max_calls}")
    status = "ok" if test_ok and ss is not None else "fail"
    return {"status": status, "summary": "\n".join(lines),
            "evidence": {"vendor": vendor, "test_device": device, "speed_device": dev(ss),
                         "test": tr.get("summary"),
                         "speed": sr.get("summary")}}


def serve(queue: Path, *, llmc: Path, recorded: Path, evidence_dir: Path, run: str,
          onemath_patch: Path, budget_usd: float, max_calls: int, model: str,
          log: Callable[[str], None] = print, poll_s: float = 5,
          remote: Callable[..., Any] | None = None) -> None:
    """Answer requests in ``queue`` until a STOP file appears there. Every request and its
    results stay in the queue folder as evidence."""
    from oneapi_agent.remote import remote_test

    from .bf16 import make_gpu_task
    from .speed import make_speed_task

    remote = remote or remote_test
    spent, calls = 0.0, 0
    while not (queue / "STOP").exists():
        reqs = sorted(p for p in queue.glob("req-*") if p.is_dir()
                      and not p.name.endswith(".tmp") and not (p / "claimed").exists())
        if not reqs:
            time.sleep(poll_s)
            continue
        req = reqs[0]
        (req / "claimed").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        r = json.loads((req / "request.json").read_text())
        vendor = r.get("vendor") if r.get("vendor") in VENDORS else "nvidia"
        est = 2 * MAX_USD_PER_TASK[vendor]
        if calls >= max_calls or spent + est > budget_usd:
            _write(req, {"status": "error", "summary": (
                f"GPU budget used: ${spent:.2f} of ${budget_usd:.2f}, {calls} of {max_calls} "
                "calls; no more GPU runs in this session. Finish with what you have measured.")})
            continue
        calls += 1
        patches = (onemath_patch,) if vendor == "nvidia" else ()
        # the request's port, laid out as make_gpu_task expects: <evidence>/<run>/port
        ev = req / "ev"
        shutil.copytree(evidence_dir / "task", ev / "task")
        shutil.copytree(req / "port", ev / run / "port")
        tdir, sdir = req / "test-task", req / "speed-task"
        make_gpu_task(ev, tdir, model, vendor=vendor, run=run, onemath_patches=patches)
        make_speed_task(llmc, req / "port", recorded, sdir, model, vendor=vendor,
                        onemath_patches=patches, runs=1 if r.get("profile") else 2)
        if r.get("profile"):
            (sdir / "profile").write_text("1\n")
        log(f"{req.name}: {vendor}{' profile' if r.get('profile') else ''}")
        out = _run_both(remote, req, vendor, {"test": tdir, "speed": sdir})
        for key in ("test", "speed"):
            spent += float(out[key][0].get("evidence", {}).get("est_cost_usd") or 0)
        res = describe(vendor, out["test"], out["speed"], spent, budget_usd, calls, max_calls)
        (req / "remote.json").write_text(json.dumps(
            {k: v[0] for k, v in out.items()}, indent=1, default=str))
        _write(req, res)
        log(f"{req.name}: {res['summary'].splitlines()[1:3]} (${spent:.2f})")


def _run_both(remote: Callable[..., Any], req: Path, vendor: str,
              tasks: dict[str, Path]) -> dict[str, tuple[dict[str, Any], Path]]:
    """Both tasks at once, each on its own rented GPU."""
    provider, public_tls = PROVIDER[vendor]
    out: dict[str, tuple[dict[str, Any], Path]] = {}

    def go(key: str, task: Path) -> None:
        into = req / f"{key}-results"
        try:
            res = remote(str(task / "task.json"), vendor=vendor, provider=provider,
                         public_tls=public_tls, max_usd=MAX_USD_PER_TASK[vendor],
                         max_minutes=40, into=str(into), log=lambda m: None)
            out[key] = ({"status": res.status, "summary": res.summary,
                         "evidence": res.evidence}, into)
        except Exception as e:  # a crashed call still gets an answer
            out[key] = ({"status": "error", "summary": f"{type(e).__name__}: {e}"}, into)

    threads = [threading.Thread(target=go, args=kv) for kv in tasks.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def _write(req: Path, result: dict[str, Any]) -> None:
    tmp = req / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(req / "result.json")
