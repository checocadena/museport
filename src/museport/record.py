"""Differential checks (plan item J1): the hidden reference is the original CUDA's own output.

Today each llm.c task's hidden reference is a C++ program built from llm.c's CPU code, written
per file. A differential check needs no such code:

1. ``recording_task``: a ``remote-test`` task that builds the original CUDA harness with
   ``nvcc`` on an NVIDIA GPU and records its outputs (the toolkit's ``record_native``), with
   no port involved.
2. ``differential_task``: the same task with the recorded outputs as its reference. They sit in
   the task directory with a small ``golden_ref.py``, outside the work directory the model's
   tools can read, and ``reference_cmd`` names them through ``{task_dir}``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .llmc import SPECS
from .port import write_inputs

GOLDEN_REF = '''"""Hidden reference of a differential check: copy the original program's
recorded output for this input. usage: golden_ref.py <input> <output>"""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
index = __import__("json").loads((here / "golden" / "index.json").read_text())
name = Path(sys.argv[1]).name
if name not in index:
    sys.exit(f"no recorded output for input {name!r}")
shutil.copyfile(here / "golden" / index[name], sys.argv[2])
'''


def _ensure_inputs(task: dict[str, Any], task_dir: Path) -> None:
    """Inputs are not committed (up to 16 MB); they are deterministic, so write them again."""
    for name in task.get("inputs", []):
        if not (task_dir / name).exists():
            source = Path(task.get("museport", {}).get("source", "")).name
            if source not in SPECS:
                raise ValueError(f"{task_dir}: {name} missing and no spec for {source!r}")
            write_inputs(SPECS[source], task_dir / name)


def recording_task(task_dir: Path, dest: Path) -> Path:
    """A ``remote-test`` task that records the original CUDA's outputs on an NVIDIA GPU."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task_dir, dest)
    task = json.loads((dest / "task.json").read_text())
    _ensure_inputs(task, dest)
    task["record_native"] = True
    task["reference_build"] = ["true"]  # nothing to build: the recording is the reference
    task.pop("sycl_sources", None)
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


def differential_task(task_dir: Path, recorded: Path, dest: Path) -> Path:
    """The task with the original CUDA's recorded outputs as its hidden reference.

    ``recorded`` is a ``remote-test`` result directory of a recording task (its
    ``summary.json`` lists ``recorded`` files with their input names and sha256)."""
    summary = json.loads((recorded / "summary.json").read_text())
    ev = summary.get("evidence", {})
    if summary.get("status") != "ok" or not ev.get("recorded"):
        raise ValueError(f"{recorded}: not a successful recording ({summary.get('summary')})")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task_dir, dest)
    task = json.loads((dest / "task.json").read_text())
    golden = dest / "golden"
    golden.mkdir()
    index = {}
    for rec in ev["recorded"]:
        src = recorded / "results" / rec["file"]
        shutil.copyfile(src, golden / src.name)
        index[rec["input"]] = src.name
    (golden / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (dest / "golden_ref.py").write_text(GOLDEN_REF)
    task["reference_build"] = ["true"]
    task["reference_cmd"] = "python3 {task_dir}/golden_ref.py {input} {output}"
    task.setdefault("museport", {})["reference"] = {
        "kind": "differential: the original CUDA's recorded output",
        "device": (ev.get("device") or {}).get("name"),
        "deterministic": ev.get("deterministic"),
        "recorded": ev["recorded"],
    }
    for stale in ("cpu_proof.c",):  # a CPU reference is no longer part of the check
        (dest / stale).unlink(missing_ok=True)
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


def check_recording(task_dir: Path, diff_dir: Path, work: Path) -> dict[str, Any]:
    """Do the recorded outputs agree with the CPU reference the task used before? Same inputs,
    same tolerance; on CPU, no GPU."""
    from oneapi_agent.toolchain import run_command
    from oneapi_agent.tools.verify import verify

    task = json.loads((task_dir / "task.json").read_text())
    diff = json.loads((diff_dir / "task.json").read_text())
    if work.exists():
        shutil.rmtree(work)
    (work / "build").mkdir(parents=True)
    for name in task.get("inputs", []):  # the task's own inputs: the recording used them
        if (task_dir / name).exists():
            shutil.copyfile(task_dir / name, work / name)
    _ensure_inputs(task, work)  # only what the task dir lacks (uncommitted Tier 1 inputs)
    build = [a.replace("{task_dir}", str(task_dir.resolve())) for a in task["reference_build"]]
    rc = run_command(build, cwd=str(work), timeout=600).returncode
    if rc != 0:
        return {"status": "error", "summary": f"CPU reference did not build (exit {rc})"}
    golden = diff["reference_cmd"].replace("{task_dir}", str(diff_dir.resolve()))
    res = verify(task["reference_cmd"], golden, task.get("inputs", []),
                 task.get("tolerance", 1e-5), rtol=task.get("rtol", 0.0),
                 mode=task.get("mode", "auto"), output_format=task.get("output_format", "auto"),
                 cwd=str(work), work_dir=str(work / "v"))
    return {"status": res.status, "summary": res.summary}


def diff_check(port_dir: Path, recordings: Path, out: Path, tasks: Path) -> dict[str, Any]:
    """For every item of a ``museport port`` run with a recording: build its differential task
    (under ``tasks``), check the recording against the CPU reference, and grade the item's port
    against the recording with the same steps as the hidden grade. Writes ``out``/summary.json
    and SUMMARY.md; the recorded outputs themselves stay in ``recordings`` (sha256 kept)."""
    from .port import grade_run

    out.mkdir(parents=True, exist_ok=True)
    port_summary = json.loads((port_dir / "summary.json").read_text())
    rows = []
    for r in port_summary["items"]:
        item = port_dir / r["item"]
        name = json.loads((item / "task" / "task.json").read_text())["name"]
        found = sorted(recordings.glob(f"*/{name}/summary.json"))
        if not found:
            rows.append({"item": r["item"], "recorded": False})
            continue
        rec = found[-1].parent
        diff = tasks / r["item"]
        differential_task(item / "task", rec, diff)
        meta = json.loads((diff / "task.json").read_text())["museport"]["reference"]
        work = tasks / ".work" / r["item"]
        before = r.get("evaluation") or {}
        rows.append({
            "item": r["item"], "recorded": True, "device": meta["device"],
            "deterministic": meta["deterministic"],
            "sha256": [x["sha256"] for x in meta["recorded"]],
            "recording_vs_cpu_reference": check_recording(item / "task", diff, work / "cpu"),
            "cpu_grade": {k: before.get(k) for k in ("verify_status", "verify_summary",
                                                     "helpers")},
            "differential_grade": grade_run(diff / "task.json", item / "port",
                                            item / "trace.jsonl", work / "port"),
        })
    summary = {"port_run": str(port_dir), "items": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(render_diff_check(summary))
    return summary


def render_diff_check(summary: dict[str, Any]) -> str:
    md = ["# Differential check: the original CUDA's recorded output as the hidden reference", "",
          f"Port run: `{summary['port_run']}`", "",
          "| item | recorded on | two runs identical | recording vs CPU reference "
          "| port vs CPU reference | port vs recorded CUDA |", "|---|---|---|---|---|---|"]
    for r in summary["items"]:
        if not r["recorded"]:
            md.append(f"| {r['item']} | not recorded | | | | |")
            continue
        cpu, dg = r["cpu_grade"], r["differential_grade"]
        md.append(f"| {r['item']} | {r['device']} | {'yes' if r['deterministic'] else 'no'} "
                  f"| {r['recording_vs_cpu_reference']['summary']} "
                  f"| {cpu.get('verify_summary')} | {dg.get('verify_summary') or dg.get('note')} |")
    return "\n".join(md) + "\n"
