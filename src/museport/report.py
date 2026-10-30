"""`museport report`: a static site built only from the committed evidence.

Pages: an overview (results per tier, every item, spend), the plan and its grounding, and one
page per item (hidden grade, variant check, GPU runs, the CUDA harness beside Muse's port, and a
turn-by-turn replay of the loop's trace). Every number on a page is read from a file under
``evidence/``; nothing is typed in by hand."""

from __future__ import annotations

import difflib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TIER_NAMES = {1: "Plain kernels", 2: "Warp/sub-group, shared memory", 3: "cuBLAS to oneMath"}
PLAN_DIR = "plan-llmc-devcuda-stream"
REPO = "https://github.com/checocadena/museport"


@dataclass
class Item:
    batch: str  # evidence directory, e.g. port-llmc-tier1
    row: dict[str, Any]
    task: dict[str, Any]
    dir: Path
    gpu: list[dict[str, Any]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.row["item"]

    @property
    def slug(self) -> str:
        return f"{self.batch}--{self.id}"

    @property
    def tier(self) -> int:
        m = re.match(r"t(\d)-", self.id)
        return int(m.group(1)) if m else 0

    @property
    def rerun(self) -> bool:
        return not re.search(r"tier\d$", self.batch)

    @property
    def correct(self) -> bool:
        ev = self.row.get("evaluation") or {}
        return ev.get("verify_status") == "ok" and ev.get("helpers") == 0

    @property
    def amd_pass(self) -> bool:
        """Passed on an AMD GPU against the reference (llm.c's CUDA does not build there)."""
        return any(g["status"] == "ok" and g["vendor"] == "amd" for g in self.gpu)

    @property
    def gpu_pass(self) -> bool:
        """Passed on an NVIDIA GPU against the reference and llm.c's own CUDA."""
        return any(g["status"] == "ok" and g["vendor"] == "nvidia" and g["vs_native"]
                   for g in self.gpu)


# ----------------------------------------------------------------------------- loading


def load(evidence: Path) -> dict[str, Any]:
    """Every port batch under ``evidence`` with its items and GPU results, and the plan."""
    items: list[Item] = []
    batches: dict[str, dict[str, Any]] = {}
    for summary in sorted(evidence.glob("port-*/summary.json")):
        bdir = summary.parent
        data = json.loads(summary.read_text())
        readme = bdir / "README.md"
        title = next((ln[2:].strip() for ln in readme.read_text().splitlines()
                      if ln.startswith("# ")), bdir.name) if readme.exists() else bdir.name
        batches[bdir.name] = {"model": data["model"], "spent_usd": data["spent_usd"],
                              "title": title}
        gpu = _gpu_results(bdir)
        for extra in EXTRA_GPU.get(bdir.name, ()):
            for task, runs in _gpu_results(evidence / extra, sub=".").items():
                gpu.setdefault(task, []).extend(runs)
        for row in data["items"]:
            task_file = bdir / row["item"] / "task" / "task.json"
            task = json.loads(task_file.read_text()) if task_file.exists() else {}
            items.append(Item(bdir.name, row, task, bdir / row["item"],
                              gpu.get(task.get("name", ""), [])))
    plan_file = evidence / PLAN_DIR / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else None
    return {"items": items, "batches": batches, "plan": plan, "gpt2": load_gpt2(evidence),
            "diff": load_diff(evidence), "specs": load_specs(evidence), "j3": load_j3(evidence),
            "k1": load_k1(evidence), "j4": load_j4(evidence), "l1": load_l1(evidence)}


def load_k1(evidence: Path) -> dict[str, Any] | None:
    """K1's loss curves (evidence/k1-curves), or None before they exist."""
    from .curves import collect

    try:
        return collect(evidence)
    except FileNotFoundError:
        return None


def load_diff(evidence: Path) -> list[dict[str, Any]]:
    """Differential checks (evidence/diff-*/, written by ``museport diff-check``)."""
    out = []
    for summary in sorted(evidence.glob("diff-*/summary.json")):
        d = summary.parent
        planted = d / "planted" / "planted.json"
        out.append({"dir": d.name, **json.loads(summary.read_text()),
                    "planted": json.loads(planted.read_text()) if planted.exists() else {}})
    return out


def load_specs(evidence: Path) -> list[dict[str, Any]]:
    """Muse-written harness specs (evidence/spec-llmc*, written by ``museport spec``): each
    file's latest run, runs taken in name order (spec-llmc, then spec-llmc-fp32). A file the
    loop left unproven but whose first spec passed a GPU proof after the run
    (``<file>/gpu-proof/gpu_proof.json``) counts as proven by that proof."""
    from .llmc import SPECS

    rows: dict[str, dict[str, Any]] = {}
    for summary in sorted(evidence.glob("spec-llmc*/summary.json")):
        d = summary.parent
        for r in json.loads(summary.read_text())["files"]:
            row = {**r, "run": d.name}
            stem = r["file"].removesuffix(".cu")
            proof = d / stem / "gpu-proof" / "gpu_proof.json"
            attempts = d / stem / "attempts.json"
            if not r["proven"] and proof.exists() and attempts.exists():
                first = json.loads(attempts.read_text())[0].get("answer")
                ok = json.loads(proof.read_text()).get(f"{stem}-a1", {})
                hand = SPECS.get(r["file"])
                if first and ok.get("status") == "ok":
                    row.update(proven=True, proven_by="gpu, after the run (first spec)",
                               variants=first["variants"], gpu_summary=ok.get("summary"))
                    if hand:
                        row["compare"] = {"variants": first["variants"] == list(hand.variants),
                                          "tol": first["tol"] == hand.tol}
            rows[r["file"]] = row
    order = {f: i for i, f in enumerate(SPECS)}
    return sorted(rows.values(), key=lambda r: order.get(r["file"], len(order)))


# J3's two programs: evidence dir, the label, and the program's name in llm.c-sycl
J3_PROGRAMS = (("j3-trainer", "llm.c's fp32 trainer", "train_gpt2fp32sycl"),
               ("j3-test", "llm.c's own test", "test_gpt2fp32sycl"))


def load_j3(evidence: Path) -> dict[str, Any] | None:
    """J3, the reassembled llm.c-sycl repo: the dev/sycl check (evidence/j3-assemble) and, per
    program, its Muse runs (run1, run2, ...; the last is the one graded), its check proof and
    its GPU results."""
    check = evidence / "j3-assemble" / "check.json"
    programs = []
    for name, label, target in J3_PROGRAMS:
        d = evidence / name
        runs = [json.loads(f.read_text()) | {"run": f.parent.name}
                for f in sorted(d.glob("run*/outcome.json"))]
        if not runs:
            continue
        proof = d / "check-proof.json"
        programs.append({"dir": name, "label": label, "target": target, "runs": runs,
                         "proof": json.loads(proof.read_text()) if proof.exists() else {},
                         "gpu": [g for gs in _gpu_results(d).values() for g in gs]})
    if not programs:
        return None
    return {"check": json.loads(check.read_text()) if check.exists() else None,
            "fresh": _fresh_clone(evidence / "j3-fresh-clone"),
            "programs": programs}


def load_j4(evidence: Path) -> dict[str, Any] | None:
    """J4, llama2.cu end to end (evidence/j4-llama2): Muse's plan and port runs, the check's
    proof, the GPU results and the fresh-clone comparison (evidence/j4-fresh-clone)."""
    d = evidence / "j4-llama2"
    runs = [json.loads(f.read_text()) | {"run": f.parent.name}
            for f in sorted(d.glob("run*/outcome.json"))]
    if not runs:
        return None
    plan = json.loads((d / "plan" / "plan.json").read_text())
    compare = evidence / "j4-fresh-clone" / "compare.txt"
    fresh = compare.read_text().splitlines() if compare.exists() else []
    port = d / runs[-1]["run"] / "port" / "llama2.cpp"
    return {"dir": d.name, "runs": runs, "plan_cost": plan.get("cost_usd"),
            "port_lines": len(port.read_text().splitlines()),
            "plan_items": len(plan.get("plan", {}).get("items", [])),
            "grounded": (plan.get("grounding") or {}).get("grounded"),
            "proof": json.loads((d / "check-proof.json").read_text())["rows"],
            "gpu": [g for gs in _gpu_results(d).values() for g in gs],
            "fresh": [ln.split(": ", 1)[0] + ": " + ln.rsplit(": ", 1)[-1] for ln in fresh],
            "fresh_ok": bool(fresh) and all(ln.endswith("True") for ln in fresh)}


L1_RUNS = ("run1", "run2", "run3", "run4", "run5", "tune", "gtune")


def _l1_speed(d: Path) -> dict[str, Any] | None:
    """A speed task's result folder: the GPU and llm.c's step timer (median ms per step) for
    the port and, on NVIDIA, for llm.c's CUDA on the same GPU."""
    summary = next(iter(sorted(d.rglob("summary.json"))), None)
    if summary is None:
        return None

    def ms(name: str) -> float | None:
        f = summary.parent / "results" / f"{name}.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())
        return r["evidence"]["results"][0].get("reported", {}).get("median") \
            if r.get("status") == "ok" else None

    s = json.loads(summary.read_text())
    port, cuda = ms("sycl_benchmark"), ms("native_benchmark")
    return {"device": s["evidence"]["device"]["name"].split(" VF ")[0],
            "vendor": s["evidence"]["vendor"], "ms": port, "cuda_ms": cuda,
            "ratio": port / cuda if port and cuda else None}


def _l1_call(summary: str) -> dict[str, Any]:
    """One gpu_test answer (the text Muse saw), as numbers."""
    test = re.search(r"^llm.c's test on (.+?): (PASS|FAIL)", summary, re.M)
    speed = re.search(r"^speed on (.+?): ([0-9.]+) ms per step", summary, re.M)
    cuda = re.search(r"CUDA trainer: ([0-9.]+) ms", summary)
    ms = float(speed.group(2)) if speed else None
    cuda_ms = float(cuda.group(1)) if cuda else None
    return {"test_device": test.group(1).split(" VF ")[0] if test else None,
            "test": test.group(2) if test else
            ("compile failed" if "compile FAILED" in summary else None),
            "device": speed.group(1).split(" VF ")[0] if speed else None,
            "ms": ms, "cuda_ms": cuda_ms,
            "ratio": ms / cuda_ms if ms and cuda_ms else None}


def load_l1(evidence: Path) -> dict[str, Any] | None:
    """L1, llm.c's bf16 trainer (evidence/l1-llmc-bf16): each run, the speed of run 5's and
    the CPU-tuned port, every gpu_test call of the GPU tuning run (marking those that built
    the final port), and our diagnostics on the final port."""
    d = evidence / "l1-llmc-bf16"
    if not (d / "gtune" / "outcome.json").exists():
        return None
    runs = {r: json.loads((d / r / "outcome.json").read_text())
            for r in L1_RUNS if (d / r / "outcome.json").exists()}
    final = d / "gtune" / "port"

    def same(port: Path) -> bool:
        files = sorted(f.relative_to(final) for f in final.rglob("*") if f.is_file())
        return files == sorted(f.relative_to(port) for f in port.rglob("*") if f.is_file()) \
            and all((final / f).read_bytes() == (port / f).read_bytes() for f in files)

    calls = []
    for req in sorted((d / "gtune" / "gpu-calls").glob("req-*")):
        c = _l1_call(json.loads((req / "result.json").read_text())["summary"])
        cost = sum(float(v.get("evidence", {}).get("est_cost_usd") or 0) for v in
                   json.loads((req / "remote.json").read_text()).values())
        c |= {"request": req.name, "vendor": json.loads((req / "request.json").read_text())
              ["vendor"], "final": same(req / "port"), "gpu_usd": cost,
              "cublas": any("dlopen" in f.read_text() for f in (req / "port").rglob("*.hpp"))}
        calls.append(c)
    diag: dict[str, dict[str, Any]] = {}
    results = d / "gtune" / "diagnostics" / "results.txt"
    if results.exists():
        for block in results.read_text().split("===== ")[1:]:
            head, _, body = block.partition("\n")
            name = "pure" if head.startswith("pure") else \
                "forced_amd" if "amd" in head else "forced_nvidia"
            if " ok" in head:
                diag[name] = _l1_call(body)  # the last ok run of each variant
    plan = d / "plan" / "plan.json"
    plan_cost = json.loads(plan.read_text()).get("cost_usd") if plan.exists() else 0.0
    return {"runs": runs, "plan_cost": plan_cost,
            "muse_usd": (plan_cost or 0.0) + sum(r.get("cost_usd") or 0.0
                                                 for r in runs.values()),
            "speed": {f"{r}_{v}": _l1_speed(d / r / "gpu" / f"speed-{v}")
                      for r in ("run5", "tune") for v in ("nvidia", "amd")},
            "calls": calls, "final": [c for c in calls if c["final"]], "diag": diag,
            "gpu_tuning_usd": sum(c["gpu_usd"] for c in calls)}


def _fresh_clone(d: Path) -> dict[str, Any] | None:
    """The llm.c-sycl Quick start, run from a fresh clone: each step's wall time (``real`` in
    its log) and whether the test printed ``overall okay: 1``."""
    if not (d / "test.log").exists():
        return None

    def secs(name: str) -> float | None:
        m = re.search(r"^real\s+(\d+)m([\d.]+)s", (d / name).read_text(), re.M)
        return int(m.group(1)) * 60 + float(m.group(2)) if m else None

    return {"okay": "overall okay: 1" in (d / "test.log").read_text(),
            "download_s": secs("download.log"), "make_s": secs("make.log"),
            "test_s": secs("test.log")}


@dataclass
class Gpt2Stage:
    """One stage of the GPT-2 training port (evidence/gpt2-stage<N>/)."""
    dir: Path
    title: str
    task: dict[str, Any]
    outcome: dict[str, Any]
    cpu_proof: dict[str, Any] | None
    gpu: list[dict[str, Any]]

    @property
    def slug(self) -> str:
        return self.dir.name

    @property
    def correct(self) -> bool:
        ev = self.outcome.get("evaluation") or {}
        return ev.get("verify_status") == "ok" and ev.get("helpers") == 0

    @property
    def label(self) -> str:
        stage = self.task.get("museport", {}).get("stage", "")
        return {"forward": "1 · Forward pass", "backward": "2 · Backward pass",
                "train": "3 · Training: 10 AdamW steps"}.get(stage, self.slug)


def _json_after_log(path: Path) -> dict[str, Any]:
    """A JSON file that may start with log lines (captured with the command's stderr)."""
    text = path.read_text()
    return json.loads(text[text.index("{"):])


def load_gpt2(evidence: Path) -> list[Gpt2Stage]:
    stages = []
    for d in sorted(evidence.glob("gpt2-stage*")):
        out_file, task_file = d / "run1" / "outcome.json", d / "task" / "task.json"
        if not (out_file.exists() and task_file.exists()):
            continue
        readme = d / "README.md"
        title = next((ln[2:].strip() for ln in readme.read_text().splitlines()
                      if ln.startswith("# ")), d.name) if readme.exists() else d.name
        proof = d / "cpu-proof.json"
        stages.append(Gpt2Stage(d, title, json.loads(task_file.read_text()),
                                json.loads(out_file.read_text()),
                                _json_after_log(proof) if proof.exists() else None,
                                [g for gs in _gpu_results(d).values() for g in gs]))
    return stages


# GPU runs of a batch's ports kept outside its own gpu/ folder: D2 ran the ports that call
# oneMath, once the GPU tasks could build it (evidence/d2-onemath-gpu/<vendor>/<device>/<task>).
EXTRA_GPU: dict[str, tuple[str, ...]] = {"port-llmc-tier23-run2": ("d2-onemath-gpu",)}


def _gpu_results(bdir: Path, sub: str = "gpu") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    # gpu/<vendor>/<device>/<task>/ (the NVIDIA runs) or gpu/<device>/<task>/ (the AMD runs)
    found = [*bdir.glob(f"{sub}/*/*/*/summary.json"), *bdir.glob(f"{sub}/*/*/summary.json")]
    for f in sorted(found):
        data = json.loads(f.read_text())
        ev = data.get("evidence", {})
        steps = {s.get("step"): s.get("status") for s in ev.get("steps", [])}
        out.setdefault(f.parent.name, []).append({
            "vendor": ev.get("vendor") or f.parent.parent.parent.name,
            # the SYCL port was also compared with the original CUDA built for this GPU
            "vs_native": steps.get("sycl_verify_vs_native") == "ok",
            "device": (ev.get("device") or {}).get("name") or f.parent.parent.name,
            "status": data.get("status"), "summary": data.get("summary", ""),
            "steps": ev.get("steps", []),
            "path": f.parent.relative_to(bdir.parent).as_posix()})
    return out


# The runs with the fixed loop (delete_file, ranged read_file, 32k output per turn, no token
# limit below the USD cap): one batch per item, the same tasks and hidden checks as the first
# runs. They are the headline results; the first runs stay as history.
CURRENT_BATCHES = ("port-llmc-tier1-run2", "port-llmc-tier23-run2")


def tier_table(items: list[Item]) -> list[dict[str, Any]]:
    """First runs only (reruns are reported beside them, not counted in place of them)."""
    return _table([i for i in items if not i.rerun])


def current_table(items: list[Item]) -> list[dict[str, Any]]:
    """The runs with the fixed loop (CURRENT_BATCHES), one per item."""
    return _table([i for i in items if i.batch in CURRENT_BATCHES])


def _table(its: list[Item]) -> list[dict[str, Any]]:
    rows = [_counts([i for i in its if i.tier == t], f"Tier {t}", TIER_NAMES[t])
            for t in (1, 2, 3)]
    rows.append(_counts(its, "All", ""))
    return rows


def _counts(its: list[Item], label: str, name: str) -> dict[str, Any]:
    return {"label": label, "name": name, "items": len(its),
            "correct": sum(i.correct for i in its),
            "success": sum(i.row.get("status") == "success" for i in its),
            "gpu": sum(i.gpu_pass for i in its),
            "amd": sum(i.amd_pass for i in its),
            "cost": round(sum(_item_spend(i.row) for i in its), 4)}


def _item_spend(row: dict[str, Any]) -> float:
    return (row.get("cost_usd") or 0.0) + sum(e.get("cost_usd") or 0.0
                                              for e in row.get("earlier_runs", []))


# ----------------------------------------------------------------------------- html


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def money(x: float | None) -> str:
    return "unknown" if x is None else f"${x:.4f}"


CSS = """
/* Layout: one reading column with a wide data band; tables and code scroll in place. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --fg: #1b1f27; --muted: #5d6675; --line: #dde1e8;
  --accent: #2a5bd7; --ok: #1d7a46; --warn: #a15c00; --bad: #b42318;
  --ok-bg: #e6f4ec; --warn-bg: #fdf1dd; --bad-bg: #fde8e6; --code-bg: #f0f2f6;
  --add: #e3f3e8; --del: #fbe9e7;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11141a; --surface: #181c24; --fg: #e4e7ee; --muted: #9aa3b2; --line: #2b313d;
  --accent: #7ea2ff; --ok: #5cc98a; --warn: #f0b04f; --bad: #ff8a7a;
  --ok-bg: #16301f; --warn-bg: #352812; --bad-bg: #3a1a17; --code-bg: #1e232d;
  --add: #173322; --del: #3a1d1a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11141a; --surface: #181c24; --fg: #e4e7ee; --muted: #9aa3b2; --line: #2b313d;
  --accent: #7ea2ff; --ok: #5cc98a; --warn: #f0b04f; --bad: #ff8a7a;
  --ok-bg: #16301f; --warn-bg: #352812; --bad-bg: #3a1a17; --code-bg: #1e232d;
  --add: #173322; --del: #3a1d1a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--f-body);
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 24px 64px; }
main { max-width: 1080px; margin: 0 auto; display: grid; gap: 28px; }
main > * { min-width: 0; }
h1, h2, h3 { font-family: var(--f-display); line-height: 1.2; text-wrap: balance; margin: 0; }
h1 { font-size: 2rem; letter-spacing: -0.01em; }
h2 { font-size: 1.3rem; margin-bottom: 10px; }
h3 { font-size: 1.05rem; }
p { margin: 0; max-width: 68ch; }
a { color: var(--accent); }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.crumb { font-size: .9rem; color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap; }
.lede { color: var(--muted); font-size: 1.05rem; }
.eyebrow { font: 600 .75rem var(--f-mono); letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); }
section { display: grid; gap: 12px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px;
  background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: .93rem; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { font: 600 .75rem var(--f-mono); letter-spacing: .05em; text-transform: uppercase;
  color: var(--muted); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.total td { font-weight: 700; }
code, pre, .mono { font-family: var(--f-mono); font-size: .85rem; }
.pill { display: inline-block; font: 600 .75rem/1.6 var(--f-mono); padding: 0 8px;
  border-radius: 999px; white-space: nowrap; }
.ok { color: var(--ok); background: var(--ok-bg); }
.warn { color: var(--warn); background: var(--warn-bg); }
.bad { color: var(--bad); background: var(--bad-bg); }
.none { color: var(--muted); background: var(--code-bg); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.facts > div { background: var(--surface); padding: 12px 14px; display: grid; gap: 6px;
  align-content: start; justify-items: start; min-width: 0; }
.facts .v { font: 600 1rem var(--f-mono); font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere; }
.facts .d { color: var(--muted); font-size: .85rem; overflow-wrap: anywhere; }
pre { margin: 0; padding: 12px 14px; background: var(--code-bg); border-radius: 6px;
  overflow-x: auto; line-height: 1.45; }
details { border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600; }
details > :not(summary) { margin: 0 14px 14px; }
.diff .a { background: var(--add); display: block; }
.diff .d { background: var(--del); display: block; }
.diff .h { color: var(--accent); display: block; }
.turns { display: grid; gap: 10px; }
.turn { border-left: 3px solid var(--line); padding: 4px 0 4px 14px; display: grid; gap: 6px; }
.turn .n { font: 600 .75rem var(--f-mono); color: var(--muted); }
.say { white-space: pre-wrap; max-width: 80ch; }
.call { display: grid; gap: 4px; }
.call .head { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.call .sum { color: var(--muted); font-size: .9rem; overflow-wrap: anywhere; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.grid2 > * { min-width: 0; }
.note { color: var(--muted); font-size: .9rem; }
footer { color: var(--muted); font-size: .85rem; border-top: 1px solid var(--line);
  padding-top: 12px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""

FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&"
         "family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700&display=swap")


def page(title: str, body: str, root: str = "") -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, '
            f'viewport-fit=cover">\n<title>{esc(title)}</title>\n'
            f'<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<link rel="stylesheet" href="{FONTS}">\n'
            f'<link rel="stylesheet" href="{root}style.css">\n</head>\n<body><main>\n{body}\n'
            f'<footer>Built by <code>museport report</code> from the committed evidence in '
            f'<a href="{REPO}">checocadena/museport</a>. Every number on this page is read '
            f'from a file under <code>evidence/</code>.</footer>\n</main></body></html>\n')


def pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def grade_pill(it: Item) -> str:
    ev = it.row.get("evaluation") or {}
    if not ev:
        return pill(it.row.get("status", "no run"), "bad")
    if it.correct:
        return pill("correct, 0 helpers", "ok")
    if ev.get("verify_status") == "ok":
        return pill(f"correct, {ev.get('helpers')} helpers", "warn")
    return pill(f"verify {ev.get('verify_status')}", "bad")


def variant_pill(it: Item) -> str:
    vc = it.row.get("variant_check")
    if not vc:
        return pill("not checked", "none")
    if vc.get("duplicated_variants"):
        return pill(f"{vc['distinct_kernel_bodies']} bodies for {vc['variants']} variants", "bad")
    n = vc["variants"]
    return pill(f"{n} variant{'s' if n != 1 else ''} kept", "ok")


def short_device(name: str) -> str:
    name = name.replace("NVIDIA ", "").replace(" Generation", "")
    return name.replace("AMD Instinct ", "").split(" gfx")[0]


def gpu_pill(it: Item) -> str:
    if not it.gpu:
        if any("onemath" in f.lower() for f in it.task.get("base_flags", [])):
            return pill("CPU only (oneMath)", "none")
        return pill("not run", "none")
    pills = []
    for g in it.gpu:
        name = short_device(g["device"])
        if g["status"] != "ok":
            pills.append(pill(f"fail: {name}", "bad"))
        else:
            pills.append(pill(f"pass: {name}" + ("" if g["vs_native"] else " (vs reference)"),
                              "ok"))
    return " ".join(pills)


def loop_pill(it: Item) -> str:
    s = it.row.get("status", "?")
    return pill(s, "ok" if s == "success" else ("bad" if s == "error" else "none"))


# ----------------------------------------------------------------------------- pages


def _tier_rows(table: list[dict[str, Any]], amd: bool) -> str:
    return "".join(
        f'<tr{" class=total" if r["label"] == "All" else ""}><td>{esc(r["label"])}'
        f'{"<br><span class=note>" + esc(r["name"]) + "</span>" if r["name"] else ""}</td>'
        f'<td class=num>{r["items"]}</td><td class=num>{r["correct"]}</td>'
        f'<td class=num>{r["success"]}</td><td class=num>{r["gpu"]}</td>'
        + (f'<td class=num>{r["amd"]}</td>' if amd else "")
        + f'<td class=num>{money(r["cost"])}</td></tr>' for r in table)


def _tier_head(amd: bool) -> str:
    return ('<div class="scroll"><table><thead><tr><th>Tier</th><th class=num>Items</th>'
            '<th class=num>Correct, 0 helpers</th><th class=num>Loop success</th>'
            '<th class=num>NVIDIA pass</th>' + ("<th class=num>AMD pass</th>" if amd else "")
            + '<th class=num>Muse cost</th></tr></thead><tbody>')


def render_index(data: dict[str, Any]) -> str:
    items: list[Item] = data["items"]
    current = [i for i in items if i.batch in CURRENT_BATCHES]
    firsts = [i for i in items if not i.rerun]
    others = [i for i in items if i.rerun and i.batch not in CURRENT_BATCHES]
    body = [
        '<header style="display:grid;gap:10px"><span class="eyebrow">Muse Port · evidence '
        'report</span><h1>Muse ports llm.c from CUDA to SYCL</h1>'
        '<p class="lede">Meta\'s <code>muse-spark-1.3</code> planned the port of '
        "<a href=\"https://github.com/karpathy/llm.c\">karpathy/llm.c</a> "
        "<code>dev/cuda</code>, then ported its kernel files one by one in an agent loop. "
        "A hidden reference the model never sees grades every port; passing ports were then "
        "run on NVIDIA GPUs against llm.c's own CUDA, and on an AMD GPU against the "
        "reference.</p>"
        '<nav class="crumb"><a href="plan.html">The plan</a><a href="#items">Every item</a>'
        '<a href="#gpt2">GPT-2</a><a href="#diff">Differential checks</a>'
        '<a href="#specs">Specs by Muse</a><a href="#j3">llm.c-sycl</a>'
        '<a href="#k1">Full training</a><a href="#j4">llama2.cu</a>'
        '<a href="#l1">bf16 trainer</a>'
        '<a href="#first-runs">The first runs</a>'
        f'<a href="{REPO}">Repository</a></nav></header>',
        _l1_section(data.get("l1")),
        '<section><h2>Results with the fixed loop</h2>' + _tier_head(True)
        + _tier_rows(current_table(items), True) + "</tbody></table></div>"
        '<p class="note"><b>Correct, 0 helpers</b>: the hidden grade. The final sources '
        "compile, match the reference within llm.c's fp32 tolerance, and no SYCLomatic "
        "(dpct) helper is left anywhere in the tree. <b>Loop success</b>: the loop's stricter "
        "rule, met inside the run on one source hash. <b>NVIDIA pass</b>: the port matched "
        "both the reference and llm.c's CUDA on a real GPU. <b>AMD pass</b>: the port matched "
        "the reference on an AMD GPU. One run per item, after the loop fixes (delete_file, "
        "ranged read_file, 32k output per turn); the first runs are below.</p></section>",
        _gpt2_section(data.get("gpt2", [])),
        _diff_section(data.get("diff", [])),
        _spec_section(data.get("specs", [])),
        _j3_section(data.get("j3")),
        _k1_section(data.get("k1")),
        _j4_section(data.get("j4")),
        '<section id="items"><h2>Every item</h2>' + _item_table(current, compare=firsts)
        + "</section>",
        '<section id="first-runs"><h2>The first runs</h2><p class="note">The same tasks and '
        "hidden checks, before the loop fixes. Most ports were already correct, but the runs "
        "spent their turns clearing SYCLomatic's output file by file and hit the toolkit's "
        "token limit.</p>" + _tier_head(False) + _tier_rows(tier_table(items), False)
        + "</tbody></table></div>" + _item_table(firsts) + "</section>",
    ]
    for batch in dict.fromkeys(i.batch for i in others):
        its = [i for i in others if i.batch == batch]
        body.append(
            f'<section><h2>{esc(data["batches"][batch]["title"])}</h2><p class="note">'
            "Another run of these items, reported beside the others. What changed is in "
            f'<a href="{REPO}/tree/main/evidence/{esc(batch)}">evidence/{esc(batch)}</a>.</p>'
            + _item_table(its, compare=firsts) + "</section>")
    spent = sum(b["spent_usd"] for b in data["batches"].values())
    gpt2 = sum(st.outcome.get("cost_usd") or 0.0 for st in data.get("gpt2", []))
    plan = data["plan"]
    plan_cost = plan.get("cost_usd") if plan else None
    body.append(
        '<section><h2>Muse spend</h2><div class="facts">'
        + "".join(f'<div><span class="eyebrow">{esc(k)}</span><span class="v">'
                  f'{money(b["spent_usd"])}</span><span class="d">{esc(b["model"])}</span></div>'
                  for k, b in data["batches"].items())
        + (f'<div><span class="eyebrow">GPT-2 stages</span><span class="v">{money(gpt2)}</span>'
           f'<span class="d">{len(data["gpt2"])} loop runs</span></div>' if gpt2 else "")
        + f'<div><span class="eyebrow">plan</span><span class="v">{money(plan_cost)}</span>'
        f'<span class="d">one grounded plan</span></div>'
        f'<div><span class="eyebrow">all of the above</span><span class="v">'
        f'{money(spent + gpt2 + (plan_cost or 0))}</span><span class="d">ports, GPT-2 and plan'
        "</span></div>"
        "</div></section>")
    return page("Muse Port Evidence", "\n".join(body))


def _check_cell(res: dict[str, Any] | None) -> str:
    """A verify result as a pill plus its worst error."""
    res = res or {}
    status = res.get("status") or res.get("verify_status")
    summary = res.get("summary") or res.get("verify_summary") or res.get("note") or ""
    m = re.search(r"worst max abs err ([0-9.e+-]+)", summary)
    bad = re.search(r"(\d+) of (\d+) element", summary)
    detail = (f"worst {m.group(1)}" if m else
              f"{int(bad.group(1)):,} of {int(bad.group(2)):,} values out of tolerance" if bad
              else summary)
    return (pill("pass" if status == "ok" else "fail", "ok" if status == "ok" else "bad")
            + f' <span class="note">{esc(detail)}</span>')


def _diff_section(diffs: list[dict[str, Any]]) -> str:
    if not diffs:
        return ""
    parts = [
        '<section id="diff"><h2>Differential checks: the original CUDA as the reference</h2>'
        '<p class="note">The hidden references above are C++ programs written from llm.c\'s '
        "CPU code. A differential check needs none: the original CUDA is built with "
        "<code>nvcc</code> on an NVIDIA GPU and its outputs are recorded, run twice to show "
        "they repeat. The recording then grades the port with the same steps as the hidden "
        "grade.</p>"]
    for d in diffs:
        rows = "".join(
            f"<tr><td><code>{esc(r['item'])}</code></td>"
            + (f"<td>{esc(short_device(r.get('device') or ''))}</td>"
               f"<td>{pill('yes', 'ok') if r.get('deterministic') else pill('no', 'warn')}</td>"
               f"<td>{_check_cell(r.get('recording_vs_cpu_reference'))}</td>"
               f"<td>{_check_cell(r.get('differential_grade'))}</td>"
               if r.get("recorded") else '<td colspan=4><span class="note">not recorded'
               "</span></td>") + "</tr>"
            for r in d["items"])
        planted = "".join(
            f"<tr><td>{esc(bug)}</td><td>{_check_cell(g.get('cpu_reference'))}</td>"
            f"<td>{_check_cell(g.get('recorded_cuda'))}</td></tr>"
            for bug, g in d.get("planted", {}).items())
        parts.append(
            f'<p class="note">Evidence: <a href="{REPO}/tree/main/evidence/{esc(d["dir"])}">'
            f'evidence/{esc(d["dir"])}</a>, ports from <code>{esc(d.get("port_run", ""))}'
            "</code>.</p><div class=\"scroll\"><table><thead><tr><th>Item</th>"
            "<th>Recorded on</th><th>Two runs identical</th><th>Recording vs CPU reference</th>"
            "<th>Muse's port vs recording</th></tr></thead><tbody>" + rows
            + "</tbody></table></div>")
        if planted:
            parts.append(
                '<h3>Planted bugs</h3><div class="scroll"><table><thead><tr><th>Bug</th>'
                "<th>vs CPU reference</th><th>vs recording</th></tr></thead><tbody>" + planted
                + '</tbody></table></div><p class="note">A bug whose error is below the '
                "tolerance passes both references: that is the limit of any tolerance-based "
                "check.</p>")
    return "".join(parts) + "</section>"


def _spec_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    proven = [r for r in rows if r["proven"]]
    strict = [r for r in proven
              if all((r.get("compare") or {}).get(k) for k in ("variants", "tol"))]
    body = "".join(
        f"<tr><td><code>{esc(r['file'])}</code></td><td>"
        + (pill("proven", "ok") + f' <span class="note">{esc(r.get("proven_by") or "cpu")}</span>'
           if r["proven"] else pill("not proven", "bad"))
        + f"</td><td>{esc(', '.join(map(str, r.get('variants') or [])))}</td><td>"
        + ((pill("yes", "ok") if all(r["compare"].get(k) for k in ("variants", "tol"))
            else pill("no", "warn")) if r["proven"] and r.get("compare") else "")
        + f"</td><td class=num>{esc(r.get('attempts', 0))}</td>"
        f"<td class=num>{money(r.get('cost_usd'))}</td>"
        f'<td><a href="{REPO}/tree/main/evidence/{esc(r["run"])}">{esc(r["run"])}</a></td></tr>'
        for r in rows)
    return (
        '<section id="specs"><h2>Hidden checks written by Muse</h2><p class="note">Every hidden '
        "check is built from a short spec per file: the CPU reference functions, the kernel "
        "variants, the inputs and sizes, the two calls and the tolerance. Muse wrote these "
        "specs from the CUDA file alone (two worked examples, not graded). A spec counts only "
        "once proven: llm.c's GPU code and CPU code must agree under it, through SYCLomatic on "
        "the CPU, or on an NVIDIA GPU where SYCLomatic fails. <b>As strict as llm.c</b>: the "
        "same tolerance and every fp32 variant, compared with the hand-written spec.</p>"
        f'<div class="facts"><div><span class="eyebrow">Proven</span><span class="v">'
        f"{len(proven)} of {len(rows)}</span></div><div><span class=\"eyebrow\">As strict as "
        f'llm.c</span><span class="v">{len(strict)} of {len(proven)}</span></div></div>'
        '<div class="scroll"><table><thead><tr><th>File</th><th>Spec</th><th>Variants</th>'
        "<th>As strict as llm.c</th><th class=num>Attempts</th><th class=num>Muse cost</th>"
        "<th>Run</th></tr></thead><tbody>" + body + "</tbody></table></div>"
        '<p class="note">Each file\'s latest run is shown; spec-llmc-fp32 reran four files with '
        "the fp32 build stated in the prompt. The cost column is that run's.</p></section>")


def _proof_rows(proof: dict[str, Any]) -> list[tuple[str, str]]:
    """A check proof as (candidate, result): the test's ``rows`` list, or the trainer's
    named entries."""
    if "rows" in proof:
        ref, atol = proof["reference_values"][1:], proof["atol"]

        def losses_off(r: dict[str, Any]) -> int:  # values[0] is the "overall okay" flag
            return sum(abs(a - b) > atol for a, b in zip(r["values"][1:], ref, strict=True))

        return [(r["candidate"], f"{r['result']}: overall okay {r['overall_okay']:g}, "
                 f"{losses_off(r)} of {len(ref)} losses out of tolerance")
                for r in proof["rows"] if not r["candidate"].startswith("Muse")]
    return [(v["what"], v["result"]) for k, v in proof.items()
            if k != "reference" and "result" in v]


def _j3_gpu_cell(prog: dict[str, Any], vendor: str, original: str = "llm.c's CUDA") -> str:
    gs = [g for g in prog["gpu"] if g["vendor"] == vendor]
    if not gs:
        return '<span class="note">not run</span>'
    g = gs[-1]
    steps = {s.get("step"): s for s in g["steps"]}
    native = steps.get("sycl_verify_vs_native")
    return (_check_cell(steps.get("sycl_verify_vs_reference") or {"status": g["status"]})
            + f'<br><span class="note">{esc(short_device(g["device"]))}</span>'
            + (f"<br>vs {esc(original)}: {_check_cell(native)}" if native else ""))


def _fresh_note(f: dict[str, Any] | None) -> str:
    if not f:
        return ""
    t = {k: f"{f[k]:.0f} s" if f.get(k) is not None else "?"
         for k in ("download_s", "make_s", "test_s")}
    return (
        '<p class="note"><b>From a fresh clone</b> (CPU device): '
        f"<code>./dev/download_starter_pack.sh</code> ({t['download_s']}), <code>make</code> "
        f"({t['make_s']}), <code>./test_gpt2fp32sycl</code> ({t['test_s']}): "
        + (pill("overall okay: 1", "ok") if f["okay"] else pill("test failed", "bad"))
        + f' (<a href="{REPO}/tree/main/evidence/j3-fresh-clone">evidence/j3-fresh-clone</a>).'
        "</p>")


def _j3_run_line(r: dict[str, Any]) -> str:
    ok = r.get("stop_reason") == "success"
    return (f"{esc(r['run'])}: " + pill("success" if ok else "no success", "ok" if ok else "warn")
            + f" {esc(r.get('iterations'))} turns, {money(r.get('cost_usd'))}")


def _j3_section(j3: dict[str, Any] | None) -> str:
    if not j3:
        return ""
    progs = j3["programs"]
    muse = sum(r.get("cost_usd") or 0.0 for p in progs for r in p["runs"])
    check = j3.get("check")
    facts = (
        (f'<div><span class="eyebrow">dev/sycl kernels</span><span class="v">{check["passed"]} '
         f'of {check["total"]}</span><span class="d">pass their hidden checks, 0 helpers'
         "</span></div>" if check else "")
        + '<div><span class="eyebrow">trainer and test</span><span class="v">'
        + f'{min(sum(g["status"] == "ok" for g in p["gpu"]) + 1 for p in progs)} targets each'
        + '</span><span class="d">pass: CPU, NVIDIA, AMD</span></div>'
        + f'<div><span class="eyebrow">Muse</span><span class="v">{money(muse)}</span>'
        f'<span class="d">{sum(len(p["runs"]) for p in progs)} loop runs</span></div>')
    rows = "".join(
        f"<tr><td>{esc(p['label'])}<br><code>{esc(p['target'])}</code></td>"
        f"<td>{_check_cell(p['runs'][-1].get('evaluation'))}</td>"
        f"<td>{_j3_gpu_cell(p, 'nvidia')}</td><td>{_j3_gpu_cell(p, 'amd')}</td>"
        f"<td>" + "<br>".join(map(_j3_run_line, p["runs"]))
        + f'</td><td><a href="{REPO}/tree/main/evidence/{esc(p["dir"])}">evidence/'
        f'{esc(p["dir"])}</a></td></tr>' for p in progs)
    proofs = "".join(
        f"<tr><td>{esc(p['label'])}</td><td>{esc(what)}</td><td>"
        + pill("pass" if res.startswith("PASS") else "fail",
               "ok" if res.startswith("PASS") else "bad")
        + f' <span class="note">{esc(res.split(": ", 1)[-1])}</span></td></tr>'
        for p in progs for what, res in _proof_rows(p["proof"]))
    return (
        '<section id="j3"><h2>llm.c-sycl: the ports put back together</h2><p class="note">'
        f'<a href="{REPO}/tree/main/llm.c-sycl">llm.c-sycl</a> is llm.c laid out as llm.c is, '
        "in SYCL, built with <code>make</code> and <code>icpx -fsycl</code>; its programs run "
        "on a CPU, an NVIDIA GPU and an AMD GPU. It holds the kernel ports in "
        "<code>dev/sycl</code>, and two "
        "programs that Muse wrote on top of its proven GPT-2 port: llm.c's fp32 trainer, with "
        "its real <code>main</code>, and llm.c's own test, which checks logits, loss and every "
        "gradient against PyTorch, then 10 training steps.</p>"
        f'<div class="facts">{facts}</div>'
        '<div class="scroll"><table><thead><tr><th>Program</th><th>CPU (hidden grade)</th>'
        "<th>NVIDIA</th><th>AMD</th><th>Muse runs</th><th>Evidence</th></tr></thead><tbody>"
        + rows + "</tbody></table></div>"
        '<p class="note"><b>Hidden grade</b>: the trainer\'s validation and training losses '
        "against llm.c's own CPU trainer; the test's <code>overall okay</code> and its 10 "
        "losses against PyTorch's; both at llm.c's tolerance, 1e-2. The GPU runs build the "
        "sources Muse last compiled, unchanged, with the toolkit's <code>gpu-test</code>; "
        "llm.c-sycl's own Makefile also builds and passes the test on both GPUs "
        f'(<a href="{REPO}/tree/main/evidence/j3-makefile-gpu">evidence/j3-makefile-gpu</a>). '
        "llm.c's CUDA does not hipify into something that builds, so on AMD there is no "
        "comparison with it.</p>"
        + _fresh_note(j3.get("fresh"))
        + "<h3>Each check, proven first</h3><div class=\"scroll\"><table><thead><tr><th>Program"
        "</th><th>Candidate</th><th>Result</th></tr></thead><tbody>" + proofs
        + "</tbody></table></div><p class=\"note\">llm.c's own CPU programs must pass and "
        "planted bugs must fail before a check grades Muse. A bug can keep every loss within "
        "1e-2 yet fail llm.c's test, so the test check also needs <code>overall okay: 1</code>."
        "</p></section>")


L1_RUN_NOTES = {
    "run1": "SYCLomatic could not parse the trainer; stopped by the toolkit's 2M-token limit "
            "before writing code",
    "run2": "the whole port written; forward pass right, training NaN at the turn limit",
    "run3": "continued run 2's files: llm.c's test passes",
    "run4": "from scratch, with a shell: stopped by the stuck rule mid-debugging",
    "run5": "from scratch, with a shell: llm.c's test passes, in one run",
    "tune": "tuning with no GPU: device memory, own attention kernels",
    "gtune": "tuning with gpu_test: profiled, measured every change on real GPUs",
}


def _x(r: float | None) -> str:
    return "" if r is None else f"{r:.2f}x"


def _dev(name: str | None) -> str:
    return esc((name or "").replace("NVIDIA ", "").replace(" Generation", "")
               .replace("GeForce ", ""))


def _l1_section(l1: dict[str, Any] | None) -> str:
    if not l1:
        return ""
    runs, sp, final, diag = l1["runs"], l1["speed"], l1["final"], l1["diag"]
    nv = [c for c in final if c["vendor"] == "nvidia" and c["ratio"]]
    amd = next((c for c in final if c["vendor"] == "amd"), None)
    lo, hi = (min(c["ratio"] for c in nv), max(c["ratio"] for c in nv)) if nv else (None, None)
    r5 = runs.get("run5", {})
    facts = (
        '<div><span class="eyebrow">Port, one run</span><span class="v">'
        f'{esc(r5.get("iterations"))} turns</span><span class="d">run 5, '
        f'{money(r5.get("cost_usd"))}: llm.c\'s test passes</span></div>'
        '<div><span class="eyebrow">NVIDIA, final</span><span class="v">'
        f'{lo:.2f}-{hi:.2f}x</span><span class="d">llm.c\'s CUDA time, same GPU (with cuBLAS '
        'interop)</span></div>'
        '<div><span class="eyebrow">Pure SYCL + oneMath</span><span class="v">'
        f'{_x((diag.get("pure") or {}).get("ratio"))}</span><span class="d">the final port, '
        "cuBLAS interop compiled out (our diagnostic)</span></div>"
        '<div><span class="eyebrow">AMD MI350X</span><span class="v">'
        f'{amd["ms"] if amd else "?"} ms</span><span class="d">per step; run 5: '
        f'{(sp.get("run5_amd") or {}).get("ms")} ms</span></div>'
        f'<div><span class="eyebrow">Muse, all of L1</span><span class="v">'
        f'{money(l1["muse_usd"])}</span><span class="d">plan and 7 runs; GPU tuning calls '
        f'{money(l1["gpu_tuning_usd"])}</span></div>')
    run_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>"
        + pill(r.get("stop_reason", "?"), "ok" if r.get("stop_reason") == "success" else "warn")
        + f"</td><td>{esc(r.get('iterations'))}</td><td>{money(r.get('cost_usd'))}</td>"
        f"<td>{esc(L1_RUN_NOTES.get(name, ''))}</td></tr>" for name, r in runs.items())

    def speed_row(label: str, s: dict[str, Any] | None) -> str:
        if not s:
            return ""
        return (f"<tr><td>{esc(label)}</td><td>{_dev(s['device'])}</td><td>{s['ms']}</td>"
                f"<td>{s.get('cuda_ms') or ''}</td><td><b>{_x(s.get('ratio'))}</b></td></tr>")
    speed_rows = (speed_row("run 5", sp.get("run5_nvidia")) + speed_row("run 5", sp.get("run5_amd"))
                  + speed_row("tuned, no GPU", sp.get("tune_nvidia"))
                  + speed_row("tuned, no GPU", sp.get("tune_amd"))
                  + "".join(speed_row("tuned with gpu_test (final)", c) for c in final)
                  + speed_row("final, cuBLAS interop compiled out (ours)", diag.get("pure")))
    call_rows = "".join(
        f"<tr><td>{i}</td><td>{esc(c['vendor'])}</td><td>"
        + (pill(c["test"], "ok" if c["test"] == "PASS" else "bad") if c["test"] else "")
        + f"</td><td>{_dev(c['device'])}</td><td>{c['ms'] or ''}</td><td>{c['cuda_ms'] or ''}"
        f"</td><td>{_x(c['ratio'])}</td><td>{'yes' if c['cublas'] else ''}</td>"
        f"<td>{'final' if c['final'] else ''}</td></tr>"
        for i, c in enumerate(l1["calls"], 1))
    diag_rows = "".join(
        f"<tr><td>{esc(label)}</td><td>{_dev(d.get('test_device'))}: "
        + pill(d.get("test") or "?", "ok" if d.get("test") == "PASS" else "bad")
        + f"</td><td>{_dev(d.get('device'))} {d.get('ms') or ''} ms {_x(d.get('ratio'))}</td></tr>"
        for label, d in (
            ("cuBLAS interop compiled out (pure SYCL + oneMath)", diag.get("pure")),
            ("fast paths forced on for llm.c's test, NVIDIA", diag.get("forced_nvidia")),
            ("fast paths forced on for llm.c's test, AMD", diag.get("forced_amd")))
        if d)
    ev = f'{REPO}/tree/main/evidence/l1-llmc-bf16'
    return (
        '<section id="l1"><h2>llm.c\'s production bf16 trainer: ported in one run, tuned on '
        "real GPUs</h2>"
        '<p class="note">llm.c\'s <code>train_gpt2.cu</code>, <code>test_gpt2.cu</code> and '
        "<code>llmc/</code>: 12 CUDA files and 4,884 lines, in bf16 mixed precision with "
        "cuBLASLt, warp reductions and stochastic rounding. The hidden check is llm.c's own "
        "test, as Muse ports it, at llm.c's bf16 tolerances: logits, all 16 gradient tensors, "
        "10 losses against PyTorch and a bit-exact checkpoint replay. A guard fails any port "
        "that loosens a tolerance. For speed, Muse got a <code>gpu_test</code> tool: it builds "
        "Muse's current code on a rented GPU and returns llm.c's test result there, llm.c's "
        "step time next to llm.c's own CUDA on the same GPU, and an nsys kernel profile. A "
        "broker outside the model's process holds the GPU keys and the budget.</p>"
        f'<div class="facts">{facts}</div>'
        "<h3>Speed: llm.c's trainer, B=4, T=1024, median ms per step over steps 6-30</h3>"
        '<div class="scroll"><table><thead><tr><th>Port</th><th>GPU</th><th>Port ms</th>'
        "<th>llm.c CUDA ms</th><th>CUDA's time ×</th></tr></thead><tbody>" + speed_rows
        + "</tbody></table></div>"
        '<p class="note"><b>Read with:</b> the NVIDIA numbers of the final port call NVIDIA\'s '
        "cuBLAS at runtime through SYCL's backend interop, for two gemm forms oneMath's cuBLAS "
        "backend lacks (batched bf16, bf16 output); pure SYCL + oneMath is the row below them. "
        "Muse uses its fast paths only for large shapes, so llm.c's small test does not run "
        "them; forced on, the test still passes (diagnostics below). Rented GPUs vary between "
        "calls, so compare ratios; each number is one run. llm.c's CUDA does not build on "
        "AMD, so AMD has no ratio.</p>"
        "<h3>Runs</h3>"
        '<div class="scroll"><table><thead><tr><th>Run</th><th>Stop</th><th>Turns</th>'
        "<th>Muse</th><th>What happened</th></tr></thead><tbody>" + run_rows
        + "</tbody></table></div>"
        "<h3>Every gpu_test call of the GPU tuning run</h3>"
        '<div class="scroll"><table><thead><tr><th>#</th><th>Vendor</th><th>llm.c\'s test'
        "</th><th>Speed GPU</th><th>Port ms</th><th>CUDA ms</th><th>×</th><th>cuBLAS interop"
        "</th><th></th></tr></thead><tbody>" + call_rows + "</tbody></table></div>"
        '<p class="note"><b>final</b>: the call built exactly the final port. On AMD the '
        "cuBLAS code compiles out. Each call keeps the code it built and the answer Muse saw."
        "</p><h3>Our diagnostics on the final port</h3>"
        '<div class="scroll"><table><thead><tr><th>Variant (a small edit by us, not Muse)'
        "</th><th>llm.c's test</th><th>Speed</th></tr></thead><tbody>" + diag_rows
        + "</tbody></table></div>"
        f'<p class="note">Evidence: <a href="{ev}">evidence/l1-llmc-bf16</a>; the GPU tuning '
        f'run: <a href="{ev}/gtune">gtune</a>.</p></section>')


def _j4_section(j4: dict[str, Any] | None) -> str:
    if not j4:
        return ""
    last = j4["runs"][-1]
    loc = j4["port_lines"]
    muse = (j4["plan_cost"] or 0.0) + sum(r.get("cost_usd") or 0.0 for r in j4["runs"])
    passes = 1 + sum(g["status"] == "ok" for g in j4["gpu"])
    facts = (
        '<div><span class="eyebrow">CUDA in the repo</span><span class="v">1 file</span>'
        '<span class="d">llama2.cu: 772 lines, 8 kernels, fp16, CUB</span></div>'
        f'<div><span class="eyebrow">Muse\'s port</span><span class="v">{loc} lines</span>'
        f'<span class="d">{esc(last.get("iterations"))} turns, one run, 0 helpers</span></div>'
        f'<div><span class="eyebrow">identical stories</span><span class="v">{passes} devices'
        '</span><span class="d">CPU, NVIDIA, AMD; byte for byte</span></div>'
        f'<div><span class="eyebrow">Muse</span><span class="v">{money(muse)}</span>'
        '<span class="d">plan and port</span></div>')
    prog = {"gpu": j4["gpu"]}
    row = (f"<tr><td><code>llama2.cpp</code><br>{_j3_run_line(last)}</td>"
           f"<td>{_check_cell(last.get('evaluation'))}</td>"
           f"<td>{_j3_gpu_cell(prog, 'nvidia', 'llama2.cu')}</td>"
           f"<td>{_j3_gpu_cell(prog, 'amd')}</td></tr>")
    proofs = "".join(
        f"<tr><td>{esc(r['candidate'])}</td><td>"
        + pill("pass" if r["result"] == "PASS" else "fail",
               "ok" if r["result"] == "PASS" else "bad")
        + ' <span class="note">'
        + esc(", ".join("identical" if c["identical"] else f"differs at byte "
                        f"{c['first_difference_byte']}" for c in r["cases"]))
        + "</span></td></tr>" for r in j4["proof"])
    fresh = ("<p class=\"note\"><b>From a fresh clone</b> of this repository, "
             "llama2-sycl's Quick start (<code>make llama2</code>, then the two prompts) on the "
             "CPU: " + (pill("both stories identical", "ok") if j4["fresh_ok"]
                        else pill("differs", "bad"))
             + f' (<a href="{REPO}/tree/main/evidence/j4-fresh-clone">evidence/j4-fresh-clone'
             "</a>).</p>") if j4["fresh"] else ""
    return (
        '<section id="j4"><h2>A second CUDA repo (one 772-line file): llama2.cu</h2>'
        '<p class="note"><a href="https://github.com/ankan-ban/llama2.cu">ankan-ban/llama2.cu'
        "</a> is Llama 2 inference in one CUDA file; the rest of the repo is llama2.c's C "
        "program and Python scripts, not CUDA. Muse Port had never run on it. Muse planned it "
        f"as {j4['plan_items']} item{'s' if j4['plan_items'] != 1 else ''}"
        + (", grounded against the scan," if j4["grounded"] else "")
        + f" for {money(j4['plan_cost'])}. The hidden check runs greedy decoding with the "
        "stories15M model on two prompts, 64 steps each, and compares the story byte for byte "
        "with llama2.cu's own output, recorded on an NVIDIA GPU. "
        f'<a href="{REPO}/tree/main/llama2-sycl">llama2-sycl</a> is llama2.cu\'s repo with '
        "the port in place of <code>llama2.cu</code>.</p>"
        f'<div class="facts">{facts}</div>'
        '<div class="scroll"><table><thead><tr><th>Port</th><th>CPU (hidden grade)</th>'
        "<th>NVIDIA</th><th>AMD</th></tr></thead><tbody>" + row + "</tbody></table></div>"
        '<p class="note">The GPU runs build the sources Muse last compiled, unchanged. On '
        "NVIDIA the port is also compared with llama2.cu itself, built with nvcc on the same "
        "GPU. llama2.cu's automatic HIP conversion did not build, so on AMD there is no such "
        f'comparison. Evidence: <a href="{REPO}/tree/main/evidence/j4-llama2">'
        "evidence/j4-llama2</a>.</p>" + fresh
        + "<h3>The check, proven first</h3><div class=\"scroll\"><table><thead><tr>"
        "<th>Candidate</th><th>Result</th></tr></thead><tbody>" + proofs
        + "</tbody></table></div><p class=\"note\">llama2.c's fp32 <code>run.c</code>, a "
        "known-good program, must produce the same stories, and planted one-line bugs must "
        "not. At 128 steps, <code>run.c</code> differs from llama2.cu's fp16 output on a "
        "near-tie token, so the check uses 64 steps; that was set before any model ran.</p>"
        "</section>")


def _k1_section(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    from .curves import chart_svg, worst

    rows = [("llm.c CUDA, second run (noise floor)", "cuda_repeat"),
            ("llm.c CUDA", "cuda_nvidia"), ("Muse SYCL", "sycl_nvidia"), ("Muse SYCL", "sycl_amd")]
    body = "".join(
        f"<tr><td>{esc(label)}</td><td>{esc(short_device(c[k]['device']))}</td>"
        f"<td class=num>{worst(c[k], c['cuda']):.4f}</td>"
        f"<td class=num>{c[k]['ms_per_step_median']:.0f} ms</td></tr>" for label, k in rows)
    # at phone width the chart keeps a readable size and scrolls inside its box
    svg = chart_svg(c).replace(
        "<svg ", '<svg style="width:100%;min-width:640px;height:auto;display:block" ', 1)
    n = len(c["cuda"]["train"])
    return (
        '<section id="k1"><h2>Full training: one epoch on NVIDIA and AMD</h2><p class="note">'
        f"GPT-2 124M trains for one epoch of tiny-shakespeare ({n} steps). llm.c's own CUDA "
        "trainer was recorded on an NVIDIA GPU; Muse's SYCL trainer from llm.c-sycl, unchanged, "
        "then ran on an NVIDIA and an AMD GPU and was compared with that recording, every val "
        "and train loss, at llm.c's tolerance (1e-2).</p>"
        f'<div class="scroll">{svg}</div>'
        '<div class="scroll"><table><thead><tr><th>Run</th><th>GPU</th>'
        "<th class=num>Worst vs recording</th><th class=num>Per step</th></tr>"
        "</thead><tbody>" + body + "</tbody></table></div>"
        '<p class="note">Times are the median of the program\'s own step timer, one run each: '
        "compare them only on the same GPU (the RTX A5000 ran both programs). Evidence: "
        f'<a href="{REPO}/tree/main/evidence/k1-curves">evidence/k1-curves</a>.</p></section>')


def _gpt2_gpu_cell(st: Gpt2Stage) -> str:
    if not st.gpu:
        return '<span class="note">CPU device only</span>'
    return "<br>".join(
        pill("pass" if g["status"] == "ok" else g["status"], "ok" if g["status"] == "ok" else "bad")
        + f" {esc(short_device(g['device']))}<br><span class=\"note\">vs PyTorch"
        + (" and llm.c's CUDA" if g["vs_native"] else "") + "</span>" for g in st.gpu)


def _gpt2_section(stages: list[Gpt2Stage]) -> str:
    if not stages:
        return ""
    rows = "".join(
        f'<tr><td><a href="gpt2/{esc(st.slug)}.html">{esc(st.label)}</a></td>'
        f'<td>{esc(st.task.get("museport", {}).get("note", ""))}</td>'
        f'<td>{pill("pass" if st.correct else "fail", "ok" if st.correct else "bad")} '
        f'<span class=mono>{esc((st.outcome.get("evaluation") or {}).get("verify_summary", ""))}'
        f'</span></td><td class=num>{esc(st.outcome.get("iterations"))}</td>'
        f'<td class=num>{money(st.outcome.get("cost_usd"))}</td>'
        f"<td>{_gpt2_gpu_cell(st)}</td></tr>" for st in stages)
    return (
        '<section id="gpt2"><h2>GPT-2: llm.c\'s training program</h2><p class="note">Muse '
        "ported llm.c's fp32 GPT-2 training file (<code>train_gpt2_fp32.cu</code>) in stages, "
        "with the real GPT-2 124M weights and llm.c's debug batch. Each stage is graded against "
        "PyTorch's numbers at llm.c's own test tolerance, and each check was first proven to "
        "pass llm.c's CPU code.</p><div class=\"scroll\"><table><thead><tr><th>Stage</th>"
        "<th>What is checked</th><th>Hidden grade of the final sources</th>"
        "<th class=num>Turns</th><th class=num>Muse cost</th><th>GPU</th></tr></thead><tbody>"
        + rows + "</tbody></table></div></section>")


def render_gpt2(st: Gpt2Stage) -> str:
    o = st.outcome
    ev = o.get("evaluation") or {}
    usage = o.get("usage") or {}
    proof = ((st.cpu_proof or {}).get("steps") or {}).get("verify_cpu_proof") or {}
    facts = [
        ("Hidden grade", pill("pass" if st.correct else "fail", "ok" if st.correct else "bad"),
         esc(f"{ev.get('verify_summary', '')}; {ev.get('helpers', '-')} helpers")),
        ("Loop", pill(o.get("stop_reason", "?"),
                      "ok" if o.get("stop_reason") == "success" else "warn"),
         esc(o.get("stop_detail", ""))),
        ("Turns and cost", f'<span class="v">{esc(o.get("iterations", "-"))} · '
         f'{money(o.get("cost_usd"))}</span>',
         esc(f"{usage.get('input_tokens', 0):,} in, {usage.get('output_tokens', 0):,} out, "
             f"{usage.get('cache_read_input_tokens', 0):,} cached") if usage else ""),
        ("Check proven on llm.c's CPU code",
         pill(proof.get("status", "?"), "ok" if proof.get("status") == "ok" else "bad"),
         esc(proof.get("summary", ""))),
        ("GPU", _gpt2_gpu_cell(st), " ".join(esc(g["summary"]) for g in st.gpu)),
        ("Tolerance", f'<span class="v">atol {esc(st.task.get("tolerance", "-"))}</span>',
         "llm.c's own test tolerance, absolute"),
    ]
    body = [
        '<nav class="crumb"><a href="../index.html">Overview</a>'
        f'<a href="../index.html#gpt2">GPT-2</a><span>{esc(st.slug)}</span></nav>',
        f'<header style="display:grid;gap:8px"><span class="eyebrow">GPT-2 · '
        f'{esc(st.task.get("museport", {}).get("source", ""))}</span><h1>{esc(st.title)}</h1>'
        f'<p class="lede">{esc(st.task.get("museport", {}).get("note", ""))}</p></header>',
        '<section class="facts">' + "".join(
            f'<div><span class="eyebrow">{esc(k)}</span><span>{v}</span>'
            f'<span class="d">{d}</span></div>' for k, v, d in facts) + "</section>",
    ]
    if st.task.get("goal"):
        body.append(f'<section><h2>The goal Muse was given</h2><p>{esc(st.task["goal"])}</p>'
                    '<p class="note">The reference (PyTorch\'s numbers) stays hidden from the '
                    "model.</p></section>")
    harness = st.dir / "task" / st.task.get("cuda_path", "-")
    ports = sorted(p for p in (st.dir / "run1" / "port").glob("*") if p.is_file())
    if harness.exists() and ports:
        body.append(
            "<section><h2>CUDA harness and Muse's SYCL port</h2><p class=\"note\">The harness "
            "includes llm.c's <code>train_gpt2_fp32.cu</code> unchanged and adds a contract "
            "<code>main</code>; Muse read llm.c's file and wrote its own SYCL program.</p>"
            f"<div class=grid2><details><summary>CUDA: {esc(harness.name)} "
            f"({len(harness.read_text().splitlines())} lines)</summary>"
            f"<pre>{esc(harness.read_text())}</pre></details>"
            + "".join(f"<details><summary>SYCL: {esc(p.name)} "
                      f"({len(p.read_text().splitlines())} lines)</summary>"
                      f"<pre>{esc(p.read_text())}</pre></details>" for p in ports)
            + "</div></section>")
    if st.gpu:
        out = ["<section><h2>On a GPU</h2>"]
        for g in st.gpu:
            steps = "".join(
                f"<tr><td><code>{esc(x.get('step'))}</code></td><td>"
                f"{pill(x.get('status', '?'), 'ok' if x.get('status') == 'ok' else 'bad')}</td>"
                f"<td class=mono>{esc(_strip_tmp(x.get('summary', '')))}</td></tr>"
                for x in g["steps"])
            out.append(f"<h3>{esc(g['device'])} ({esc(g['vendor'])})</h3>"
                       f'<div class="scroll"><table><tbody>{steps}</tbody></table></div>')
        out.append("</section>")
        body.append("".join(out))
    trace = st.dir / "run1" / "trace.jsonl"
    if trace.exists():
        body.append(_replay(trace))
    body.append(f'<p class="note">Source: <a href="{REPO}/tree/main/evidence/{st.slug}">'
                f"evidence/{esc(st.slug)}</a>.</p>")
    return page(f"{st.slug} · Muse Port", "\n".join(body), root="../")


def _item_table(items: list[Item], compare: list[Item] | None = None) -> str:
    first = {i.id: i for i in compare or []}
    head = ("<tr><th>Item</th><th>Hidden grade</th><th>Variants</th><th>Loop</th>"
            "<th class=num>Turns</th><th class=num>Cost</th><th>GPU</th>"
            + ("<th>First run</th>" if compare is not None else "") + "</tr>")
    rows = []
    for it in items:
        extra = ""
        if compare is not None:
            f = first.get(it.id)
            extra = (f'<td><a href="items/{f.slug}.html">{grade_pill(f)}</a></td>' if f
                     else "<td>-</td>")
        rows.append(
            f'<tr><td><a href="items/{it.slug}.html"><code>{esc(it.id)}</code></a><br>'
            f'<span class="note">{esc(it.row.get("file", ""))}</span></td>'
            f"<td>{grade_pill(it)}</td><td>{variant_pill(it)}</td><td>{loop_pill(it)}</td>"
            f'<td class=num>{esc(it.row.get("iterations", "-"))}</td>'
            f'<td class=num>{money(_item_spend(it.row))}</td><td>{gpu_pill(it)}</td>{extra}</tr>')
    return (f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}'
            "</tbody></table></div>")


def render_item(it: Item, evidence: Path) -> str:
    ev = it.row.get("evaluation") or {}
    vc = it.row.get("variant_check") or {}
    meta = it.task.get("museport", {})
    usage = it.row.get("usage") or {}
    facts = [
        ("Hidden grade", grade_pill(it), esc(ev.get("verify_summary", ""))),
        ("Variant check", variant_pill(it),
         esc(f"{vc.get('kernels', '-')} kernels, {vc.get('distinct_kernel_bodies', '-')} "
             f"distinct bodies, {vc.get('onemath_calls', 0)} oneMath calls") if vc else ""),
        ("Loop", loop_pill(it), esc(it.row.get("detail", it.row.get("error", "")))),
        ("Turns and cost", f'<span class="v">{esc(it.row.get("iterations", "-"))} · '
         f'{money(it.row.get("cost_usd"))}</span>',
         esc(f"{usage.get('input_tokens', 0):,} in, {usage.get('output_tokens', 0):,} out, "
             f"{usage.get('cache_read_input_tokens', 0):,} cached") if usage else ""),
        ("GPU", gpu_pill(it), " ".join(esc(g["summary"]) for g in it.gpu)),
        ("Tolerance", f'<span class="v">atol {esc(it.task.get("tolerance", "-"))}</span>',
         "plus rtol FLT_EPSILON·|ref|, llm.c's fp32 rule"),
    ]
    body = [
        f'<nav class="crumb"><a href="../index.html">Overview</a><a href="../plan.html">Plan</a>'
        f'<span>{esc(it.batch)}</span></nav>',
        f'<header style="display:grid;gap:8px"><span class="eyebrow">Tier {it.tier} · '
        f'{esc(meta.get("source", it.row.get("file", "")))}</span><h1>{esc(it.id)}</h1>'
        + (f'<p class="lede">{esc(meta["note"])}</p>' if meta.get("note") else "") + "</header>",
        '<section class="facts">' + "".join(
            f'<div><span class="eyebrow">{esc(k)}</span><span>{v}</span>'
            f'<span class="d">{d}</span></div>' for k, v, d in facts) + "</section>",
    ]
    for e in it.row.get("earlier_runs", []):
        body.append(f'<p class="note">An earlier run of this item: {esc(e.get("status"))} in '
                    f'{esc(e.get("iterations"))} turns, {money(e.get("cost_usd"))}'
                    f'{" (" + esc(e["error"]) + ")" if e.get("error") else ""}.</p>')
    if it.task.get("goal"):
        body.append(f'<section><h2>The goal Muse was given</h2><p>{esc(it.task["goal"])}</p>'
                    f'<p class="note">Variants {esc(meta.get("variants", []))}; sizes '
                    f'{esc(meta.get("sizes", {}))}. The reference and its inputs stay hidden '
                    "from the model.</p></section>")
    body.append(_code_section(it))
    if it.gpu:
        body.append(_gpu_section(it))
    trace = it.dir / "trace.jsonl"
    if trace.exists():
        body.append(_replay(trace))
    links = [f'<a href="{REPO}/tree/main/evidence/{it.batch}/{it.id}">evidence files</a>']
    body.append(f'<p class="note">Source: {" · ".join(links)}.</p>')
    return page(f"{it.id} · Muse Port", "\n".join(body), root="../")


def _port_files(it: Item) -> list[Path]:
    d = it.dir / "port"
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and "migrated" not in p.parts)


def _code_section(it: Item) -> str:
    harness = it.dir / "task" / it.task.get("cuda_path", "-")
    ports = _port_files(it)
    if not harness.exists() or not ports:
        return ""
    cuda = harness.read_text()
    sycl = ports[0].read_text()
    diff = list(difflib.unified_diff(cuda.splitlines(), sycl.splitlines(), harness.name,
                                     ports[0].name, lineterm="", n=2))
    lines = []
    for ln in diff:
        cls = ("h" if ln.startswith("@@") else "a" if ln.startswith("+") else
               "d" if ln.startswith("-") else "")
        lines.append(f'<span class="{cls}">{esc(ln)}</span>' if cls else esc(ln))
    files = "".join(f"<details><summary>{esc(p.relative_to(it.dir / 'port'))} "
                    f"({len(p.read_text().splitlines())} lines)</summary>"
                    f"<pre>{esc(p.read_text())}</pre></details>" for p in ports)
    leftovers = sorted(p.name for p in (it.dir / "port").rglob("*")
                       if p.is_file() and "migrated" in p.parts)
    left = (f'<p class="note">Also in the tree at the end: migrated/ '
            f'{esc(", ".join(leftovers))} (SYCLomatic output, stubbed or left by Muse).</p>'
            if leftovers else "")
    return (
        "<section><h2>CUDA harness and Muse's SYCL port</h2><div class=grid2>"
        f"<details><summary>CUDA: {esc(harness.name)} ({len(cuda.splitlines())} lines)</summary>"
        f"<pre>{esc(cuda)}</pre></details>{files}</div>{left}"
        f'<details><summary>Diff, CUDA to SYCL ({len(diff)} lines)</summary>'
        f'<pre class="diff">{chr(10).join(lines)}</pre></details></section>')


def _gpu_section(it: Item) -> str:
    out = ["<section><h2>On a GPU</h2>"]
    for g in it.gpu:
        steps = "".join(f"<tr><td><code>{esc(s.get('step'))}</code></td><td>"
                        f"{pill(s.get('status', '?'), 'ok' if s.get('status') == 'ok' else 'bad')}"
                        f"</td><td class=mono>{esc(_strip_tmp(s.get('summary', '')))}</td></tr>"
                        for s in g["steps"])
        out.append(f"<h3>{esc(g['device'])} ({esc(g['vendor'])})</h3>"
                   f'<div class="scroll"><table><tbody>{steps}</tbody></table></div>')
    out.append("</section>")
    return "".join(out)


def _strip_tmp(s: str) -> str:
    return re.sub(r"/tmp/tmp\.\w+/", "", s)


def _replay(trace: Path) -> str:
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    turns: dict[int, list[dict[str, Any]]] = {}
    stop = None
    for e in events:
        if e.get("type") == "stop":
            stop = e
        elif "iteration" in e and e.get("type") in ("model_response", "tool_result",
                                                    "best_snapshot"):
            turns.setdefault(int(e["iteration"]), []).append(e)
    out = ['<section><h2>Replay, turn by turn</h2><p class="note">From '
           "<code>trace.jsonl</code>: what Muse wrote, each tool it called, and what the tool "
           "returned.</p><div class=turns>"]
    for n in sorted(turns):
        parts = [f'<div class="turn"><span class="n">TURN {n}</span>']
        for e in turns[n]:
            if e["type"] == "model_response":
                for c in e.get("content", []):
                    if c.get("type") == "text" and c.get("text", "").strip():
                        parts.append(f'<div class="say">{esc(c["text"].strip())}</div>')
            elif e["type"] == "tool_result":
                r = e.get("result", {})
                st = r.get("status", "?")
                parts.append(
                    f'<div class="call"><div class="head"><code>{esc(e["name"])}</code>'
                    f'{_args(e.get("input", {}))}{pill(st, "ok" if st == "ok" else "bad")}</div>'
                    f'<div class="sum">{esc(_strip_run(r.get("summary", "")))}</div>'
                    + _content(e) + "</div>")
            elif e["type"] == "best_snapshot":
                parts.append(f'<div class="sum">Best so far: {esc(e.get("verify_summary"))}, '
                             f'{esc(e.get("helpers"))} helpers.</div>')
        parts.append("</div>")
        out.append("".join(parts))
    out.append("</div>")
    if stop:
        out.append(f'<p><b>Stop:</b> {esc(stop.get("stop_reason"))}: '
                   f'{esc(stop.get("stop_detail"))}</p>')
    out.append("</section>")
    return "".join(out)


def _args(inp: dict[str, Any]) -> str:
    shown = {k: v for k, v in inp.items() if k != "content"}
    return f'<span class="sum mono">{esc(json.dumps(shown))}</span>' if shown else ""


def _content(e: dict[str, Any]) -> str:
    text = (e.get("input") or {}).get("content")
    if not isinstance(text, str):
        return ""
    return (f"<details><summary>Written: {len(text.splitlines())} lines</summary>"
            f"<pre>{esc(text)}</pre></details>")


def _strip_run(s: str) -> str:
    return re.sub(r"/\S*/runs/\S*?/work/", "", s)


def render_plan(plan: dict[str, Any]) -> str:
    g = plan["grounding"]
    p = plan["plan"]
    checks = [(k.replace("_", " "), len(v)) for k, v in g.items() if isinstance(v, list)]
    usage = plan.get("usage") or {}
    rows = "".join(
        f'<tr><td><code>{esc(i["id"])}</code></td><td class=num>{esc(i.get("tier"))}</td>'
        f'<td>{esc(", ".join(i.get("files", [])))}</td><td>{esc(i.get("approach", ""))}'
        + "".join(f'<br><span class="note">{esc(m.get("from"))} → {esc(m.get("to"))}: '
                  f'{esc(m.get("notes", ""))}</span>' for m in i.get("library_mappings", []))
        + f'</td><td>{esc(", ".join(i.get("depends_on", [])))}</td></tr>'
        for i in p.get("items", []))
    oos = "".join(f"<li>{esc(o if isinstance(o, str) else json.dumps(o))}</li>"
                  for o in p.get("out_of_scope", []))
    body = [
        '<nav class="crumb"><a href="index.html">Overview</a></nav>',
        '<header style="display:grid;gap:8px"><span class="eyebrow">museport plan · '
        f'{esc(plan.get("model"))}</span><h1>The porting plan</h1>'
        f'<p class="lede">{esc(p.get("summary", ""))}</p></header>',
        '<section class="facts">'
        f'<div><span class="eyebrow">Grounded</span><span>'
        f'{pill("yes" if g["grounded"] else "no", "ok" if g["grounded"] else "bad")}</span>'
        f'<span class="d">{g["issues"]} issues found by the checks</span></div>'
        f'<div><span class="eyebrow">Files placed</span><span class="v">'
        f'{g["coverage"]["covered"]}/{g["coverage"]["files"]}</span>'
        f'<span class="d">every CUDA file planned or left out on purpose</span></div>'
        f'<div><span class="eyebrow">Items</span><span class="v">{g["items"]}</span>'
        f'<span class="d">ordered by dependency</span></div>'
        f'<div><span class="eyebrow">Cost</span><span class="v">{money(plan.get("cost_usd"))}'
        f'</span><span class="d">{usage.get("input_tokens", 0):,} in, '
        f'{usage.get("output_tokens", 0):,} out</span></div></section>',
        '<section><h2>Grounding checks</h2><p class="note">The plan is checked against '
        "<code>museport scan</code>, which reads the repository without a model. Each row "
        "counts what the check found wrong.</p><div class=scroll><table><tbody>"
        + "".join(f"<tr><td>{esc(k)}</td><td class=num>{n}</td></tr>" for k, n in checks)
        + "</tbody></table></div></section>",
        "<section><h2>Items</h2><div class=scroll><table><thead><tr><th>Id</th>"
        "<th class=num>Tier</th><th>Files</th><th>Approach</th><th>Depends on</th></tr>"
        f"</thead><tbody>{rows}</tbody></table></div></section>",
    ]
    if oos:
        body.append(f"<section><h2>Left out on purpose</h2><ul>{oos}</ul></section>")
    return page("Muse Port Plan", "\n".join(body))


def build(evidence: Path, out: Path) -> dict[str, Any]:
    """Write the site under ``out``; returns what was written."""
    data = load(evidence)
    (out / "items").mkdir(parents=True, exist_ok=True)
    (out / "style.css").write_text(CSS.lstrip())
    (out / "index.html").write_text(render_index(data))
    pages = ["index.html"]
    if data["plan"]:
        (out / "plan.html").write_text(render_plan(data["plan"]))
        pages.append("plan.html")
    for it in data["items"]:
        name = f"items/{it.slug}.html"
        (out / name).write_text(render_item(it, evidence))
        pages.append(name)
    if data["gpt2"]:
        (out / "gpt2").mkdir(exist_ok=True)
    for st in data["gpt2"]:
        name = f"gpt2/{st.slug}.html"
        (out / name).write_text(render_gpt2(st))
        pages.append(name)
    return {"pages": pages, "tiers": tier_table(data["items"])}
