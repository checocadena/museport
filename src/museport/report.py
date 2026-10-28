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
        for row in data["items"]:
            task_file = bdir / row["item"] / "task" / "task.json"
            task = json.loads(task_file.read_text()) if task_file.exists() else {}
            items.append(Item(bdir.name, row, task, bdir / row["item"],
                              gpu.get(task.get("name", ""), [])))
    plan_file = evidence / PLAN_DIR / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else None
    return {"items": items, "batches": batches, "plan": plan, "gpt2": load_gpt2(evidence)}


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


def _gpu_results(bdir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    # gpu/<vendor>/<device>/<task>/ (the NVIDIA runs) or gpu/<device>/<task>/ (the AMD runs)
    found = [*bdir.glob("gpu/*/*/*/summary.json"), *bdir.glob("gpu/*/*/summary.json")]
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


def gpu_pill(it: Item) -> str:
    if not it.gpu:
        if any("onemath" in f.lower() for f in it.task.get("base_flags", [])):
            return pill("CPU only (oneMath)", "none")
        return pill("not run", "none")
    pills = []
    for g in it.gpu:
        name = g["device"].replace("NVIDIA ", "").replace(" Generation", "")
        name = name.replace("AMD Instinct ", "").split(" gfx")[0]
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
        '<a href="#gpt2">GPT-2</a><a href="#first-runs">The first runs</a>'
        f'<a href="{REPO}">Repository</a></nav></header>',
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


def _gpt2_gpu_cell(st: Gpt2Stage) -> str:
    if not st.gpu:
        return '<span class="note">CPU device only</span>'
    return "<br>".join(
        pill("pass" if g["status"] == "ok" else g["status"], "ok" if g["status"] == "ok" else "bad")
        + f" {esc(g['device'].replace('NVIDIA ', ''))}<br><span class=\"note\">vs PyTorch"
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
