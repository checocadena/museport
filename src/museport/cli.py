"""Muse Port: Meta's Muse models plan and port CUDA projects to SYCL, and every step is checked.

Commands: scan (CUDA inventory, no model), plan (Muse writes a grounded porting plan), loop (the
toolkit's agent loop on one task, driven by Muse), port (Muse ports a plan's items against
hidden references), gpu-tasks (remote-test tasks for the ports that pass), report (a static
site built from the committed evidence), gpt2-task (the GPT-2 training port's stage-1
task), record-tasks and diff-task (differential checks: the original CUDA's recorded output as
the hidden reference), spec (Muse writes a file's harness spec, which is then
proven)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .scan import render_markdown, scan

DEFAULT_MODEL = "muse-spark-1.3"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="museport", description=__doc__)
    p.add_argument("--version", action="version", version=f"museport {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("scan", help="inventory a CUDA repository (no model, no compiler)")
    sp.add_argument("repo", help="path to the repository")
    sp.add_argument("--json", action="store_true", help="print JSON instead of Markdown")

    pp = sub.add_parser("plan", help="Muse writes a porting plan; it is checked against the scan")
    pp.add_argument("repo", help="path to the repository (or a subdirectory of it)")
    pp.add_argument("--out", default="museport-out", help="output directory")
    pp.add_argument("--model", default=DEFAULT_MODEL)
    pp.add_argument("--provider", default="meta", choices=["meta", "baseten"])
    pp.add_argument("--max-cost", type=float, default=1.0,
                    help="refuse if the worst-case cost exceeds this many USD (default 1.0)")
    pp.add_argument("--max-output", type=int, default=32_000,
                    help="max output tokens, reasoning included (default 32000)")
    pp.add_argument("--dry-run", action="store_true",
                    help="no model: a scripted server answers with the mechanical baseline plan")

    lp = sub.add_parser("loop", help="the toolkit's agent loop on one task, driven by Muse "
                                     "(streamed, with a USD cap)")
    lp.add_argument("task", help="toolkit task file (task.json)")
    lp.add_argument("--model", default=DEFAULT_MODEL)
    lp.add_argument("--provider", default="meta", choices=["meta", "baseten"])
    lp.add_argument("--max-iterations", type=int, default=8)
    lp.add_argument("--max-cost", type=float, default=1.0,
                    help="stop once the run's cost exceeds this many USD (default 1.0); "
                         "checked between turns")
    lp.add_argument("--runs-dir", default="runs")
    lp.add_argument("--baseline", action="store_true",
                    help="ablation: the toolkit's arm without feedback (no verify, scan_helpers "
                         "or benchmark; the run ends on finish; graded afterwards)")

    po = sub.add_parser("port", help="Muse ports each plan item of a tier with the toolkit's "
                                     "loop; a hidden reference checks every result")
    po.add_argument("plan", help="plan.json written by `museport plan`")
    po.add_argument("--cuda-dir", required=True, help="the directory the plan was made for")
    po.add_argument("--out", default="museport-port", help="tasks, reports and ports")
    po.add_argument("--items", help="comma-separated plan item ids, of any tier (default: all "
                                    "of --tier)")
    po.add_argument("--tier", type=int, default=1, help="plan tier to port (default 1)")
    po.add_argument("--model", default=DEFAULT_MODEL)
    po.add_argument("--max-iterations", type=int, default=12)
    po.add_argument("--max-cost", type=float, default=1.0, help="USD cap per item (default 1)")
    po.add_argument("--total-cap", type=float, default=20.0,
                    help="USD cap for the whole batch (default 20)")
    po.add_argument("--runs-dir", default="runs/port")
    po.add_argument("--tasks-only", action="store_true",
                    help="write the tasks (and run --check); no model")
    po.add_argument("--check", action="store_true",
                    help="prove each task first: SYCLomatic's raw output must pass the hidden "
                         "reference (needs c2s and icpx)")

    po.add_argument("--max-output", type=int, default=32_000,
                    help="max output tokens per turn, reasoning included (default 32000)")
    po.add_argument("--max-total-tokens", type=int, default=2_000_000,
                    help="the toolkit's per-run token limit (default 2000000, so the USD caps "
                         "decide; the toolkit's own default is 600000)")
    po.add_argument("--strict-goal", action="store_true",
                    help="tell Muse to clear SYCLomatic's output early and to keep each "
                         "variant's own algorithm (for reruns)")
    po.add_argument("--baseline", action="store_true",
                    help="ablation: no verify, scan_helpers or benchmark; ends on finish")
    po.add_argument("--no-migrate", action="store_true",
                    help="ablation: run without SYCLomatic (no migrate tool; Muse writes the "
                         "port from the CUDA harness)")

    gp = sub.add_parser("gpu-tasks", help="remote-test tasks for the ports that pass the hidden "
                                          "grade (run them with `oneapi-agent remote-test`)")
    gp.add_argument("port_dir", help="the --out directory of `museport port`")
    gp.add_argument("--out", help="where to write the tasks (default: <port_dir>/gpu-tasks)")
    g2 = sub.add_parser("gpt2-task", help="write a GPT-2 training port stage's task (forward or "
                                          "backward pass) and prove its hidden check on the CPU")
    g2.add_argument("--llmc", required=True, help="the llm.c checkout")
    g2.add_argument("--data", help="where gpt2_124M.bin and gpt2_124M_debug_state.bin are "
                                   "(default: --llmc)")
    g2.add_argument("--out", default="gpt2-task", help="task directory to write")
    g2.add_argument("--stage", choices=["forward", "backward", "train"], default="forward")
    g2.add_argument("--check", action="store_true",
                    help="prove the check: llm.c's CPU train_gpt2.c must pass it (no model)")

    g3 = sub.add_parser("gpt2-gpu-task", help="a remote-test task for a passing GPT-2 stage "
                                              "(data fetched and oneMath built on the GPU)")
    g3.add_argument("stage_dir", help="e.g. evidence/gpt2-stage3")
    g3.add_argument("--out", required=True, help="task directory to write")
    g3.add_argument("--model", default=DEFAULT_MODEL)
    g3.add_argument("--vendor", choices=["nvidia", "amd"], default="nvidia",
                    help="oneMath backend to build on the machine: cuBLAS or rocBLAS")

    rc = sub.add_parser("record-tasks", help="remote-test tasks that record the original CUDA's "
                                             "outputs on an NVIDIA GPU (no port involved)")
    rc.add_argument("port_dir", help="the --out directory of `museport port`")
    rc.add_argument("--out", required=True, help="where to write the tasks")
    dt = sub.add_parser("diff-task", help="a task whose hidden reference is the original CUDA's "
                                          "recorded output (from a record-tasks run)")
    dt.add_argument("task_dir", help="the item's task directory")
    dt.add_argument("recorded", help="the remote-test result directory of its recording")
    dt.add_argument("--out", required=True, help="task directory to write")

    dc = sub.add_parser("diff-check", help="grade a port run's items against their recorded "
                                           "CUDA outputs and check each recording against the "
                                           "CPU reference (no model, no GPU)")
    dc.add_argument("port_dir", help="the --out directory of `museport port`")
    dc.add_argument("recordings", help="the --into directory of the recording runs")
    dc.add_argument("--out", required=True, help="evidence directory to write")
    dc.add_argument("--tasks", required=True, help="where to write the differential tasks")

    sp = sub.add_parser("spec", help="Muse writes each CUDA file's harness spec; each is checked "
                                     "and proven (SYCLomatic + icpx vs the CPU reference)")
    sp.add_argument("--cuda-dir", required=True, help="llm.c's dev/cuda directory")
    sp.add_argument("files", nargs="*", help="files to spec (default: every file museport has a "
                                             "hand-written spec for, except the examples)")
    sp.add_argument("--out", default="evidence/spec-llmc")
    sp.add_argument("--model", default=DEFAULT_MODEL)
    sp.add_argument("--total-cap", type=float, default=2.0, help="USD cap for the whole run")
    sp.add_argument("--file-cap", type=float, default=0.4, help="USD cap per file")
    sp.add_argument("--gpu-proof", action="store_true",
                    help="when SYCLomatic's output fails the CPU proof, prove the spec with "
                         "llm.c's CUDA on a rented NVIDIA GPU (Runpod, about $0.003 each)")

    rv = sub.add_parser("review", help="Muse reviews each finished port from the code alone "
                                       "(no test results); verdicts compared with the hidden "
                                       "grade and the variant check")
    rv.add_argument("evidence", nargs="?", default="evidence")
    rv.add_argument("--out", default="evidence/review-muse")
    rv.add_argument("--model", default=DEFAULT_MODEL)
    rv.add_argument("--total-cap", type=float, default=1.5, help="USD cap for all reviews")

    rp = sub.add_parser("report", help="a static site built only from the committed evidence "
                                       "(overview, plan, one page per item with a replay)")
    rp.add_argument("evidence", nargs="?", default="evidence", help="the evidence directory")
    rp.add_argument("--out", default="site", help="where to write the site (default: site)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "scan":
        try:
            report = scan(args.repo)
        except FileNotFoundError as e:
            print(f"museport: {e}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2) if args.json else render_markdown(report), end="")
        return 0
    if args.cmd == "plan":
        return _plan(args)
    if args.cmd == "loop":
        return _loop(args)
    if args.cmd == "port":
        return _port(args)
    if args.cmd == "gpu-tasks":
        return _gpu_tasks(args)
    if args.cmd == "report":
        return _report(args)
    if args.cmd == "gpt2-task":
        return _gpt2_task(args)
    if args.cmd == "review":
        from pathlib import Path

        from .review import run_reviews

        summary = run_reviews(Path(args.evidence), Path(args.out), model=args.model,
                              total_cap_usd=args.total_cap,
                              log=lambda m: print(m, file=sys.stderr))
        print(f"[review] spent ${summary['spent_usd']:.4f}", file=sys.stderr)
        return 0
    if args.cmd in ("record-tasks", "diff-task", "diff-check"):
        return _record(args)
    if args.cmd == "spec":
        from pathlib import Path

        from .llmc import SPECS
        from .spec import EXAMPLES, gpu_proof, run_specs

        files = args.files or [f for f in SPECS if f not in EXAMPLES]
        summary = run_specs(files, Path(args.cuda_dir), Path(args.out), model=args.model,
                            total_cap_usd=args.total_cap, file_cap_usd=args.file_cap,
                            prove_on_gpu=gpu_proof if args.gpu_proof else None,
                            log=lambda m: print(m, file=sys.stderr))
        proven = sum(r["proven"] for r in summary["files"] if r["file"] in files)
        print(f"[spec] {proven}/{len(files)} proven, spent ${summary['spent_usd']:.4f}",
              file=sys.stderr)
        return 0
    if args.cmd == "gpt2-gpu-task":
        from pathlib import Path

        from .gpt2 import make_gpu_task

        print(make_gpu_task(Path(args.stage_dir), Path(args.out), args.model, args.vendor))
        return 0
    return 2


def _gpt2_task(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .gpt2 import check_task, make_task

    llmc = Path(args.llmc)
    try:
        task = make_task(llmc, Path(args.data) if args.data else llmc, Path(args.out),
                         args.stage)
    except FileNotFoundError as e:
        print(f"museport: {e}", file=sys.stderr)
        return 2
    print(f"[gpt2-task] wrote {task}", file=sys.stderr)
    if not args.check:
        return 0
    import tempfile

    result = check_task(task, llmc, Path(tempfile.mkdtemp(prefix="museport-gpt2-check-")))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .report import build

    result = build(Path(args.evidence), Path(args.out))
    print(f"[report] wrote {len(result['pages'])} page(s) to {args.out}/", file=sys.stderr)
    return 0


def _record(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .record import diff_check, differential_task, recording_task

    if args.cmd == "diff-check":
        s = diff_check(Path(args.port_dir), Path(args.recordings), Path(args.out),
                       Path(args.tasks))
        bad = [r["item"] for r in s["items"] if r["recorded"]
               and r["differential_grade"].get("verify_status") != "ok"]
        print(f"[diff-check] {len(s['items'])} item(s); not passing: {bad or 'none'}",
              file=sys.stderr)
        return 0
    if args.cmd == "diff-task":
        print(differential_task(Path(args.task_dir), Path(args.recorded), Path(args.out)))
        return 0
    root = Path(args.port_dir)
    for item in sorted(p.parent.parent for p in root.glob("*/task/task.json")):
        path = recording_task(item / "task", Path(args.out) / item.name)
        print(f"[record-tasks] {item.name}: {path}", file=sys.stderr)
    return 0


def _gpu_tasks(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .port import make_gpu_task

    root = Path(args.port_dir)
    out = Path(args.out) if args.out else root / "gpu-tasks"
    summary = json.loads((root / "summary.json").read_text())
    for r in summary["items"]:
        ev = r.get("evaluation") or {}
        if ev.get("verify_status") != "ok" or ev.get("helpers") != 0:
            print(f"[gpu-tasks] {r['item']}: skipped (hidden grade not passed)", file=sys.stderr)
            continue
        task = json.loads((root / r["item"] / "task" / "task.json").read_text())
        if any("onemath" in f.lower() for f in task.get("base_flags", [])):
            print(f"[gpu-tasks] {r['item']}: skipped (the port calls oneMath, which the GPU "
                  "hosts do not have)", file=sys.stderr)
            continue
        path = make_gpu_task(root / r["item"], out / r["item"], summary["model"])
        print(f"[gpu-tasks] {r['item']}: {path}", file=sys.stderr)
    return 0


def _port(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .port import run_items

    data = json.loads(Path(args.plan).read_text())
    summary = run_items(data.get("plan", data), Path(args.cuda_dir), Path(args.out),
                        model=args.model,
                        items=args.items.split(",") if args.items else None,
                        max_iterations=args.max_iterations, max_cost_usd=args.max_cost,
                        total_cap_usd=args.total_cap, runs_dir=Path(args.runs_dir),
                        tasks_only=args.tasks_only, check=args.check, tier=args.tier,
                        strict_goal=args.strict_goal, max_output_tokens=args.max_output,
                        max_total_tokens=args.max_total_tokens, no_migrate=args.no_migrate,
                        baseline=args.baseline,
                        log=lambda m: print(m, file=sys.stderr))
    print(json.dumps(summary, indent=2))
    bad = {"harness_check_failed", "error"}
    return 1 if any(r["status"] in bad for r in summary["items"]) else 0


def _loop(args: argparse.Namespace) -> int:
    from .loop import run, turn_overshoot_usd

    print(f"[loop] {args.model}: up to {args.max_iterations} turns; stops once cost exceeds "
          f"${args.max_cost:.2f} (one more turn can add about "
          f"${turn_overshoot_usd(args.model):.2f})", file=sys.stderr)
    try:
        outcome = run(args.task, model=args.model, provider=args.provider,
                      max_iterations=args.max_iterations, max_cost_usd=args.max_cost,
                      runs_dir=args.runs_dir, baseline=args.baseline)
    except RuntimeError as e:
        print(f"museport: {e}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["stop_reason"] in ("success", "model_finished") else 1


def _plan(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .plan import ScriptedPlanTransport, make_plan, render_plan_markdown

    try:
        report = scan(args.repo)
    except FileNotFoundError as e:
        print(f"museport: {e}", file=sys.stderr)
        return 2
    transport = ScriptedPlanTransport(report) if args.dry_run else None
    model = f"scripted-dry-run ({args.model})" if args.dry_run else args.model
    try:
        result = make_plan(report, model=args.model, provider=args.provider,
                           max_output_tokens=args.max_output, max_cost_usd=args.max_cost,
                           transport=transport)
    except RuntimeError as e:
        print(f"museport: {e}", file=sys.stderr)
        return 2
    result.model = model  # a dry run never passes for a real model's plan
    if args.dry_run:
        result.cost_usd = 0.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scan.json").write_text(json.dumps(report, indent=2))
    (out / "plan.json").write_text(json.dumps({
        "model": result.model, "provider": args.provider,
        "usage": result.usage if result.usage_reported else None,
        "cost_usd": result.cost_usd, "plan": result.plan, "grounding": result.grounding},
        indent=2))
    (out / "plan.md").write_text(render_plan_markdown(result))
    g = result.grounding
    print(f"[plan] {result.model}: {g['items']} item(s), grounded={g['grounded']}, "
          f"{g['coverage']['covered']}/{g['coverage']['files']} files placed, "
          f"{g['issues']} issue(s); wrote {out}/plan.md")
    return 0 if g["grounded"] else 1


if __name__ == "__main__":
    sys.exit(main())
