"""E1/C6: Tier 1, three runs per item in each arm, from the committed batch summaries.

- **Feedback:** the agent loop with verify, scan_helpers and benchmark (`museport port`).
- **Baseline:** `museport port --baseline`, with no checks; the run ends when Muse calls finish.

Every final port gets the same hidden grade. "Correct" means it passes the hidden check with 0
dpct helpers. "Copied variant" is the variant check: fewer distinct kernel bodies than the file
has variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

ARMS = {"feedback": ("port-llmc-tier1-run2", "port-llmc-tier1-run3", "port-llmc-tier1-run4"),
        "baseline": ("port-llmc-tier1-baseline", "port-llmc-tier1-baseline-2",
                     "port-llmc-tier1-baseline-3")}


def _correct(row: dict[str, Any]) -> bool:
    ev = row.get("evaluation") or {}
    return ev.get("verify_status") == "ok" and ev.get("helpers") == 0


def _run(evidence: Path, batch: str) -> dict[str, Any]:
    s = json.loads((evidence / batch / "summary.json").read_text())
    rows = s["items"]
    return {"batch": batch, "spent_usd": s["spent_usd"], "items": len(rows),
            "correct": sum(map(_correct, rows)),
            "loop_success": sum(r["status"] == "success" for r in rows),
            "copied_variant": sorted(r["item"] for r in rows
                                     if (r.get("variant_check") or {}).get("duplicated_variants")),
            "mean_turns": round(mean(r["iterations"] for r in rows), 1),
            "retried_after_api_error": sorted(r["item"] for r in rows
                                              if any(e.get("status") == "error"
                                                     for e in r.get("earlier_runs", []))),
            "by_item": {r["item"]: {"correct": _correct(r), "status": r["status"],
                                    "turns": r["iterations"], "cost_usd": r["cost_usd"],
                                    "helpers": (r.get("evaluation") or {}).get("helpers")}
                        for r in rows}}


def collect(evidence: Path) -> dict[str, Any]:
    """Each arm's runs, and per item how many of its runs are correct."""
    arms = {arm: [_run(evidence, b) for b in batches] for arm, batches in ARMS.items()}
    items = list(arms["feedback"][0]["by_item"])
    per_item = {it: {arm: sum(r["by_item"][it]["correct"] for r in runs)
                     for arm, runs in arms.items()} for it in items}
    return {"arms": arms, "per_item": per_item,
            "totals": {arm: {"correct": sum(r["correct"] for r in runs),
                             "of": sum(r["items"] for r in runs),
                             "per_run": [r["correct"] for r in runs],
                             "spent_usd": round(sum(r["spent_usd"] for r in runs), 4)}
                       for arm, runs in arms.items()}}


def markdown(c: dict[str, Any]) -> str:
    """The README's tables."""
    lines = ["| Arm | Run | Correct, 0 helpers | Loop success | Copied variant | Mean turns | "
             "Cost |", "|---|---|---|---|---|---|---|"]
    for arm, runs in c["arms"].items():
        for r in runs:
            loop = f"{r['loop_success']} / {r['items']}" if arm == "feedback" else "n/a"
            copied = ", ".join(f"`{i[3:]}`" for i in r["copied_variant"]) or "none"
            lines.append(f"| {arm} | [{r['batch']}](../{r['batch']}) | {r['correct']} / "
                         f"{r['items']} | {loop} | {copied} | {r['mean_turns']} | "
                         f"${r['spent_usd']:.2f} |")
    lines += ["", "| Item | Feedback: correct runs | Baseline: correct runs |", "|---|---|---|"]
    for it, n in c["per_item"].items():
        lines.append(f"| `{it}` | {n['feedback']} / 3 | {n['baseline']} / 3 |")
    return "\n".join(lines) + "\n"
