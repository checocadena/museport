"""Collect every number the demo shows from the committed evidence (none typed by hand).

usage: python3 demo/make_data.py [evidence_dir] > demo/data.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from museport.report import current_table, load, tier_table


def _gpu(evidence: Path, vendor_dir: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for f in sorted(evidence.glob(f"port-llmc-tier1-run2/gpu/{vendor_dir}summary.json")):
        d = json.loads(f.read_text())
        steps = {s["step"]: s for s in d["evidence"]["steps"]}
        ref = steps.get("sycl_verify_vs_reference", {})
        err = re.search(r"worst max abs err ([0-9.e+-]+)", ref.get("summary", ""))
        out[f.parent.name.removeprefix("llmc_")] = {
            "device": d["evidence"]["device"]["name"],
            "pass": d["status"] == "ok",
            "vs_native": steps.get("sycl_verify_vs_native", {}).get("status") == "ok",
            "worst_err": float(err.group(1)) if err else None}
    return out


def _replay(trace: Path) -> list[dict[str, object]]:
    """One entry per turn: the model's line of text and each tool result's name and status."""
    turns: dict[int, dict[str, object]] = {}
    for line in trace.read_text().splitlines():
        e = json.loads(line)
        if "iteration" not in e:
            continue
        t = turns.setdefault(int(e["iteration"]), {"turn": int(e["iteration"]), "say": "",
                                                    "tools": []})
        if e["type"] == "model_response":
            text = " ".join(c.get("text", "") for c in e.get("content", [])
                            if c.get("type") == "text").strip()
            t["say"] = text[:110]
        elif e["type"] == "tool_result":
            r = e.get("result", {})
            summary = re.sub(r"/\S+/", "", r.get("summary", ""))[:90]
            t["tools"].append({"name": e["name"], "status": r.get("status"), "summary": summary})
    return [turns[k] for k in sorted(turns)]


def main(evidence: Path) -> dict[str, object]:
    data = load(evidence)
    plan = json.loads((evidence / "plan-llmc-devcuda-stream/plan.json").read_text())
    run1 = {r["item"]: r for r in json.loads(
        (evidence / "port-llmc-tier1/summary.json").read_text())["items"]}
    run2 = {r["item"]: r for r in json.loads(
        (evidence / "port-llmc-tier1-run2/summary.json").read_text())["items"]}

    def tier1(rows: dict[str, dict[str, object]]) -> dict[str, object]:
        ev = [r.get("evaluation") or {} for r in rows.values()]
        return {"loop_success": sum(r.get("status") == "success" for r in rows.values()),
                "correct": sum(e.get("verify_status") == "ok" and e.get("helpers") == 0
                               for e in ev),
                "turns_mean": round(sum(r["iterations"] for r in rows.values()) / len(rows), 1),
                "cost": round(sum(r["cost_usd"] for r in rows.values()), 4),
                "items": len(rows)}

    def stage(name: str) -> dict[str, object] | None:
        """A GPT-2 stage's run, or None when its evidence is not in this checkout."""
        f = evidence / name / "run1/outcome.json"
        if not f.is_file():
            return None
        g = json.loads(f.read_text())
        return {"status": g["stop_reason"], "turns": g["iterations"], "cost": g["cost_usd"],
                "verify": g["evaluation"]["verify_summary"],
                "verify_ok": g["evaluation"]["verify_status"] == "ok",
                "helpers": g["evaluation"]["helpers"],
                "replay": _replay(evidence / name / "run1/trace.jsonl")}

    nomigrate = evidence / "port-llmc-tier1-nomigrate/summary.json"
    g1 = json.loads((evidence / "gpt2-stage1/run1/outcome.json").read_text())
    proof1 = json.loads((evidence / "gpt2-stage1/cpu-proof.json").read_text())
    return {
        "plan": {"items": plan["grounding"]["items"], "files": plan["grounding"]["coverage"],
                 "issues": plan["grounding"]["issues"], "cost": plan["cost_usd"],
                 "model": plan["model"]},
        "tiers": tier_table(data["items"]),  # the first runs
        "current": current_table(data["items"]),  # the runs with the fixed loop
        "tier1_first": tier1(run1),
        "tier1_fixed": tier1(run2),
        "nvidia": _gpu(evidence, "nvidia/*/*/"),
        "amd": _gpu(evidence, "amd-*/*/"),
        "gpt2_stage1": {"status": g1["stop_reason"], "turns": g1["iterations"],
                        "cost": g1["cost_usd"], "verify": g1["evaluation"]["verify_summary"],
                        "helpers": g1["evaluation"]["helpers"],
                        "cpu_proof": proof1["steps"]["verify_cpu_proof"]["summary"],
                        "replay": _replay(evidence / "gpt2-stage1/run1/trace.jsonl")},
        "gpt2_stage2": stage("gpt2-stage2"),
        "gpt2_stage3": stage("gpt2-stage3"),
        "tier1_nomigrate": tier1({r["item"]: r for r in json.loads(nomigrate.read_text())["items"]})
        if nomigrate.is_file() else None,
        "native_exp_check": json.loads(
            (evidence / "gpt2-stage1/native-exp-check.json").read_text()),
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1] if len(sys.argv) > 1 else "evidence")), indent=1))
