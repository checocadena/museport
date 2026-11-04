"""Every number the film shows, read from the committed evidence (none typed by hand).

usage: PYTHONPATH=src python3 demo/remotion/make_film_data.py > demo/remotion/public/film-data.json
(run from the repository root)

It extends demo/make_data.py's data with what the film adds: the kernel-file count including
the attention ports, the portable trainer's speeds by GPU, the oneMath fix's unit tests, and
the two terminal sessions the film replays (llm.c-sycl's test and llama2-sycl's run).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"


def _make_data():  # demo/make_data.py is a script, not a package module
    spec = importlib.util.spec_from_file_location("make_data", ROOT / "demo" / "make_data.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _attention() -> list[dict[str, object]]:
    rows = []
    for d in ("port-llmc-attention", "port-llmc-attention-forward"):
        for r in json.loads((EV / d / "summary.json").read_text())["items"]:
            ev = r.get("evaluation") or {}
            ok = ev.get("verify_status") == "ok" and ev.get("helpers") == 0
            if not ok:
                continue  # the cut-off forward run; its retry is the other folder
            gpu = []
            for s in sorted((EV / d / "gpu").glob("*/*/summary.json")):
                g = json.loads(s.read_text())
                steps = {x["step"]: x for x in g["evidence"]["steps"]}
                gpu.append({"device": g["evidence"]["device"]["name"],
                            "pass": g["status"] == "ok",
                            "vs_native": steps.get("sycl_verify_vs_native", {}).get(
                                "status") == "ok"})
            rows.append({"item": r["item"], "gpu": gpu})
    return rows


def _onemath() -> dict[str, object]:
    log = (ROOT / "upstream/onemath/unit-tests/onemath-cublas-blas-tests.log").read_text()
    passed = [int(n) for n in re.findall(r"\[  PASSED  \] (\d+) tests\.", log)]
    return {"pr": "uxlfoundation/oneMath#793", "passed_before": min(passed),
            "passed_after": max(passed)}


def _session(path: Path, keep: list[str]) -> list[str]:
    """The lines of a committed terminal log that the film replays, in order."""
    lines = [ln.rstrip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if any(re.search(k, ln) for k in keep)]


def main() -> dict[str, object]:
    data = _make_data().main(EV)
    cur = {r["label"]: r for r in data["current"]}["All"]
    att = _attention()
    portable = data["l1"]["portable"]
    data["film"] = {
        "devcuda": {
            "files": data["plan"]["files"]["files"],
            "attempted": cur["items"] + len(att),
            "correct": cur["correct"] + len(att),
            "nvidia": cur["gpu"] + sum(1 for a in att if any(
                g["pass"] and g["vs_native"] for g in a["gpu"])),
        },
        "portable_speeds": [{"device": s["device"], "ratio": s["ratio"], "ms": s["ms"],
                             "cuda_ms": s["cuda_ms"]}
                            for s in sorted(portable["speeds"], key=lambda s: s["ratio"] or 0)
                            if s.get("ratio")],
        # AMD: the one-run port's time per step, then the portable version's (no vendor code)
        "amd": {"device": "AMD Instinct MI350X", "before_ms": data["l1"]["amd"]["run5"]["ms"],
                "after_ms": next(s["ms"] for s in portable["speeds"]
                                 if s["device"] == "AMD Instinct MI350X")},
        # the one-run port, before tuning: its time per step as a multiple of llm.c's CUDA
        "run5": {"device": data["l1"]["nvidia"]["run5"]["device"],
                 "ratio": data["l1"]["nvidia"]["run5"]["ratio"]},
        # Muse's own GPU calls during the GPU tuning run, rented and deleted by the broker
        "gpu_calls": {"calls": data["l1"]["gtune"]["gpu_calls"],
                      "usd": data["l1"]["gtune"]["gpu_usd"]},
        "onemath": _onemath(),
        "j3_test": _session(EV / "j3-fresh-clone/test.log",
                            [r"^Device", r"OK \(LOGITS\)", r"^LOSS OK", r"^TENSOR OK",
                             r"^step \d", r"^overall okay"]),
        "j4_session": _session(EV / "j4-fresh-clone/session.log",
                               [r"^\$ make llama2", r"^icpx", r"^\$ \./llama2 .*Once upon",
                                r"^Once upon a time", r"^achieved tok/s"])[:5],
    }
    return data


if __name__ == "__main__":
    json.dump(main(), sys.stdout, indent=1)
