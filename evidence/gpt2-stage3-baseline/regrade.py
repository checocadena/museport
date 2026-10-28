"""Rerun the toolkit's after-run grade (AgentLoop.evaluate's steps) on a copy of a run's work dir.

usage: python3 regrade.py <task.json> <copy of the run's work dir> <the run's trace.jsonl>
"""

import json
import sys
from pathlib import Path

from oneapi_agent.loop import LoopConfig, Toolbox, source_hash
from oneapi_agent.toolchain import run_command
from oneapi_agent.tools.scan_helpers import scan_helpers

task, work, trace = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = LoopConfig.from_task_file(task, model="muse-spark-1.3", evaluate=True, baseline=True)
calls = [json.loads(line) for line in trace.read_text().splitlines() if '"tool_call"' in line]
last = [c for c in calls if c.get("name") == "compile"][-1]
ref = run_command(cfg.reference_build, cwd=str(work), timeout=600)
box = Toolbox(cfg, work)
ev = {"source_hash": source_hash(work), "reference_build_rc": ref.returncode,
      "last_compile_input": last["input"]}
compiled = box.t_compile(last["input"])
ev["compile"] = [compiled.status, compiled.summary[:200]]
verified = box.t_verify({})
ev["verify_status"], ev["verify_summary"] = verified.status, verified.summary
scan = scan_helpers(str(work))
ev["helpers"] = int(scan.evidence.get("total", 0))
print(json.dumps(ev, indent=2))
