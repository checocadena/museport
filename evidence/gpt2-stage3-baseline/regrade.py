"""Re-run the toolkit's after-run grade (AgentLoop.evaluate's steps) on a copy of a run's work dir."""
import hashlib, json, sys
from pathlib import Path
from oneapi_agent.loop import LoopConfig, Toolbox, source_hash
from oneapi_agent.toolchain import run_command
from oneapi_agent.tools.scan_helpers import scan_helpers

task, work, trace = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = LoopConfig.from_task_file(task, model="muse-spark-1.3", evaluate=True, baseline=True)
compiles = [json.loads(l) for l in trace.read_text().splitlines() if '"tool_call"' in l]
last = [c for c in compiles if c.get("name") == "compile"][-1]
ref = run_command(cfg.reference_build, cwd=str(work), timeout=600)
box = Toolbox(cfg, work)
ev = {"source_hash": source_hash(work), "reference_build_rc": ref.returncode,
      "last_compile_input": last["input"]}
c = box.t_compile(last["input"]); ev["compile"] = [c.status, c.summary[:200]]
v = box.t_verify({}); ev["verify_status"], ev["verify_summary"] = v.status, v.summary
sc = scan_helpers(str(work)); ev["helpers"] = int(sc.evidence.get("total", 0))
print(json.dumps(ev, indent=2))
