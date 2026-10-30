"""``museport loop``: the toolkit's agent loop, driven by Muse over a streaming connection,
with a hard per-run USD cap.

The toolkit's own ``oneapi-agent loop`` sends each turn without streaming (so a turn longer
than about 30 s fails with HTTP 502, and its transport retries that paid request), and its USD
cap is fixed at $5. This wrapper changes neither: it hands the toolkit's ``AgentLoop`` an
``OpenAICompatModel`` built on ``stream_transport`` and a ``LoopConfig`` with the caller's cap.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import oneapi_agent.loop as tk
from oneapi_agent.loop import AgentLoop, LoopConfig, Toolbox, model_pricing, preflight_real_model
from oneapi_agent.models import OpenAICompatModel, Transport
from oneapi_agent.result import ToolResult

from .gpubroker import gpu_tool, toolbox_factory
from .stream import stream_transport

# Per turn, reasoning included. The toolkit's default is 16000; Muse wrote whole llm.c ports of
# up to 24278 output tokens in one write_file, and at 16000 such writes were cut off (Oct 27).
MAX_OUTPUT_TOKENS = 32_000
# The toolkit also stops a run at 600000 tokens in total. museport's USD caps bound every run,
# so this limit is raised out of their way (it stopped a run below its USD cap on Oct 27).
MAX_TOTAL_TOKENS = 2_000_000


# Added to the goal in the toolkit's baseline arm (`--baseline`): the goal still names verify
# and scan_helpers, which that arm takes away.
BASELINE_NOTE = (
    " In this run you have no verify, scan_helpers or benchmark tool, so you cannot test the "
    "output yourself; the loop does not check success either. Call finish when you believe the "
    "port is complete and correct: a hidden reference then grades your final sources against "
    "the same requirements."
)


def build(task_file: str | Path, *, model: str, provider: str = "meta",
          max_iterations: int = 8, max_cost_usd: float = 1.0,
          max_output_tokens: int = MAX_OUTPUT_TOKENS, max_total_tokens: int = MAX_TOTAL_TOKENS,
          baseline: bool = False, shell: bool = False,
          transport: Transport | None = None) -> tuple[LoopConfig, OpenAICompatModel]:
    # evaluate: after the run, the toolkit grades the final sources with a hidden compile,
    # verify and whole-tree helper scan (harness-side; the model never sees it).
    cfg = LoopConfig.from_task_file(task_file, max_iterations=max_iterations,
                                    max_cost_usd=max_cost_usd, model=model, evaluate=True,
                                    baseline=baseline)
    if baseline:
        cfg.goal += BASELINE_NOTE
    cfg.provider = provider
    cfg.max_total_tokens = max_total_tokens
    cfg.shell = shell  # the toolkit's `run` tool: bash in the work dir
    client = OpenAICompatModel(model, provider=provider, max_tokens=max_output_tokens,
                               transport=transport or stream_transport(fill_missing_usage=True,
                                                                       mark_empty=True))
    return cfg, client


class SafeToolbox(Toolbox):
    """The toolkit's Toolbox, but a tool call missing a required argument becomes an error
    result the model sees, like the toolkit's other bad-input errors, instead of a KeyError
    that ends the run. It happens when a turn's output budget cuts a tool call off mid-stream
    (seen on 2026-10-26: write_file without "path")."""

    def __call__(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            return super().__call__(name, args)
        except KeyError as e:
            return ToolResult(name, "error", f"missing argument {e} (the call may have been "
                                             "cut off by the output limit; send it again, "
                                             "smaller if needed)")


class NoMigrateToolbox(SafeToolbox):
    """SafeToolbox without SYCLomatic: migrate is refused, as the toolkit refuses tools a run
    does not offer."""

    def __call__(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "migrate":
            return ToolResult(name, "error", "tool 'migrate' is not available in this run")
        return super().__call__(name, args)


# The toolkit's procedure line, as it reads without migrate (the rest of the prompt is kept).
NO_MIGRATE_PROCEDURE = (
    "Procedure: read the CUDA source -> write the SYCL port yourself (migrate/SYCLomatic is not\n"
    "available) -> compile -> fix errors -> scan_helpers -> compile -> verify\n"
    "-> benchmark -> finish."
)


ELIDED = "[older tool result shortened by the loop to save context; read the file or run the tool \
again if you need it]"


class CompactingModel:
    """Keeps a long run's prompt bounded. The toolkit's loop sends the whole conversation every
    turn; once it passes ``trigger_chars`` (about 300k tokens), every tool result and every
    written file's content older than the last ``keep_turns`` turns is cut to its first ``head``
    characters. It compacts in steps, not every turn, so the cached prefix stays the same
    between compactions and most of each prompt is still read from the cache."""

    def __init__(self, inner: Any, *, trigger_chars: int = 1_000_000, keep_turns: int = 40,
                 head: int = 400) -> None:
        self.inner, self.trigger, self.keep, self.head = inner, trigger_chars, keep_turns, head
        self.compactions = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def create(self, *, system: str, messages: list[dict[str, Any]], tools: Any) -> Any:
        if len(json.dumps(messages)) > self.trigger:
            self.compact(messages)
        return self.inner.create(system=system, messages=messages, tools=tools)

    def compact(self, messages: list[dict[str, Any]]) -> None:
        """In place, so the loop's own list stays the compacted one."""
        for m in messages[1:max(1, len(messages) - 2 * self.keep)]:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result" and isinstance(b.get("content"), str):
                    if len(b["content"]) > self.head and not b["content"].endswith(ELIDED):
                        b["content"] = b["content"][:self.head] + "..." + ELIDED
                elif b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    body = inp.get("content")
                    if isinstance(body, str) and len(body) > self.head and \
                            not body.endswith(ELIDED):
                        inp["content"] = body[:self.head] + "..." + ELIDED
        self.compactions += 1
        print(f"[loop] context compacted ({self.compactions}): "
              f"{len(json.dumps(messages))} chars", file=sys.stderr)


class NoMigrateLoop(AgentLoop):
    """The toolkit's AgentLoop with SYCLomatic taken away: the migrate tool is neither offered
    nor run, and the system prompt and first message no longer tell the model to start with
    it. Everything else (tools, checks, success rule, caps) is the toolkit's own."""

    def _initial_message(self) -> str:
        return super()._initial_message().replace(
            "Start with migrate.", "Start by writing your SYCL port (migrate is not available).")

    def run(self) -> dict[str, Any]:
        with _without_migrate():
            return super().run()


class FinishGate:
    """Notes whether the model's latest answer calls ``finish`` (a tuning run's success needs
    it; see TuningLoop). Wraps the model client like CompactingModel and can wrap it."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.finish_requested = inner, False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def create(self, *, system: str, messages: list[dict[str, Any]], tools: Any) -> Any:
        resp = self.inner.create(system=system, messages=messages, tools=tools)
        self.finish_requested = any(u.get("name") == "finish"
                                    for u in (getattr(resp, "tool_uses", None) or []))
        return resp


class TuningLoop(NoMigrateLoop):
    """A run that starts from a port that already passes (seeded) and tunes it. The toolkit
    stops at the first turn where compile, verify and zero helpers hold on the current
    sources, which a passing seed meets at once; here success also needs the model to call
    ``finish`` in that turn, so the run ends when the model says it is done and the test
    still passes."""

    gate: FinishGate

    def success(self) -> bool:
        return super().success() and self.gate.finish_requested


@contextmanager
def _shell_not_stuck() -> Iterator[None]:
    """The toolkit stops a run when one failure repeats 3 turns in a row. With the shell tool
    that rule fired on scratch work: L1 run 4 stopped at turn 132, mid-debugging, after three
    different commands failed the same way (a debug program whose compile error a `| head`
    hid). Failed `run` commands no longer count; compile and verify failures still do."""
    orig = tk.failure_signature
    tk.failure_signature = lambda r: None if r.tool == "run" else orig(r)
    try:
        yield
    finally:
        tk.failure_signature = orig


@contextmanager
def _without_migrate() -> Iterator[None]:
    schemas, prompt = tk.tool_schemas, tk.SYSTEM_PROMPT
    start = prompt.index("Procedure:")
    end = prompt.index("\n\n", start)
    tk.SYSTEM_PROMPT = prompt[:start] + NO_MIGRATE_PROCEDURE + prompt[end:]
    tk.tool_schemas = lambda *a, **k: [t for t in schemas(*a, **k) if t["name"] != "migrate"]
    try:
        yield
    finally:
        tk.tool_schemas, tk.SYSTEM_PROMPT = schemas, prompt


def turn_overshoot_usd(model: str, input_tokens: int = 50_000,
                       max_output_tokens: int = MAX_OUTPUT_TOKENS) -> float:
    """What one more turn can add after the cap is checked: the cap is enforced between turns,
    so a run can end up to one turn above it (full output budget, ``input_tokens`` input)."""
    p = model_pricing(model)
    return (input_tokens * p.input + max_output_tokens * p.output) / 1e6


def run(task_file: str | Path, *, model: str, provider: str = "meta", max_iterations: int = 8,
        max_cost_usd: float = 1.0, runs_dir: str | Path = "runs",
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        max_total_tokens: int = MAX_TOTAL_TOKENS, no_migrate: bool = False,
        baseline: bool = False, shell: bool = False, compact: bool = False,
        tune: bool = False, gpu_queue: str | Path | None = None) -> dict[str, Any]:
    """``no_migrate``: the ablation without SYCLomatic (see NoMigrateLoop). ``baseline``: the
    toolkit's arm without feedback (no verify, scan_helpers or benchmark; ends on finish).
    ``gpu_queue`` (with ``tune``): the gpu_test tool, answered by a GPU broker on that queue."""
    preflight_real_model(model, provider=provider)
    cfg, client = build(task_file, model=model, provider=provider,
                        max_iterations=max_iterations, max_cost_usd=max_cost_usd,
                        max_output_tokens=max_output_tokens,
                        max_total_tokens=max_total_tokens, baseline=baseline, shell=shell)
    if compact:
        client = CompactingModel(client)
    if tune:
        gate = FinishGate(client)
        factory: Any = NoMigrateToolbox
        if gpu_queue:
            factory = toolbox_factory(NoMigrateToolbox, Path(gpu_queue).resolve())
        tuner = TuningLoop(cfg, gate, runs_dir=runs_dir, toolbox_factory=factory)
        tuner.gate = gate
        seed_work(task_file, tuner.work)
        with _shell_not_stuck(), (gpu_tool() if gpu_queue else nullcontext()):
            return tuner.run()
    agent = (NoMigrateLoop(cfg, client, runs_dir=runs_dir, toolbox_factory=NoMigrateToolbox)
             if no_migrate else
             AgentLoop(cfg, client, runs_dir=runs_dir, toolbox_factory=SafeToolbox))
    seed_work(task_file, agent.work)
    if shell:
        with _shell_not_stuck():
            return agent.run()
    return agent.run()


def seed_work(task_file: str | Path, work: Path) -> list[str]:
    """Copy the task's ``museport.seed`` folder (relative to the task dir) into the work dir:
    files the model starts from, e.g. its own earlier, proven port (J3)."""
    task_file = Path(task_file)
    seed = json.loads(task_file.read_text()).get("museport", {}).get("seed")
    if not seed:
        return []
    src = task_file.parent / seed
    copied = []
    for f in sorted(src.rglob("*")):
        if f.is_file():
            dest = work / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            copied.append(str(f.relative_to(src)))
    return copied
