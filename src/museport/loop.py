"""``museport loop``: the toolkit's agent loop, driven by Muse over a streaming connection,
with a hard per-run USD cap.

The toolkit's own ``oneapi-agent loop`` sends each turn without streaming (so a turn longer
than about 30 s fails with HTTP 502, and its transport retries that paid request), and its USD
cap is fixed at $5. This wrapper changes neither: it hands the toolkit's ``AgentLoop`` an
``OpenAICompatModel`` built on ``stream_transport`` and a ``LoopConfig`` with the caller's cap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oneapi_agent.loop import AgentLoop, LoopConfig, Toolbox, model_pricing, preflight_real_model
from oneapi_agent.models import OpenAICompatModel, Transport
from oneapi_agent.result import ToolResult

from .stream import stream_transport

# Per turn, reasoning included. The toolkit's default is 16000; Muse wrote whole llm.c ports of
# up to 24278 output tokens in one write_file, and at 16000 such writes were cut off (Oct 27).
MAX_OUTPUT_TOKENS = 32_000
# The toolkit also stops a run at 600000 tokens in total. museport's USD caps bound every run,
# so this limit is raised out of their way (it stopped a run below its USD cap on Oct 27).
MAX_TOTAL_TOKENS = 2_000_000


def build(task_file: str | Path, *, model: str, provider: str = "meta",
          max_iterations: int = 8, max_cost_usd: float = 1.0,
          max_output_tokens: int = MAX_OUTPUT_TOKENS, max_total_tokens: int = MAX_TOTAL_TOKENS,
          transport: Transport | None = None) -> tuple[LoopConfig, OpenAICompatModel]:
    # evaluate: after the run, the toolkit grades the final sources with a hidden compile,
    # verify and whole-tree helper scan (harness-side; the model never sees it).
    cfg = LoopConfig.from_task_file(task_file, max_iterations=max_iterations,
                                    max_cost_usd=max_cost_usd, model=model, evaluate=True)
    cfg.provider = provider
    cfg.max_total_tokens = max_total_tokens
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


def turn_overshoot_usd(model: str, input_tokens: int = 50_000,
                       max_output_tokens: int = MAX_OUTPUT_TOKENS) -> float:
    """What one more turn can add after the cap is checked: the cap is enforced between turns,
    so a run can end up to one turn above it (full output budget, ``input_tokens`` input)."""
    p = model_pricing(model)
    return (input_tokens * p.input + max_output_tokens * p.output) / 1e6


def run(task_file: str | Path, *, model: str, provider: str = "meta", max_iterations: int = 8,
        max_cost_usd: float = 1.0, runs_dir: str | Path = "runs",
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        max_total_tokens: int = MAX_TOTAL_TOKENS) -> dict[str, Any]:
    preflight_real_model(model, provider=provider)
    cfg, client = build(task_file, model=model, provider=provider,
                        max_iterations=max_iterations, max_cost_usd=max_cost_usd,
                        max_output_tokens=max_output_tokens,
                        max_total_tokens=max_total_tokens)
    return AgentLoop(cfg, client, runs_dir=runs_dir, toolbox_factory=SafeToolbox).run()
