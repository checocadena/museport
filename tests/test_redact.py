"""Credentials in run evidence are redacted by variable name."""

from pathlib import Path

from museport.redact import redact, redact_tree, secret_hits

ENV = {"MY_API_KEY": "sk-test-0123456789abcdef", "GH_TOKEN": "ghp_0123456789abcdef",
       "PATH": "/usr/bin", "MAX_THINKING_TOKENS": "32000000000"}


def test_values_are_replaced_by_name() -> None:
    out = redact("env: MY_API_KEY=sk-test-0123456789abcdef PATH=/usr/bin", ENV)
    assert "sk-test" not in out and "<redacted:MY_API_KEY>" in out and "/usr/bin" in out


def test_tree_is_redacted_and_rechecked(tmp_path: Path) -> None:
    (tmp_path / "trace.jsonl").write_text('{"stdout": "GH_TOKEN=ghp_0123456789abcdef"}\n')
    (tmp_path / "clean.txt").write_text("nothing here\n")
    assert secret_hits(tmp_path, ENV) == [(str(tmp_path / "trace.jsonl"), "GH_TOKEN")]
    assert redact_tree(tmp_path, ENV) == [str(tmp_path / "trace.jsonl")]
    assert secret_hits(tmp_path, ENV) == []
