"""museport llmc-sycl: the reassembled repo's layout and Makefile (no compiler)."""

from __future__ import annotations

import json
from pathlib import Path

from museport.assemble import KERNELS, assemble

ROOT = Path(__file__).resolve().parents[1]


def _fake_llmc(d: Path) -> Path:
    for f in ("llmc/utils.h", "llmc/tokenizer.h", "llmc/dataloader.h", "llmc/rand.h", "LICENSE",
              "dev/download_starter_pack.sh"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(f"// {f}\n")
    return d


def test_every_kernel_port_is_copied_with_a_make_rule(tmp_path):
    out = tmp_path / "llm.c-sycl"
    r = assemble(ROOT / "evidence", _fake_llmc(tmp_path / "llm.c"), out)
    assert len(r["kernels"]) == len(KERNELS) == 19
    make = (out / "dev" / "sycl" / "Makefile").read_text()
    for k in r["kernels"]:
        for s in k["sources"]:
            assert (out / "dev" / "sycl" / k["kernel"] / s).exists()
        assert f"build/{k['kernel']}:" in make
    onemath = {k["kernel"] for k in r["kernels"] if k["onemath"]}
    assert onemath == {"matmul_forward", "matmul_backward", "trimat_forward"}
    assert make.count("$(ONEMATH_FLAGS)") == 3
    # softmax_forward comes from the rerun that passes, not run 2's failing port
    prov = json.loads((out / "dev" / "sycl" / "PROVENANCE.json").read_text())
    sm = next(k for k in prov if k["kernel"] == "softmax_forward")
    assert sm["evidence"] == "evidence/port-llmc-softmax-rerun-2/t2-softmax-forward"
    assert (out / "llmc" / "dataloader.h").exists() and (out / "LICENSE").exists()
