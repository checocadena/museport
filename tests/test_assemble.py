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


def test_assemble_reproduces_the_committed_repo(tmp_path):
    # every committed llm.c-sycl file comes out of assemble(), byte for byte, except README.md
    # (hand-written) and llm.c's own files (from an llm.c checkout; faked here)
    import subprocess

    out = tmp_path / "llm.c-sycl"
    r = assemble(ROOT / "evidence", _fake_llmc(tmp_path / "llm.c"), out)
    assert r["programs"] == ["test_gpt2_fp32.cpp", "train_gpt2_fp32.cpp"]
    tracked = subprocess.run(["git", "ls-files", "llm.c-sycl"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    from_llmc = {"LICENSE", "dev/download_starter_pack.sh", *(f"llmc/{h}" for h in
                                                              ("utils.h", "tokenizer.h",
                                                               "dataloader.h", "rand.h"))}
    ours = [f.removeprefix("llm.c-sycl/") for f in tracked]
    ours = [f for f in ours if f != "README.md" and f not in from_llmc]
    assert len(ours) == 27
    for f in ours:
        assert (out / f).read_bytes() == (ROOT / "llm.c-sycl" / f).read_bytes(), f
    assert (out / "dev" / "download_starter_pack.sh").stat().st_mode & 0o111


def test_assemble_refuses_a_trainer_that_is_not_the_graded_one(tmp_path):
    import shutil

    import pytest

    ev = tmp_path / "evidence"
    for d in ("j3-test/run3/port", "j3-trainer/run1/port"):
        shutil.copytree(ROOT / "evidence" / d, ev / d)
    from museport.assemble import _programs

    out = tmp_path / "out"
    out.mkdir()
    assert _programs(ev, out) == ["test_gpt2_fp32.cpp", "train_gpt2_fp32.cpp"]
    t = ev / "j3-test" / "run3" / "port" / "train_gpt2_fp32.cpp"
    t.write_text(t.read_text().replace("0.044715f", "0.04715f", 1))
    with pytest.raises(ValueError, match="not the graded trainer"):
        _programs(ev, out)
