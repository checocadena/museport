#!/usr/bin/env bash
# Demo, no model, no cost: Muse's gelu_forward port passes the hidden check; plant a typical
# porting slip (x^3 written as x^2 in GELU) and the same check fails. Needs the toolkit's toolchain (icpx).
# usage: demo/planted_bug.sh   (from the repository root, after `source <toolkit>/scripts/oneapi-env.sh`)
set -euo pipefail
item=evidence/port-llmc-tier1-run2/t1-gelu-forward
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$item"/task/* "$work"/ && cp "$item"/port/gelu_forward_sycl.cpp "$work"/
python3 -c "
from pathlib import Path; from museport.port import write_inputs; from museport.llmc import SPECS
write_inputs(SPECS['gelu_forward.cu'], Path('$work/input.bin'))"
cd "$work" && mkdir -p build
g++ -O2 -std=c++17 gelu_forward_ref.cpp -o build/reference
check() {
  icpx -fsycl -O2 gelu_forward_sycl.cpp -o build/candidate
  oneapi-agent --summary verify --ref "build/reference {input} {output} 4 256 768" \
    --cand "build/candidate {input} {output} 4 256 768" --input input.bin \
    --tolerance 1e-5 --rtol 1.1920929e-07 --mode array || true
}
echo "\$ verify Muse's port against the hidden reference"
check
echo
echo "\$ plant a bug: x^3 -> x^2 in GELU"
sed -i 's/0\.044715f \* xi \* xi \* xi/0.044715f * xi * xi/' gelu_forward_sycl.cpp
grep -n "0.044715f" gelu_forward_sycl.cpp
check
