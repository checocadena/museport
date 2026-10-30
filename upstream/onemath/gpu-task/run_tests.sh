# museport: oneMath's own BLAS unit tests on the cuBLAS backend, before and after the bf16
# gemm patch. Sourced by gpu-test as the task's gpu_setup; everything printed lands in
# setup.log, which comes back with the results.
# This file is *sourced* by the runner, so everything runs in a subshell: no variable, option
# or directory change may leak into the runner, which packs the results afterwards. (Two runs
# lost their logs: `set -u` leaked, and so did a variable named HERE, the runner's own output
# directory.)
(
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONEMATH_COMMIT=585e610ebf8101fab67c223c033611264efcd21e   # uxlfoundation/oneMath develop, 2026-10-23
LAPACK_TAG=v3.12.1
CXX_BIN="${SYCL_CXX:-$(command -v icpx || command -v clang++)}"
C_BIN="$(dirname "$CXX_BIN")/clang"
[ -x "$C_BIN" ] || C_BIN="$(command -v icx || command -v clang || command -v gcc)"
echo "== compiler: $CXX_BIN"; "$CXX_BIN" --version | head -1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -m pip install -q cmake ninja 2>/dev/null \
  || python3 -m pip install -q --break-system-packages cmake ninja 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
command -v gfortran >/dev/null || { apt-get update -qq && apt-get install -y -qq gfortran >/dev/null; }
W="$(mktemp -d)"
t0=$(date +%s); el() { echo "[$(( $(date +%s) - t0 )) s]"; }

echo "== reference BLAS (Reference-LAPACK $LAPACK_TAG, CBLAS, 32- and 64-bit index) $(el)"
git clone -q --depth 1 --branch "$LAPACK_TAG" https://github.com/Reference-LAPACK/lapack.git "$W/lapack"
for idx in True False; do
  cmake -S "$W/lapack" -B "$W/lapack/build-$idx" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$W/ref" -DCBLAS=True -DLAPACK=True -DLAPACKE=False \
    -DBUILD_INDEX64=$idx -DBUILD_SHARED_LIBS=True -DBUILD_TESTING=OFF >/dev/null \
    && cmake --build "$W/lapack/build-$idx" -j"$(nproc)" --target install >/dev/null \
    || echo "REFERENCE BLAS BUILD FAILED (index64=$idx)"
done
ls "$W/ref/lib" | head -20

echo "== oneMath $ONEMATH_COMMIT, cuBLAS backend, BLAS tests $(el)"
SRC="$W/oneMath"
git init -q "$SRC" && git -C "$SRC" fetch -q --depth 1 https://github.com/uxlfoundation/oneMath.git "$ONEMATH_COMMIT" \
  && git -C "$SRC" checkout -q FETCH_HEAD
cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" -DCMAKE_C_COMPILER="$C_BIN" \
  -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_CUBLAS_BACKEND=ON \
  -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=ON -DBUILD_EXAMPLES=OFF \
  -DREF_BLAS_ROOT="$W/ref" > "$W/cmake.log" 2>&1 || { echo "CMAKE FAILED"; tail -40 "$W/cmake.log"; }
grep -i "cblas\|warning" "$W/cmake.log" | head -5

run_suite() {  # <label> <gtest filter>
  for api in ct rt; do
    b="$SRC/build/bin/test_main_blas_$api"
    [ -x "$b" ] || { echo "$1 $api: no binary"; continue; }
    ONEAPI_DEVICE_SELECTOR='cuda:*' "$b" --gtest_filter="$2" > "$W/$1-$api.log" 2>&1
    echo "-- $1, $api API, filter '$2': exit $?"
    grep -E "^\[  (PASSED|SKIPPED|FAILED) +\]|tests? ran" "$W/$1-$api.log"
    grep -E "^\[ +(OK|SKIPPED|FAILED) +\] .*Bfloat16" "$W/$1-$api.log" | head -12
    grep -E "^\[  FAILED  \] [A-Za-z]" "$W/$1-$api.log" | head -40
  done
}

echo "== build (unpatched) $(el)"
cmake --build "$SRC/build" -j"$(nproc)" > "$W/build1.log" 2>&1 || { echo "BUILD FAILED"; tail -60 "$W/build1.log"; }
echo "== tests, unpatched $(el)"
run_suite unpatched-gemm '*Gemm*'
run_suite unpatched-all '*'

echo "== apply the patch and rebuild $(el)"
git -C "$SRC" apply "$HERE/onemath-cublas-bf16-gemm.patch" && echo "applied onemath-cublas-bf16-gemm.patch" \
  || echo "PATCH DID NOT APPLY"
git -C "$SRC" diff --stat
cmake --build "$SRC/build" -j"$(nproc)" > "$W/build2.log" 2>&1 || { echo "REBUILD FAILED"; tail -60 "$W/build2.log"; }
echo "== tests, patched $(el)"
run_suite patched-gemm '*Gemm*'
run_suite patched-all '*'
echo "== bf16 gemm tests, patched, full output $(el)"
for api in ct rt; do grep -E "Bfloat16" "$W/patched-gemm-$api.log"; done
echo "== done $(el)"
)
