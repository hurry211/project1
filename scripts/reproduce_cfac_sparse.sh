#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ $# -ne 1 ]]; then
  echo "Supply a new output directory" >&2
  exit 2
fi
out="$1"
cpu="$(python -c 'import os; print(min(os.sched_getaffinity(0)))')"
mkdir "$out"
./scripts/build_solver.sh
cmake --build build/cpp --target cfac_pcg
python -m pytest -q tests/test_cfac_sparse.py
taskset -c "$cpu" python -m research.cfac.sparse_diagnostics --binary build/cpp/cfac_pcg --directory "$out/fixed-primal"
taskset -c "$cpu" python -m research.cfac.sparse_study --cohort transfer --output "$out/transfer"
taskset -c "$cpu" python -m research.cfac.sparse_study --cohort screen --output "$out/screen"
for cohort in transfer screen; do
  archive="$out/$cohort/receipt.json.gz"
  digest="$(sha256sum "$archive" | cut -d' ' -f1)"
  python -m research.cfac.sparse_study --replay "$archive" --sha256 "$digest"
  python -m research.cfac.sparse_analysis "$archive" --output "$out/$cohort-summary.json"
done
for n in 32 64 100 160 320; do
  rc=0
  taskset -c "$cpu" timeout 100 python -m research.cfac.sparse_diagnostics \
    --binary build/cpp/cfac_pcg --directory "$out/scale-$n" --scale "$n" || rc=$?
  printf '%s,%s\n' "$n" "$rc" >>"$out/scale-status.csv"
done
