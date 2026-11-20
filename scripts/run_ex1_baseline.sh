#!/usr/bin/env bash
# Build the UNMODIFIED official MFEM Example 1 (examples/ex1.cpp from the pinned
# source tree) against the pinned install and run it, preserving evidence.
#
# Output: runs/baseline_ex1/<UTC stamp>/ with build/run commands, stdout,
# stderr, exit codes, binary hash, source hash and the produced mesh/solution.
# Note: ex1 solves -lap(u) = 1 (NOT the manufactured-solution benchmark).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="${THIRD_PARTY:-$ROOT/third_party}"
CXX="${CXX:-g++}"
SRC="$THIRD_PARTY/mfem-src"; INST="${MFEM_INSTALL:-$THIRD_PARTY/mfem-install}"
OUT="$ROOT/runs/baseline_ex1/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
cd "$OUT"

{
  echo "mfem_git=$(git -C "$SRC" describe --tags --long 2>/dev/null) $(git -C "$SRC" rev-parse HEAD)"
  echo "ex1_source_sha256=$(sha256sum "$SRC/examples/ex1.cpp" | cut -d' ' -f1)"
  echo "libmfem_sha256=$(sha256sum "$INST/lib/libmfem.a" | cut -d' ' -f1)"
  echo "cxx=$("$CXX" --version | head -1)"
  echo "uname=$(uname -srm)"
} > provenance.txt

BUILD=("$CXX" -O3 -std=c++17 -I"$INST/include" "$SRC/examples/ex1.cpp" -o ex1 -L"$INST/lib" -lmfem)
printf '%q ' "${BUILD[@]}" > build_command.txt; echo >> build_command.txt
"${BUILD[@]}" > build_stdout.txt 2> build_stderr.txt
echo $? > build_exit_code.txt
[ "$(cat build_exit_code.txt)" = "0" ] || { echo "ex1 build failed; see $OUT"; exit 1; }
sha256sum ex1 | cut -d' ' -f1 > ex1_sha256.txt

run_case () {  # name, mesh file, extra args...
  local name="$1" mesh="$2"; shift 2
  local cmd=(./ex1 -m "$mesh" -no-vis "$@")
  mkdir -p "$name"
  printf '%q ' "${cmd[@]}" > "$name/command.txt"; echo >> "$name/command.txt"
  ( cd "$name" && ../ex1 -m "$mesh" -no-vis "$@" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "$name exit=$(cat "$name/exit_code.txt")"
}
run_case star_default "$SRC/data/star.mesh"
run_case square_disc_order2 "$SRC/data/square-disc.mesh" -o 2
echo "evidence in $OUT"
