#!/usr/bin/env bash
# Build cpp/poisson_mms and cpp/poisson_amr against the pinned MFEM install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INST="${MFEM_INSTALL:-${THIRD_PARTY:-$ROOT/third_party}/mfem-install}"
cmake -S "$ROOT/cpp" -B "$ROOT/build/cpp" -DMFEM_DIR="$INST/lib/cmake/mfem" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build/cpp" -j "${JOBS:-2}"
"$ROOT/build/cpp/poisson_mms" --list-cases
