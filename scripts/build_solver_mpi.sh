#!/usr/bin/env bash
# Build the parallel driver against the separate MPI MFEM installation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL="${MFEM_MPI_INSTALL:-${MPI_THIRD_PARTY:-$ROOT/third_party/mpi}/mfem-mpi-install}"
BUILD="${MPI_BUILD_DIR:-$ROOT/build/mpi}"
cmake -S "$ROOT/cpp/mpi" -B "$BUILD" -DMFEM_DIR="$INSTALL/lib/cmake/mfem" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j "${JOBS:-2}"
"$BUILD/poisson_mpi" --build-info
