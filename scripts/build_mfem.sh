#!/usr/bin/env bash
# Fetch and build the pinned MFEM release (serial, no optional dependencies).
#
#   MFEM_TAG       release tag            (default v4.10)
#   MFEM_COMMIT    expected commit hash   (default d964264cdb9a13e94a201b6c236c7721e0c8765f)
#   THIRD_PARTY    where to put src/build/install (default <project>/third_party)
#   JOBS           parallel make jobs      (default 2)
#
# Nothing is installed system-wide; everything lives under $THIRD_PARTY.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MFEM_TAG="${MFEM_TAG:-v4.10}"
MFEM_COMMIT="${MFEM_COMMIT:-d964264cdb9a13e94a201b6c236c7721e0c8765f}"
THIRD_PARTY="${THIRD_PARTY:-$ROOT/third_party}"
JOBS="${JOBS:-2}"
SRC="$THIRD_PARTY/mfem-src"; BLD="$THIRD_PARTY/mfem-build"; INST="${MFEM_INSTALL:-$THIRD_PARTY/mfem-install}"
mkdir -p "$THIRD_PARTY"
exec 9>"$THIRD_PARTY/.mfem-build.lock"
flock -n 9 || { echo "Another build owns $THIRD_PARTY" >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$MFEM_TAG" https://github.com/mfem/mfem.git "$SRC"
fi
actual="$(git -C "$SRC" rev-parse HEAD)"
if [ "$actual" != "$MFEM_COMMIT" ]; then
  echo "ERROR: MFEM checkout is $actual, expected pinned $MFEM_COMMIT" >&2
  exit 1
fi

if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=normal)" ]; then
  echo "ERROR: MFEM source tree has local changes: $SRC" >&2
  exit 1
fi
cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_MPI=OFF \
  -DMFEM_PRECISION=double -DBUILD_SHARED_LIBS=OFF -DMFEM_USE_METIS=OFF \
  -DMFEM_USE_OPENMP=OFF -DMFEM_USE_CUDA=OFF -DMFEM_USE_HIP=OFF \
  -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF -DCMAKE_INSTALL_PREFIX="$INST"
cmake --build "$BLD" -j "$JOBS"
cmake --install "$BLD"
echo "MFEM installed to $INST"
