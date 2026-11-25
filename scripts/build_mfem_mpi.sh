#!/usr/bin/env bash
# Build the reviewed Ubuntu Noble/amd64 MPI profile in a separate prefix.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
root="$(realpath -m "${MPI_THIRD_PARTY:-$repo/third_party/mpi}")"
source_dir="$(realpath -m "${MPI_MFEM_SOURCE:-$root/mfem-src}")"
build="$root/mfem-mpi-build"
install="$(realpath -m "${MFEM_MPI_INSTALL:-$root/mfem-mpi-install}")"
revision=d964264cdb9a13e94a201b6c236c7721e0c8765f
export CC="${CC:-gcc-13}" CXX="${CXX:-g++-13}"

. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != noble || "$(dpkg --print-architecture)" != amd64 ]]; then
  echo 'The reviewed MPI package lock requires Ubuntu 24.04 amd64.' >&2
  exit 1
fi
config=include/mfem/config/_config.hpp
if [[ -f "$install/$config" ]] && ! grep -qx '#define MFEM_USE_MPI' "$install/$config"; then
  echo "Existing MFEM installation is serial: $install" >&2
  exit 1
fi

mkdir -p "$root/packages" "$root/sysroot"
exec 9>"$root/.mpi-build.lock"
flock -n 9 || { echo "Another MPI build owns $root" >&2; exit 1; }

mpi=4.1.6-7ubuntu2
hypre=2.28.0-8build2
metis=5.1.0.dfsg-7build3
# Download headers without installing development packages or changing alternatives.
runtime=("openmpi-bin=$mpi" "libopenmpi3t64=$mpi" "libhypre-2.28.0=$hypre" "libmetis5=$metis")
for requirement in "${runtime[@]}"; do
  name="${requirement%=*}"
  observed="$(dpkg-query -W -f='${db:Status-Status} ${Version}' "$name" 2>/dev/null || true)"
  if [[ "$observed" != "installed ${requirement#*=}" ]]; then
    echo "Required Ubuntu runtime: $requirement (found '$observed')" >&2
    exit 1
  fi
done

cd "$root/packages"
extract_headers() {
  local name="$1" version="$2" digest="$3" deb="${1}_${2}_amd64.deb"
  if [[ ! -f "$deb" ]]; then
    apt-get download "${name}:amd64=$version"
  fi
  echo "$digest  $deb" | sha256sum --check --status || { echo "Digest mismatch: $deb" >&2; exit 1; }
  dpkg-deb -x "$deb" "$root/sysroot"
}
extract_headers libopenmpi-dev "$mpi" 0468721220ae0ec0969f12c41092a4dc370dbf2d14cbdb8b02aa43888f845acc
extract_headers libhypre-dev "$hypre" 311d73e61bb992f25fdcec990ad506118e751388c4d186624dc5921880620d7d
extract_headers libmetis-dev "$metis" 2cc70ea0fce0aa7d7601b5fccf9f645d8fb57786183c6a52db6766bc8f48b188

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 --branch v4.10 https://github.com/mfem/mfem.git "$source_dir"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" || -n "$(git -C "$source_dir" status --porcelain)" ]]; then
  echo "Requires clean MFEM source at $revision: $source_dir" >&2
  exit 1
fi
lib=/usr/lib/x86_64-linux-gnu
inc="$root/sysroot/usr/include"
mpi_inc="$root/sysroot$lib/openmpi/include"
cmake -S "$source_dir" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_INSTALL_PREFIX="$install" -DBUILD_SHARED_LIBS=OFF -DMFEM_PRECISION=double \
  -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DMETIS_VERSION_5=ON -DMFEM_USE_METIS_5=ON \
  -DMFEM_USE_CUDA=OFF -DMFEM_USE_HIP=OFF -DMFEM_USE_OPENMP=OFF \
  -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF \
  "-DMPI_CXX_COMPILER_INCLUDE_DIRS=$mpi_inc;$mpi_inc/openmpi" -DMPI_CXX_HEADER_DIR="$mpi_inc" \
  '-DMPI_CXX_LIB_NAMES=mpi_cxx;mpi' -DMPI_mpi_LIBRARY="$lib/libmpi.so.40" \
  -DMPI_mpi_cxx_LIBRARY="$lib/libmpi_cxx.so.40" \
  -DHYPRE_INCLUDE_DIRS="$inc/hypre" -DHYPRE_LIBRARIES="$lib/libHYPRE-2.28.0.so" \
  -DMETIS_INCLUDE_DIRS="$inc" -DMETIS_LIBRARIES="$lib/libmetis.so.5"
for feature in MPI METIS METIS_5 DOUBLE; do
  grep -qx "#define MFEM_USE_$feature" "$build/config/_config.hpp" || { echo "Missing $feature in build" >&2; exit 1; }
done
cmake --build "$build" -j "${JOBS:-2}"
cmake --install "$build"

python3 - "$install" "$revision" "$CXX" "${runtime[@]}" "libopenmpi-dev=$mpi" "libhypre-dev=$hypre" "libmetis-dev=$metis" <<'PY'
from hashlib import file_digest
import json
from pathlib import Path
from subprocess import check_output
import sys

install, revision, compiler, *requirements = sys.argv[1:]
prefix = Path(install)
def digest(file):
    with file.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()

header = prefix / "include/mfem/config/_config.hpp"
if f'g{revision}"' not in header.read_text() or '#define MFEM_VERSION_STRING "4.10.0"' not in header.read_text():
    raise SystemExit("Installed MFEM revision/version disagrees with the reviewed source")
record = {
    "schema_version": 1, "mfem_commit": revision, "mfem_version": "4.10.0",
    "packages": dict(item.split("=", 1) for item in requirements),
    "compiler": check_output([compiler, "--version"], text=True).splitlines()[0],
    "library_sha256": digest(prefix / "lib/libmfem.a"), "config_sha256": digest(header),
}
output = prefix / "share/mfem/mpi-toolchain.json"
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(f"MPI MFEM installed to {prefix}. Toolchain record: {output}")
PY
