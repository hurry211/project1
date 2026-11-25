# Serial and parallel MFEM build profiles

The Stage 3 MPI build targets x86-64 Ubuntu 24.04, including the Ubuntu 24.04 WSL2 distribution. MFEM uses double precision, MPI, hypre and the METIS 5 partitioning API. The MPI installation has its own prefix alongside the existing serial build. The solver and archive checks use the observed build flags, package versions and artifact digests.

## Provision the parallel runtime

Run these commands in the Ubuntu shell with the repository as the working directory. The Ubuntu `universe` package index supplies the numerical packages. On a new machine, provision the following packages once.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git gcc-13 g++-13 cmake make python3 util-linux \
  openmpi-bin=4.1.6-7ubuntu2 libopenmpi3t64=4.1.6-7ubuntu2 \
  libhypre-2.28.0=2.28.0-8build2 libmetis5=5.1.0.dfsg-7build3
```

## Build the MPI installation

```bash
export CC=gcc-13 CXX=g++-13 JOBS=2
export MPI_THIRD_PARTY="$PWD/third_party/mpi"
export MFEM_MPI_INSTALL="$MPI_THIRD_PARTY/mfem-mpi-install"
bash scripts/build_mfem_mpi.sh
cat "$MFEM_MPI_INSTALL/share/mfem/mpi-toolchain.json"
```

Set `MPI_THIRD_PARTY` to an absolute path on the Linux filesystem to keep compilation there when the repository lives on a mounted Windows drive. Keep this directory after installation. The MFEM CMake export references the versioned development headers in its `sysroot`. The first build downloads the three Noble development packages, checks their SHA-256 digests and extracts the headers locally. Repeated builds reuse the verified archives. To reuse an existing clean MFEM checkout, set `MPI_MFEM_SOURCE` before running the build. The required commit is `d964264cdb9a13e94a201b6c236c7721e0c8765f`. [Build script and package checksums](../scripts/build_mfem_mpi.sh)

The build checks the generated header for MPI, METIS, the METIS 5 interface and double precision. Specifying only the METIS library and include directories in the pinned MFEM release can leave the legacy partitioner interface selected. The reviewed configuration sets and checks the v5 ABI explicitly. The toolchain record contains the MFEM source revision, compiler banner, seven package versions, static archive digest and installed configuration-header digest. Archive digests identify an individual build. The package pins cover the direct parallel dependencies. Ubuntu resolves their transitive runtime packages from its enabled archives.
