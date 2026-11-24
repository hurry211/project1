# Matched serial and MPI Poisson studies

The `mpi-study` command solves the existing sine Poisson problem on generated P1 triangles with homogeneous Dirichlet data. The serial control uses MFEM CG with Gauss–Seidel. The MPI driver uses a fully assembled HypreParMatrix and MPI CG with hypre BoomerAMG. Both use the same mesh connectivity, source, boundary data, relative tolerance of `1e-12`, zero absolute tolerance and maximum of 20,000 iterations. Changing the preconditioner can change the iteration count. [Serial request](../specs/poisson_sin_sin_p1_tri.json) · [Parallel assembly](../cpp/mpi/poisson_mpi.cpp)

## Run and replay

Use the Linux Python environment and serial executable from the README. Provision the separate parallel installation, then build its driver and run the paired experiment on a host with at least four available MPI slots. Each campaign takes a fresh output directory. Use a Linux filesystem for the raw exports when measuring performance. [MPI build instructions](MPI_BUILD.md)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash scripts/build_solver_mpi.sh
python -m sciagent matrix --backend mpi_cpu
python -m pytest -q --require-mfem --require-mpi
python -m sciagent mpi-study --directory runs/mpi-study --repetitions 8 > mpi-receipt.json
digest="$(python -c "import json; print(json.load(open('mpi-receipt.json'))['manifest_sha256'])")"
python -m sciagent verify-mpi runs/mpi-study --manifest-sha256 "$digest"
```

Keep the receipt separately from the archived directory. Replay requires the Python environment and the whole campaign tree. It checks file digests, reruns the serial verifier, reconstructs every MPI field and recomputes the statistics using the captured timing records. Replay can run on a different path with the native executables unavailable. The `mpi-study` entry point defines the paired experiment. Existing `run` requests continue to use their serial capability contract. [Orchestrator](../sciagent/mpi_study.py)

## Acceptance before timing

1. Reproduce the frozen serial benchmark at 8, 16, 32 and 64 cells per side, then reverify its archive using its digest.
2. Run all four resolutions on 1, 2 and 4 ranks. Each process exports its owned triangles, vertex values and physical boundary edges. Coordinates map local nodes to the serial lattice. Every triangle must occur once, boundary edges must belong to their owning cells, and repeated nodal values must agree within `1e-12`.
3. Check the merged field against the matched serial field and a SciPy direct solve. The verifier derives the five-point interior stiffness matrix for this triangulation and integrates the source separately. It independently computes L2 and H1 errors using a Duffy transformation and the analytic gradient. It also reconstructs the constrained algebraic residual with the native load quadrature rule. The original field, norm and order gates apply. [Verifier](../verifier/mpi.py) · [Field, mesh and claim regressions](../tests/test_mpi_verifier.py)
4. Start timing after all twelve screening runs pass. Each timed solve also passes the full per-level field checks before it contributes to the report. A failed sample ends the campaign with a failed manifest. [Protocol regressions](../tests/test_mpi_protocol.py)

## Reading the performance measurements

Strong scaling fixes `n=64`, or 8,192 global triangles, and divides the one-rank median time by each rank count's median time. Weak scaling compares `n=32` on one rank with `n=64` on four ranks. Both allocate an average of 2,048 triangles per rank. The reported weak efficiency is the one-rank time divided by the four-rank time. True DOF counts and the distribution of shared nodes vary with resolution and partitioning.

The runner rotates the order of the four benchmark configurations across repetitions. The `r4_n64` sample set is used in both comparisons. The native solve measurement is the maximum rank time in CG, including lazy preconditioner setup. Launcher wall time includes process startup, the replicated input mesh, METIS partitioning, assembly, solving, export and shutdown. Per-sample workflow time additionally includes the Python checks of the parallel field and the matched serial field. The receipt's wall time includes environment probing, creation and replay of the serial control, screening, all timing samples, verification and final inventory hashing. Timing measurements describe one host, MPI binding policy and workload. [Measured results](RESULTS.md#stage-3-mpi-poisson-2026-11-24)
