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

## MPI launch and phase measurements

The optional diagnostic times the same Open MPI launcher with the same executable in two modes. The null control starts MPI, runs one barrier and finalizes. Numerical trials run the P1 problem with BoomerAMG. Each rank measures mesh construction and partitioning, finite-element assembly and solver setup, the pre-solve barrier, CG including lazy AMG setup, postprocessing and collectives, and rank-local field export. Every numerical trial must pass the independent serial-field and SciPy checks. The Python clock measures the launcher call and the independent verification. Rank-phase tables report the maximum within each phase. Different ranks can attain those maxima, so adding the columns can exceed the actual longest rank window. [Experiment](../benchmarks/mpi_phases.py) · [Native rank clocks](../cpp/mpi/poisson_mpi.cpp)

In WSL2, eight measured blocks plus one discarded warm-up block interleaved three null-launch controls and six numerical configurations. The launcher used `--bind-to none`. Each rank recorded the inherited CPU mask of 12, 16, 18 and 20, representing four physical cores on this host. Each process used one numerical thread. Median null-launch wall time rose by 250 ms from one to four ranks. At 64 cells per side the verified workflow increased by 209 ms, even as the longest rank-local numerical window fell by 27 ms. The null result demonstrates substantial rank-dependent cost outside numerical work on this host. Separate samples and non-additive medians preclude assigning an exact fraction of the increase to startup. [Every sample, environment and replay digest](../evidence/e1/mpi-phases.json)

| MPI ranks | Null launch (ms) | Verified workflow at n=32 (ms) | Verified workflow at n=64 (ms) | Longest rank window at n=64 (ms) | CG including lazy AMG setup at n=64 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 402.26 | 458.29 | 585.09 | 68.25 | 4.85 |
| 2 | 464.11 | 513.60 | 606.46 | 42.88 | 4.68 |
| 4 | 652.72 | 674.94 | 793.71 | 41.22 | 5.07 |

The capture passed 12 convergence screening solves and 54 individually verified phase-profiled solves including warm-ups. Digest-pinned replay verified the campaign with the native executable paths set to unavailable locations. To repeat on another Linux system, select four available physical cores and use a fresh output directory on the Linux filesystem. `CPU_LIST` holds their logical CPU numbers. `SHA` holds the receipt digest.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/build_solver_mpi.sh
taskset -c "$CPU_LIST" python -m benchmarks.mpi_phases --directory runs/mpi-phases --repetitions 8
python -m benchmarks.mpi_phases --directory runs/mpi-phases --replay-sha256 "$SHA"
```
