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

## Larger strong-scaling campaign with E3 verification

The E4 experiment solves the same sine P1 problem at 128, 256 and 512 cells per side. The finest grid contains 524,288 triangles and 263,169 DOFs. The reference for every MPI field uses the independent five-point stiffness derivation, chunked high-order error/load integration and the E3 residual-checked SA-AMG-PCG solver. The quadrature workers process disjoint batches under a bounded thread pool and reduce contributions in canonical cell order. The E3 serial verifier separately assembles and verifies the serial control. Rank ownership, the exported field, boundary traces, native stopping claims, both error norms, accuracy targets, reconstructed residual and finest-pair convergence are mandatory. The two final targets are L2 at most `2/512²` and H1 seminorm at most `5/512`. [MPI verifier](../verifier/mpi.py) · [AMG residual checks](../verifier/reference_solver.py)

The compact driver partitions actual Cartesian element centroids into 1×1, 2×1 or 2×2 tiles. Each rank writes every local vertex's coordinates and field value once as three little-endian binary64 numbers. Three little-endian int32 local indices encode each owned triangle. Physical boundary edges retain their existing text format. A separate legacy arm uses METIS and the original text exports with the same streaming verifier. The single-node launcher starts once per mesh level. Each rank still constructs the input Cartesian mesh before building its partition. This leaves a replicated-memory cost at larger node and rank counts. [Native partition and export](../cpp/mpi/poisson_mpi.cpp)

Each arm runs all three levels with one BLAS thread. The serial arm uses one physical core, and each MPI arm uses the corresponding number of physical cores from the same set. The measured interval includes native process launch, assembly, solve, export, independent numerical checks, convergence and hashing the raw trial files. Each MPI trial re-reads an accepted serial control. The campaign records the initial cost of producing and validating that shared control separately. A warm-up block precedes five scored blocks. The seeded arm order rotates to cover each within-block position once. The intervals resample complete paired differences. The sample size and shared WSL2 host limit interpretation. Serial GS and MPI BoomerAMG use different preconditioners and export formats. The MPI one-rank comparison isolates strong scaling within the parallel implementation. [Capture, selection and replay](../benchmarks/mpi_scaling.py)

From the serial and MPI installations described above, choose four logical CPU IDs on different physical cores and an empty output path on a Linux filesystem. Use the SHA-256 printed by capture as PIN. The offline replay rechecks the control and every trial. Prospective selection checks the pinned calibration, executable, hardware, loaded code and numerical libraries, records the decision, then runs and independently verifies a new study. It considers a change from serial only when the candidate's upper paired interval shows at least 5% lower verified time with at least five scored pairs.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$CPUS" python -m benchmarks.mpi_scaling --directory "$RUN" --scale large
taskset -c "$CPUS" python -m benchmarks.mpi_scaling --directory "$RUN" --replay-sha256 "$PIN"
taskset -c "$CPUS" python -m sciagent select-mpi --calibration "$RUN" --report-sha256 "$PIN" --directory "$NEW_STUDY"
```

## Measured E4 results

The large calibration used logical CPUs 12, 14, 16 and 18 on separate physical cores. Each row gives medians across five scored studies. Each study verified one field at each of the three mesh sizes. Wall time for the prospective choice also includes the decision itself. [Design and source hashes](../evidence/e4/mpi-design.json) · [Paired comparisons](../evidence/e4/mpi-report.json) · [Phase and field comparison results](../evidence/e4/mpi-summary.json)

| Large study | Native process time (s) | Entire verified study (s) |
|---|---:|---:|
| Serial, GS | 12.821 | 25.317 |
| 1 rank, BoomerAMG | 7.486 | 15.933 |
| 2 ranks, BoomerAMG | 6.207 | 14.712 |
| 4 ranks, BoomerAMG | 4.643 | 11.064 |
| 4 ranks, BoomerAMG, METIS and text exports | 5.720 | 12.651 |

Four compact-output ranks achieved **1.44× verified strong scaling** over one. The median paired difference was −5.53 s, with a 95% paired bootstrap interval of −10.72 to −1.96 s. The two-rank interval included zero. Comparing four ranks to the serial GS configuration gave a 2.29× ratio of medians. That comparison includes the effects of the different solver, preconditioner, field format and verification implementation. Across every scored compact four-rank level, the maximum relative field difference from the independent AMG reference was 1.44 × 10⁻¹³ and from the serial control was 1.48 × 10⁻¹³. [Per-trial numerical results and artifact digests](../evidence/e4/mpi-trials.jsonl.gz)

At four ranks, the median of the summed rank-maximum export times across the three meshes decreased from 0.426 s (text) to 0.014 s (compact). The median end-to-end difference for this combined partition and export change was −1.77 s, with a 95% paired interval of −7.94 to +2.35 s. The five paired blocks resolve the four-versus-one scaling effect more clearly than the difference between the two four-rank export paths. [Measured phase summaries](../evidence/e4/mpi-summary.json)

The policy selected four ranks for a fresh large study. It spent 0.685 s validating the calibration and recording the decision, then completed all three verified levels in 16.646 s (17.331 s including the decision). The raw bundle retains the executable pair, frozen versions of the tracked timing and numerical modules, control, six complete blocks, selected field and receipt. Later changes to a pinned module require fresh calibration for a new production decision. The replay command can check archived fields from these frozen experiments independently of the native executables. [Decision](../evidence/e4/mpi-decision.json) · [Fresh-study receipt](../evidence/e4/mpi-selection.json) · [Archive digest and size](../evidence/e4/mpi-summary.json)

A second five-block calibration used the release source tree, meshes of 32, 64 and 128 cells per side, and logical CPUs 6, 8, 10 and 20. Median verified times were 1.124 s for serial, 2.169 s for one rank, 2.134 s for two ranks, 2.614 s for four ranks and 2.824 s for the four-rank text export. The paired four-rank minus serial interval was +0.20 to +1.63 s. The public `select-mpi` command chose serial, verified the fresh study in 2.239 s and returned an accepted receipt in 2.595 s including the decision. Independent replay reproduced all 31 calibration studies and the selected study. CPU sets and capture times differ between campaigns, so the two size points establish a policy choice on each tested setup without estimating a precise crossover size. [Release design](../evidence/e4/mpi-small-design.json) · [Paired statistics](../evidence/e4/mpi-small-report.json) · [Selected study](../evidence/e4/mpi-small-selection.json) · [Raw archive digest](../evidence/e4/mpi-small-summary.json)

Digest-pinned numerical replay checked 31 calibration studies for each size range, including the shared controls and warm-ups. Both prospective selections also passed a separate replay of their artifacts, levels, convergence and timing records. The 49 focused MPI tests passed with the native executable required. [Replay and test record](../evidence/e4/mpi-replay.json)

This WSL2 campaign reserved four physical cores and addressed one host. The host exposes a Windows Intel Arc graphics adapter through a D3D12 bridge. Both MFEM installations used here were CPU builds. The scaling figures cover single-node CPU solves and parallel field verification. [Hardware and build inventory](../evidence/e4/mpi-hardware.json)
