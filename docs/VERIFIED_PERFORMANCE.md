# Time to an independently verified MFEM solution

The performance selector chooses between two preconditioners for the same scalar Poisson operator and must deliver a field accepted by the independent numerical verifier. The benchmark measures the time spent to achieve that result, including unsuccessful attempts. The selection command accepts a previously scored local calibration campaign and records its decision before launching the final study. [Paired campaign](../benchmarks/e1.py) · [Selection policy](../benchmarks/e1_analysis.py) · [Live selector](../benchmarks/e1_select.py)

## Experimental contract

Both arms use CG, P1 triangles, full assembly, double precision, a positive scalar operator with essential Dirichlet constraints, and the same native binary. The preconditioner is the only solver change. The allowance is one full study per trial, with a 300 s limit per native level and an external 900 s limit for the whole trial. For a finest mesh of size `n`, both arms must pass the original convergence, boundary, reference and self-report checks, plus an L2 error at most `2/n²` and an H1 seminorm error at most `5/n`. The latter is a benchmark qualification against the independently integrated error. A timeout, a failed full verdict or an unavailable required reference excludes the candidate from the winners.

| Workload | Purpose | Manufactured case | Mesh levels | Finest DOFs |
|---|---|---|---|---:|
| C1 | Calibrate | Sine, homogeneous Dirichlet | 8, 16, 32, 64 | 4,225 |
| C2 | Calibrate | Sine, homogeneous Dirichlet | 16, 32, 64, 128 | 16,641 |
| H1 | Test size extrapolation | Sine, homogeneous Dirichlet | 64, 128, 256, 512 | 263,169 |
| H2 | Test case transfer | Cosine-exponential, mixed boundary data | 16, 32, 64, 128 | 16,641 |

Each block runs GS, Jacobi and a second GS control in randomized order on one Linux logical CPU. Three warm-up blocks precede 30 scored blocks for each workload. Every run uses a fresh process and directory, with one thread for BLAS and OpenMP. The endpoint includes process startup, environment and spec checks, native computation and export, independent verification, and study archive creation. Corpus scheduling, deduplicating finalized identical artifacts and a separate offline replay fall outside each arm's elapsed time. Process peak RSS is a process-lifetime high-water mark. Native, worker and MPI peaks cannot be added to estimate simultaneous memory use.

The measured capture used logical CPU 4 on one WSL2 host. Related build, regression and diagnostic jobs used other cores during parts of the experiment. All arms shared the same filesystem and host resources. The rotated order and A/A control expose some of the resulting variation. Timing conclusions are specific to this host and workload.

The primary estimate is the median paired difference in verified wall time (Jacobi minus the first GS arm). The 95% percentile interval resamples 30 complete blocks, separately for each workload. The prespecified practical margin is the larger of 20 ms and 5% of the GS median. A signed A/A interval checks ordering bias. A supplementary sensitivity check measures the magnitude of A/A variation and can veto a switch. Report failures, incomplete blocks and their costs alongside qualified-pair timings. [Scoring and noise diagnostics](../benchmarks/e1_analysis.py)

## Run and replay

Use the Ubuntu serial MFEM environment in the README. Capture on a Linux filesystem. Choose an available logical CPU for `CPU` and a fresh `RUN` directory. The calibration and holdout cases run sequentially. The full experiment is a long-running measurement. For a functional check, add `--scale compact --warmups 0 --repetitions 1`. This short run exercises the workflow.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$CPU" python -m benchmarks.e1 --directory "$RUN" --cpu "$CPU" \
  --scale memo --seed 20261124 --warmups 3 --repetitions 30 --compact-archives
sha256sum "$RUN/campaign.json"
python -m benchmarks.e1_analysis --output scored.json analyze "$RUN/campaign.json"
python -m benchmarks.e1_audit --directory "$RUN" --campaign-sha256 "$DIGEST" --output replay.json
```

Set `DIGEST` to the separately retained SHA-256 from capture. The audit checks the schedule, journal, worker receipts and artifact inventories for every eligible study. By default it numerically replays the first eligible scored study of every workload and arm. Add `--all` to repeat full numerical verification for every eligible study including warm-ups. The archive records per-level iterations, residuals, independently computed errors, binary and manifest digests, process wall times and peak RSS. Native timers overlap the native process envelope. The current worker additionally captures cold startup, preparation, post-verification qualification and exit timing at cross-process boundaries. The original E1 capture recorded that outer portion as a single remainder. [Timing boundary regression](../tests/test_e1_phase_budget.py)

## Use the calibrated decision

```bash
taskset -c "$CPU" python -m sciagent select-solver --workload H2 \
  --calibration-directory "$RUN" --campaign-sha256 "$DIGEST" \
  --directory runs/selected-mixed --wall-budget 300
```

The selector fits its policy on C1 and C2 only. The environment and frozen case determine which solvers are feasible. Source fingerprints, Python and numerical library versions, MFEM build identity, hardware, thread constraints and the reviewed memory envelope bound the transfer of timing information. The reference DOF cap also applies. A materially faster Jacobi calibration, consistent timing and a projected saving larger than pilot cost and the practical margin are needed for an automatic switch. Insufficient calibration supports the fixed default. An explicitly authorized single solver can still be used when it is eligible. The policy cannot alter the PDE, relax an accuracy target or turn a pilot field into an accepted solution.

Use `--measure-pilots` to execute both preconditioners on the first two levels and charge their wall time even if calibration selected the fixed default. The forecast uses iteration growth, time per sweep and the calibrated solve-time scale. The raw pilot fields and the decision are retained. In prospective runs the decision and all pilots share the final study's external time budget. The receipt points to the full study and records the decision digest. A one-off live run establishes that the policy can execute. Repeated paired runs supply the performance comparison.

## Lower-cost verification and other comparisons

The full verifier reuses a basis only for exactly identical ordered reference coordinates and computes the Jacobian of each affine triangle once. It preserves quadrature-point Jacobians for quadrilaterals. It locates Neumann edge owners using the bounds of the exported boundary segments and continues to check the actual endpoints. The paired verifier benchmark runs the older and current verifier on identical retained fields and requires equal hashes of the **complete verdict** for every timed pair. [Verifier measurement](../benchmarks/verification_cost.py) · [Element and boundary regressions](../tests/test_fem_quadrature.py)

A separate experiment times a cheap rejection screen alongside full verification on 208 existing numerical challenge trials. It reads solver stopping and quadrature metadata. Every candidate that passes this screen still requires full verification. Clean controls, missed faults, runtime and estimated cost with escalation are reported separately. The screen has no authority to accept a result. [Screen experiment](../benchmarks/screen_cost.py) · [Challenge protocol](VERIFIER_CHALLENGE.md)

For assembly comparisons, the program compares Q1 and Q2 full and partial assembly with the same Jacobi action. The independent reference covers three manufactured cases and both generated and imported meshes. For parallel comparisons the matched serial control, one-rank MPI baseline, empty MPI launches and per-rank timers distinguish numerical work from process costs on the tested host. [Assembly experiment](../benchmarks/assembly_comparison.py) · [MPI measurements and commands](MPI.md)

## Facts resolved from the local revision

The benchmark host exposed 22 logical CPUs in WSL2. The serial measurements bound each study to one logical CPU. The MPI diagnostic used four logical CPUs on different physical cores. The MPI driver writes owned partitions on each rank and the Python verifier merges them. CG timing includes lazy BoomerAMG setup. The stored launcher time also covers partitioning, assembly, export and shutdown. These boundaries explain why a break-even rank count or problem size cannot be extrapolated from a single CG measurement. [MPI driver](../cpp/mpi/poisson_mpi.cpp) · [Rank-local diagnostics](../evidence/e1/mpi-phases.json)

The pinned MFEM 4.10 `GSSmoother` constructor used by the serial driver defaults to a symmetric sweep. The compiled case registry already accepts the cosine-exponential mixed-boundary problem on P1 triangles, and the retained Stage 2 scikit-fem comparison covers it with both preconditioners. [Serial construction](../cpp/poisson_mms.cpp) · [Matched field evidence](../evidence/stage2/coverage.json)
