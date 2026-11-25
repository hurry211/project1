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

## Measure a verifier change from submission to acceptance

The revision experiment keeps the MFEM executable, solver (CG with symmetric Gauss–Seidel), requests, mesh levels, tolerances and reference solve fixed. H1 stresses the large affine triangle mesh. H2 exercises transfer to a mixed Dirichlet–Neumann problem. Each revision runs a fresh native study in every block. The reported clock spans request preparation through the verified worker's exit. An additional process repeatedly verifies identical archived fields and measures the verifier's time and peak resident memory in isolation. [Live acquisition](../benchmarks/verified_revision.py) · [Archived-field measurement](../benchmarks/verification_cost.py)

Both comparisons discard one warm-up pair per workload and counterbalance the order of the two revisions across scored blocks. The paired statistic is the median within-block difference, candidate minus baseline. Each 95% percentile interval uses 4,096 resamples of complete pairs within the workload. In fresh studies, the numerical comparison retains every check, convergence value, export digest and native result field after normalizing the per-run directory and three native wall-clock fields. On an identical archived input, the entire verdict must have the same hash. The offline audit checks each inventory against its recorded manifest and repeats the independent numerical verification on representative archives. [Offline audit](../benchmarks/verified_revision_audit.py)

The verifier reads the element table into typed arrays. It evaluates expression and gradient temporaries in batches of up to 65,536 quadrature points and retains the quadrature densities for the original full-array summations. Ordered nodal layouts, quadrature weights and physical gradients must match bit for bit before the corresponding local computations can be reused. Variable diffusivity still receives pointwise integration. Global sparse assembly and the direct reference solve continue to scale with mesh size. [Finite-element data](../verifier/fem.py) · [Integration and reference solve](../verifier/verifier.py)

For a new acquisition using the historical baseline and candidate, export the pinned source trees from the repository. Build the matching serial native binary first and set its absolute path below. The original E2 design records binary SHA-256 `e2a75e0eff6dcbd40ae13515498f7d3a4fe3756b8bdef41aca126397bd8b6438`. Use a Linux filesystem for RUN.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export POISSON_MMS_BIN="$PWD/build/cpp/poisson_mms"
export THIRD_PARTY="$PWD/third_party"
CPU="$(python -c 'import os; print(min(os.sched_getaffinity(0)))')"
RUN="$(mktemp -d)"
mkdir "$RUN/baseline" "$RUN/candidate"
git archive 6b5761cf4765a8bc7028922348202d6b6cf12076 | tar -x -C "$RUN/baseline"
git archive ceb485cb9836c73ce846f62d49ca9591fcb83bfc | tar -x -C "$RUN/candidate"
cd "$RUN/candidate"
taskset -c "$CPU" python -m benchmarks.verified_revision \
  --baseline "$RUN/baseline" --candidate "$RUN/candidate" --cpu "$CPU" \
  --blocks 12 --seed 20261125 --directory "$RUN/live"
sha256sum "$RUN/live/design.json" "$RUN/live/report.json"
```

Retain those two SHA-256 values separately. Set DESIGN and REPORT to the respective digests when auditing the live records. To measure isolated verification on the newly generated fields, select the first scored baseline attempt of each workload. The published replay used the corresponding retained E1 attempts.

```bash
python -m benchmarks.verified_revision_audit --directory "$RUN/live" \
  --design-sha256 "$DESIGN" --report-sha256 "$REPORT" --output "$RUN/audit.json"
H1="$(find "$RUN/live/H1_000_baseline/studies" -type d -name attempt_01)"
H2="$(find "$RUN/live/H2_000_baseline/studies" -type d -name attempt_01)"
taskset -c "$CPU" python -m benchmarks.verification_cost \
  --baseline "$RUN/baseline" --candidate "$RUN/candidate" --attempt "$H1" --attempt "$H2" \
  --blocks 30 --seed 20261124 --expect-verdict ACCEPTED --output "$RUN/replay.json"
```

## Replaying historical E2 evidence on the current branch

The audit compares the complete saved verdict, including its version and diagnostic text. For the original E2 capture, run the audit from commit `ceb485cb9836c73ce846f62d49ca9591fcb83bfc` using the locked numerical environment. The design and report digests are independently published in the E2 receipt. Point `E2_LIVE` to the restored directory containing the original `design.json`, `report.json` and study folders. [Original audit receipt](../evidence/e2/live-audit.json)

```bash
# From the repository root, with the project's Linux Python environment active.
E2_CODE="$(mktemp -d)"
git archive ceb485cb9836c73ce846f62d49ca9591fcb83bfc | tar -x -C "$E2_CODE"
(cd "$E2_CODE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.verified_revision_audit --directory "$E2_LIVE" \
    --design-sha256 8ade49e156696eb5ab3b8548cb04af09a7f09e4facbd1fce92fa4b5215869fa9 \
    --report-sha256 221178e7fe3ea8e30e15a2fbd63bee84cd50f7a055b0c4b39f238f15d84dcc99 \
    --output "$E2_CODE/replay.json")
cmp "$E2_CODE/replay.json" evidence/e2/live-audit.json
```

On 2026-11-25 this procedure checked all 52 original inventories and replayed four representative fields. The complete audit was byte-identical to the published file. A separate replay using verifier 0.6.0 accepted the original H2 block-zero baseline with matching input hashes and numerical values. Its verdict differed from the saved one in the verifier version and four descriptions of the reference solve, one per mesh level. The historical verifier was 0.4.0. [Reproduction, source hashes and differences](../evidence/e4/e2-replay.json)

A fresh current-code functional control used two identical source trees as the two arms, compact C1, one warm-up pair and two scored pairs. All six records passed, the audit replayed both block-zero fields, and all runs had the same normalized numerical digest. The median paired wall-time difference for these identical arms was +0.229 s on the measured host. The run tests acquisition, independent replay and timing noise. Use `--workloads C1 --scale compact --blocks 2` with both revision arguments pointing at the current checkout to reproduce the control. [Current A/A receipt](../evidence/e4/e2-replay.json)

## Fresh E3 and E4 endpoints on the same binary

On 2026-11-25 both arms used the E4 controller and timing driver from `56eccf5` and the same serial MFEM binary (SHA-256 `fdbe3f731f7ad158d086b9cd4409111a29b7eb3f2a8511f0af32fd02999cf65b`). The baseline used the verifier tree from `07dca40`, and the candidate used the E4 verifier from `56eccf5`. Fresh H1 and H2 solves ran under identical requests on logical CPU 10, with one thread and no concurrent local solver jobs. Each workload had one excluded warm-up pair and four counterbalanced scored pairs. [Driver](../benchmarks/verified_revision.py) · [Timings, source hashes and numerical receipts](../evidence/e4/endpoint-e3-e4.json.gz)

| Workload | Verifier | Endpoint median (s) | Native processes (s) | Verification phases (s) | Other endpoint work (s) | Verifier peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H1, 263,169 DOFs | E3 | 19.117 | 9.611 | 8.585 | 0.843 | 674.9 |
| H1, 263,169 DOFs | E4 | 18.586 | 9.605 | 8.004 | 0.877 | 673.7 |
| H2, 16,641 DOFs | E3 | 1.917 | 0.471 | 0.745 | 0.703 | 133.0 |
| H2, 16,641 DOFs | E4 | 1.932 | 0.468 | 0.741 | 0.735 | 133.1 |

The endpoint starts at request preparation and ends after the worker exits with a qualified study. The native column covers all four native levels, including assembly, solve and export. The verification column sums the non-overlapping verification phases including the verifier process overhead. The remaining time covers startup, controller work and archiving. Column medians are computed independently and need not add to the endpoint median. The median of the maximum native process peak per trial was about 160 MiB for H1 and 47 MiB for H2.

All 20 studies passed the frozen accuracy, reference and process gates. The offline audits checked all 20 inventories and numerically replayed both block-zero fields of each workload. Each workload had one identical normalized numerical fingerprint across both verifier versions and all repetitions. The median within-block E4 minus E3 endpoint difference was **−0.595 s** for H1 (95% paired bootstrap interval **−0.992 to +1.564 s**) and **+0.007 s** for H2 (**−0.082 to +0.079 s**). Both intervals include zero. The lower observed H1 verification median therefore supports a phase-level observation, while the fresh endpoints leave the overall speed advantage unresolved at four pairs. H1 spent about half its endpoint in the unchanged native driver.

The raw archive also retains a separate interrupted initial H1 acquisition. Eight studies completed with matching numerical fingerprints, followed by a WSL kernel stall during the ninth worker. The local host was also running Gaussian and MPI work on other cores. That acquisition has no completed campaign score. After WSL recovered, the complete H1 and H2 schedules above ran from fresh directories.

## Facts resolved from the local revision

The benchmark host exposed 22 logical CPUs in WSL2. The serial measurements bound each study to one logical CPU. The MPI diagnostic used four logical CPUs on different physical cores. The MPI driver writes owned partitions on each rank and the Python verifier merges them. CG timing includes lazy BoomerAMG setup. The stored launcher time also covers partitioning, assembly, export and shutdown. These boundaries explain why a break-even rank count or problem size cannot be extrapolated from a single CG measurement. [MPI driver](../cpp/mpi/poisson_mpi.cpp) · [Rank-local diagnostics](../evidence/e1/mpi-phases.json)

The pinned MFEM 4.10 `GSSmoother` constructor used by the serial driver defaults to a symmetric sweep. The compiled case registry already accepts the cosine-exponential mixed-boundary problem on P1 triangles, and the retained Stage 2 scikit-fem comparison covers it with both preconditioners. [Serial construction](../cpp/poisson_mms.cpp) · [Matched field evidence](../evidence/stage2/coverage.json)
