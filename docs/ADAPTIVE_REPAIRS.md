# Verified adaptive selection on the E4 workstation

All six policies met the seven frozen task targets in each of three randomized repetitions. The comparison includes the original sharp Gaussian target of 0.00004 and the mixed-boundary target of 0.0000001. The revised fixed policy also passed every trial, which makes a matched time-to-solution comparison possible. Across the seven tasks the cost predictor and deterministic shortlist policy had confidence intervals spanning parity with the fixed control. Calling the local language model increased elapsed time on these workloads. [Comparison and per-trial evidence](../evidence/e4/adaptive-comparison.json)

## Numerical repairs

The native RHS, boundary and reported-error integration rules now use at least degree 20 on the registered medium and sharp Gaussian when the grid has at most 32 divisions. The independent verifier selects its rule from the frozen expression and uses at least 13 Gauss points per axis on those meshes. It also checks the order claimed by the native binary. The sharp Q3 Cartesian sequence with 8, 16 and 32 divisions achieved a finest L2 error of 0.00000745690 with 9,409 DOFs. Native and independent integrations satisfied the original self-report agreement checks.

The Q2 adaptive run uses a marking fraction of 0.6. The native driver checks true DOFs before solving each refined level. Across ten solved meshes it reached 2,209 true DOFs and L2 error 0.0000286546, inside the frozen 4,000-DOF budget and accuracy goal. The driver can terminate earlier when the next refined mesh exceeds the cap. The independent run checks the boundary partition, native case, order, solver, assembly, actual true DOFs and accuracy. [Focused numerical and failure regressions](../tests/test_adaptive_repairs.py)

## Policy and costs

The deterministic fixed control uses the requested topology with full assembly and CG plus Gauss–Seidel. Its fixed sequence tries order 3 through grid size 32, order 4 through 32, then order 3 through 64, subject to the original budgets. The predictor ranks eligible alternatives using measured time, error and memory. Its features distinguish partial assembly, Jacobi and unpreconditioned CG. Pilot measurements apply to the matching order, element, assembly and preconditioner. The deterministic advisor and live advisor share the same shortlist, retry sequence and fallback.

Four calibration tasks contributed 56 accepted measurements, including matched preconditioners and assemblers at the same order and mesh. These cost 79.77 s in total before fitting and are excluded from online policy timings. In-sample log10 RMSEs were 0.3011 for L2 error, 0.1708 for elapsed time and 0.0823 for peak memory. Generalization was assessed by running the full policies on five previous test tasks and two transfer tasks. The original sharp Gaussian informed the numerical repair. The two transfer tasks vary the boundary partition or the unstructured element and seed. These measurements cover one WSL2 host and one serial worker affinity.

| Policy | Accepted / trials | Median external seconds | Max worker-tree MiB | Fixed time / policy time (task-cluster 95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Fixed control | 21 / 21 | 1.635 | 202.82 | 1.000 by definition |
| Expert rules | 21 / 21 | 3.044 | 594.31 | 0.773 [0.232, 0.889] |
| Predictor | 21 / 21 | 1.847 | 202.71 | 0.989 [0.892, 1.120] |
| Pilot autotuner | 21 / 21 | 3.032 | 346.03 | 0.563 [0.243, 0.608] |
| Deterministic advisor | 21 / 21 | 1.624 | 203.01 | 1.058 [0.978, 1.115] |
| Live advisor | 21 / 21 | 4.407 | 203.13 | 0.413 [0.332, 0.470] |

The last column first pairs accepted trials on task, target and repetition, then takes the median ratio within each task and across tasks. Bootstrap samples keep all repeats of a task together. Ratios above one favor the named policy. For timing, the parent measures creation of the worker through its exit, including pilots, verification, decisions and unsuccessful retries. Each worker has a 120 s wall deadline and a 2 GiB per-process address-space cap. The peak estimate takes the largest of the Python worker high-water mark, child high-water mark and a 10 ms sample of total simultaneous RSS. Shared pages can count twice and a short simultaneous peak can fall between samples. External Ollama server memory is outside this measurement. Actual model inference and waiting time are included. The pinned Qwen2.5 model nominated a candidate in 21/21 trials. Each choice matched the deterministic leading candidate. Its median paired incremental cost was 2.277 s. [Live model ablation](../evidence/e4/advisor-ablation.json)

## Preconditioner and assembly search

The regular search CLI accepts an explicit solver list and uses the compiled catalog. Schema 0.4 requests can bind the additional research cases. Candidate execution still goes through the standard controller, which freezes inputs and independently verifies each field. Search records the seeded execution order, trial times, manifest pins, feasibility and matched time differences. With three repeats of the same sine problem, all fifteen Q1 runs missed their 0.0001 target, and all fifteen Q2 runs passed. Partial assembly plus Gauss–Seidel is capability-excluded. On this small case, median end-to-end times for Q2 were 1.268 s (full GS), 1.371 s (full Jacobi), 1.488 s (full unpreconditioned), 1.181 s (partial Jacobi) and 1.231 s (partial unpreconditioned). The selected configuration depends on the problem, hardware and timing variation. These results do not establish that partial assembly or Jacobi improve the historical 263,169-DOF experiment. [Search evidence](../evidence/e4/strategy-search.json) and [30 authenticated verdict replays](../evidence/e4/strategy-audit.json)

## Reproduce or replay

Use Linux or WSL2 with the pinned serial MFEM build and the project's Python environment. From the checkout, select a fresh location on a Linux filesystem. Enable a local Ollama server with the model and digest below to exercise the live arm.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export POISSON_MMS_BIN=/absolute/path/to/poisson_mms
export POISSON_AMR_BIN=/absolute/path/to/poisson_amr
taskset -c 4 python -m benchmarks.e3_adaptive --repeats 3 \
  --out-dir /tmp/e4-new/report --work-dir /tmp/e4-new/raw \
  --ollama-model qwen2.5:1.5b \
  --ollama-digest 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b
taskset -c 8 python -m sciagent search evidence/e4/strategy-request.json \
  --solvers cg_gs cg_jacobi cg_none --repeats 3 --output /tmp/e4-new/search
```

Select an available logical CPU on hosts with a different affinity layout. The raw campaign and regular CLI outputs remain on disk after execution.

For exact historical replay, obtain the release asset named in the tracked archive metadata, authenticate it and extract it into an empty Linux directory. The archive includes the compact comparison, every native field and verdict, the captured Python and C++ sources, and all 30 standard controller archives. The captured verifier preserves the check details used during measurement, including high-order near-roundoff diagnostics. The aggregate report and its root inventory are anchored outside the archive. [Archive digest](../evidence/e4/adaptive-archive.json) and [full replay receipt](../evidence/e4/adaptive-audit.json)

```bash
echo '731e22c4bd02086e1e24d174bf282858a6603b2e24128b69df285b6716902552  adaptive-evidence.tar.gz' | sha256sum -c -
mkdir restored
tar -xzf adaptive-evidence.tar.gz -C restored
cd restored/adaptive-final-src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m benchmarks.adaptive_audit --report ../adaptive-final-report/adaptive_comparison.json \
  --report-sha256 68d585c4b44c3cb3fc57c195926ef95889b85025115ad51001f2deca4c9c0e9f \
  --raw ../adaptive-final --all --workers 4
```

The recorded replay authenticated 12,048 files and all 126 strategy receipts. It independently reproduced 290 complete numerical verdicts, including calibration runs, pilot fields and unsuccessful intermediate attempts. Of these, 221 were accepted, 51 were rejected and 18 were retry-eligible. The endpoint for a strategy is its last qualified attempt.

## Local Q2 assembly control at 263,169 DOFs

This acquisition repeats the historical offset-Gaussian Q2 workload with CG–Jacobi at relative tolerance 1e-12, exact Dirichlet data, and grid divisions 64, 128 and 256. Both arms use the same three independent reference solves, convergence gates, and an explicit finest L2 target of 1e-6 from the historical Gaussian strategy search. The original assembly table used the same formulation and convergence checks with no additional absolute L2 target. The new run uses the E4 verifier on WSL2, a single worker bound to logical CPU 6 with one BLAS thread, and four blocks in the order partial/full, full/partial, partial/full, full/partial. The first block warms both paths. Each timing ends after writing the controller manifest. Three measured replications per arm were scheduled. Other host workloads can still share processor cache and memory bandwidth. [Original configuration and results](../evidence/scaling-stage/gaussian.json)

Before the first solve, the frozen E4 predictor selected full assembly from the two admissible Q2 choices. Its forecasts were 11.03 s for full, 11.75 s for partial and 257 MiB for each. The calibration topped out at 16,641 DOFs, so this 263,169-DOF prediction extrapolates beyond the fitted scale. The first matched pair already required about 1.25 GiB for the verifier. A kernel watchdog subsequently reported soft lockups on this WSL2 host, and Ubuntu required a restart. The entire local timing cohort is excluded from speedup and deployment decisions. Five of the eight scheduled runs produced authenticated accepted verdicts. The archive also contains the unfinished sixth run. Every slot appears in the receipt, and all 275 archived files were verified after extraction. [Interrupted local campaign and raw archive digest](../evidence/e4/gaussian-assembly-interrupted.json)

## Hosted Q2 full versus partial assembly

The same offset Gaussian was tested on an Ubuntu 24.04 GitHub Actions runner with an Intel Xeon 6975P-C. The workflow pinned the entire benchmark to logical CPU 0 with one BLAS thread. Both arms used Q2, CG–Jacobi, the same independent reference solves, and the original convergence checks. The workflow required finest L2 error at most 1e-6 and equivalent per-level L2 and H1 errors for each measured pair. The CLI exited successfully. The 16 runs include two warmups and six scored trials at each size. [Workflow and pinned acquisition source](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45616524974)

| Finest DOFs | Assembly | Assembly across levels (ms) | CG across levels (s) | Verification (s) | Verified total (s) | Native / verifier peak (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 66,049 | Full | 72.75 | 0.831 | 3.807 | 5.257 | 51.27 / 419.28 |
| 66,049 | Partial | 22.16 | 1.712 | 3.796 | 6.128 | 31.46 / 419.26 |
| 263,169 | Full | 308.52 | 7.607 | 18.786 | 28.632 | 178.55 / 1,286.65 |
| 263,169 | Partial | 88.67 | 14.697 | 18.752 | 35.618 | 96.92 / 1,287.61 |

Entries are medians across three measured pairs at each size. The paired full/partial end-to-end time ratios were 0.860, 0.857 and 0.861 at 66,049 DOFs and 0.808, 0.808 and 0.804 at 263,169 DOFs. The two assemblers took the same number of finest-level CG iterations (849 and 1,677 at the two sizes), with verified finest L2 errors of approximately 4.68e-7 and 5.85e-8. Partial assembly saved native memory and assembly time, while its more expensive CG iterations increased the time to a verified answer. Full assembly was faster in every measured pair on this host. The predeclared local predictor also nominated full at 263,169 DOFs, but its 11 s absolute forecast substantially underestimated both hosted medians. Generalization to other preconditioners, accelerators or smaller memory limits requires measurements under those conditions.

The released tarball retains the runner's original bytes and SHA-256. The public receipt contains the source commit, GitHub artifact digest, archive digest, 16 individual manifest pins, paired measurements and four independent cross-host replays. Replays use the final full/partial pair at each size and compare acceptance, individually named numerical checks, thresholds and per-level errors to the stored records. [Hosted summary and replay receipts](../evidence/e4/gaussian-hosted.json)

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0 python -m benchmarks.performance --gaussian --repeats 3 \
  --output /tmp/gaussian-reproduction
```
