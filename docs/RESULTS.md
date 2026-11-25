# Numerical results

## E3.1: Strict Paired Scaling to >1M DOFs, AMR Hardening & Risk-Aware Decision

Branch `research/e3-scaling-adaptive` (`07dca409`) extends E2 (`ceb485cb`) cleanly ([Full E3.1 Report](E3_PERFORMANCE_ADAPTIVE.md) · [Advisor Report](导师阶段汇报-第三阶段-E3.1.md) · [Raw Evidence](../evidence/e3/)):

- **Strict 4k -> 1.64M DOF Paired Scaling** (`scaling_4k_to_1m.json`, 5 randomized blocks under `RLIMIT_AS=4096 MiB`): Frozen E2 direct hits OOM above `~590k` DOFs, whereas E3 `e3_sa_amg_pcg` verifies `1,050,625` DOFs in `70.30 s` (`1507.5 MiB`) and mature `e3_pyamg` (`pyamg==5.4.1`) verifies `1,050,625` DOFs in `66.41 s` (`1182.2 MiB`) and reaches `1,640,961` DOFs (`n=1280`, `5/5` accepted, `122.26 s`, `1833.8 MiB`).
- **Independent AMR & Complex Meshes** (`complex_meshes_and_amr.json`): Unstructured Delaunay/quad variable-`kappa` `P1–P4`/`Q1–Q4`, strict non-zero Neumann Ex6 AMR & `lshape_2d` verified from frozen specs (`1e-6` self-report tol).
- **Separated Risk-Aware Adaptive Search** (`adaptive_comparison.json`, 6 calibration -> 6 held-out × 5 repeats): `hybrid_guardrail_predictor` achieves `30/30` completion rate (`100.0%`, failed cost `0.00 s`) vs strong `fixed_baseline` Q2 (`15/30`), with accepted-only paired speedup `2.435x` across `15` mutually accepted pairs.

## Verifier revision at 263169 degrees of freedom

On a single WSL2 host (Intel Core Ultra 7 155H), each pair used the same P1 triangle problem, frozen accuracy targets, MFEM binary, preconditioner, thread limits and independently assembled reference solve. H1 used four mesh levels up to 512 cells per side and 263,169 DOFs. H2 used mixed boundary conditions up to 128 cells per side and 16,641 DOFs. [Protocol and commands](VERIFIED_PERFORMANCE.md#measure-a-verifier-change-from-submission-to-acceptance) · [Frozen design](../evidence/e2/live-design.json)

**Fresh MFEM studies.** One warm-up pair and 12 scored pairs per workload yielded 52 independently accepted studies. The numerical fingerprints matched across every run of each workload after normalizing the per-run location and three native timers. The offline audit checked all 52 inventories, replayed the first scored study from each revision and workload, and reproduced every reported score. Times cover submission through the verified worker’s exit. Δ is the median within-block difference (updated minus previous). [Per-study records](../evidence/e2/live-report.json.gz) · [Offline audit](../evidence/e2/live-audit.json)

| Workload | Previous (s) | Updated (s) | Median paired Δ (s) | 95% paired interval (s) | Change in median time |
|---|---:|---:|---:|---:|---:|
| H1 | 30.812 | 23.714 | -7.643 | [-8.377, -6.892] | -23.0% |
| H2 | 2.370 | 2.047 | -0.310 | [-0.516, -0.178] | -13.6% |

**Verifier replay on identical fields.** Each workload supplied 30 scored pairs plus one warm-up pair. The verifier returned a byte-identical serialized verdict within every pair, including every error norm, threshold, check and input hash. Peak RSS is the fresh Python worker’s lifetime high-water mark. The replay clock covers the independent verification call. [All timings, memory readings and hashes](../evidence/e2/replay.json)

| Workload | Previous (s) | Updated (s) | Ratio of medians | 95% paired Δ (s) | Previous peak (GiB) | Updated peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 18.499 | 11.696 | 1.58× | [-7.189, -6.706] | 2.417 | 1.183 |
| H2 | 0.775 | 0.478 | 1.62× | [-0.300, -0.283] | 0.207 | 0.128 |

The median live H1 phases changed from 3.58 to 1.42 s for field loading, 8.71 to 4.19 s for error integration and 6.17 to 5.47 s for the independent reference. The revision bounds temporary expression arrays, reuses exactly repeated element work and releases assembly buffers before the sparse direct solve. Global quadrature densities and sparse factors still consume memory proportional to the problem or factor size. The timing jobs used separate physical cores and shared the host and memory system with one another and some regression jobs. The intervals describe within-session variability. Cross-host reproducibility of the speedup remains open.

**Prospective solver decisions.** The updated source completed and numerically replayed all 198 calibration trials on C1 and C2. In C1 the A/A timing noise exceeded the calibrated margin, and in C2 the two preconditioners were within the practical margin. Both default holdout runs selected CG with symmetric Gauss–Seidel and independently passed using the ordinary 3,072 MiB process address-space cap. The optional four H2 pilot solves forecast Jacobi. The policy retained GS because calibration did not support an automatic switch. All three final fields passed pinned numerical replay. A fourth mixed-boundary study supplied the older calibration. The selector detected the two changed verifier sources, disabled automatic switching and delivered another accepted field. [Earlier calibration control](../evidence/e2/stale-selection.json) · [Decision receipts](../evidence/e2/selection.json) · [Calibration design](../evidence/e2/calibration-design.json) · [198 observations](../evidence/e2/calibration-campaign.json.gz) · [Calibration audit](../evidence/e2/calibration-audit.json.gz)

**Regression coverage.** The strict Python, serial MFEM and MPI suite passed 421 tests with zero skips. All 31 original study states and retry decisions matched, including 158 error values and 83 sets of byte-identical field exports. The 284 challenge trials and 14 independent reference-checker controls reproduced the preceding revision’s decisions and scores. The 42-row scikit-fem comparison and both Stage 4 replay reports matched their published bytes. The preceding E1 audit checked all 396 inventories and recomputed 12 representative studies, and the MPI replay passed at 1, 2 and 4 ranks. [Validation records](../evidence/e2/validation.json) · [Challenge trials](../evidence/e2/challenge.json.gz) · [Planning replays](../evidence/e2/planning-replay.json)

## Verified performance experiments

The end-to-end paired preconditioner study uses real MFEM and independent accuracy targets on one pinned CPU. It compares the reviewed Gauss–Seidel default with Jacobi and includes an independently executed second Gauss–Seidel arm in each block to measure timing noise. The preregistered workload definitions and digests were recorded before the first timed run. [Protocol and commands](VERIFIED_PERFORMANCE.md) · [Acquisition configuration](../evidence/e1/paired-design.json)

**Preconditioners at matched accuracy.** The campaign completed 396 independently accepted runs. Each of four workloads supplied 30 scored three-arm blocks and three discarded warm-up blocks. Times below are seconds. The difference and interval use within-block subtraction, so subtracting the displayed marginal medians can give a different value. [Paired scores and uncertainty](../evidence/e1/paired-analysis.json) · [Compressed per-run records](../evidence/e1/paired-campaign.json.gz)

| Workload and finest size | GS median | Jacobi median | Median paired difference | 95% paired interval | Practical margin | Primary result |
|---|---:|---:|---:|---:|---:|---|
| C1, 64 | 1.068 | 1.083 | -0.003 | [-0.028, +0.033] | 0.053 | Within margin |
| C2, 128 | 1.969 | 1.977 | -0.013 | [-0.035, +0.038] | 0.098 | Within margin |
| H1, 512 | 28.473 | 27.805 | -0.850 | [-1.665, -0.217] | 1.424 | Inconclusive |
| H2, 128 | 2.292 | 2.209 | -0.061 | [-0.160, +0.022] | 0.115 | Inconclusive |

On H1 the paired interval favors Jacobi relative to zero, but it includes savings smaller than the prespecified 1.42 s practical margin. At 263,169 DOFs the finest solves took 467 iterations with GS and 845 with Jacobi. Across all four levels of this workload the median native CG times were 3.80 s and 3.01 s respectively. Full numerical verification occupied a median 62.0% of external wall time. The median peak RSS of the controller/verifier worker was 2.42 GiB. On H2, the median absolute A/A difference was 150 ms and 17/30 A/A differences exceeded the margin. The absolute-noise sensitivity analysis was added after these data were seen.

**Prospective selection.** C1 and C2 supplied no material Jacobi advantage, so the policy retained GS and skipped pilots by default. A separately pinned calibration report contained only the 198 C1/C2 records, including warm-ups. Its source attestation reproduces the implementation and environment used for the measured verifier. Four subsequent full CLI invocations used the two held-out workloads with and without compulsory diagnostic pilots. The decision was recorded before execution. All four results passed an additional independent numerical replay. [Decisions, charged costs and study digests](../evidence/e1/live-selection.json)

| Held-out workload | Default CLI (s) | CLI with forced pilots (s) | Four pilots alone (s) | Selected solver |
|---|---:|---:|---:|---|
| H1 | 32.612 | 31.992 | 0.818 | cg_gs |
| H2 | 3.304 | 3.267 | 0.082 | cg_gs |

Each live cell contains one observation. The CLI clock includes interpreter startup, calibration checks, hardware and binary checks, decision making, pilots when requested, and the final verified run. These observations demonstrate the execution path and expose its overhead. The paired study above supplies the repeated performance comparison.

Using only the archived first two levels, the retrospective forecast picked the faster complete workflow in 22/30 H1 pairs and 20/30 H2 pairs. Against the fastest verified arm in each observed block, the default policy's median excess times were 0.850 and 0.061 s respectively. The hindsight reference includes ordinary timing noise.

**Cost of full numerical verification.** On identical retained 8–64 level fields, the optimized verifier reduced median verification time from 340 to 221 ms on P1 sine triangles and from 367 to 219 ms on mixed-boundary Q1 quadrilaterals. These are 1.54× and 1.67× speedups of the verification step. Each of the 12 paired repetitions per case had an identical complete verdict hash between implementations. The median paired differences (new minus old) were −116 ms and −148 ms, with separate paired 95% bootstrap intervals of [−123, −100] and [−164, −127] ms. [Individual times and verdict hashes](../evidence/e1/verifier-cost.json)

**Low-cost screening.** In a shadow experiment on warm local archives, stopping and quadrature metadata identified 36 of 162 faulty inputs and rejected none of 46 clean controls. The other 126 faults and all 46 clean controls needed full verification. The screen took 0.044 s across all 208 trials. The measured full checks took 9.600 s in total. Charging all screen costs and the full checks for escalated inputs gives a projected 7.882 s for this fault-heavy mix (17.9% lower). A clean candidate pays both costs. All final study acceptances in the solver experiments used full verification. [Per-trial results](../evidence/e1/screen-cost.json)

**Assembly and parallelism.** Full and partial assembly using matched Jacobi actions passed 24 studies with 72 independently checked fields, spanning Q1/Q2, three cases and two mesh sources. The 36 matched levels agreed on CG iteration counts, and the largest relative field difference was `4.59e-14`. With eight timed pairs per order, the end-to-end intervals for partial minus full crossed zero for Q1 and Q2. On the four-core MPI diagnostic, increasing ranks from one to four increased the median empty launch from 402 to 653 ms. At `n=64`, the longest rank-local numerical window fell from 68 to 41 ms while the verified workflow rose from 585 to 794 ms. The empty launch isolates real rank-dependent process cost. Its duration cannot be subtracted from another campaign to assign an exact causal fraction. [Assembly measurements](../evidence/e1/assembly.json) · [MPI samples and independent replay](../evidence/e1/mpi-phases.json)

All **385 tests passed** with real serial and MPI MFEM enabled and zero skips. Fresh builds reproduced both native executable digests. The final audit checked all 396 study inventories and reran full numerical verification for the first eligible scored study in each workload/arm cell (12 replays). All 284 historical challenge decisions, 14 checker calibrations, 31 serial study states, and the Stage 2, 3 and 4 archive regressions reproduced. [Audit of raw study records](../evidence/e1/paired-audit.json) · [Builds, tests and portable archive digest](../evidence/e1/validation.json)

## Stage 4 controlled planning, 2026-11-24

The paired benchmark used 19 fixed English and Chinese requests, a deterministic first-eligible-template rule, and a locally installed Qwen2.5 1.5B model served by Ollama 0.40.17. Five eligible tasks specified a unique template, nine expressed a preference among multiple authorized templates, and five required human review. Only the task text and eligible options were supplied to the model. Each eligible execution went through the real MFEM controller and independent numerical replay. [Corpus and scoring protocol](PLANNING_BENCHMARK.md) · [Local model identity](LOCAL_PLANNER_MODEL.md)

| Outcome | Rule baseline, first / repeat | Local model, first / repeat |
|---|---:|---:|
| Authorized, numerically verified studies out of 14 | 14 / 14 | 14 / 14 |
| Tasks with the intended template out of 14 | 8 / 8 | 10 / 13 |
| Preference matches out of 9 | 3 / 3 | 5 / 8 |
| Correctly held review requests out of 5 | 5 / 5 | 5 / 5 |
| Unauthorized acceptances, count | 0 / 0 | 0 / 0 |
| Proposal retries and numerical retries, count | 0 in both runs | 0 in both runs |
| Workflow seconds for all 19 requests | 18.71 / 17.97 | 31.26 / 31.74 |
| Measured input tokens | 0 / 0 | 3,465 / 3,465 |
| Measured output tokens | 0 / 0 | 208 / 204 |

The model improved preference matching on this panel, with an additional 12.55 s in the first capture and 13.77 s in the repeat. In the first capture it corrected four rule selections and missed two preferences that the rule satisfied. Across live captures, three model nominations changed despite byte-identical prompts, matching model and weight digests, temperature 0 and seed 42. The model ran locally with zero hosted API charges. Electricity and hardware cost were unmeasured. The five review decisions were made deterministically before any model call. [First capture](../evidence/stage4/planning-results.json) · [Repeat capture](../evidence/stage4/replication-results.json)

The four supplemental controls included conflicting template IDs and an attempted change to a validation parameter. All were rejected. In a separate regression the planner selected an authorized template and the real solver returned success on every level after injecting a wrong source. The independent verifier rejected the solution with `REFERENCE_MISMATCH`. Another regression changed only the selected file's line endings before execution and the controller rejected its byte digest. [Workflow and numerical regressions](../tests/test_planning_run.py)

On WSL2, all **319 tests passed** with real serial and MPI MFEM enabled and zero skips. After relocating both planner campaigns and disabling network connections and subprocesses, the verifier rechecked 76 decision archives, including 56 numerical studies, and regenerated each campaign's own report byte for byte. The first campaign was captured during integration. The repeat and both offline replays used code commit `d8161e2`. The original 31 serial decisions, 158 error values, 284 challenge decisions and 14 verifier calibrations reproduced. The Stage 2 archive reproduced its 42-study, 126-field report exactly. The Stage 3 archive passed replay for all 12 screening solves and 32 timing samples. [Validation record and portable archive digest](../evidence/stage4/validation.json)

## Stage 3 MPI Poisson, 2026-11-24

Code commit `16f6b29` passed **256 tests with zero failures and zero skips** on Ubuntu 24.04 WSL2. The MPI driver used the pinned MFEM 4.10.0 source, Open MPI 4.1.6, hypre 2.28.0 and METIS 5.1.0. A second driver build in a clean Linux directory produced the same binary digest. This campaign ran on one Intel Core Ultra 7 155H host with 22 visible logical CPUs, one numerical thread per process and MPI binding disabled. [Toolchain, test logs and digests](../evidence/stage3/validation.json)

The P1 sine Poisson study passed all 12 screening solves at 8, 16, 32 and 64 cells per side on 1, 2 and 4 MPI ranks. The maximum relative nodal difference across screening solves was `1.875e-13` against serial MFEM and `6.207e-11` against the SciPy direct solve. At 64 cells per side all rank counts gave L2 error `3.379923e-4` and H1 seminorm error `5.451370e-2`. The finest-pair orders were `1.99836` and `0.99931`. On four ranks the finest mesh partition had 2,032, 2,051, 2,053 and 2,056 triangles. [Per-rank numerical results and timing samples](../evidence/stage3/results.json)

**Strong scaling at 8,192 triangles.** Each row gives medians over eight independently verified samples. The speedup denominator is the corresponding median at the current rank count, and the numerator is the one-rank MPI median. Launcher time includes MPI startup, assembly, partitioning, solving and export. Verified workflow time also includes independent numerical checks.

| MPI ranks | CG solve (ms) | Launcher (ms) | Verified workflow (ms) | Solve speedup | Workflow speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.592 | 452.5 | 513.9 | 1.00× | 1.00× |
| 2 | 3.511 | 488.2 | 556.2 | 1.31× | 0.92× |
| 4 | 2.793 | 634.7 | 696.9 | 1.64× | 0.74× |

**Weak scaling at 2,048 triangles per rank on average.** From one rank at `n=32` to four at `n=64`, median verified workflow time increased from 424.8 ms to 696.9 ms, for **61.0% weak efficiency**. The complete campaign, including serial reproduction, screening, 32 timed solves and final inventory hashing, took **25.48 s**. These measurements establish single-host behavior for the tested resolutions. The solver starts from a replicated serial mesh and the Python verifier runs on one process. [Protocol and measurement boundaries](MPI.md)

The portable archive passed digest-pinned replay after relocation with both native executables unavailable. Process tests detected nonconvergence when CG was limited to one iteration, preserved existing outputs and confirmed that the timeout path terminates both ranks even when the launcher stalls. All 31 original serial decisions, 158 archived error values and 83 sets of native field exports reproduced. The Stage 2 coverage report was byte-identical on replay. The 284 historical challenge decisions and 14 checker calibrations also reproduced. [Archive receipt and regression evidence](../evidence/stage3/validation.json)

## Stage 2 scalar diffusion and Gmsh, 2026-11-23

On WSL2 Ubuntu 24.04, code commit `40873c2` passed **217 tests with zero failures and zero skips**. The two CI selectors ran locally with 141 Python tests and 76 MFEM tests. The pinned serial double MFEM 4.10.0 installation used GCC 13.3.0. A clean out-of-tree driver build on the Linux filesystem and the tested executable had identical SHA-256 digests. The CI workflow now runs the scalar coverage benchmark. This local-only revision has no hosted CI run. [Build, test commands, digests and local archive locations](../evidence/stage2/validation.json)

All **42 studies** in the reviewed matrix passed on their first attempt, spanning three manufactured cases, seven solver configurations and two mesh sources. Each ran at 8, 16 and 32 cells per side. For all 126 fields a separately regenerated scikit-fem mesh agreed on element connectivity, boundary tags and DOF coordinates. The largest relative nodal difference from the scikit-fem solution was `2.273e-9`. The largest relative difference in the independently integrated L2 and H1 errors was `1.436e-10`. A second pass using the saved manifest digests and raw archives reproduced the coverage report byte for byte. [Per-configuration results](../evidence/stage2/coverage.json) · [Generation and replay commands](../README.md#variable-diffusion-and-named-mesh-examples)

| Mixed variable diffusion at 32 cells per side | L2 error | H1 seminorm error | Finest L2 / H1 orders |
|---|---:|---:|---:|
| Generated P1 triangles, full assembly | 2.897880e-3 | 2.553787e-1 | 1.986 / 0.994 |
| Imported P2 quadrilaterals, partial assembly | 2.191353e-5 | 4.547010e-3 | 2.996 / 1.998 |

The full matrix exposed a quadrature defect on P2 triangles. MFEM's default stiffness rule underintegrated the variable coefficient. The first implementation differed from both independent solvers by approximately `2.33e-5` at eight cells per side. Using an explicit rule of order `2p+1` reduced the largest SciPy field difference across all three P2 triangle levels to `1.33e-10`. Separate regressions check the actual solution and reject a missing or inconsistent native quadrature claim. [Before and after measurements](../evidence/stage2/validation.json) · [Native regression](../tests/test_integration_mfem.py)

Both deliberate diffusion faults exited successfully at every level. At 32 cells per side, omitting the coefficient-gradient source term changed the discrete field by **3.57%**, and omitting the top Neumann coefficient changed it by **66.89%**, relative to the SciPy reference. The verifier rejected both by its original `1e-7` threshold. Replaying a Gmsh archive succeeded after removing the input file. Altering the archived mesh failed digest verification, and substituting a geometrically valid mesh with the opposite triangle diagonal also failed the numerical topology check. [Fault measurements](../evidence/stage2/validation.json) · [Gmsh regressions](../tests/test_gmsh.py)

All 31 historical study states and codes matched Stage 1. The comparison replayed 22 historical and 22 fresh attempts, reproduced 158 error values exactly, and found byte-identical mesh and field exports at all 83 levels with native results. The Stage 1 challenge repeated 215 rejected faults and 69 accepted controls with the same per-trial decisions and gates. All 14 separate checker calibrations matched. These challenge denominators cover the original Poisson corpus. The Stage 2 fault cases and scikit-fem comparisons are reported separately. [Paired manifests](../evidence/stage2/parity.json) · [Challenge counts and digests](../evidence/stage2/validation.json)

## Stage 2 registry and native diagnostics, 2026-11-23

The WSL2 Ubuntu 24.04 configuration rebuilt the C++ driver against the pinned double-precision serial MFEM 4.10 installation. **167 tests passed with zero skips** on code commit `eca008f`. The P1 quickstart and its digest-checked replay both returned `ACCEPTED`. The fresh 284-trial verifier challenge matched the previous per-trial decisions and gate results, and all 14 deliberate reference-checker mutations were detected. [Local validation record](../evidence/stage2-registry/validation.json)

Two comparisons used the full 31-request demo. Against the imported results, all study decisions matched and the largest relative difference in the 68 accepted L2/H1 errors was `1.18e-13`. Against the retained Stage 1 WSL2 runs, all final states, codes, frozen hashes, attempt verdicts, failure codes and 158 available L2/H1 error values matched exactly. All 83 native level results preserved their original non-timing fields, and the six mesh/field export files were byte-identical at every such level. The independent verifier replayed 22 historical and 22 fresh attempts with passing archive integrity. [Paired archives and diagnostics](../evidence/stage2-registry/parity.json)

| Measurement | Result |
|---|---|
| Native residual on seven positive configurations, four levels each | Three of 28 unpreconditioned ratios exceeded the CG relative tolerance of `1e-12`. The largest was `1.4011e-12`. All seven studies passed the existing criteria. |
| P1 sine study on 64 cells per side | Outward gradient flux `7.99357660`, integrated volume source `8.00000000`, signed imbalance `−0.00642340`. |
| Early-stop fault and wrong-source fault at eight cells per side | The true relative residuals were `9.4202e-3` and `3.8546e-13`, respectively. Independent numerical verification rejected both studies. |
| Nonfinite exported field | Algebraic measurements describe the solved vector before export corruption. Field fluxes and imbalance are JSON `null` and the verifier rejects the field. |

For a fresh run, use the build instructions in the README, then run `python -m pytest -q --require-mfem`, `RUNS_DIR=runs/stage2-demo ./scripts/run_demo.sh` and `python scripts/compare_baseline.py runs/stage2-demo`. Start with an empty output directory. To repeat the exact WSL2 archive comparison, pass the directory of the pinned Stage 1 runs and the fresh directory to [compare_archives.py](../evidence/stage2-registry/compare_archives.py). The validation record includes the binary hash and location of the local raw reports.

## Stage 1 verifier challenge, 2026-11-20

The baseline verifier accepted 28/98 fixed numerical faults and 17/64 seeded draws. Four fault families accounted for every false accept. After fixing the Cartesian geometry and native CG and quadrature checks, the verifier rejected all 162 numerical fault trials. All 46 independently certified clean numerical trials passed. For the 64 sampled faults the two-sided exact 95% upper bound on the false-accept rate is 5.60% under the declared generator. All 53 archive faults were rejected and all 23 clean archive trials passed. Fourteen additional calibration trials exposed deliberately incorrect reference assemblies. [Per-gate results, uncertainty and fault witnesses](VERIFIER_CHALLENGE.md)

The starting revision passed 138 tests in 32.12 s. The repaired checkout passed **151 tests in 39.71 s** in the same WSL2 distribution with the real MFEM binary. All 31 original study states and attempt decisions reproduced. The 68 accepted L2 and H1 error values matched the historical record to a maximum relative difference of `1.18e-13`. The finest P1 triangle orders were 1.9983631008284015 for L2 and 0.999311941962252 for the H1 seminorm. Replaying the 22 saved Stage 0 attempts and the 22 fresh Stage 1 attempts with their manifest digests reproduced every decision. [Original-study comparison](../evidence/stage1/canonical-comparison.json) · [Archive replays](../evidence/stage1/archive-replays.json) · [Local test transcript](../evidence/stage1/transcript.txt)

Comparing the two WSL2 canonical runs directly, all 158 available finite L2/H1 values across all attempts had identical float64 representations. The only change in a diagnostic list was the additional `CG_RECORD_INCONSISTENT` code on the existing early-stop fault. [Local comparison](../evidence/stage1/transcript.txt)

GitHub Actions passed 90 Python tests and 61 native integration tests on code commit `8910b4f`. The clean MFEM source build took 20 min 37 s. The 36-second challenge step generated fresh native controls on Python 3.12.16. All 284 individual verdicts, failure codes, reached/failed gates and aggregate metrics matched WSL2, and all 14 reference-checker calibrations gave the expected reference mismatch. The retained hosted JSON has SHA-256 `5a0c0f416d3057c8337eec7ead31f8972d2fad1ba56492e8cc7cf52fda2791fe`. [CI logs](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/44859215191) · [Hosted report and WSL2 control bundle](https://github.com/rodriguezrobertbfrkx6857-sudo/project/releases/tag/stage1-verifier-2026-11-20)

## Independent WSL2 reproduction, 2026-11-20

The original checkout passed 62 Python tests and skipped 34 native tests before building the solver. With MFEM 4.10 built from the pinned commit, the same checkout passed all 96 tests in 21.67 s. The Stage 0 checkout passed **138 tests in 30.81 s**, with zero failures and zero skips. Its unit and native jobs selected 84 and 54 tests respectively. With the binary deliberately absent, 84 passed and all 54 native tests reported the missing executable. Strict integration mode returned a nonzero status before collection. [Execution transcript](../evidence/stage0-wsl2/transcript.txt)

The final P1 triangle benchmark was numerically accepted and replayed from its archive with the manifest digest supplied separately. The digest covers the inventory of 52 files. [Quickstart output](../evidence/stage0-wsl2/quickstart.json)

| Cells per side | DOFs | Independent L2 error | Independent H1 seminorm error | L2 order | H1 order |
|---:|---:|---:|---:|---:|---:|
| 8 | 81 | 2.11327734e-2 | 4.31798283e-1 | — | — |
| 16 | 289 | 5.37743501e-3 | 2.17536336e-1 | 1.97449 | 0.98910 |
| 32 | 1,089 | 1.35043625e-3 | 1.08975424e-1 | 1.99349 | 0.99725 |
| 64 | 4,225 | 3.37992335e-4 | 5.45137045e-2 | 1.99836 | 0.99931 |

The complete demo produced 8 accepted studies, 1 exhausted budget, 2 blocked backends, 7 invalid specifications and 13 failures. All 31 terminal states, codes and attempt decisions matched the imported records. Across the final accepted attempts, the largest relative difference in the 68 L2 and H1 error values was $1.18\times10^{-13}$. These are cross-environment differences measured against the original records. Both runs used the frozen scientific acceptance criteria. [Machine-readable comparison and manifest digests](../evidence/stage0-wsl2/comparison.json) · [All WSL2 study tables and failure gates](../evidence/stage0-wsl2/studies.md)

Every stored attempt was then replayed with its run's manifest digest. All 22 archives passed the integrity check and reproduced their recorded decisions, including 8 numerical acceptances, 9 rejections and 5 retry-eligible attempts. An additional replay succeeded after moving the quickstart directory and pointing both MFEM paths at absent locations. On the real P1, P2, Q1 and Q2 meshes at $n=8$, quadrature recovered the unit area to $2.22\times10^{-16}$, the basis functions summed to one to $2.67\times10^{-15}$, and their physical gradients summed to zero to $2.22\times10^{-14}$. [Replay and finite-element checks](../evidence/stage0-wsl2/transcript.txt)

MFEM's unmodified Example 1 reproduced the original diagnostic on the square-disc mesh. It wrote exit code 0 after 200 PCG iterations and printed `PCG: No convergence!`. The star mesh had 20,801 unknowns and reached its criterion in 111 iterations. [Execution transcript](../evidence/stage0-wsl2/transcript.txt)

On GitHub Actions, the Python job passed all 84 selected tests in 9.85 s. The native job built MFEM from source and passed all 54 selected integration tests in 19.14 s. The hosted quickstart returned `ACCEPTED` on replay with L2 order 1.9983631008284015 and H1 seminorm order 0.999311941962252. [Hosted run and logs](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/44776289242)

## Original Claude experimental record

The following tables were generated by `scripts/summarize_runs.py` from Claude's original run archive.

## Environment of these runs

* MFEM: `4.10.0` git `tags/v4.10-0-gd964264cdb9a13e94a201b6c236c7721e0c8765f`; libmfem.a SHA-256 `45e334fe910124b42a23035adb1ee267bbd9a68422c5f4177028cf5141d7c474`
* Solver binary SHA-256: `9aba0c1d449edc61fe53bb51ad488ac8b7dcee9f5d16e4b2fdf539bd692e4960`
* Compiler: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; CMake: cmake version 3.28.3; Python 3.13.18; numpy 2.5.3; scipy 1.18.1
* OS: Linux-6.18.44-fc-v726-x86_64-with-glibc2.39; CPUs: 2; RAM: 8024 MB
* MFEM build flags: MFEM_USE_MPI=False, MFEM_USE_METIS=False, MFEM_USE_METIS_5=False, MFEM_USE_SUITESPARSE=False, MFEM_USE_LAPACK=False, MFEM_USE_OPENMP=False, MFEM_USE_CUDA=False, MFEM_USE_HIP=False, MFEM_USE_RAJA=False, MFEM_USE_OCCA=False, MFEM_USE_PETSC=False, MFEM_USE_ZLIB=False, MFEM_USE_DOUBLE=True, MFEM_USE_SINGLE=False

## Terminal states

| Spec | Final state | Code | Attempts | Run directory |
|---|---|---|---|---|
| `specs/budget_demo_sin_sin_p1_tri.json` | BUDGET_EXHAUSTED | BUDGET_EXHAUSTED | 2 | `runs/20261120T155135958265Z_budget_demo_sin_sin_p1_tri` |
| `specs/negative/duplicate_attribute.json` | INVALID_SPEC | INVALID_SPEC | 0 | `runs/20261120T155147782554Z_duplicate_attribute` |
| `specs/negative/fault_drop_dirichlet_attr.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155148156171Z_fault_drop_dirichlet_attr` |
| `specs/negative/fault_early_stop_lie.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155149205353Z_fault_early_stop_lie` |
| `specs/negative/fault_nan_value.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155150244828Z_fault_nan_value` |
| `specs/negative/fault_self_report_lie.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155150920930Z_fault_self_report_lie` |
| `specs/negative/fault_wrong_source.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155152036436Z_fault_wrong_source` |
| `specs/negative/inconsistent_mms_source.json` | INVALID_SPEC | MMS_INCONSISTENT | 0 | `runs/20261120T155153118500Z_inconsistent_mms_source` |
| `specs/negative/malicious_expression.json` | INVALID_SPEC | INVALID_EXPRESSION | 0 | `runs/20261120T155153431479Z_malicious_expression` |
| `specs/negative/missing_boundary.json` | INVALID_SPEC | INVALID_SPEC | 0 | `runs/20261120T155153768822Z_missing_boundary` |
| `specs/negative/non_doubling_levels.json` | INVALID_SPEC | INVALID_SPEC | 0 | `runs/20261120T155154176578Z_non_doubling_levels` |
| `specs/negative/not_verified_order3.json` | FAILED | NOT_VERIFIED_COMBINATION | 0 | `runs/20261120T155154559030Z_not_verified_order3` |
| `specs/negative/preasymptotic_strict_band.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155155100213Z_preasymptotic_strict_band` |
| `specs/negative/strict_band_resolved.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155155707632Z_strict_band_resolved` |
| `specs/negative/swapped_boundary_labels.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155156387621Z_swapped_boundary_labels` |
| `specs/negative/unassigned_mesh_attribute.json` | FAILED | VALIDATION_FAILED | 1 | `runs/20261120T155157553882Z_unassigned_mesh_attribute` |
| `specs/negative/unbound_case.json` | FAILED | UNSUPPORTED_PROBLEM | 0 | `runs/20261120T155158065044Z_unbound_case` |
| `specs/negative/unknown_field.json` | INVALID_SPEC | INVALID_SPEC | 0 | `runs/20261120T155158565311Z_unknown_field` |
| `specs/negative/unsupported_backend_cuda.json` | BLOCKED | UNSUPPORTED_BACKEND | 0 | `runs/20261120T155158921312Z_unsupported_backend_cuda` |
| `specs/negative/unsupported_backend_mpi.json` | BLOCKED | UNSUPPORTED_BACKEND | 0 | `runs/20261120T155159381213Z_unsupported_backend_mpi` |
| `specs/negative/unsupported_bc_value.json` | INVALID_SPEC | INVALID_SPEC | 0 | `runs/20261120T155159978072Z_unsupported_bc_value` |
| `specs/negative/unsupported_pa_with_gs.json` | FAILED | UNSUPPORTED_COMBINATION | 0 | `runs/20261120T155200406916Z_unsupported_pa_with_gs` |
| `specs/negative/unsupported_solver_umfpack.json` | FAILED | UNSUPPORTED_SOLVER | 0 | `runs/20261120T155200975722Z_unsupported_solver_umfpack` |
| `specs/poisson_cos_exp_mixed_p1_quad.json` | ACCEPTED | OK | 1 | `runs/20261120T155136678345Z_poisson_cos_exp_mixed_p1_quad` |
| `specs/poisson_cos_exp_mixed_p1_quad_pa.json` | ACCEPTED | OK | 1 | `runs/20261120T155137876054Z_poisson_cos_exp_mixed_p1_quad_pa` |
| `specs/poisson_cos_exp_mixed_p2_quad.json` | ACCEPTED | OK | 1 | `runs/20261120T155139062995Z_poisson_cos_exp_mixed_p2_quad` |
| `specs/poisson_cos_exp_mixed_p2_quad_pa.json` | ACCEPTED | OK | 1 | `runs/20261120T155139943452Z_poisson_cos_exp_mixed_p2_quad_pa` |
| `specs/poisson_sin_sin_p1_tri.json` | ACCEPTED | OK | 1 | `runs/20261120T155140827686Z_poisson_sin_sin_p1_tri` |
| `specs/poisson_sin_sin_p1_tri_jacobi.json` | ACCEPTED | OK | 1 | `runs/20261120T155142045852Z_poisson_sin_sin_p1_tri_jacobi` |
| `specs/poisson_sin_sin_p2_tri.json` | ACCEPTED | OK | 1 | `runs/20261120T155143161774Z_poisson_sin_sin_p2_tri` |
| `specs/retry_demo_sin_sin_p1_tri.json` | ACCEPTED | OK | 4 | `runs/20261120T155144030514Z_retry_demo_sin_sin_p1_tri` |

## Convergence studies (accepted runs)

### `specs/poisson_cos_exp_mixed_p1_quad.json`

mesh `quad`, p=1, assembly `full`, solver `cg_gs`; expected orders L2=2, H1-semi=1

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1/8 | 81 | 14 | 1.6256e-02 | 1.6256e-02 | 4.5158e-01 | — | — | 8.40e-13 |
| 16 | 1/16 | 289 | 26 | 4.0696e-03 | 4.0696e-03 | 2.2610e-01 | 1.9980 | 0.9980 | 1.93e-13 |
| 32 | 1/32 | 1089 | 48 | 1.0177e-03 | 1.0177e-03 | 1.1309e-01 | 1.9995 | 0.9995 | 6.62e-13 |
| 64 | 1/64 | 4225 | 91 | 2.5446e-04 | 2.5446e-04 | 5.6549e-02 | 1.9999 | 0.9999 | 2.04e-12 |

### `specs/poisson_cos_exp_mixed_p1_quad_pa.json`

mesh `quad`, p=1, assembly `partial`, solver `cg_jacobi`; expected orders L2=2, H1-semi=1

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1/8 | 81 | 17 | 1.6256e-02 | 1.6256e-02 | 4.5158e-01 | — | — | 7.07e-13 |
| 16 | 1/16 | 289 | 38 | 4.0696e-03 | 4.0696e-03 | 2.2610e-01 | 1.9980 | 0.9980 | 1.63e-13 |
| 32 | 1/32 | 1089 | 76 | 1.0177e-03 | 1.0177e-03 | 1.1309e-01 | 1.9995 | 0.9995 | 7.20e-13 |
| 64 | 1/64 | 4225 | 152 | 2.5446e-04 | 2.5446e-04 | 5.6549e-02 | 1.9999 | 0.9999 | 1.20e-12 |

### `specs/poisson_cos_exp_mixed_p2_quad.json`

mesh `quad`, p=2, assembly `full`, solver `cg_gs`; expected orders L2=3, H1-semi=2

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1/4 | 81 | 18 | 3.4864e-03 | 3.4864e-03 | 9.0590e-02 | — | — | 4.33e-12 |
| 8 | 1/8 | 289 | 35 | 4.3919e-04 | 4.3919e-04 | 2.2785e-02 | 2.9888 | 1.9913 | 3.41e-13 |
| 16 | 1/16 | 1089 | 69 | 5.5008e-05 | 5.5008e-05 | 5.7048e-03 | 2.9971 | 1.9978 | 8.53e-13 |
| 32 | 1/32 | 4225 | 136 | 6.8794e-06 | 6.8794e-06 | 1.4267e-03 | 2.9993 | 1.9995 | 2.42e-12 |

### `specs/poisson_cos_exp_mixed_p2_quad_pa.json`

mesh `quad`, p=2, assembly `partial`, solver `cg_jacobi`; expected orders L2=3, H1-semi=2

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1/4 | 81 | 21 | 3.4864e-03 | 3.4864e-03 | 9.0590e-02 | — | — | 4.34e-12 |
| 8 | 1/8 | 289 | 46 | 4.3919e-04 | 4.3919e-04 | 2.2785e-02 | 2.9888 | 1.9913 | 1.46e-13 |
| 16 | 1/16 | 1089 | 91 | 5.5008e-05 | 5.5008e-05 | 5.7048e-03 | 2.9971 | 1.9978 | 1.02e-12 |
| 32 | 1/32 | 4225 | 180 | 6.8794e-06 | 6.8794e-06 | 1.4267e-03 | 2.9993 | 1.9995 | 1.59e-12 |

### `specs/poisson_sin_sin_p1_tri.json`

mesh `tri`, p=1, assembly `full`, solver `cg_gs`; expected orders L2=2, H1-semi=1

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1/8 | 81 | 13 | 2.1133e-02 | 2.1133e-02 | 4.3180e-01 | — | — | 6.21e-11 |
| 16 | 1/16 | 289 | 22 | 5.3774e-03 | 5.3774e-03 | 2.1754e-01 | 1.9745 | 0.9891 | 2.46e-13 |
| 32 | 1/32 | 1089 | 40 | 1.3504e-03 | 1.3504e-03 | 1.0898e-01 | 1.9935 | 0.9973 | 3.01e-14 |
| 64 | 1/64 | 4225 | 74 | 3.3799e-04 | 3.3799e-04 | 5.4514e-02 | 1.9984 | 0.9993 | 3.20e-14 |

### `specs/poisson_sin_sin_p1_tri_jacobi.json`

mesh `tri`, p=1, assembly `full`, solver `cg_jacobi`; expected orders L2=2, H1-semi=1

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1/8 | 81 | 6 | 2.1133e-02 | 2.1133e-02 | 4.3180e-01 | — | — | 6.20e-11 |
| 16 | 1/16 | 289 | 26 | 5.3774e-03 | 5.3774e-03 | 2.1754e-01 | 1.9745 | 0.9891 | 2.42e-13 |
| 32 | 1/32 | 1089 | 56 | 1.3504e-03 | 1.3504e-03 | 1.0898e-01 | 1.9935 | 0.9973 | 2.32e-14 |
| 64 | 1/64 | 4225 | 114 | 3.3799e-04 | 3.3799e-04 | 5.4514e-02 | 1.9984 | 0.9993 | 2.51e-14 |

### `specs/poisson_sin_sin_p2_tri.json`

mesh `tri`, p=2, assembly `full`, solver `cg_gs`; expected orders L2=3, H1-semi=2

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1/4 | 81 | 14 | 4.3276e-03 | 4.3276e-03 | 1.2939e-01 | — | — | 1.78e-10 |
| 8 | 1/8 | 289 | 27 | 5.4806e-04 | 5.4806e-04 | 3.3387e-02 | 2.9812 | 1.9544 | 1.71e-13 |
| 16 | 1/16 | 1089 | 49 | 6.8739e-05 | 6.8739e-05 | 8.4191e-03 | 2.9951 | 1.9875 | 7.31e-14 |
| 32 | 1/32 | 4225 | 93 | 8.6005e-06 | 8.6005e-06 | 2.1095e-03 | 2.9986 | 1.9968 | 3.51e-14 |

### `specs/retry_demo_sin_sin_p1_tri.json`

mesh `tri`, p=1, assembly `full`, solver `cg_gs`; expected orders L2=2, H1-semi=1

| n | h | DOFs | CG its | L2 error (verifier) | L2 (MFEM self-report) | H1-semi error (verifier) | L2 order | H1 order | rel. diff vs reference solve |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1/4 | 25 | 5 | 7.9075e-02 | 7.9075e-02 | 8.3855e-01 | — | — | 1.61e-08 |
| 8 | 1/8 | 81 | 13 | 2.1133e-02 | 2.1133e-02 | 4.3180e-01 | 1.9037 | 0.9575 | 6.21e-11 |
| 16 | 1/16 | 289 | 22 | 5.3774e-03 | 5.3774e-03 | 2.1754e-01 | 1.9745 | 0.9891 | 2.46e-13 |
| 32 | 1/32 | 1089 | 40 | 1.3504e-03 | 1.3504e-03 | 1.0898e-01 | 1.9935 | 0.9973 | 3.01e-14 |
| 64 | 1/64 | 4225 | 74 | 3.3799e-04 | 3.3799e-04 | 5.4514e-02 | 1.9984 | 0.9993 | 3.20e-14 |
| 128 | 1/128 | 16641 | 138 | 8.4522e-05 | 8.4522e-05 | 2.7260e-02 | 1.9996 | 0.9998 | 7.81e-14 |

## Bounded retry history

### `specs/budget_demo_sin_sin_p1_tri.json` → BUDGET_EXHAUSTED (BUDGET_EXHAUSTED)

| Attempt | Mesh levels | Finest L2 error | Verdict | Codes |
|---|---|---|---|---|
| 1 | [4, 8, 16] | 5.3774e-03 | RETRY_ELIGIBLE | ['ACCURACY_NOT_MET'] |
| 2 | [4, 8, 16, 32] | 1.3504e-03 | RETRY_ELIGIBLE | ['ACCURACY_NOT_MET'] |

Frozen accuracy target `max_l2_error_finest` = 1e-09; frozen ValidationSpec SHA-256 `652a86fc83732d26a37b9728203950e7cf87fdc9a12dd7dfdb147f982b50a920` (unchanged across attempts).

### `specs/retry_demo_sin_sin_p1_tri.json` → ACCEPTED (OK)

| Attempt | Mesh levels | Finest L2 error | Verdict | Codes |
|---|---|---|---|---|
| 1 | [4, 8, 16] | 5.3774e-03 | RETRY_ELIGIBLE | ['ACCURACY_NOT_MET'] |
| 2 | [4, 8, 16, 32] | 1.3504e-03 | RETRY_ELIGIBLE | ['ACCURACY_NOT_MET'] |
| 3 | [4, 8, 16, 32, 64] | 3.3799e-04 | RETRY_ELIGIBLE | ['ACCURACY_NOT_MET'] |
| 4 | [4, 8, 16, 32, 64, 128] | 8.4522e-05 | ACCEPTED | [] |

Frozen accuracy target `max_l2_error_finest` = 0.0001; frozen ValidationSpec SHA-256 `4ccf408a0c1260632fff37c9e5cf828ffaf34f70345a22ea67252e977907ca71` (unchanged across attempts).

## Fault × gate detection matrix

Each row is a negative study that executed MFEM; columns are verifier check names; ✗ = check failed (fault detected by that gate).

| Negative study | Solver exit(s) | Verdict | `finest_pair_l2_order` | `reference_solve_agreement` | `finest_pair_h1_semi_order` | `dirichlet_nodal_values` | `monotone_error_decrease` | `finite_solution` | `self_report_l2` | `self_report_h1_semi` | `boundary_side_length` | `boundary_label_geometry` | `result_json_present` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fault_drop_dirichlet_attr | OK | REJECTED | ✗ | ✗ | ✗ | ✗ | ✗ |  |  |  |  |  |  |
| fault_early_stop_lie | OK | REJECTED | ✗ | ✗ |  |  |  |  |  |  |  |  |  |
| fault_nan_value | NON_FINITE_SOLUTION | REJECTED |  |  |  |  |  | ✗ |  |  |  |  |  |
| fault_self_report_lie | OK | REJECTED |  |  |  |  |  |  | ✗ | ✗ |  |  |  |
| fault_wrong_source | OK | REJECTED | ✗ | ✗ | ✗ |  | ✗ |  |  |  |  |  |  |
| preasymptotic_strict_band | OK | REJECTED | ✗ | ✗ | ✗ |  |  |  | ✗ | ✗ |  |  |  |
| strict_band_resolved | OK | REJECTED | ✗ |  |  |  |  |  |  |  |  |  |  |
| swapped_boundary_labels | OK | REJECTED |  | ✗ |  | ✗ |  |  |  |  | ✗ | ✗ |  |
| unassigned_mesh_attribute | BOUNDARY_COVERAGE | REJECTED |  |  |  |  |  |  |  |  |  |  | ✗ |

