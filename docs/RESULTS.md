# Numerical results

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

