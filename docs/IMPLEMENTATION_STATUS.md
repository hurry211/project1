# Implementation Status

## Stage 1 handoff, 2026-11-20

Verifier 0.3.0 checks the native integration-order fields and the consistency of the CG stopping record. It checks the Cartesian partition in the interior, complementary triangles and the complete P1/P2 node lattice. Native numerical regressions and geometry tests cover the newly rejected inputs. The solver, manufactured functions, validation defaults and seven supported configurations retain their Stage 0 implementation. [Verifier changes](../verifier/verifier.py) · [Mesh implementation](../verifier/fem.py) · [Acceptance contract](MVP_SPEC.md)

The optional challenge package regenerates the seven control meshes using scikit-fem and compares all 21 levels to a separately assembled solution and analytic error integrals. Both verifier revisions were scored on byte-identical native inputs. The challenge is part of the native GitHub Actions job, which retains its JSON report as an artifact. The local environment ran 151 tests with zero skips and replayed the saved Stage 0 and Stage 1 canonical attempts. [Measured results](RESULTS.md) · [Generation rules and trust boundaries](VERIFIER_CHALLENGE.md)

The private-repository evidence release contains the full WSL2 control bundle and the independently generated GitHub Actions report. The bundle digest and both local scored reports are versioned. The local run directories are under `/root/mfem-stage1` in the `Ubuntu` WSL2 distribution. [Bundle checksum](../evidence/stage1/controls.sha256) · [Evidence release](https://github.com/rodriguezrobertbfrkx6857-sudo/project/releases/tag/stage1-verifier-2026-11-20)

The next low-risk Stage 2 step is to register the two current manufactured solutions through a reviewed case registry and demonstrate numerical parity for all 31 requests. A new PDE or mesh family will need its own independent construction, boundary tests and convergence evidence. The live SciPy reference solve still runs in the controller process and has a DOF cap. A separately enforced time and memory budget remains open.

## Stage 0 handoff, 2026-11-20

The independent reproduction started at `34d3d66`. The original suite passed on a new MFEM source build. The updated suite exercises the same two manufactured solutions, seven supported configurations, numerical thresholds and bounded refinement studies. The original implementation record follows this section.

| Component | Stage 0 finding and change | Evidence |
|---|---|---|
| Mathematical path | Reviewed the C++ gradient and source for both manufactured solutions, Dirichlet interpolation, the Neumann flux sign, the Galerkin assembly and CG stopping conditions. The verifier reconstructs the nodal field with inverse maps and a Vandermonde basis, integrates with Gauss/Duffy rules and solves a separate SciPy system. Seven native positive configurations reproduced their accepted verdicts. | [WSL2 study tables](../evidence/stage0-wsl2/studies.md) |
| Process and policy | The original controller applied the DOF cap only to a retry. It now checks initial levels, caps a level timeout at the remaining wall budget and checks frozen inputs again after solver execution. A native nonzero exit cannot produce numerical acceptance. | `tests/test_workflow.py`, `tests/test_archive_mfem.py` |
| Solver output | The original binary exited 0 with zero files when given a missing output directory. Each export now checks its stream after close and reports exit 7 on failure. | `test_binary_reports_output_failure` and [execution transcript](../evidence/stage0-wsl2/transcript.txt) |
| Independent verifier | The previous implementation accepted a string convergence flag, incorrect solver/DOF claims and equal-length boundary segments with a gap and overlap. Typed claims, checked indices, finite coordinates, a Cartesian boundary partition and malformed-input verdicts now gate these cases. | `tests/test_verifier_synthetic.py` |
| Archive | Replay checks the complete file inventory and the stored verdict, matches frozen inputs to the request and reconstructs the permitted execution for the selected attempt. The report generator checks the inventory before writing. A relocated clean run and an original-format run both replayed successfully. | `tests/test_archive_mfem.py`, [quickstart verdict](../evidence/stage0-wsl2/quickstart.json) |
| Build and CI | The Python wheel lock includes all transitive hashes for CPython 3.12 on Linux x86-64. The MFEM source build checks the commit and working tree and locks its build root. CMake requires MFEM 4.10.0. Separate CI jobs run the Python suite and a clean native build with strict integration tests. | [setup and commands](../README.md), [CI workflow](../.github/workflows/verify.yml) |

**Local configuration.** Windows had Git, GitHub CLI, MinGW and multiple Python installations. The default WSL2 distribution was Ubuntu 20.04.6 with GCC 9.4 and Python 3.8. The independent runs used the separate Ubuntu 24.04.1 WSL2 distribution, kernel 6.6.87.2, GCC 13.3.0, CMake 3.28.3 and Python 3.12.3. The machine reported 22 logical CPUs and 15,794 MiB of RAM. The final clean build used six compile jobs, a Linux-filesystem dependency prefix and the pinned serial double-precision MFEM commit. Clone, configure, build and installation took 272 seconds. The installed library and solver hashes are in the [generated study tables](../evidence/stage0-wsl2/studies.md).

The existing workstation build can be resumed in its `Ubuntu` WSL2 distribution with the commands below. The unabridged native outputs are in `runs/stage0-verified/` and the clean build log is in `runs/stage0-logs/clean-build.log`. New machines can follow the [repository setup](../README.md).

```bash
cd '/mnt/d/github项目/project'
source /root/mfem-stage0/locked/bin/activate
export THIRD_PARTY=/root/mfem-stage0/clean/third_party CXX=g++-13 OPENBLAS_NUM_THREADS=1
python -m sciagent quickstart
```

**Remaining limits.** The requester, the local solver and verifier binaries, and the shared mathematical specification remain trust inputs. A study author can choose validation overrides before freezing. Every published positive benchmark here required the independent reference solve. A writer who controls the directory can replace both a manifest and its local checksum. For independent provenance, retain the digest outside that directory. Mesh checks cover the supported Cartesian geometry and boundary partition. Interior mesh topology and shared formulation errors still need independent challenges. The solver subprocess has an address-space limit and timeout. The in-process SciPy reference solve has a DOF gate and can run past the wall deadline.

**CI permissions.** The repository API reported Actions enabled and default workflow permissions set to read. The workflow requests `contents: read` and disables credential persistence at checkout. Each native job builds the pinned MFEM source. Only Python package downloads use a CI cache.

**Hosted reproduction.** The initial push exposed a YAML parsing error in the pip install step. Commit `9573a56` fixed the command scalar. Both jobs then passed on Ubuntu 24.04. The hosted MFEM source build took 18 min 58 s, and the hosted quickstart reproduced the WSL2 finest-pair orders under Python 3.12.16. [Successful GitHub Actions run](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/44776289242)

**Follow-up.** The Stage 1 challenge above measures the false-accept and false-reject rates under a declared sampling distribution.

## Original Claude implementation record

The following sections describe Claude's original environment, measurements and proposed next actions at import. The Stage 0 record above supersedes the earlier milestone ordering.

## 1. Environment (measured, `python3 -m sciagent env`, `runs/*/environment.json`)

| Item | Finding | How determined |
|---|---|---|
| OS / arch | Ubuntu 24.04.5 LTS, Linux 6.18.44, x86_64 | `/etc/os-release`, `uname -a` |
| CPU / RAM / GPU | 2 cores, 8024 MB, no swap; no GPU (`nvidia-smi` absent) | `nproc`, `/proc/meminfo` |
| Compiler / build | g++ 13.3.0, CMake 3.28.3, GNU Make | `--version` |
| Python | 3.13.18; numpy 2.5.3, scipy 1.18.1, pydantic 2.13.5, pytest 9.1.1; **sympy absent** | imports |
| MPI | **absent**: no `mpirun`, no MPI headers/packages | `which`, `dpkg -l` |
| MFEM | not preinstalled; cloned tag `v4.10` (commit `d964264cdb9a13e94a201b6c236c7721e0c8765f`, released 2026-09-01; newest tag listed by `git ls-remote` at audit time) | `git rev-parse`, `CHANGELOG` |
| MFEM configuration | serial, double precision; `MFEM_USE_{MPI,METIS,SUITESPARSE,LAPACK,OPENMP,CUDA,HIP,RAJA,OCCA,PETSC,ZLIB}` all **off** | installed `config/_config.hpp` |
| Network | GitHub reachable via `git`; Python package installation disabled for this account | observed |

### MFEM build record

* Command: `THIRD_PARTY=/home/claude/third_party_clean ./scripts/build_mfem.sh` (CMake, `Release`, `-DMFEM_USE_MPI=OFF`, examples/miniapps off, `make -j2`, local install prefix).
* Result: exit 0, **714 s** wall time, 0 compiler warnings; `libmfem.a` SHA-256 `45e334fe910124b42a23035adb1ee267bbd9a68422c5f4177028cf5141d7c474`. Log: `runs/mfem_build/build_mfem_sh.log`.
* **Provenance incident (resolved):** the first build in this session was written by two concurrent `make` processes in one build tree (a trailing `make install` from a job whose `/usr/bin/time` wrapper failed, plus a second `make -j2`). Its results were moved to `runs_superseded/` with an explanatory `NOTE.md`. All evidence in `runs/` comes from the clean rebuild above. All 18 executed studies produced **bitwise-identical** error values on both builds.

## 2. File inventory

| Path | Purpose |
|---|---|
| `cpp/poisson_mms.cpp` | MFEM v4.10 solver: Cartesian mesh, $H^1$ order $p$, Dirichlet/Neumann from manufactured solution, strict BC partition (exit 3), PA+GS refusal (exit 4), `CGSolver` with checked convergence (exit 5), NaN check (exit 6), MFEM self-reported L2/$H^1$-semi errors, raw-data export, GLVis output, labeled fault injection |
| `cpp/manufactured_cases.hpp` | Reviewed registry of manufactured cases (`sin_sin`, `cos_exp`) with canonical expression strings |
| `cpp/CMakeLists.txt` | Builds `poisson_mms` against an MFEM install via `find_package(MFEM)` |
| `sciagent/spec.py` | Pydantic v2 contracts: `ProblemSpec`, `ValidationSpec`, `RetryPolicy`, `ExecutionSpec`, `StudySpec`; frozen, `extra="forbid"`, canonical-JSON SHA-256 |
| `sciagent/capability.py` | Tested/unsupported/not-verified compatibility matrix |
| `sciagent/environment.py` | Environment and MFEM-build probe (flags, version, git, library and binary hashes) |
| `sciagent/mfem_runner.py` | CMake build, case binding (exact expression match), level execution with timeout and `RLIMIT_AS`, stdout/stderr/exit-code capture |
| `sciagent/controller.py` | State machine, frozen-spec hash checks per attempt, bounded `append_refined_level` retry, manifest |
| `sciagent/evidence.py` | Append-only evidence store (write-once files, JSONL events) |
| `sciagent/__main__.py` | CLI: `env`, `build`, `cases`, `matrix`, `run`, `verify` |
| `verifier/expr.py` | Whitelisted AST expression evaluator; 4th-order FD gradient and Laplacian |
| `verifier/fem.py` | Independent artifact loader, monomial/Vandermonde Lagrange basis, Gauss/Duffy quadrature, affine/bilinear geometry and inverse maps |
| `verifier/verifier.py` | Gates A–D, convergence analysis, typed verdict with failure codes, shared-assumption disclosure, input hashes |
| `specs/*.json` (9) | Positive studies, the retry demo and the budget demo |
| `specs/negative/*.json` (22) | Schema violations, inconsistent/malicious expressions, unsupported backends/solvers/combinations, unbound case, BC label swap, unassigned attribute, 5 injected faults, pre-asymptotic and strict-band studies |
| `tests/` | 96 tests: schema/expressions (22), capability (17), verifier on synthetic data (15), workflow (8), MFEM integration (34) |
| `scripts/` | `build_mfem.sh`, `build_solver.sh`, `run_ex1_baseline.sh`, `run_demo.sh`, `make_specs.py`, `summarize_runs.py` |
| `runs/` | Real evidence (Section 4) |
| `docs/` | `CLAUDE_CRITICAL_REVIEW.md`, `MVP_SPEC.md`, `IMPLEMENTATION_STATUS.md`, `RESULTS.md` (generated) |

## 3. Commands executed and their outcomes

| Command | Outcome |
|---|---|
| `./scripts/build_mfem.sh` | exit 0, 714 s |
| `./scripts/run_ex1_baseline.sh` | ex1 built (exit 0) and run on two meshes, both exit 0; see 4.1 |
| `./scripts/build_solver.sh` | exit 0, no warnings with `-Wall -Wextra` |
| `./scripts/run_demo.sh` | 31 studies; every one reached its expected terminal state (`runs/demo_summary.txt`) |
| `python3 scripts/summarize_runs.py` | wrote `docs/RESULTS.md` from the 31 runs |
| `python3 -m pytest -q` | **96 passed** in 19.2 s (`runs/pytest_output.txt`) |
| `python3 -m sciagent verify runs/20261120T155140827686Z_poisson_sin_sin_p1_tri/attempts/attempt_01` (earlier equivalent on superseded run) | verdict reproduced from artifacts alone; bitwise-identical error values |

## 4. Verified results

### 4.1 Official Example 1 baseline (`runs/baseline_ex1/20261120T155127Z/`)

Unmodified `examples/ex1.cpp` (SHA-256 `8a584173…`) compiled with `g++ -O3 -std=c++17 … -lmfem`.

| Case | Command | Unknowns | Outcome |
|---|---|---|---|
| `star_default` | `./ex1 -m data/star.mesh -no-vis` | 20 801 | exit 0; PCG reached its (squared) tolerance in 111 iterations |
| `square_disc_order2` | `./ex1 -m data/square-disc.mesh -no-vis -o 2` | 79 616 | **exit 0 although stdout reports `PCG: No convergence!`** after the 200-iteration cap |

This is a real demonstration of review risk #1: a zero exit status from the official example does not imply a converged linear solve.

### 4.2 Mandated manufactured-solution benchmark (`runs/20261120T155140827686Z_poisson_sin_sin_p1_tri/`)

$-\Delta u = 2\pi^2\sin(\pi x)\sin(\pi y)$, $u=0$ on $\partial(0,1)^2$; P1 triangles; CG + symmetric Gauss–Seidel, rel. tol. $10^{-12}$. Errors are the verifier's independent values; MFEM's self-reports agree to < $10^{-6}$ relative.

| n | DOFs | CG its | $\lVert u-u_h\rVert_{L^2}$ | $\lvert u-u_h\rvert_{H^1}$ | L2 order | H1 order | MFEM vs. reference solve |
|---|---|---|---|---|---|---|---|
| 8 | 81 | 13 | 2.1133e-02 | 4.3180e-01 | — | — | 6.2e-11 |
| 16 | 289 | 22 | 5.3774e-03 | 2.1754e-01 | 1.9745 | 0.9891 | 2.5e-13 |
| 32 | 1089 | 40 | 1.3504e-03 | 1.0898e-01 | 1.9935 | 0.9973 | 3.0e-14 |
| 64 | 4225 | 74 | 3.3799e-04 | 5.4514e-02 | 1.9984 | 0.9993 | 3.2e-14 |

Verdict: **ACCEPTED**. Finest-pair orders 1.9984 (band [1.8, 2.5]) and 0.9993 (band [0.8, 1.5]). Total controller wall time 0.78 s.

### 4.3 Other accepted studies (full tables in `docs/RESULTS.md`)

| Study | Finest-pair L2 / H1 order | Max MFEM-vs-reference diff |
|---|---|---|
| P2 triangles, `sin_sin` | 2.9986 / 1.9968 | 1.8e-10 |
| P1 triangles, CG+Jacobi | 1.9984 / 0.9993 | 6.2e-11 |
| Q1 quads, `cos_exp`, Neumann top, full assembly | 1.9999 / 0.9999 | 2.0e-12 |
| Q1 quads, same, **partial assembly** + Jacobi | 1.9999 / 0.9999 | 1.2e-12 |
| Q2 quads, same, full assembly | 2.9993 / 1.9995 | 4.3e-12 |
| Q2 quads, same, partial assembly + Jacobi | 2.9993 / 1.9995 | 4.3e-12 |

### 4.4 Bounded retry (`runs/20261120T155144030514Z_retry_demo_sin_sin_p1_tri/`)

Frozen target $\lVert u-u_h\rVert_{L^2} \le 10^{-4}$ on the finest level, `max_attempts = 4`. Attempts 1–3 (finest n = 16, 32, 64; finest L2 = 5.38e-3, 1.35e-3, 3.38e-4) were `RETRY_ELIGIBLE`; attempt 4 (n up to 128, L2 = 8.45e-5) was **ACCEPTED**. Every attempt directory and verdict is preserved; `events.jsonl` records `unchanged: ["problem","validation","policy"]` for each retry; the ValidationSpec hash is identical across attempts. The budget demo (target $10^{-9}$, `max_attempts = 2`) ends in **BUDGET_EXHAUSTED** after two preserved attempts.

### 4.5 Negative tests and the quality gate

Rejected before execution (no solver run): 5 schema violations, inconsistent MMS source (`MMS_INCONSISTENT`, measured mismatch 0.91), malicious expression (`INVALID_EXPRESSION`), MPI and CUDA backends (`BLOCKED/UNSUPPORTED_BACKEND`), UMFPACK (`UNSUPPORTED_SOLVER`), partial assembly + Gauss–Seidel (`UNSUPPORTED_COMBINATION`), order 3 (`NOT_VERIFIED_COMBINATION`), a valid manufactured problem with no compiled case (`UNSUPPORTED_PROBLEM`).

Rejected by the verifier after real MFEM execution (detection matrix in `docs/RESULTS.md`):

| Fault | Solver exit | Detected by |
|---|---|---|
| Wrong source (×1.1) | 0 | reference solve, monotonicity, both order bands |
| Dropped Dirichlet attribute (silent natural BC) | 0 | Dirichlet nodal check, reference solve, monotonicity, both order bands |
| NaN in solution | 6 | finiteness |
| CG stopped after 3 iterations **but reported converged** | 0 | reference solve, L2 order band |
| Self-reported errors replaced by 0 | 0 | self-report consistency (only these gates) |
| Swapped boundary labels (mixed BC) | 0 | label geometry, side length, Dirichlet nodal check, reference solve |
| Mesh attribute 4 unassigned (attribute 5 declared) | 3 | C++ coverage guard → no artifacts → verifier `MISSING_ARTIFACT` |
| Strict order band, resolved meshes | 0 | L2 order band only |

In every row the solver's own success flag was ignored as evidence; four of the faults exit 0 with `solver_converged=true`.

**Measurement finding.** On the n = 1, 2 meshes of `preasymptotic_strict_band`, MFEM's self-reported L2 error (order-6 rule) differs from the verifier's by 6.7e-3 (n=1) and 9.3e-6 (n=2) relative. At n = 1, $u_h\equiv 0$, so the exact error is $\lVert u\rVert_{L^2}=1/2$: the verifier measures 0.500008 and MFEM 0.49666. The self-report and reference-solve gates therefore flag quadrature-dominated coarse meshes, which is the intended behavior, and a test pins this value.

## 5. Acceptance criteria (from `docs/MVP_SPEC.md` §9)

| # | Criterion | Status |
|---|---|---|
| 1 | Real MFEM builds and runs; version and commands recorded | **DONE** |
| 2 | Inputs and BC handling explicit and machine-checkable | **DONE** |
| 3 | Manufactured problem actually solved by MFEM | **DONE** |
| 4 | Errors on ≥ 3 meshes with documented order computation | **DONE** (4 meshes; 6 for the retry) |
| 5 | Independent verifier with typed verdict | **DONE** |
| 6 | Invalid/unsupported inputs rejected explicitly | **DONE** |
| 7 | Reproducible configuration, logs, versions, status per run | **DONE** |
| 8 | Rerun commands and automated tests | **DONE** |
| 9 | Unmet items labelled | see Section 6 |

## 6. Blocked or not done

| Item | Status | Reason |
|---|---|---|
| MPI parallel benchmark (Priority 7) | **BLOCKED** | no MPI implementation installed, MFEM built without MPI/hypre/METIS; installing system packages is outside the allowed scope |
| GPU backends | **BLOCKED** | no GPU device or CUDA/HIP toolkit |
| Symbolic derivation of $f$ from $u_\ast$ | **NOT DONE** | sympy unavailable and package installs are disabled; FD-based consistency check used instead |
| C++ code generation from the IR (arbitrary manufactured solutions) | **NOT DONE** | out of MVP scope; registry + exact binding used |
| Unstructured / Gmsh meshes, curved geometry, 3D | **NOT DONE** | out of MVP scope |
| Order ≥ 3 | **NOT DONE** (rejected as `not_verified`) | verifier is order-generic but not yet exercised |
| LLM planner interface | **NOT DONE** (deliberately) | deterministic baseline first |
| Literature review for research directions | **NOT DONE** | |

## 7. Next actions

1. **MPI path:** in an environment with an MPI implementation, build MFEM with `MFEM_USE_MPI=ON` + hypre + METIS, add `ParMesh`/`HypreBoomerAMG` to `poisson_mms` behind a new capability row, and require nodal agreement with the serial run (same mesh, partition-independent comparison via exported coordinates) plus identical convergence verdicts.
2. **Generated manufactured cases:** compile C++ coefficient code from a small symbolic IR (or an exact-arithmetic Python generator), keep the verifier on the independent string path, and add a mutation-testing harness that randomly perturbs generated code to measure the verifier's false-accept rate.
3. **Mesh generality:** accept Gmsh meshes with named physical groups, replace the unit-square side labels with geometric predicates per physical group, and extend the reference solve to curved (high-order geometry) elements; then add a second PDE family (e.g. variable-coefficient diffusion or linear elasticity) through the same contracts.
