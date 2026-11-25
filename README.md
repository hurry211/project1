# MFEM scientific agent for verified finite-element studies

A deterministic workflow that turns a frozen, machine-checked problem specification into **real MFEM computations** and lets an **independent verifier** decide whether the result is accepted. No LLM is used anywhere in the numerical path.

The CFAC research branch adds a small CPU prototype that certifies energy error for P1 Poisson fields without an exact solution. It can also determine whether further iteration on a fixed P1/RT0 pair can meet a tolerance. [Mathematical specification](docs/CFAC_MATHEMATICAL_SPEC.md) · [Implementation report](docs/CFAC_IMPLEMENTATION_REPORT.md) · [Experiments and replay](docs/CFAC_EXPERIMENTAL_EVIDENCE.md) · [Open problems](docs/CFAC_OPEN_PROBLEMS.md)

An optional RT1 flux reconstruction can certify the same P1 field after a proof that the RT0 pair cannot meet tolerance. At a target of 0.15 the enriched policy accepted on the original mesh. At 0.125 the RT1 trial missed and the controller still refined. [RT1 method, controls and replay](docs/CFAC_RT1_ENRICHMENT.md)

The stopping study compares fixed residual stopping, certified algebraic error balancing and CFAC at identical energy tolerances. The experiment measures delivery of the checked field from a fresh process and records failed attempts. [Matched-accuracy results and replay](docs/CFAC_STOPPING_RESULTS.md) · [Frozen comparison protocol](docs/CFAC_STOPPING_PROTOCOL.md)

An optional CFAC schedule reserves four actions for a checked trial when progress slows on a sufficient refined pair. It recovered the strict signed-stripe case in six of six runs. On three new load families, both CFAC schedules delivered 30 of 36 submissions and classical stopping delivered all 36. [Reserve study, timings and replay](docs/CFAC_RESERVE_RESULTS.md)

The `cfac_retry` policy also checks algebraic error before a refined RT1 trial when the RT0 pair is provably futile. It recovered the historical checkerboard and a new edge patch. It matched classical coverage on 16 of 18 physical problems, with a higher total delivery cost. An offline reference returned checked fields for the remaining two cases on the same meshes. [Results, field bounds and reproduction](docs/CFAC_FUTILE_RESULTS.md)

```
StudySpec (JSON) ─► schema + MMS consistency ─► freeze & hash ─► environment probe ─► capability matrix
     ─► bind to compiled C++ case ─► MFEM poisson_mms (per mesh level, time/memory limited)
     ─► independent verifier (own basis, quadrature, reference solve) ─► ACCEPTED / RETRY / FAILED
     ─► bounded refinement retry (accuracy target only) ─► append-only evidence in runs/
```

* `docs/CLAUDE_CRITICAL_REVIEW.md`: independent review, top-10 risk table, architecture recommendation
* `docs/MVP_SPEC.md`: supported problem, data contracts, state machine, frozen acceptance criteria
* `docs/IMPLEMENTATION_STATUS.md`: inventory, environment, verified results, blockers, next actions
* `docs/RESULTS.md`: generated from `runs/`, with error tables, retry history and the fault × gate detection matrix
* `docs/E3_PERFORMANCE_ADAPTIVE.md`: E3 1.05M-DOF streaming+AMG verifier, unstructured/Example 6 AMR, and separated adaptive strategy research

The performance benchmark measures time from a submitted request to an independently accepted solution. The E2 comparison includes fresh MFEM studies, verifier-only timings and peak memory at 263,169 degrees of freedom. [Results](docs/RESULTS.md#verifier-revision-at-263169-degrees-of-freedom) · [Reproduction commands](docs/VERIFIED_PERFORMANCE.md#measure-a-verifier-change-from-submission-to-acceptance)

## November 2026 review follow-up

E4 adds independently replayable heat, wave and elasticity studies in 2D and 3D. The scalar verifier reaches 1.64 million DOFs under a 3 GiB address-space cap. On one host the four-rank MPI/hypre study reduced verified time from 15.93 to 11.06 seconds relative to one rank. An FP64 GPU audit independently integrated the exported MPI fields. The seven-task adaptive study met all 126 accuracy targets and measured the incremental cost of real model nominations.

[Sixteen findings and reproduction commands](docs/ISSUE_RESOLUTION.md) · [Adaptive study and archive](docs/ADAPTIVE_REPAIRS.md) · [Raw release assets](https://github.com/rodriguezrobertbfrkx6857-sudo/project/releases/tag/e4-verified-studies-20261125)

## Requirements

The reproduced configuration uses x86-64 Ubuntu 24.04, CPython 3.12, GCC 13, CMake 3.28 and double-precision serial MFEM 4.10. The solver uses POSIX process resource limits. On Windows, select the Ubuntu 24.04 WSL2 distribution shown by `wsl -l -v` and run the following commands in its Bash shell. A checkout in the Linux home directory keeps compilation and the many small evidence files on the Linux filesystem.

## Reproduce

Clone this repository inside Ubuntu. For an existing checkout, start at the directory containing this README.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends git g++-13 cmake make time python3.12-venv
# New checkout
git clone https://github.com/rodriguezrobertbfrkx6857-sudo/project.git
cd project
# From the repository root
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements-linux-py312.lock
python -m pip check
export CC=gcc-13 CXX=g++-13 OPENBLAS_NUM_THREADS=1 JOBS=2
python -m pytest -q -m 'not mfem and not mpi'

./scripts/build_mfem.sh
./scripts/build_solver.sh
python -m pytest -q -m 'not mpi' --require-mfem
python -m sciagent quickstart
```

`quickstart` reports the toolchain, runs the P1 triangle study at 8, 16, 32 and 64 cells per side, and replays the stored attempt. Exit 0 requires both numerical acceptance and successful replay. Its JSON includes the attempt directory, the finest-pair orders and the manifest digest. To check a copy of the archive on another machine, preserve the **whole run directory** and keep its manifest digest separately.

```bash
python -m sciagent verify <attempt_dir> --manifest-sha256 <saved_digest>
```

Replace the placeholders with the values printed by `quickstart`. The replay uses Python and the raw files and runs without the native solver. New runs inventory every regular file under the run directory, including the stored verdict, frozen policy, events, logs and native MFEM exports. A separately retained digest detects replacement of the inventory and its local checksum together. Original 0.1.0 archives report the narrower `legacy-input-hashes` mode.

The MFEM source pin is `d964264cdb9a13e94a201b6c236c7721e0c8765f`. The build script checks its checkout and takes an exclusive build lock. `THIRD_PARTY` selects the source, build and installation root. `MFEM_INSTALL` overrides its installation prefix and `POISSON_MMS_BIN` selects the solver binary. Use absolute paths for overrides. The Python lock identifies the tested Linux x86-64 wheels and transitive versions. `requirements-dev.txt` retains the original dependency ranges for other environments.

GitHub Actions runs Python unit tests and a separate clean MFEM source build on Ubuntu 24.04. The native job uses `-m mfem --require-mfem`, which fails immediately if the solver is missing. Plain `python -m pytest -q` keeps the original developer behavior and lists the reason for each skipped integration test.

The native CI job also runs a 284-trial verifier challenge and 14 deliberate reference-checker mutations. To run it locally, use a new control directory for each fresh native capture. Retained archives support Python-only replay of the same inputs. [Design, measured error rates and replay instructions](docs/VERIFIER_CHALLENGE.md)

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-challenge.lock
python -m benchmarks.robustness --generate-controls runs/challenge-controls --report runs/challenge.json
```

```bash
./scripts/run_ex1_baseline.sh
python -m sciagent run specs/poisson_sin_sin_p1_tri.json
RUNS_DIR=runs/demo ./scripts/run_demo.sh
python scripts/summarize_runs.py runs/demo --output runs/demo/report.md
python scripts/compare_baseline.py runs/demo
```

The demo includes expected rejections and one exhausted retry budget. Use an empty demo directory for the historical comparison. It checks the 31 studies, frozen inputs, attempt decisions and original failure codes, and reports the relative difference in accepted error norms. `python -m sciagent env`, `cases` and `matrix` expose environment details, compiled manufactured cases and the capability matrix.

## Performance and extension experiments

Run these commands after building the native solver. Each command takes a fresh output directory.

```bash
python -m benchmarks.performance --output runs/performance --repeats 5
python -m benchmarks.performance --gaussian --output runs/gaussian --repeats 3
python -m sciagent search specs/experiments/strategy_poisson.json --output runs/strategy --repeats 3
python -m sciagent run specs/extensions/anisotropic_1000_q1.json --runs-dir runs/anisotropic
python -m sciagent run specs/extensions/layered_unit_drop_10000.json --runs-dir runs/layered
```

The verifier runs in a separate POSIX process with a 300-second wall limit and a 3072 MiB address-space limit by default. A study can freeze separate values in `policy.verifier_timeout_seconds` and `policy.verifier_memory_mb`. Live verification also respects the remaining study wall budget. A timed-out or failed verifier cannot accept a run.

An accuracy-only retry copies the previous independently verified levels into the new attempt, checks every copied file's digest, and verifies the newly added level. It reevaluates the study-level convergence and accuracy gates over all levels. Archive replay recomputes the independent checks for every level, including copied levels.

The first study covers assembly, retry reuse and the anisotropic case. The follow-up measures the cost of replaying identical archived fields and extends the CPU tests to a localized load and an aligned material interface. [First study](docs/PERFORMANCE_STAGE.md) · [Follow-up protocol and raw data](docs/VERIFIER_SCALING.md) · [导师阶段汇报](docs/导师阶段汇报-第二阶段.md)

`python -m sciagent cases` shows the formulation, boundary traces and manufactured expressions compiled into the binary. Runs that reach case binding save this catalog in `case_registry.json`. Each new native level's `result.json` includes the true algebraic residual of the constrained system and integrated outward flux on all four sides. The generated Markdown report shows their totals for accepted studies. Use these measurements to investigate solver accuracy and flux convergence. The independent reference solve and frozen convergence criteria still determine acceptance. [Field definitions](docs/MVP_SPEC.md#native-diagnostics)

## Select a reviewed study

Use `python -m sciagent plan-catalog` to inspect nine pinned serial templates. A task JSON specifies the authorized physics and discretization constraints. The default rule or an optional local language model nominates a template. With `--execute`, the established controller runs it and the independent verifier checks the numerical evidence. [Planning CLI, review decisions and replay](docs/PLANNER.md) · [Rules versus local-model results](docs/RESULTS.md#stage-4-controlled-planning-2026-11-24)

## Time to an independently verified solution

`python -m sciagent select-solver` uses a pinned calibration campaign to select an authorized serial preconditioner, execute it and verify the result. It can use two-level native pilots when calibration justifies their cost. The paired benchmark includes cold worker startup, solving, exporting, checking and archiving. [Workloads, operating limits and reproduction](docs/VERIFIED_PERFORMANCE.md) · [Experimental results](docs/RESULTS.md#verified-performance-experiments)

## Verified MPI Poisson benchmark

The separate Stage 3 MPI workflow verifies the original P1 triangle Poisson benchmark using 1, 2 and 4 processes. Its timing campaign begins after all ranks pass the matched serial comparison and independent SciPy solve at four resolutions. See the parallel guide for the second MFEM build, the `mpi-study` command and digest-checked offline replay. [MPI setup and experiment](docs/MPI.md) · [Measured single-host scaling](docs/RESULTS.md#stage-3-mpi-poisson-2026-11-24)

## Variable diffusion and named mesh examples

Schema `0.3` adds a reviewed scalar diffusion case with `kappa = 1+x` and an optional, digest-pinned Gmsh 2.2 input. The first command solves the mixed-boundary case on generated triangles. The second imports a named quadrilateral mesh and uses P2 partial assembly. Both run at 8, 16 and 32 cells per side. The matrix command lists the reviewed MFEM revision, build flags, preconditioner and verification coverage for each configuration. [Equation and mesh contract](docs/MVP_SPEC.md#10-stage-2-scalar-diffusion-and-named-meshes)

```bash
python -m sciagent run specs/stage2/diffusion_mixed_p1_tri.json
python -m sciagent run specs/stage2/diffusion_gmsh_p2_quad.json
python -m sciagent matrix --case sin_sin_variable --mesh-source gmsh22
```

The optional scikit-fem benchmark exercises all three cases on both mesh sources and the seven original Stage 2 solver configurations. The subsequent matched assembly experiment also certifies full-assembly Jacobi on Q1 and Q2 quadrilaterals. Use a fresh capture directory and the locked dependencies from `requirements-challenge.lock`. Replay takes the original manifest digests from the saved report and works with the retained raw archives. [Matched assembly evidence](evidence/e1/assembly.json)

```bash
python -m benchmarks.stage2 --directory runs/scalar-coverage --report runs/scalar-coverage.json
python -m benchmarks.stage2 --directory runs/scalar-coverage --replay runs/scalar-coverage.json --report runs/scalar-replay.json
```

## Layout

```
cpp/                  MFEM C++ solvers (poisson_mms.cpp, poisson_amr.cpp) and manufactured-case registry
sciagent/             spec.py (contracts), capability.py, environment.py, mfem_runner.py, gmsh.py,
                      adaptive.py, controller.py (state machine + bounded retry), evidence.py, __main__.py (CLI)
verifier/             independent verifier: expr.py (whitelisted expressions, FD derivatives),
                      fem.py (chunked Lagrange basis/quadrature/geometry), reference_solver.py (direct & SA-AMG-PCG),
                      amr.py (independent hanging-node C^T K C verifier), verifier.py (gates, verdict)
benchmarks/           performance, challenge, stage2/stage3/stage4 and E3 scalability/AMR/adaptive suites
specs/                example studies; specs/negative/ = invalid, unsupported and fault-injection studies
tests/                unit tests (schema, capability, verifier on synthetic data), workflow tests,
                      integration tests against the real MFEM binaries
scripts/              build_mfem.sh, build_solver.sh, run_ex1_baseline.sh, run_demo.sh,
                      make_specs.py, summarize_runs.py
evidence/             published benchmark archives and SVG figures (e1/, e2/, e3/, stage1..stage4/)
runs/                 evidence: baseline_ex1/, mfem_build/, one directory per study run
third_party/          MFEM source/build/install (not part of this project; BSD-3-Clause, see its LICENSE)
```

## Writing a study spec

```json
{
  "schema_version": "0.2",
  "problem": {
    "id": "poisson_unit_square_mms",
    "equation": "-laplacian(u) = f",
    "domain": "unit_square_2d",
    "exact_solution": "sin(pi*x)*sin(pi*y)",
    "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
    "boundaries": [
      {"attribute": 1, "label": "bottom", "kind": "dirichlet", "value": "from_exact_solution"},
      {"attribute": 2, "label": "right",  "kind": "dirichlet", "value": "from_exact_solution"},
      {"attribute": 3, "label": "top",    "kind": "dirichlet", "value": "from_exact_solution"},
      {"attribute": 4, "label": "left",   "kind": "dirichlet", "value": "from_exact_solution"}
    ]
  },
  "execution": {"mesh_type": "tri", "polynomial_order": 1, "mesh_levels": [8, 16, 32, 64],
                "backend": "serial_cpu", "assembly": "full", "solver": "cg_gs"},
  "validation": {},
  "policy": {}
}
```

Empty `validation` / `policy` objects use the frozen defaults documented in `docs/MVP_SPEC.md` §5–6 and the verifier budgets above. A problem runs only if its equation and expressions match a compiled case in `cpp/manufactured_cases.hpp`.

## What "ACCEPTED" means

The reviewed studies check the mesh, boundary data, solver record and independently integrated field errors. Their reference solution comes from a separate SciPy assembly and sparse direct solve (or residual-verified SA-AMG-PCG at large scale in E3). Smooth cases check the finest-pair convergence rates. For layered diffusion, the aligned Q1 space contains the analytic solution. Each level must satisfy frozen limits on nodal error, material flux, the one-sided interface flux jump and energy. The verdict applies to the declared mathematical model and discretization.

## E3 Deep Performance Optimization & Adaptive Decision Research

Branch `research/e3-scaling-adaptive` extends E2 across three validated pillars documented in [`docs/E3_PERFORMANCE_ADAPTIVE.md`](docs/E3_PERFORMANCE_ADAPTIVE.md) with raw archives and SVG plots in [`evidence/e3/`](evidence/e3/):

- **Verifier Scalability to 1.05M DOFs** (`verifier/reference_solver.py`, `verifier/fem.py`, `verifier/verifier.py`, `benchmarks/e3_scaling.py`): Chunked ASCII I/O, bounded-slice quadrature/assembly, and Oettli–Prager backward-stability verified SA-AMG-PCG verify `1,050,625` DOFs (`5.44e-14` relative agreement vs MFEM) within `1,081.4 MiB` peak RSS (`38.36 s`), eliminating SuperLU direct OOM above ~500k DOFs under a 3 GiB cap.
- **Unstructured Gmsh, High-Order P3/Q3–P4/Q4 & MFEM Example 6 AMR** (`cpp/poisson_amr.cpp`, `sciagent/gmsh.py`, `verifier/amr.py`, `benchmarks/e3_complex_meshes.py`): Validates unstructured Delaunay triangles and distorted quads (`P1–P4`, `Q1–Q4`), Zienkiewicz–Zhu AMR with independent geometric hanging-node $C^T K C$ constraint verification on `gaussian_steep` and `lshape_singularity`, and empirical `repeated_element_period` speedup (`95x–1,087x`) vs degradation (`period == E`).
- In E3 the predictor and offline adviser each met the accuracy target in 12 of 15 trials. The fixed policy met it in zero trials. An equal-quality speedup against the fixed policy is undefined.

## License note

Project code: provided as-is for the requesting user (no license chosen yet). MFEM is used unmodified as an external dependency under its own BSD-3-Clause license (`third_party/mfem-src/LICENSE`); `scripts/run_ex1_baseline.sh` compiles MFEM's own `examples/ex1.cpp` in place without copying it.
