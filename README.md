# mfem-scientific-agent: a verified MFEM Poisson vertical slice

A deterministic workflow that turns a frozen, machine-checked problem specification into **real MFEM computations** and lets an **independent verifier** decide whether the result is accepted. No LLM is used anywhere in the numerical path.

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

## Requirements

The reproduced configuration uses x86-64 Ubuntu 24.04, CPython 3.12, GCC 13, CMake 3.28 and double-precision serial MFEM 4.10. The solver uses POSIX process resource limits. On Windows, select the Ubuntu 24.04 WSL2 distribution shown by `wsl -l -v` and run the following commands in its Bash shell. A checkout in the Linux home directory keeps compilation and the many small evidence files on the Linux filesystem.

## Reproduce

With GitHub access to this private repository, clone it inside Ubuntu. For an existing checkout, start at the directory containing this README.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends git g++-13 cmake make python3.12-venv
# New checkout
git clone https://github.com/rodriguezrobertbfrkx6857-sudo/project.git
cd project
# From the repository root
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements-linux-py312.lock
python -m pip check
export CC=gcc-13 CXX=g++-13 OPENBLAS_NUM_THREADS=1 JOBS=2
python -m pytest -q -m 'not mfem'

./scripts/build_mfem.sh
./scripts/build_solver.sh
python -m pytest -q --require-mfem
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

## Layout

```
cpp/                  MFEM C++ solver (poisson_mms.cpp) and manufactured-case registry
sciagent/             spec.py (contracts), capability.py, environment.py, mfem_runner.py,
                      controller.py (state machine + bounded retry), evidence.py, __main__.py (CLI)
verifier/             independent verifier: expr.py (whitelisted expressions, FD derivatives),
                      fem.py (own Lagrange basis/quadrature/geometry), verifier.py (gates, verdict)
specs/                example studies; specs/negative/ = invalid, unsupported and fault-injection studies
tests/                unit tests (schema, capability, verifier on synthetic data), workflow tests,
                      integration tests against the real MFEM binary
scripts/              build_mfem.sh, build_solver.sh, run_ex1_baseline.sh, run_demo.sh,
                      make_specs.py, summarize_runs.py
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

Empty `validation` / `policy` objects mean the frozen defaults documented in `docs/MVP_SPEC.md` §5–6. A problem runs only if its expressions match a compiled case in `cpp/manufactured_cases.hpp` exactly.

## What "ACCEPTED" means

"Numerical (code) verification passed for this manufactured-solution benchmark": MFEM's discrete solution matches an independently assembled and solved discrete problem, boundary data and labels are correct, error norms recomputed independently match MFEM's claims, and both norms converge at the theoretical rate on the finest pair. It says nothing about physical validity.

## License note

Project code: provided as-is for the requesting user (no license chosen yet). MFEM is used unmodified as an external dependency under its own BSD-3-Clause license (`third_party/mfem-src/LICENSE`); `scripts/run_ex1_baseline.sh` compiles MFEM's own `examples/ex1.cpp` in place without copying it.
