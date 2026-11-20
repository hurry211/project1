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

Linux, g++ ≥ 11 (tested with 13.3), CMake ≥ 3.16 (tested with 3.28), git, Python ≥ 3.10 (tested with 3.13) with `numpy`, `scipy`, `pydantic>=2`, `pytest`. No MPI, GPU or optional MFEM dependency is needed.

## Reproduce

```bash
cd mfem-scientific-agent

# 1. Fetch and build the pinned MFEM release (v4.10, commit d964264c...) into ./third_party
#    (714 s measured with 2 cores; nothing is installed system-wide)
./scripts/build_mfem.sh

# 2. Baseline: build and run the UNMODIFIED official MFEM Example 1, evidence in runs/baseline_ex1/
./scripts/run_ex1_baseline.sh

# 3. Build the project's MFEM solver (cpp/poisson_mms.cpp) and list its compiled cases
./scripts/build_solver.sh

# 4. Run the mandated benchmark end to end (exit code 0 only if ACCEPTED)
python3 -m sciagent run specs/poisson_sin_sin_p1_tri.json

# 5. Run every positive and negative spec, then regenerate docs/RESULTS.md
./scripts/run_demo.sh
python3 scripts/summarize_runs.py

# 6. Tests: unit (no MFEM needed) + integration (real MFEM; skipped if step 3 not done)
python3 -m pytest -q

# 7. Re-verify any stored attempt from its artifacts alone
python3 -m sciagent verify runs/<run_dir>/attempts/attempt_01
```

Useful extras: `python3 -m sciagent env` (environment probe), `python3 -m sciagent matrix` (capability matrix), `python3 -m sciagent cases`.

Environment variables: `MFEM_INSTALL` (default `./third_party/mfem-install`), `POISSON_MMS_BIN` (default `./build/cpp/poisson_mms`), `THIRD_PARTY`, `JOBS`.

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
