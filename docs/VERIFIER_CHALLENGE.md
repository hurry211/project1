# Stage 1 verifier challenge

## Experiment

The two verifiers were scored on identical native archives, trial draws and fault witnesses. The earlier implementation is commit `46eebcd` (verifier 0.2.0). The repaired implementation is `8910b4f` (verifier 0.3.0). Each report identifies the Python sources by SHA-256, the seven control manifests and the native binary. The local run used WSL2 Ubuntu 24.04, CPython 3.12.3, MFEM 4.10.0 and scikit-fem 12.0.2. [Before report](../evidence/stage1/verifier-before.json) · [After report](../evidence/stage1/verifier-after.json)

Each of the seven canonical positive requests was run through the controller with its first three mesh levels. The two mesh sequences were 8, 16, 32 for order one and 4, 8, 16 for order two. Every validation threshold came from the original request. The controller's terminal state and injected-fault labels were excluded from construction of the clean labels. The separate 31-study comparison used every original mesh level and retry. [Control generator](../benchmarks/controls.py) · [Canonical comparison](../evidence/stage1/canonical-comparison.json)

The third implementation reconstructs each mesh with scikit-fem, using the canonical Cartesian knots and diagonal. It matches physical node coordinates by a bijection and checks the native cell vertices, orientation, local DOF incidence and labelled boundary segments. Its Poisson solve uses scikit-fem basis functions, analytic gradients, order-12 integration, Dirichlet values from the exact solution and the analytic top flux for the mixed problem. It checks both norms of the exported field, the nodal field against the regenerated Galerkin solution, the Dirichlet values and the convergence bands. Across 21 levels the largest relative nodal difference was `1.783e-10` and the largest relative error-norm difference was `7.932e-7`. Both met the frozen limits of `1e-7` and `1e-6`. The maximum Dirichlet difference was zero. [Oracle source](../benchmarks/reference.py) · [scikit-fem API](https://scikit-fem.readthedocs.io/en/stable/api.html)

## How the trials are generated

Every trial starts from a fresh copy. Numerical trials call the live verifier with required native run records. Archive trials call archived replay with the expected manifest digest obtained from the control index before making the copy. A numerical rejection receives credit only when an evaluated gate fails. An unexpected exception is recorded as `ERROR` and fails the campaign. Acceptance of a fault counts as a false accept. Every outcome other than acceptance of a valid control counts as a false reject. The JSON also records retries and errors separately if either occurs. [Campaign and scoring](../benchmarks/robustness.py) · [Metric implementation](../benchmarks/scoring.py)

| Numerical fault | Construction and independent witness |
|---|---|
| Reversed volume source | Solve a new system with the opposite volume load and original boundary conditions. Integrate honest error claims for that new field and measure its distance from the canonical solution. |
| Incorrect stiffness | Scale the stiffness by 1.0001, solve again and recompute honest error claims. The field difference must exceed the frozen reference tolerance. |
| Underintegrated error norms | Integrate the unchanged field with one centroid per cell. The resulting discrepancy must exceed the self-report tolerance. |
| False quadrature declaration | Replace the reported RHS integration order by one after recording the native order. |
| Boundary permutation | Exchange bottom and top attributes, preserving their counts. Count the segments on the wrong side. |
| Exchanged DOF values | Exchange the most unequal interior nodal values. Record the measured difference. |
| Duplicate element | Replace an interior element and its local DOF records with another element from the same mesh while preserving row and element counts. Record both IDs. |
| Displaced interior mesh | Move vertex and nodal x coordinates by `2e-10 * 16*x*(1-x)*y*(1-y)`. Boundaries stay fixed. The measured displacement exceeds the coordinate tolerance. |
| NaN and infinity | Replace one randomly chosen interior value with each nonfinite type in separate trials. Count the affected nodes. |
| Inconsistent CG residual | Keep the convergence flag and set the final norm to the nonzero initial norm. Record the ratio to the requested stopping threshold. |
| Impossible CG iteration count | Set the count to one above the requested maximum. |
| Wrong cached result | Take a genuine native result from another configuration at the same mesh size. Record which configuration fields differ. |
| Inconsistent manufactured source | Negate the source in the mathematical input and evaluate the change at (1/4, 1/4) against the analytic source. |

The separate clean numerical controls use original exports and consistent permutations of both global DOFs and elements. Each permuted export must decode to the same field at every coordinate, with the same cell incidence. Clean archive trials relocate the directory before replay. Archive faults either disable the required reference solve, raise the allowed retry count or present a complete internally consistent donor archive under the original digest. [Mutation source](../benchmarks/mutations.py)

The stopping-record check follows MFEM's preconditioned norm and its relative stopping test. The native runner requests zero absolute tolerance. On a claimed convergence, the reported final norm must be at most the requested relative tolerance times the initial norm, allowing one representable floating-point step for the reported square roots. [Pinned MFEM CG implementation](https://github.com/mfem/mfem/blob/d964264cdb9a13e94a201b6c236c7721e0c8765f/linalg/solvers.cpp#L895-L924)

The fixed panel crosses seven configurations with each family. Its numerical faults affect the coarsest level. The seeded sample draws configurations and families uniformly with replacement within each of the four trial types. It also chooses one of the configuration's three levels uniformly. The public seed is `20261120`. Draw seeds, the affected levels, the fault witnesses, evaluated gates, failed gates and final decisions are recorded per trial. The fixed panel is reported as a census. Exact two-sided 95% Clopper–Pearson intervals in the sampled cohort are conditional on this generator and these seven controls. Repeated inputs are possible. Intervals on individual gates are marginal and have no simultaneous-coverage adjustment.

## Results

| Trial class | Fixed panel | Sampled cohort | Stage 0 errors, panel / sample | Stage 1 errors, panel / sample | Stage 1 sampled error rate, exact 95% interval |
|---|---:|---:|---:|---:|---:|
| Numerical faults, false accepts | 98 | 64 | 28 / 17 | 0 / 0 | 0–5.60% |
| Correct numerical studies, false rejects | 14 | 32 | 0 / 0 | 0 / 0 | 0–10.89% |
| Archive faults, false accepts | 21 | 32 | 0 / 0 | 0 / 0 | 0–10.89% |
| Correct relocated archives, false rejects | 7 | 16 | 0 / 0 | 0 / 0 | 0–20.59% |

The previous verifier accepted all seven panel examples in each of four families. It accepted coherent interior warps, false RHS quadrature declarations, CG iteration counts beyond the maximum and residual records inconsistent with claimed convergence. Its sampled numerical false-accept rate was 17/64 (26.56%, exact 95% interval 16.30–39.09%). The repaired verifier rejected 98/98 fixed and 64/64 sampled numerical faults. The sampled union detection interval was 94.40–100%. All 53 archive faults were rejected and all 23 clean archive trials passed. The 284 scored trials on each verifier produced zero crashes or retries. [Per-trial evidence](../evidence/stage1/verifier-after.json)

Selected fixed-panel numerical gates are shown below. The full table in each JSON report includes zero-hit gates, sampled uncertainties and per-family and per-configuration outcomes. The denominator for overall detection is 98. The evaluated column counts trials where the gate ran at the affected level or at study scope. An early return can prevent later checks from running. Multiple gates can detect the same fault.

| Gate | Detected | Evaluated | Sole detection |
|---|---:|---:|---:|
| Manufactured source consistency | 7 | 98 | 7 |
| CG stopping record | 14 | 91 | 14 |
| Quadrature declarations | 10 | 91 | 7 |
| Finite nodal values | 14 | 91 | 14 |
| Cartesian cell partition | 12 | 75 | 0 |
| Cartesian node lattice | 7 | 75 | 0 |
| Boundary label geometry | 7 | 75 | 0 |
| Independent L2 self-report comparison | 24 | 75 | 0 |
| Independent H1 self-report comparison | 24 | 75 | 0 |
| Independent reference solve agreement | 30 | 75 | 14 |
| Union of all numerical gates | 98 | 98 | — |

A separate calibration deliberately perturbed the production reference checker itself. Reversing its volume load and scaling its stiffness by 1.25 each caused all seven independently certified clean studies to fail only the reference agreement check. Every affected level exceeded the frozen tolerance. The smallest relative discrepancy was 0.05672. These 14 deliberate checker failures per revision are reported separately from the 284 production-verifier trials.

## Trust boundaries

| Shared input or implementation | What the measurements establish | Residual assumption |
|---|---|---|
| PDE, boundary meanings and analytic truth | Separate code for the two closed-form fields, gradients and mixed flux checks a stated manufactured problem. | Agreement on the same incorrectly specified physical model can pass all three implementations. Physical validity needs other evidence. |
| Mesh and DOF mapping | The live verifier checks Cartesian cell coverage, complementary triangle pairs, nodal lattice coverage and boundary intervals. The third implementation generates its own connectivity and checks each native cell against it. | The live verifier integrates the exported coordinates and permits either conforming cell diagonal. The independent control oracle pins the native benchmark diagonal. Curved meshes and untested element orders require their own geometry checks. |
| Discretization, quadrature and solve | MFEM, the production verifier and the third implementation build their bases and forms through separate code paths. A one-point integration error and two deliberately faulty reference assemblies were observable. | The two Python implementations share NumPy, SciPy sparse direct solves and floating-point arithmetic. All three use the same Galerkin weak form. The oracle's high-order integration also has finite error. |
| Solver records and binary | Field agreement, native process records, iteration bounds and a consistent stopping inequality constrain accepted outputs. The binary and control manifests have recorded digests. | A plausible iteration log or quadrature-order label cannot establish which instructions an untrusted binary executed. A compiler, MFEM dependency or host under common control can affect production and capture. |
| Evidence custody | Replaying with an independently held digest detects replacement of an entire otherwise consistent archive. Frozen policy and validation must match the recorded request. | A checksum retained only beside the archive can be replaced with it. A digest establishes equality to the retained bytes. Origin and identity require a trusted distribution channel or independently held signature. |
| Statistical scope | The measured rates cover the documented fault families, amplitudes, mesh levels and sampling distribution. | Unknown faults, combinations of faults, other compilers and new problems have no measured population rate here. Authorization of validation overrides before a request is frozen remains a policy decision. |

## Reproduction

Start from the WSL2 or Ubuntu environment in the README. The extra hash-locked wheel is used by the challenge harness. The production controller and verifier run with the original Python lock. Set `OPENBLAS_NUM_THREADS=1` for the published local comparison.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-challenge.lock
python -m pip check
export OPENBLAS_NUM_THREADS=1
out=$(mktemp -d)
python -m benchmarks.robustness --generate-controls "$out/controls" --report "$out/after.json"
```

For exact-input replay, download the archived controls and verify the compressed bytes against the digest in this checkout. This path also works on a machine without the MFEM binary.

```bash
repo=$(pwd)
out=$(mktemp -d)
gh release download stage1-verifier-2026-11-20 --pattern mfem-stage1-controls.tar.gz --dir "$out"
(cd "$out" && sha256sum -c "$repo/evidence/stage1/controls.sha256")
tar -xzf "$out/mfem-stage1-controls.tar.gz" -C "$out"
python -m benchmarks.robustness --controls "$out/controls" --report "$out/after.json"
git worktree add --detach "$out/stage0-src" 46eebcd
cp -R benchmarks "$out/stage0-src/"
(cd "$out/stage0-src" && python -m benchmarks.robustness --controls "$out/controls" --report "$out/before.json")
```

The final command exits 1 on the measured Stage 0 false accepts after writing its complete report. Runs with freshly generated controls have new timestamps, process timings and manifest digests. [Retained control archives](https://github.com/rodriguezrobertbfrkx6857-sudo/project/releases/tag/stage1-verifier-2026-11-20) · [Control bundle checksum](../evidence/stage1/controls.sha256)
