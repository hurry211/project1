# MVP Specification: Verified MFEM Poisson Vertical Slice

**Version:** 0.2 (schema `0.2`), November 2026
**Status of this document:** normative for the code in this repository. The acceptance thresholds in Section 6 were fixed *before* the first MFEM run of the manufactured-solution studies. Any later change must be committed as a new `ValidationSpec` version, never edited inside a run.

## 1. Supported problem

The steady scalar Poisson problem on the unit square,

$$
-\Delta u = f \ \text{in}\ \Omega=(0,1)^2,\qquad
u = u_\ast \ \text{on}\ \Gamma_D,\qquad
\nabla u\cdot n = \nabla u_\ast\cdot n \ \text{on}\ \Gamma_N,
$$

with $\Gamma_D \neq \emptyset$, $\Gamma_D\cup\Gamma_N=\partial\Omega$, $\Gamma_D\cap\Gamma_N=\emptyset$, and a manufactured exact solution $u_\ast$ with $f=-\Delta u_\ast$.

Compiled manufactured cases (`cpp/manufactured_cases.hpp`):

| Case id | $u_\ast$ | $f$ | Purpose |
|---|---|---|---|
| `sin_sin` | $\sin(\pi x)\sin(\pi y)$ | $2\pi^2\sin(\pi x)\sin(\pi y)$ | Mandated benchmark; homogeneous Dirichlet |
| `cos_exp` | $\cos(\pi x)e^{y}$ | $(\pi^2-1)\cos(\pi x)e^{y}$ | Distinct data on every side; used with a Neumann side so that label swaps and dropped BCs change the solution |

**Discretization.** Conforming $H^1$ Lagrange elements of order $p\in\{1,2\}$ (MFEM `H1_FECollection`, default Gauss–Lobatto nodal basis) on an $n\times n$ Cartesian mesh (`Mesh::MakeCartesian2D`) of triangles or quadrilaterals. Standard Galerkin weak form; Dirichlet data by nodal interpolation (`GridFunction::ProjectBdrCoefficient`), Neumann data by `BoundaryNormalLFIntegrator` with $\nabla u_\ast$.

**Mesh boundary attributes (verified in MFEM v4.10 source, `Mesh::Make2D`):** 1 = bottom ($y=0$), 2 = right ($x=1$), 3 = top ($y=1$), 4 = left ($x=0$). The verifier does not rely on this table: it checks every exported boundary segment against the side declared in the `ProblemSpec`.

## 2. Assumptions

1. Serial CPU execution only (the probed MFEM build has no MPI, CUDA, HIP, SuiteSparse or hypre).
2. Boundary data are always derived from the exact solution (`"value": "from_exact_solution"`). Any other boundary value is rejected at schema level.
3. Expressions use a whitelist grammar (`x`, `y`, `pi`, `e`, numbers, `+ - * / **`, `sin cos tan exp log sqrt sinh cosh tanh`).
4. A `ProblemSpec` runs only if its expression strings match a compiled C++ case exactly (whitespace-insensitive). No code generation happens in the MVP.

## 3. Data contracts

| Object | Owner | Mutability | Location in run directory |
|---|---|---|---|
| `ProblemSpec` | user | frozen at `SPEC_VALIDATED`; SHA-256 recorded and re-checked before every attempt | `frozen/problem.json` |
| `ValidationSpec` | user/reviewer | frozen; the controller has no code path that writes it | `frozen/validation.json` |
| `RetryPolicy` | user/reviewer | frozen | `frozen/policy.json` |
| `ExecutionSpec` | controller | new version per attempt, only via `append_refined_level` | `attempts/attempt_NN/execution.json` |
| Level artifacts | MFEM solver | written once per level directory (`exist_ok=False`) | `attempts/attempt_NN/levels/nNNNN/` |
| Verdict | verifier | written once per attempt | `attempts/attempt_NN/verdict.json` |
| Events | controller | append-only JSON lines | `events.jsonl` |
| Manifest | controller | written once at termination | `manifest.json` |

Schema definitions are in `sciagent/spec.py` (pydantic v2, `frozen=True`, `extra="forbid"`).

### Solver artifact contract (`cpp/poisson_mms.cpp` → verifier)

| File | Content |
|---|---|
| `result.json` | solver *claims*: parameters, convergence flag, iterations, norms, self-reported errors, injected-fault label, timings, MFEM version/git string |
| `elements.txt` | per element: geometry and vertex coordinates in MFEM vertex order |
| `element_dofs.txt` | per (element, local dof): global dof id and physical node coordinates |
| `dof_values.txt` | nodal values of $u_h$ (17 significant digits) |
| `boundary.txt` | per boundary segment: attribute and endpoint coordinates |
| `mesh.mesh`, `solution.gf` | native MFEM/GLVis output (not read by the verifier) |
| `stdout.txt`, `stderr.txt`, `command.json`, `run_record.json` | written by the runner |

Solver exit codes: 0 OK, 2 invalid arguments, 3 boundary-coverage error, 4 unsupported combination, 5 CG not converged, 6 non-finite solution.

## 4. State machine

```
RECEIVED ─► SPEC_VALIDATED ─► ENVIRONMENT_CHECKED ─► READY ─► RUNNING ─► VERIFYING ─┬─► ACCEPTED
   │              │                    │               │          ▲                 ├─► RETRY_ALLOWED ─┐
   ▼              ▼                    ▼               ▼          └─────────────────┼──────────────────┘
INVALID_SPEC  INVALID_SPEC          BLOCKED       BLOCKED/FAILED                    ├─► FAILED
(schema)      (MMS check)        (MFEM/binary)  (capability/binding)                └─► BUDGET_EXHAUSTED
```

Terminal codes: `INVALID_SPEC`, `MMS_INCONSISTENT`, `INVALID_EXPRESSION`, `MFEM_NOT_FOUND`, `BUILD_MISSING`, `BUILD_FAILURE`, `UNSUPPORTED_BACKEND`, `UNSUPPORTED_SOLVER`, `UNSUPPORTED_COMBINATION`, `NOT_VERIFIED_COMBINATION`, `UNSUPPORTED_PROBLEM`, `SPEC_TAMPERED`, `SOLVER_DIVERGED`, `VALIDATION_FAILED`, `RESOURCE_LIMIT`, `BUDGET_EXHAUSTED`, `OK`.

## 5. Corrective loop

* **Trigger:** verifier verdict `RETRY_ELIGIBLE`, which is produced only when *every* check passes except the pre-committed accuracy target `max_l2_error_finest`.
* **Action:** append one level $2n_{\text{finest}}$ to `mesh_levels`; rerun all levels in a new attempt directory.
* **Bounds:** `max_attempts` (default 3, ≤ 6), `max_dofs_per_level` (default 300 000, checked before launching), `max_total_wall_seconds` (default 900 s), per-level timeout (300 s) and address-space limit (3 GiB, `RLIMIT_AS`).
* **Never changed by the loop:** PDE, boundary conditions, exact solution, source, validation thresholds, retry policy. A `RETRY_ALLOWED` event records `unchanged: ["problem", "validation", "policy"]`.
* All failed attempts remain on disk with their own verdicts.

## 6. Acceptance criteria (frozen defaults of `ValidationSpec`)

### 6.1 Per-level checks (all must pass)

| Check | Code on failure | Threshold | Rationale |
|---|---|---|---|
| Solver claims match frozen spec (n, p, mesh, BC sets, expressions) | `CLAIM_SPEC_MISMATCH` | exact | a run of a different problem is not evidence |
| Solver reported convergence | `SOLVER_NOT_CONVERGED` | true | necessary, never sufficient |
| Artifacts present, parseable, ids contiguous | `MISSING_ARTIFACT` / `MALFORMED_ARTIFACT` | — | |
| All nodal values finite | `NON_FINITE_SOLUTION` | — | |
| Global-dof coordinates agree across elements | `MALFORMED_ARTIFACT` | $10^{-12}$ | export/numbering consistency |
| Element count equals $n^2$ (quad) or $2n^2$ (tri) | `MESH_MISMATCH` | exact | |
| Mesh boundary attributes = declared attributes | `BOUNDARY_COVERAGE` | exact | no silent natural BC |
| Every segment of attribute *a* lies on the side declared for *a* | `BOUNDARY_LABEL_MISMATCH` | $10^{-12}$ | catches label swaps |
| Each side fully covered (total length 1) | `BOUNDARY_COVERAGE` | $10^{-10}$ | |
| $\max \lvert u_h - u_\ast\rvert$ at Dirichlet nodes (located geometrically) | `DIRICHLET_VIOLATION` | $10^{-10}$ | nodal interpolation is exact up to rounding |
| Positive Jacobian at all quadrature points | `INVALID_MESH` | $>0$ | |
| Self-reported L2 and $H^1$-seminorm errors agree with the verifier | `SELF_REPORT_MISMATCH` | rel. $10^{-6}$ | both integrate the same $u_h$ with high-order rules; disagreement means a wrong claim or export |
| Independent reference solve: $\lVert u_h^{\text{MFEM}} - u_h^{\text{ref}}\rVert_\infty / \lVert u_h^{\text{ref}}\rVert_\infty$ | `REFERENCE_MISMATCH` | $10^{-7}$ | algebraic error at CG rel. tol. $10^{-12}$ plus load-vector quadrature differences are expected far below this; a wrong source, dropped BC or premature stop is far above |

### 6.2 Study-level checks

| Check | Code | Threshold | Rationale |
|---|---|---|---|
| MMS consistency $\max\lvert -\Delta u_\ast - f\rvert/(1+\lvert f\rvert)$ at 256 fixed points (4th-order FD, $h=10^{-3}$) | `MMS_INCONSISTENT` | $10^{-6}$ | FD truncation ≈ $10^{-10}$ for these cases |
| Number of levels | `INSUFFICIENT_LEVELS` | ≥ 3 | |
| Monotone decrease of both error norms | `NON_MONOTONE_CONVERGENCE` | strict | |
| Observed L2 order on the finest pair | `CONVERGENCE_ORDER_OUT_OF_BAND` | $[p+1-0.2,\ p+1+0.5]$ | smooth solution, quasi-uniform mesh; asymmetric band tolerates superconvergent pre-asymptotic behavior but not sub-optimal rates |
| Observed $H^1$-seminorm order on the finest pair | `CONVERGENCE_ORDER_OUT_OF_BAND` | $[p-0.2,\ p+0.5]$ | idem |
| Finest L2 error ≤ `max_l2_error_finest` (optional) | `ACCURACY_NOT_MET` → `RETRY_ELIGIBLE` | user-set | the only retryable failure |

$p_{\text{obs}} = \log(E_{h}/E_{h/2})/\log 2$.

### 6.3 Error measurement method

The verifier reconstructs $u_h$ on each element from the exported nodal values with its own Lagrange basis (Vandermonde inversion on $P_p$ or $Q_p$ at the inverse-mapped node coordinates), integrates with its own Gauss rule ($m=p+4$ points per direction; Duffy-collapsed on triangles), evaluates $u_\ast$ from the frozen expression string, and evaluates $\nabla u_\ast$ by 4th-order central differences ($h=10^{-3}$, truncation ≲ $10^{-10}$). MFEM's self-reported errors use `ComputeL2Error` / `ComputeGradError` with `IntRules.Get(geom, 2p+4)`.

## 7. Numerical verification vs. physical validation

* **Numerical (code) verification** — this MVP: the code solves the stated discrete problem correctly and the discretization converges at the theoretical rate for a manufactured solution.
* **Physical-model validation** — out of scope: whether $-\Delta u = f$ with these data models any physical system. No physical claim is made, and accepted verdicts say "numerical (code) verification passed for this manufactured-solution benchmark".

## 8. Capability matrix

`tested` = exercised by `tests/test_integration_mfem.py` in this environment. Everything else is rejected.

| Mesh | p | Assembly | Solver | Status |
|---|---|---|---|---|
| tri | 1 | full | CG + symmetric Gauss–Seidel (`cg_gs`) | tested |
| tri | 2 | full | `cg_gs` | tested |
| quad | 1 | full | `cg_gs` | tested |
| quad | 2 | full | `cg_gs` | tested |
| tri | 1 | full | CG + Jacobi (`cg_jacobi`) | tested |
| quad | 1 | partial | `cg_jacobi` (`OperatorJacobiSmoother`) | tested |
| quad | 2 | partial | `cg_jacobi` | tested |
| any | any | partial | `cg_gs` | unsupported (GS needs an assembled `SparseMatrix`) |
| any | any | any | `umfpack`, `hypre_amg` | unsupported (not in MFEM build) |
| any | any | any | backend `mpi_cpu`, `cuda`, `hip` | unsupported (MFEM built without MPI/CUDA/HIP) |
| other | | | | not_verified |

## 9. MVP acceptance criteria (project level)

| # | Criterion | Evidence required |
|---|---|---|
| 1 | Real MFEM builds and runs; exact version and commands recorded | `runs/baseline_ex1/*`, `environment.json` |
| 2 | Mathematical inputs and BC handling explicit and machine-checked | `specs/*.json`, schema tests |
| 3 | The manufactured problem itself is solved by MFEM | `result.json` per level |
| 4 | Errors at ≥ 3 meshes with documented order computation | `verdict.json` |
| 5 | Verifier independently inspects data and returns a typed verdict | verifier code + synthetic-data tests |
| 6 | Invalid/unsupported inputs rejected explicitly | negative specs + tests |
| 7 | Every run has configuration, logs, versions, unambiguous status | run directory layout |
| 8 | Rerun commands and automated tests | `README.md`, `pytest` |
| 9 | Unmet criteria marked NOT DONE/BLOCKED | `docs/IMPLEMENTATION_STATUS.md` |
