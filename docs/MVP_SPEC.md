# Verified MFEM scalar diffusion specification

This specification covers the original Poisson contract (schema `0.2`) and the Stage 2 scalar diffusion extension (schema `0.3`), November 2026. Section 10 defines the additional coefficient, manufactured field and mesh input. Section 6 gives the default acceptance thresholds fixed before the first MFEM studies. Changes to those criteria require a new `ValidationSpec` version.

## 1. Original Poisson problem

The steady scalar Poisson problem on the unit square,

$$
-\Delta u = f \ \text{in}\ \Omega=(0,1)^2,\qquad
u = u_\ast \ \text{on}\ \Gamma_D,\qquad
\nabla u\cdot n = \nabla u_\ast\cdot n \ \text{on}\ \Gamma_N,
$$

with $\Gamma_D \neq \emptyset$, $\Gamma_D\cup\Gamma_N=\partial\Omega$, $\Gamma_D\cap\Gamma_N=\emptyset$, and a manufactured exact solution $u_\ast$ with $f=-\Delta u_\ast$.

The original Poisson cases appear in `cpp/manufactured_cases.hpp`.

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
| Manifest | controller | written once at termination with a relative-path SHA-256 inventory of all run artifacts | `manifest.json` |
| Manifest digest | controller | checksum of the complete manifest bytes | `manifest.sha256` |
| Compiled case registry | controller | snapshot of the solver catalog used for case binding, covered by the manifest inventory | `case_registry.json` |

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

Solver exit codes are 0 for success, 2 for invalid arguments, 3 for a boundary-coverage error, 4 for an unsupported combination, 5 for CG non-convergence, 6 for a non-finite solution and 7 for an artifact write or close failure.

Completed-run replay checks the archive inventory, the saved decision and the recomputed numerical verdict. The frozen models must agree with the original request. Each execution must match the initial execution plus the number of level doublings authorized by preceding retry decisions. An archive mismatch yields `REJECTED / EVIDENCE_INTEGRITY`. A caller can supply a separately saved manifest SHA-256 to detect coordinated changes to the archive and its local checksum. Historical 0.1.0 runs have verifier-input and frozen-model hashes with narrower coverage.

### Native diagnostics

The version-2 compiled catalog provides the original common Poisson formulation and a case-specific formulation for variable diffusion. Each lists the equation, domain, scalar H¹ space, diffusion integrator and essential and natural traces. The variable formulation also declares its coefficient. The controller records the selected case ID in the run manifest. The independent verifier evaluates the expressions in the frozen request.

The native `result.json` object `diagnostics` has `schema_version: 1`. These solver measurements carry no acceptance thresholds.

| Field | Measurement |
|---|---|
| `algebraic_residual.absolute_l2` | Euclidean norm of `A X − B` after CG, using the actual constrained MFEM operator and vectors returned by `FormLinearSystem`. The operator is applied in both full and partial assembly. |
| `algebraic_residual.rhs_l2`, `relative_l2` | Euclidean norm of `B` and the ratio of the absolute residual to that norm. For a zero RHS the absolute norm remains available and the ratio is JSON `null`. CG's preconditioned initial and final norms remain separate fields. |
| `boundary_flux.outward_by_side`, `outward_total` | The signed integral of `−κ grad(u_h)·n` on each exterior side (bottom, right, top, left) and their sum. Here κ=1 for the original Poisson cases. Gradients are taken from adjacent elements. |
| `boundary_flux.source_integral`, `signed_imbalance` | The volume integral of the RHS coefficient used in assembly, including a deliberately scaled source in fault tests. The imbalance is the outward total minus this volume integral. No normalization is required for a cancelling source. |

A nonfinite exported field yields `null` fluxes and imbalance. The source integral is still recorded. The algebraic residual precedes field recovery and any injected export corruption. For conforming H¹ fields the gradient flux generally has discretization error. An accurately solved incorrect RHS can also yield a small native residual. Scientific acceptance continues to use the independent checks in Section 6. Older archives can be replayed without the catalog snapshot or diagnostic fields.

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
| Solver claims match frozen spec (n, p, mesh, BC sets, expressions, assembly, solver, iteration cap, relative tolerance) | `CLAIM_SPEC_MISMATCH` | exact, allowing the documented expression whitespace | binds the computation to the requested configuration |
| Recorded process completed at the requested level | `SOLVER_EXIT` | zero exit and no timeout | required for controller runs and archive replay |
| Solver reported convergence | `SOLVER_NOT_CONVERGED` | true | necessary, never sufficient |
| Native CG iteration and stopping record | `CG_RECORD_INCONSISTENT` | integer iterations within the requested budget, zero requested absolute tolerance, finite nonnegative norms | A convergence claim requires the final norm to meet the relative stopping threshold. One floating-point step accounts for reporting the square roots of MFEM's squared norms. |
| Native RHS and error quadrature declarations | `QUADRATURE_CLAIM_MISMATCH` | both integers equal to $2p+4$ | the integration rule specified by the supported C++ producer |
| Artifacts present, parseable, ids contiguous | `MISSING_ARTIFACT` / `MALFORMED_ARTIFACT` | — | |
| All nodal values finite | `NON_FINITE_SOLUTION` | — | |
| Global-dof coordinates agree across elements | `MALFORMED_ARTIFACT` | $10^{-12}$ | export/numbering consistency |
| Element count equals $n^2$ (quad) or $2n^2$ (tri) | `MESH_MISMATCH` | exact | |
| Every Cartesian cell has one quadrilateral with cyclic edges or two complementary triangles | `MESH_MISMATCH` | complete coverage with vertex tolerance $10^{-12}$ | permits either triangle diagonal and consistent numbering changes |
| P1/P2 global nodes cover the requested Cartesian lattice bijectively | `MESH_MISMATCH` | coordinate tolerance $10^{-12}$ | checks each requested physical node location |
| Geometry matches the requested type and reported/exported DOF counts equal $(np+1)^2$ | `MESH_MISMATCH` | exact | dimension of the supported conforming Cartesian spaces |
| Mesh boundary attributes = declared attributes | `BOUNDARY_COVERAGE` | exact | no silent natural BC |
| Every segment of attribute *a* lies on the side declared for *a* | `BOUNDARY_LABEL_MISMATCH` | $10^{-12}$ | catches label swaps |
| Each side fully covered (total length 1) | `BOUNDARY_COVERAGE` | $10^{-10}$ | |
| Sorted side intervals partition $[0,1]$ at the requested Cartesian knots | `BOUNDARY_COVERAGE` | endpoint tolerance $10^{-12}$ | catches a gap compensated by an overlapping segment |
| $\max \lvert u_h - u_\ast\rvert$ at Dirichlet nodes (located geometrically) | `DIRICHLET_VIOLATION` | $10^{-10}$ | nodal interpolation is exact up to rounding |
| Positive Jacobian at all quadrature points | `INVALID_MESH` | $>0$ | |
| Self-reported L2 and $H^1$-seminorm errors agree with the verifier | `SELF_REPORT_MISMATCH` | rel. $10^{-6}$ | both integrate the same $u_h$ with high-order rules; disagreement means a wrong claim or export |
| Independent reference solve: $\lVert u_h^{\text{MFEM}} - u_h^{\text{ref}}\rVert_\infty / \lVert u_h^{\text{ref}}\rVert_\infty$ | `REFERENCE_MISMATCH` | $10^{-7}$ | algebraic error at CG rel. tol. $10^{-12}$ plus load-vector quadrature differences are expected far below this; a wrong source, dropped BC or premature stop is far above |

### 6.2 Study-level checks

| Check | Code | Threshold | Rationale |
|---|---|---|---|
| MMS consistency $\max\lvert -\nabla\cdot(\kappa\nabla u_\ast)-f\rvert/(1+\lvert f\rvert)$ at 256 fixed points (4th-order FD, $h=10^{-3}$) | `MMS_INCONSISTENT` | $10^{-6}$ | independent differentiation of the frozen exact field and coefficient |
| Number of levels | `INSUFFICIENT_LEVELS` | ≥ 3 | |
| Monotone decrease of both error norms | `NON_MONOTONE_CONVERGENCE` | strict | |
| Observed L2 order on the finest pair | `CONVERGENCE_ORDER_OUT_OF_BAND` | $[p+1-0.2,\ p+1+0.5]$ | smooth solution, quasi-uniform mesh; asymmetric band tolerates superconvergent pre-asymptotic behavior but not sub-optimal rates |
| Observed $H^1$-seminorm order on the finest pair | `CONVERGENCE_ORDER_OUT_OF_BAND` | $[p-0.2,\ p+0.5]$ | idem |
| Finest L2 error ≤ `max_l2_error_finest` (optional) | `ACCURACY_NOT_MET` → `RETRY_ELIGIBLE` | user-set | the only retryable failure |

$p_{\text{obs}} = \log(E_{h}/E_{h/2})/\log 2$.

### 6.3 Error measurement method

The verifier reconstructs $u_h$ from exported nodal values using its own Lagrange basis. It inverts a $P_p$ or $Q_p$ Vandermonde matrix at the inverse-mapped node coordinates of each element. Its Gauss rule uses $m=p+4$ points per direction with a Duffy map on triangles. It evaluates $u_\ast$ from the frozen expression and $\nabla u_\ast$ by fourth-order central differences with $h=10^{-3}$. MFEM's self-reported errors use `ComputeL2Error` and `ComputeGradError` with `IntRules.Get(geom, 2p+4)`.

## 7. Numerical verification vs. physical validation

* **Numerical (code) verification** — this MVP: the code solves the stated discrete problem correctly and the discretization converges at the theoretical rate for a manufactured solution.
* **Physical-model validation** — out of scope: whether $-\Delta u = f$ with these data models any physical system. No physical claim is made, and accepted verdicts say "numerical (code) verification passed for this manufactured-solution benchmark".

## 8. Capability matrix

The rows below describe the discretization and solver. An executable configuration also depends on the case, mesh source, MFEM revision and build flags. The Stage 2 coverage driver exercises those combinations and compares fields and error norms against scikit-fem. [Coverage driver](../benchmarks/stage2.py)

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

## 10. Stage 2 scalar diffusion and named meshes

For the variable case, schema `0.3` requires `equation = "-div(kappa*grad(u)) = f"` and `kappa = "1+x"`. On the unit square, $1\leq\kappa\leq2$. Schema `0.2` represents the original Poisson equation with implicit unit diffusivity. Omitting the coefficient from its serialized representation preserves historical problem hashes. [Study definition](../sciagent/spec.py)

The compiled `sin_sin_variable` case uses

$$
u_\ast=\sin(\pi x)\sin(2\pi y),\qquad
f=5\pi^2(1+x)u_\ast-\pi\cos(\pi x)\sin(2\pi y).
$$

The example prescribes $u_\ast$ on the bottom, right and left and the outward conormal derivative $g_N=\kappa\nabla u_\ast\cdot n=2\pi(1+x)\sin(\pi x)$ on the top. With test functions vanishing on the Dirichlet boundary, the reference solver assembles

$$
\int_\Omega \kappa\nabla u_h\cdot\nabla v_h
= \int_\Omega f v_h + \int_{\Gamma_N}g_Nv_h.
$$

The strong-form check differentiates the frozen expressions and evaluates $-\kappa\Delta u_\ast-\nabla\kappa\cdot\nabla u_\ast$. The verifier checks the variable solver claims `case_equation` and `case_diffusivity` against the frozen problem. Its reported $H^1$ seminorm is the ordinary gradient error. The scikit-fem coverage calculation uses the equivalent load $\int_\Omega\kappa\nabla u_\ast\cdot\nabla v_h$, providing a separate check of the MFEM source and boundary integrals. [Example request](../specs/stage2/diffusion_mixed_p1_tri.json) · [Independent verifier](../verifier/verifier.py) · [Operator and conormal regressions](../tests/test_variable_diffusion.py)

The variable operator explicitly integrates the stiffness form with MFEM order $2p+1$ for triangles and quadrilaterals. On P2 triangles the integrand can have degree three. The verifier checks the recorded `stiffness_quadrature_order` against the declared polynomial order and separately solves the weighted discrete problem from the frozen expressions. [Measured underintegration regression](RESULTS.md#stage-2-scalar-diffusion-and-gmsh-2026-11-23)

A schema `0.3` study can provide `external_mesh.file` and `external_mesh.sha256`. The supported input is an ASCII Gmsh MSH 2.2 planar Cartesian unit square with a single named surface, four named sides and straight-sided elements of one family. Each requested grid is obtained by uniform refinement of that mesh. The request binds physical tags to the boundary labels in `ProblemSpec`. The input digest and the archived mesh bytes bind replay to the supplied geometry. [Named mesh example](../specs/stage2/diffusion_gmsh_p2_quad.json) · [Mesh preflight](../sciagent/gmsh.py)

Place the `.msh` beside the request and use its basename. Compute the lowercase SHA-256 over the original bytes, including line endings. The preflight accepts files up to 8 MiB with the four sections `MeshFormat`, `PhysicalNames`, `Nodes` and `Elements` in that order. Use linear line elements on the boundary and either linear triangles or linear quadrilaterals throughout the surface. Give every element a physical tag in `1..65535` and a positive elementary entity tag. The four one-dimensional groups are named `bottom`, `right`, `top` and `left`. Their numeric tags can differ from MFEM's generated square. [Gmsh MSH 2 file format](https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029)

For a base grid with `b` cells per side, each requested level must equal `b * 2**k` with integer `k >= 0`. Surface connectivity must cover each base cell with one counterclockwise quadrilateral or two counterclockwise complementary triangles. Different cells may use different diagonals. Every perimeter interval needs its matching named segment. The controller runs this preflight before probing MFEM, archives the exact input as `frozen/mesh.msh`, and passes the archived path and refinement count to the executable. The verifier checks refined connectivity against the source cell diagonals in addition to the boundary and Cartesian partition checks. Archived studies can be reverified after the original input is moved or removed. [Gmsh regression tests](../tests/test_gmsh.py)
