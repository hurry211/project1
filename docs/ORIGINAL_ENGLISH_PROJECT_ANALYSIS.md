# MFEM-Based Scientific Computing Agent
## Independent Technical Review and First-Iteration Implementation Plan

**Document purpose:** Provide a technical basis for an independent Claude review and a small but real implementation attempt.

**Status:** Proposed design, not an implemented or validated system. The following report distinguishes ideas in the original Chinese planning document from the recommended engineering adjustments. Neither the existence of a working prototype nor compatibility of any specific hardware/software combination should be inferred from this report.

**Original document:** `MFEM科学计算智能体初步规划.md` (provided by the project owner).

---

## 1. Executive Summary

The proposed system combines **MFEM as a numerical finite-element engine**, an **agent/controller as a workflow organizer**, and an **independent verification component as the quality gate**. The objective is not merely to translate natural-language prompts into C++ code. It is to produce reproducible scientific-computing runs with explicit mathematical assumptions, validated numerical methods, controlled execution, inspectable evidence, and clearly qualified acceptance decisions.

This direction is credible as an engineering project. MFEM provides a broad finite-element foundation and documented tutorial examples; important surrounding capabilities—request interpretation, geometry authoring, job orchestration, experiment tracking, validation governance, and auditability—must be built outside MFEM or integrated from third-party tools.

**Main recommendation:** Do not start with ten fully autonomous LLM agents, general CAD ingestion, arbitrary multiphysics modeling, and automatic GPU optimization. Start with a deterministic **end-to-end vertical slice** around one carefully specified Poisson problem. Demonstrate a real MFEM run, independent checks, structured output, meaningful failure detection, and a bounded recovery policy. Add natural-language planning, MPI, GPU, and additional PDE families only after the baseline passes.

The report's prioritized deliverables are:

1. A versioned, structured mathematical and computational specification.
2. A robust wrapper for an authentic MFEM executable.
3. A separately implemented, evidence-based validation gate.
4. A reproducible run manifest and machine-readable results.
5. Positive tests, negative tests, and a documented path to parallel execution.

## 2. Source Proposal vs. Recommended Scope

### 2.1 Capabilities proposed in the source document

The source identifies ten responsibilities: mathematical modeling; geometry and meshing; initial/boundary conditions; discretization; solver and preconditioner choice; environment adaptation; operating conditions and parameters; parallel execution and monitoring; independent verification; visualization and reporting. It also proposes a unified intermediate representation, a state-machine workflow, and full run provenance.

Those responsibilities remain useful as **architecture-level domains**. They do not imply ten models or ten separately deployed services in the first implementation.

### 2.2 Narrower first-iteration target (recommendation)

Support a fixed, explicit subset:

- **PDE:** stationary scalar Poisson equation.
- **Domain:** unit square in two dimensions, with a mesh supported by the chosen MFEM implementation.
- **Finite element:** conforming scalar H1, initially polynomial order 1.
- **Boundary treatment:** homogeneous Dirichlet conditions for the manufactured-solution benchmark.
- **Execution:** local CPU, serial execution first; MPI only as a separately validated optional extension.
- **Validation:** input checks, real solver status, independently computed error metrics, and mesh-refinement convergence.
- **Artifacts:** machine-readable inputs/results, logs, version metadata, and a concise validation report.

### 2.3 Explicit non-goals for the first iteration

- Free-form derivation of arbitrary PDE weak forms by an LLM.
- Unrestricted CAD/STEP/IGES ingestion, automated geometry repair, or complicated interfaces.
- Unverified solver/preconditioner combinations.
- General nonlinear, time-dependent, electromagnetics, and multiphysics support.
- Claims of physical-world validation without appropriate experimental or benchmark evidence.
- Unrestricted retries, silently changed physical assumptions, or unlimited compute spending.
- Production security, authentication, or cluster-scale deployment.

## 3. System Architecture

A three-plane design makes responsibilities and trust boundaries explicit.

### 3.1 Control plane: problem interpretation and workflow policy

A controller takes a user request and generates a candidate structured specification; checks which assumptions are supplied and which are unresolved; selects from a **reviewed capability matrix**; and transitions through an explicit state machine. If an LLM is used, it should propose actions rather than directly certify numerical correctness.

### 3.2 Execution plane: deterministic tools and MFEM

Tools inspect the environment, validate input schemas, resolve boundary mappings, construct or locate meshes, build/run MFEM programs, collect output, and optionally generate visualizations. The execution plane should be usable **without an LLM** so that its baseline reliability is measurable.

### 3.3 Assurance plane: validation and evidence

A validation process reads frozen specifications and actual output artifacts, performs its own checks, and issues a typed verdict. The agent must not be able to alter the acceptance thresholds during a failed run. A separately written verification implementation is preferable to reusing the exact same logic as the solver.

### 3.4 Recommended MVP component map

| Component | First-iteration responsibility | Suggested implementation |
|---|---|---|
| Specification validator | Reject missing, conflicting, or unsupported fields | JSON Schema/Pydantic or equivalent |
| Controller | Deterministic states, bounded retry rules | Python workflow code |
| MFEM adapter | Build/invoke a real MFEM-based Poisson solver | C++ MFEM code with subprocess wrapper |
| Mesh / BC inspector | Compare declared and actual boundary attributes | Explicit checks before assembly |
| Result extractor | Produce stable numerical output and metadata | CSV/JSON plus MFEM field output |
| Independent verifier | Compute errors, convergence, and acceptance | Separate Python numerical checks or equivalent |
| Evidence store | Preserve runs, tool versions, and decisions | Local per-run directories and manifest |
| Optional LLM interface | Map user requests to supported specs | Thin agent interface, later stage |

The original ten domains can become independent services when their complexity and measured value justify separation.

## 4. Mathematical Reference Problem

The recommended benchmark is a **manufactured-solution test**, separate from the default source term in MFEM Example 1.

Consider

\[
-\Delta u=f \quad \text{in } \Omega=(0,1)^2,
\qquad u=0 \quad \text{on } \partial\Omega.
\]

Choose a smooth exact solution

\[
u_*(x,y)=\sin(\pi x)\sin(\pi y).
\]

The corresponding right-hand side is

\[
f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y).
\]

**Important:** MFEM's original `ex1` solves `-Delta u = 1` with homogeneous Dirichlet boundaries. Running it unchanged does **not** implement the manufactured-solution benchmark above. The prototype must either adapt the MFEM equation/source term (and document precisely how) or use another supported MFEM path that actually solves the stated test problem.

Use at least three mesh resolutions. A meaningful validation record should include the L2 error, and preferably the H1 seminorm error, approximated using appropriate numerical quadrature or documented exact/validated integration. For mesh sizes `h` and `h/2`, the observed L2 convergence order is

\[
p_{\mathrm{obs}}=\frac{\log(E_h/E_{h/2})}{\log 2}.
\]

For a smooth problem and first-order conforming elements on regular meshes, the expected asymptotic L2 error order is approximately 2 and H1 seminorm error order approximately 1, assuming adequate quadrature and sufficiently small algebraic error. These are conditional expectations—not universal success thresholds. Fix the actual tolerances and mesh-resolution policy before running the evaluation.

A second benchmark should deliberately use distinguishable boundary data or a reviewed mixed-boundary setup to detect boundary-label swaps that an all-zero Dirichlet test cannot reveal.

## 5. Structured Data Contracts

The original document proposes a broad nested schema. The following decomposition is a recommended implementation refinement:

- **ProblemSpec**: physically meaningful problem definition; versioned and locked once approved.
- **ExecutionSpec**: selected mesh resolution, finite-element order, assembly mode, solver settings, and resources; may be changed under approved policies.
- **ValidationSpec**: precommitted assertions, reference definition, numerical tolerances, and tests; cannot be relaxed by the executor.
- **RunManifest**: actual code and environment versions, commands, files, timings, numerical output, errors, and the verifier's verdict; append-only for a given run.

Illustrative machine-readable specification (fields and accepted values require implementation-time schema definitions):

```json
{
  "schema_version": "0.1",
  "problem": {
    "id": "poisson_unit_square_mms",
    "equation": "-laplacian(u) = f",
    "domain": "unit_square_2d",
    "source": "2*pi*pi*sin(pi*x)*sin(pi*y)",
    "exact_solution": "sin(pi*x)*sin(pi*y)",
    "boundaries": [
      {"selector": "all_external", "kind": "Dirichlet", "value": "0"}
    ]
  },
  "execution": {
    "element_family": "H1",
    "polynomial_order": 1,
    "backend": "serial_cpu",
    "mesh_levels": [2, 3, 4],
    "solver": "reviewed_poisson_default"
  },
  "validation": {
    "require_solver_success": true,
    "require_boundary_coverage": true,
    "require_finite_outputs": true,
    "require_l2_convergence_study": true,
    "expected_l2_order": "approximately_2_in_asymptotic_regime"
  }
}
```

This example intentionally avoids hard-coding a universal convergence-order threshold. The implementation must add explicit numeric policies, mesh conventions, types, range restrictions, and a documented reason for its chosen tolerances.

**Boundary mapping is mandatory:** every actual MFEM boundary attribute must map to a physical label, mathematical boundary condition, and implementation pathway. Unassigned attributes must not silently become an unintended natural boundary condition.

## 6. End-to-End Workflow and State Machine

Recommended states:

`RECEIVED -> SPEC_VALIDATED -> ENVIRONMENT_CHECKED -> READY -> BUILDING -> PRECHECK -> RUNNING -> VERIFYING -> {ACCEPTED, RETRY_ALLOWED, HUMAN_REVIEW, FAILED, BLOCKED}`

Transitions must be explicit. Each transition writes a timestamped event with inputs, reasons, and resulting status.

1. **Receive:** Save original request and identify unsupported/undetermined requirements.
2. **Formalize:** Freeze the accepted `ProblemSpec` and `ValidationSpec`.
3. **Check environment:** Locate MFEM, compiler, optional MPI, runtime dependencies, and supported backend.
4. **Prepare:** Resolve mesh and boundary attributes; reject unsupported combinations.
5. **Precheck:** Build and run a small baseline case; ensure outputs can be parsed.
6. **Execute:** Run the selected actual MFEM program within resource and time limits.
7. **Verify:** Compare against the frozen specification, recompute metrics, and assess convergence/constraints.
8. **Decide:** Accept, try an approved numerical configuration adjustment, ask for human review, or stop with a structured error.
9. **Record:** Preserve every attempt, including invalid or failed runs, without presenting them as validated scientific results.

A status code of zero, a visually plausible field plot, or a reduced linear residual is not sufficient evidence of correctness.

### Retry policy

Define a small, explicit maximum number of attempts and total runtime/compute budget. The controller may refine the mesh when the discretization error is excessive, or select an approved solver configuration after a documented solver failure. It must not change the physical PDE, change boundary conditions, or loosen acceptance criteria silently. Failure cases need dedicated codes such as `INVALID_SPEC`, `UNSUPPORTED_BACKEND`, `BUILD_FAILURE`, `SOLVER_DIVERGED`, `VALIDATION_FAILED`, `RESOURCE_LIMIT`, and `BUDGET_EXHAUSTED`.

## 7. Verification, Validation, and Independent Evidence

The original plan has five review levels. For the first iteration, make at least these checks executable:

| Gate | Check | Required evidence |
|---|---|---|
| A. Input | Schema, units if applicable, boundary coverage, mesh validity, no unresolved required fields | Machine-readable validation log |
| B. Execution | Real binary ran; no NaN/Inf; result files complete; solver returned meaningful status | Command, exit code, logs, numerical metrics |
| C. Numerical | Manufactured-solution error at several mesh levels; observed convergence trend | Error table, order estimate, verification code |
| D. Reference | Compare with a reviewed analytic result and, where feasible, an independent numerical implementation | Reference definition and discrepancy report |
| E. Provenance | Input, code revision/hash, library version, compiler, environment, and rerun instructions | Run manifest and archived artifacts |

The wider project can later add conservation, positivity, temporal convergence, sensitivity, uncertainty quantification, and experiment-based model validation **where mathematically relevant**. Not every discretization guarantees elementwise conservation or positivity, so checks must be method-specific.

Independent verification is a **design aspiration** that requires technical enforcement. A verifier that merely reads a solver-generated `passed=true` flag is not independent. If the solver and verifier share the same mistaken model assumptions, even cross-checking may fail to detect the error. Freeze and expose the original problem definition and identify shared assumptions in every verdict.

Use language such as "numerical verification passed for this benchmark" rather than "the physical model is validated" unless independent physical evidence exists.

## 8. Solver, Assembly, and Hardware Compatibility

Solver-selection rules must be conditional, not just keyword-based:

- Conjugate Gradient requires an appropriate symmetric positive-definite problem and compatible preconditioning.
- MINRES requires a symmetric operator and appropriate preconditioner properties.
- Krylov and preconditioner choices depend on boundary constraints, possible nullspaces, algebraic system properties, and discretization.
- Partial assembly does not automatically support every preconditioner that expects an explicitly assembled global matrix.
- MPI and GPU implementations introduce dependency, element type, operator, and backend restrictions; support should be demonstrated with small correctness tests.

Build a **tested compatibility matrix** keyed by exact MFEM/dependency versions, equation, finite-element space, mesh element, assembly mode, solver, preconditioner, and backend. Mark combinations `tested`, `unsupported`, or `not yet verified`. Do not convert a library's broad feature list into a guarantee that all combinations work together.

The first performance objective should be *complete end-to-end time to obtain an accepted answer at fixed accuracy*, not merely short solver runtime. Track compile time separately, and measure retries and verification cost.

## 9. Concrete First-Round Implementation Tasks

Claude should inspect its actual workspace and complete the first feasible vertical slice, in the following order:

**Task 1 — Repository and environment audit.** Identify operating system, architecture, compiler, CMake/Make, Python, MFEM availability and version, whether source checkout/build is available, optional MPI/hypre/METIS, and available compute limits. Record evidence, not guesses.

**Task 2 — Establish a real baseline.** Compile and run MFEM Example 1 (or a documented, equivalently small official Poisson example) using supported build steps. Capture exact commands, standard output/error, exit code, and installed library version. Do not claim a completed solver run if only a mock or downloaded example source exists.

**Task 3 — Build the project skeleton.** Introduce schema(s), a controller/runner, MFEM adapter, verifier, tests, and per-run output directories. Keep upstream MFEM source distinct from project code when possible.

**Task 4 — Manufactured solution.** Implement the exact PDE defined in Section 4 in actual MFEM code; compute numerical errors at three or more mesh sizes; write results to a stable parseable format. If blocked, document the reason and preserve the real baseline instead of fabricating numbers.

**Task 5 — Validation gate and negative tests.** Test at least one invalid or missing boundary specification, one unsupported configuration, and one deliberately failed/insufficient-accuracy case. Check that the system rejects failures rather than issuing a passing verdict.

**Task 6 — Bounded configuration retry.** Only if Task 5 is functional, demonstrate an explicit, finite mesh-refinement retry based on a frozen numerical acceptance policy.

**Task 7 — Optional parallel comparison.** If MPI-enabled MFEM and dependencies are actually available, run the supported parallel Poisson path and compare with a reference under a documented tolerance. Do not infer MPI availability from the existence of `mpirun` alone.

### Proposed repository structure

```text
mfem-scientific-agent/
  README.md
  docs/
    technical_review.md
    implementation_status.md
    decisions.md
  specs/
    poisson_unit_square.json
    schema.json
  cpp/
    CMakeLists.txt
    poisson_mms.cpp
  agent/
    controller.py
    environment.py
    mfem_runner.py
  validation/
    verifier.py
    policies.py
  tests/
    test_specs.py
    test_validation_failures.py
    test_workflow.py
  scripts/
    run_demo.sh
  runs/
    .gitkeep
```

The filenames are a proposed organization, not evidence that these files already exist. The agent may choose a simpler arrangement when justified by the environment and explain deviations.

## 10. Acceptance Criteria for the First Demonstration

The initial vertical slice is successful only if **all applicable** criteria are supported by actual evidence:

1. A real MFEM executable builds and runs locally, with recorded exact version and command.
2. The mathematical inputs and BC handling are explicit and machine-checkable.
3. The manufactured-solution problem—not an unrelated default example—is actually solved.
4. Numerical errors for multiple meshes are reported, with a documented convergence calculation.
5. The verification component independently inspects result data and returns a typed verdict.
6. Invalid or unsupported inputs yield explicit rejection rather than silent default substitution.
7. Every run has reproducible configuration, logs, version metadata, and an unambiguous status.
8. The demonstration has a documented rerun command and automated tests wherever practical.
9. Any failed or unattempted criteria are clearly marked `NOT DONE` or `BLOCKED`, never implicitly treated as successful.

## 11. Research Opportunities (Not Yet Established Contributions)

This architecture may support research questions beyond ordinary systems engineering. Their novelty and practical significance **remain to be established through literature review and comparative experiments**.

**A. Verification-guided repair:** Does access to independently computed error/BC/convergence feedback improve verified task completion compared with terminal-output-only feedback?

**B. Constraint-preserving planning:** Does freezing the physical problem and validation policy reduce silent specification drift during multi-step automated scientific computation?

**C. Cost-aware solver/workflow selection:** Can a constrained controller reduce total cost per accepted result at fixed error tolerance compared with fixed rules and generic tool-use baselines?

For each investigation, predefine datasets, seed handling, failure modes, evaluation metrics, and baselines, and include ablations removing verification feedback or schema constraints. Avoid presenting a new orchestration system as a novel scientific algorithm without evidence.

## 12. Main Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Scope expands to all PDEs too early | No reliable end-to-end result | Freeze a single benchmark and acceptance gates |
| Wrong or missing boundary labels | Solver converges for the wrong PDE | Compare actual mesh labels against immutable ProblemSpec |
| Solver output is trusted as its own proof | False acceptance | Separate verifier using reference solution and raw output |
| MFEM/dependency/backend incompatibility | Build or runtime failures | Environment inventory and tested compatibility matrix |
| Unbounded retries | Wasted compute, hidden instability | Max attempts, runtime budget, typed stop reasons |
| Verification criteria are relaxed on failure | Misleading success | Immutable ValidationSpec and audit trail |
| Performance is compared at different errors | Invalid speedups | Compare cost at fixed quality/accuracy |
| No independent reference | Mathematical correctness cannot be determined | Use a manufactured solution and audited benchmark cases |
| LLM interface precedes deterministic baseline | Hard-to-debug failures | Build and test tools without an LLM first |

## 13. Decisions Required After the Initial Review

A reviewer should recommend explicit decisions on:

1. The exact version/tag/commit of MFEM to support initially.
2. Whether the available environment can compile MFEM C++ directly.
3. Whether the first milestone is *official Example 1 automation* or *full manufactured-solution validation*. The latter is the preferred end-to-end target, but the former is a useful intermediate deliverable.
4. Exact acceptance metrics, mesh levels, and error tolerances for the supported benchmark.
5. Whether MPI verification is possible now or must remain a later milestone.
6. The minimal structured schema and retry budget to freeze for the first implementation.

## 14. Technical References for Verification

The source document already points to MFEM's broader capabilities. For this first implementation, consult and verify the following directly against the selected version:

- MFEM project and current releases: https://mfem.org/
- Build instructions: https://mfem.org/building/
- Official examples, especially Example 0 and Example 1: https://mfem.org/examples/
- Example source for Poisson (`ex1.cpp`): https://github.com/mfem/mfem/blob/master/examples/ex1.cpp
- Boundary condition handling and natural boundary defaults: https://mfem.org/fem_bc/
- Solver tutorials and AMG/parallel demonstrations: https://mfem.org/tutorial/solvers/
- Installation guide with prerequisites: https://github.com/mfem/mfem/blob/master/INSTALL

Web documentation may change. Pin an actual project dependency version and validate any command against that version's source tree. Do not blindly copy outdated build instructions or assume optional dependencies are installed.

---

**Final assessment:** The original plan has a valuable architecture-level insight: a scientific-computing agent needs structured specifications, deterministic numerical tools, independent verification, and provenance. The next milestone should produce **a real, reproducible, independently checked Poisson computation** before pursuing autonomous multi-agent behavior or large-scale heterogeneous HPC.
