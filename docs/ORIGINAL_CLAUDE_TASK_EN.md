# Claude Task: Independently Review and Implement a Small MFEM Scientific Computing Agent Prototype

You are acting as (1) a scientific computing / numerical PDE reviewer, (2) an MFEM-oriented C++ engineer, and (3) an agent-workflow and verification engineer.

I am providing two project documents:
- `MFEM科学计算智能体初步规划.md`: the original, broad Chinese proposal.
- `MFEM_Scientific_Computing_Agent_Review_EN.md`: an English technical review and first-iteration implementation plan.

**Your task is NOT merely to summarize them. Independently challenge the architecture and, using the tools and environment actually available to you, do tangible implementation and validation work in this session.** The English review is a proposal, not unquestionable ground truth. Preserve the original project's long-term ambition, but prioritize an executable vertical slice.

## A. Independent critical review

1. Read both files closely. Separate: (i) documented requirements from the original proposal, (ii) engineering recommendations in the English review, and (iii) your own new conclusions.
2. Assess technical feasibility, mathematical correctness, architecture, MFEM API compatibility, solver/preconditioner limitations, hardware assumptions, development complexity, and likely hidden failure modes.
3. Identify the **10 most important risks or gaps**, with severity, evidence, and practical fixes. Do not simply endorse the design.
4. Compare a 10-agent architecture with a simpler orchestrator + deterministic tools + independent verifier. Recommend the smallest architecture that can genuinely work.
5. Define measurable MVP acceptance criteria and distinguish numerical verification from physical-model validation.
6. Verify relevant claims against official MFEM documentation or the specific installed source/version whenever possible; clearly label claims you could not check.

## B. Start real implementation immediately

Do not stop after producing a plan. Inspect your actual filesystem, installed tools, and permissions, then begin the most feasible sequence below.

**Priority 1 — Environment inventory.** Determine OS, C++ compiler, CMake/Make, Python, MFEM installation or checkout, optional MPI/hypre/METIS, available CPU/GPU, and resource constraints. Record exact commands, detected versions, and blockers. Do not invent them.

**Priority 2 — Real MFEM baseline.** If MFEM is available or can be obtained and built safely with reasonable local resources, build and run official Poisson Example 1 or an equivalently small official example. Preserve the actual stdout/stderr, exit code, build configuration, and executable path. Do not call a mocked run a genuine MFEM execution.

**Priority 3 — Minimal project skeleton.** Create a clean, maintainable project under a sensible local directory, without overwriting unrelated work. At minimum, provide a typed or schema-validated problem specification, a deterministic command-line runner, basic environment detection, a validator, and a run-evidence folder structure. Use Python for orchestration if helpful, and real MFEM C++ for the solver.

**Priority 4 — Mathematical verification.** Move beyond the unmodified Example 1 baseline: implement a 2D unit-square Poisson problem in MFEM with

    -Δu = 2π² sin(πx)sin(πy),  u = 0 on ∂(0,1)²,
    u_exact(x,y) = sin(πx)sin(πy).

Perform a mesh-refinement study on at least three resolutions, calculate numerical error metrics from actual solution data or appropriate MFEM error routines, and estimate observed convergence order. Explain the integration/measurement method and numerical tolerances. If you cannot modify/build the MFEM code, say exactly why and leave this task marked BLOCKED rather than substituting a standalone NumPy toy solver as though it were MFEM.

**Priority 5 — Negative tests and a true quality gate.** Verify that missing/mismatched boundary declarations, unsupported solver/backend combinations, and intentional failures are rejected. The verifier must not merely accept the solver's self-declared success flag. Track the immutable physical specification separately from adjustable execution settings.

**Priority 6 — Limited corrective loop.** If the baseline is working, implement an explicit, bounded retry triggered by a predefined numerical acceptance condition (e.g., mesh refinement). Preserve all failed attempts. Never silently change the PDE, boundary conditions, or verification thresholds to achieve success.

**Priority 7 — Optional MPI path.** Only if a compatible MPI-enabled MFEM build and required dependencies are present, execute a comparable parallel benchmark and check correctness against a reference. GPU work is optional and should not precede the CPU correctness baseline.

## C. Implementation quality requirements

- Work with the **actual installed MFEM version** and its supported build system; do not hallucinate APIs or assume all assembly/preconditioner/backend combinations are compatible.
- Keep third-party MFEM code separate from project-specific code when practical. Prefer pinning versions and preserving upstream licenses.
- Do not require an LLM or an online API merely to run the numerical baseline. The deterministic workflow must function independently.
- Establish a clear contract between `ProblemSpec`, adjustable execution settings, frozen validation criteria, and append-only run records.
- Use explicit resource limits and bounded retries; do not incur paid cloud costs, launch large HPC jobs, perform system-wide invasive installations, or push to remote repositories without authorization.
- Use safe, reversible local development. Do not modify or delete unrelated files.
- Provide unit tests for schema and failure handling and integration tests for real MFEM execution where feasible.
- Never fabricate logs, speedups, numerical accuracy, completed tests, or implementation status.
- When the environment blocks a stage, proceed with the best independently useful artifacts (schemas, adapters, tests, build scripts, documentation). Clearly label what was not executed.
- Do not repeatedly ask for clarification about nonessential preferences. Make conservative, documented assumptions and proceed.

## D. Expected deliverables

Produce concrete files whenever filesystem access is available:

1. `docs/CLAUDE_CRITICAL_REVIEW.md` — independent technical assessment, a prioritized risk table, and justified architecture changes.
2. `docs/MVP_SPEC.md` — precise supported problem, assumptions, states, data contracts, and acceptance criteria.
3. `docs/IMPLEMENTATION_STATUS.md` — file inventory, environment findings, test commands, verified results, blockers, and next actions.
4. Actual source, build instructions, controller/runner, validator, example specs, and test cases for the smallest feasible vertical slice.
5. `runs/` or equivalent — **real output evidence** if a simulation was actually executed, including logs, metrics, version information, and a run manifest.
6. `README.md` — exact commands to reproduce what you managed to run.

If you cannot create files, deliver complete file contents or patches with clearly identified paths. Do not claim that uncreated files exist.

## E. Required final response format

Write your final answer in English with these sections:

1. **Independent Verdict** — What is strong, wrong, missing, and out of scope.
2. **Architecture Recommendation** — The minimum credible design and rationale.
3. **Environment and MFEM Audit** — What was actually found and verified.
4. **Work Completed** — Exact files created/modified and what each does.
5. **Commands and Test Evidence** — Real build/run/test commands and actual results, or explicit BLOCKED notes.
6. **Numerical Verification Results** — Error table, convergence-order calculation, and verdict if actually measured; otherwise NOT RUN.
7. **Open Risks and Limitations** — Remaining issues, not inflated claims.
8. **Next Three High-Value Actions** — Actionable and prioritized.

**Success means demonstrating a trustworthy increment, however small. Prefer one genuine MFEM computation with a defensible verification record over a large speculative multi-agent framework. Begin by reading the documents and inspecting the available environment, then implement what you can.**
