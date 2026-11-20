# Codex development handoff — MFEM Scientific Computing Agent

## Project goal

Turn the **existing** deterministic, verified MFEM Poisson vertical slice into a robust, extensible scientific-computing agent system. Keep the principle **planner proposes, independent verifier accepts**. Preserve scientific correctness, explicit failure modes, and reproducible evidence over breadth or agent count.

## Before editing

1. Inspect `README.md`, `docs/MVP_SPEC.md`, `docs/IMPLEMENTATION_STATUS.md`, `docs/CLAUDE_CRITICAL_REVIEW.md`, `docs/RESULTS.md` and `docs/INDEPENDENT_REVIEW_2026-11-20.md`.
2. Read `cpp/`, `sciagent/`, `verifier/`, `specs/`, `tests/` and the shell scripts. Do not assume Claude's report is independently verified.
3. Check OS and tools. For local Windows development prefer WSL2/Ubuntu with a documented reproducible workflow; do not silently rely on native Windows builds.
4. Run `python -m pytest -q` and record counts of passes/skips/failures. If feasible, build the pinned MFEM release and run full MFEM integration checks.
5. Read the full original proposal for long-term direction, but avoid implementing ten agents or a universal PDE generator in the next patch.

## Stage 0 — Stabilize the baseline (first coding milestone)

- Reproduce the source build and establish a consistent pinned dependency/OS toolchain (dev container, WSL2 setup, or equivalent).
- Add automated Python unit CI and a separate MFEM integration job (or clearly explain why it must be manual). Verify GitHub Actions permissions before deployment.
- Fix defects revealed by local runs; preserve current accepted numerical results and expected failure modes.
- Add checks of artifact hashes, corrupted/missing/reordered run files, and tampered verdicts.
- Ensure no false 'all tests passed' claims when native integration tests are skipped.
- Add a small CLI quick-start command that reports tool availability, executes one supported study and produces a concise verdict.

**Acceptance:** a new developer can follow README from a clean environment, run unit tests, run a real MFEM Poisson example, generate and reverify stored evidence, and independently explain every skipped test.

## Stage 1 — Prove verifier robustness

- Build an explicit mutation suite: sign/source changes, boundary permutations, DOF/elements corruption, altered solver convergence claims, spurious NaNs/Infs, wrong quadrature, unallowed policy relaxation, incorrect cached artifacts.
- Build separate clean-positive test cases to measure false rejects.
- Report per-gate detection, union detection, false accepts/rejects and test denominators with uncertainty. Reject circular 'expected status' assertions that merely reproduce fixture labels.
- Add trust-boundary analysis of shared specs, reference formulations, mesh sources, accepted compiler outputs and file hashes.
- Keep failing cases as named regression tests.

**Acceptance:** evidence-based verifier metrics with known test-generation rules and machine-readable test outputs, and no regression in the original Poisson benchmarks.

## Stage 2 — Make the mathematical scope extensible

- Add a safe restricted formula/DSL or a reviewed equation template registry, with correct C++ generation and independent verification. Do not use unrestricted `eval` or shell interpolation.
- Extend manufactured solutions beyond the two hard-coded functions; include independently checked analytic derivatives where possible.
- Handle variable-coefficient diffusion as the next equation. Start with manufactured truth and fixed geometry, not arbitrary free-form multiphysics.
- Add Gmsh named boundary groups on simple real meshes, geometry validation, and refined test cases.
- Design the capability matrix so each configuration has exact version, build flags, solver/preconditioner assumptions and verification coverage.

**Acceptance:** new supported mathematical cases can be registered without editing the workflow controller, and every new case has independent positive/negative tests.

## Stage 3 — Parallel and performance path

- In an environment with MPI, hypre, and METIS, add an MPI solver and serial-versus-parallel cross-check at matched discretization and tolerance.
- Report strong/weak scaling only after numerical agreement passes. Measure total workflow cost, not just solver kernel time.
- Add GPU/partial-assembly combinations only after testing them explicitly against the capability matrix, not as speculative labels.

**Acceptance:** reproducible, validation-gated parallel experiments with measured rather than assumed speedups.

## Stage 4 — Controlled AI planning and user workflow

- Introduce one optional LLM planner that produces **candidate** specs or chooses from a reviewed allowlist. A human approves changes to physical models or validation thresholds.
- Keep numerical solvers, validators, artifact integrity, sandboxing and tool invocations deterministic.
- Benchmark LLM versus rule-only orchestration on verified completion, false acceptance, retries, cost and runtime.
- Build clear CLI/API output and practical examples before considering multiple specialized agents.

**Acceptance:** the optional LLM cannot bypass validation or alter frozen acceptance criteria, and its contribution is measured against a deterministic baseline.

## Work style and deliverables

- Work in small, reviewable commits. Read the current repository state and avoid destructive overwrites.
- Prioritize real code, automated tests, reproducibility and performance evidence over speculative architecture prose.
- Preserve all existing tests and expected negative outcomes.
- Do not use APIs that would expose user code, proprietary data or credentials.
- Do not perform expensive remote compute jobs without authorization.
- Update `docs/IMPLEMENTATION_STATUS.md`, `docs/RESULTS.md` only from real execution data; add new reproducible scripts for any new benchmarks.
- At each milestone report: files changed, commands executed, passed/failed/skipped tests, actual logs, remaining blockers and the next milestone.

## Suggested first Codex instruction

> Read `CODEX_HANDOFF.md` and the existing source. Start **Stage 0** now. Audit the build/verification path, reproduce the Python tests and (if available) the MFEM integration tests, implement a pinned clean-environment setup plus CI and artifact-integrity regression tests, then update the README and implementation status with actual evidence. Do not invent execution results or jump to MPI/GPU/ten-agent design before the baseline is reproducible.
