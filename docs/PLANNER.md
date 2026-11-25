# Candidate planning and numerical acceptance

The planning entry point selects one of nine reviewed serial studies. These cover the original seven Poisson requests and the two Stage 2 variable-diffusion examples. The registry binds an identifier to exact study bytes and, for the named mesh, its SHA-256. The MPI performance campaign retains its own `mpi-study` entry point. [Reviewed registry](../sciagent/planner.py)

## Submit a task

The required `case` and optional `mesh_type`, `mesh_source`, `polynomial_order`, `assembly` and `solver` fields specify what the caller authorizes. A `description` states preferences among the eligible templates. Put requirements in the structured fields. For example, `solver: "cg_jacobi"` requires Jacobi, while a preference for Jacobi in the description allows another authorized selection. A nonempty `change_request` asks for human review before proposal or execution. [Preference example](../specs/stage4/choose_preconditioner.json) · [Review example](../specs/stage4/review_tolerance.json)

```bash
python -m sciagent plan-catalog
python -m sciagent plan specs/stage4/choose_diffusion.json --directory runs/plan-preview
python -m sciagent plan specs/stage4/choose_diffusion.json --directory runs/plan-executed --execute
python -m sciagent verify-plan runs/plan-executed --manifest-sha256 <digest-from-execution>
```

Use a fresh directory for each request. The default rule selects the first eligible template in identifier order. The optional local model receives descriptions of eligible templates and the user's preference. Its reply can contain one field, `template_id`, holding a reviewed identifier or `null` for abstention. Supply the locally installed model tag and its full digest when choosing this mode. The numerical controller inherits its established process limits and refinement policy. [Model installation and measurements](LOCAL_PLANNER_MODEL.md)

```bash
python -m sciagent plan specs/stage4/choose_preconditioner.json \
  --mode llm --model qwen2.5:1.5b \
  --model-digest 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b \
  --directory runs/preconditioner-preview
```

## Interpret the outcome

| State | Meaning |
|---|---|
| `PLANNED` | The deterministic gate approved the nomination. Inspect the pinned input in `template/study.json`. |
| `ACCEPTED` | Execution completed and the independent numerical verifier replayed the accepted study. |
| `REVIEW_REQUIRED` | The request changes an approved contract or lacks a matching reviewed configuration. The archive contains the task and decision. |
| `REJECTED` | The planner abstained or its proposal failed the format or authorization checks. |
| `FAILED` | The provider, native study or evidence checks failed. Use the decision code and retained records to diagnose the failure. |

For a new physical model, changed validation threshold or larger resource budget, a human reviews the proposed change in the ordinary repository revision process. Review the mathematical specification, tests and effect on acceptance before authorizing a new registry digest. A subsequent request starts a fresh archive. Existing requests and archives retain their original criteria.

## Evidence and replay

The archive contains the task, mode, deterministic decision and manifest. Model calls also record the prompt, raw reply, installed model identity, sampling settings, usage and timings. The controller checks the approved study digest on the bytes it actually consumes. Numerical acceptance requires a digest-matched child study, verified input and output inventories, and independent replay of the attempts. Archive replay recomputes authorization and checks recorded proposals, the template bytes and the numerical results. It runs with the local model service and the native executable unavailable. Keep the manifest digest outside the archive to authenticate the capture. [Orchestrator and replay checks](../sciagent/planning_run.py) · [Fault regressions](../tests/test_planning_run.py)

The measured comparison includes independent preference answers and review controls. The task description can guide the model to a better or worse eligible choice, which is scored separately from numerical correctness. [Benchmark protocol](PLANNING_BENCHMARK.md) · [Measured results](RESULTS.md#stage-4-controlled-planning-2026-11-24)

## Repeated requests with a pinned proposal

Use `--mode bounded` to select the sole eligible template directly. A request with multiple eligible templates invokes the pinned local model. For a repeated preference, a caller can bind the new request to the manifest digest of a previous successful live planning preview. Use the digest printed by the preview command above.

```bash
python -m sciagent plan specs/stage4/choose_preconditioner.json \
  --mode bounded --model qwen2.5:1.5b \
  --model-digest 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b \
  --proposal-directory runs/preconditioner-preview --proposal-manifest-sha256 "$PREVIEW_SHA" \
  --directory runs/repeated-preconditioner --execute
```

The task, model, prompt, options and reviewed template pin are checked again. Each repeated execution computes and verifies a fresh solution. Its archive carries a copy of the pinned source proposal evidence for offline replay. The receipt records the route and counts new inference calls and tokens. Choosing a new preview can yield a different eligible template. [Repeat benchmark](../benchmarks/planner_repeats.py) · [Regression tests](../tests/test_planner_reuse.py)
