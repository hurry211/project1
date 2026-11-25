# Paired planning benchmark

The Stage 4 benchmark compares local language-model selection with the deterministic rule that selects the first lexicographically sorted eligible template. Each method receives the same structured authorization and advisory description. Both pass their selected ID through the same eligibility check, study controller and numerical verifier. [Planner](../sciagent/planner.py) · [Benchmark driver](../benchmarks/planning.py)

## Task panel

The ordered panel contains 19 English and Chinese requests. Before live inference, the author fixed each task's permitted template IDs, preferred ID and expected disposition. Only the request embedded in each row reaches the planner. Changing the corpus bytes invalidates the benchmark's pinned digest. [Task panel](../benchmarks/data/planning_tasks.json)

| Group | Count | Measured outcome |
|---|---:|---|
| Unique reviewed configuration | 5 | The specified template produces an accepted, replayable MFEM result. |
| Two or more authorized options | 9 | The accepted result also uses the template requested in the free-text description. Examples distinguish full from partial assembly, P1 from P2, diagonal from Gauss–Seidel preconditioning and a generated mesh from a supplied mesh. |
| Human review required | 5 | The deterministic preflight holds an unavailable configuration, unknown physics, or requested change to the model or validation criteria. This group exercises orchestration before model inference. |

Four separately fixed gate controls exercise the decision taken *after* a candidate is supplied. They cover a valid ID with the wrong polynomial order, a valid ID with the wrong mesh source, an unrelated reviewed physics template, and an additional field requesting a validation change. Each requires rejection. These controls are recorded separately from the paired model and rule outcomes. [Gate controls](../benchmarks/data/planning_gate_controls.json)

## Capture and offline replay

Prepare the serial MFEM study environment and the local model as described in the repository. Supply the installed model name and its verified digest in the Ubuntu shell. Use a new output directory for every capture and new paths for the two reports. [Local model setup](LOCAL_PLANNER_MODEL.md)

```bash
python -m benchmarks.planning --directory runs/paired --model "$MODEL" \
  --model-digest "$DIGEST" --report runs/paired-results.json
sha256sum runs/paired-results.json
```

Keep the printed SHA-256 separately. To re-score a relocated archive, set `REPORT_SHA` to that retained value and run the following command. The offline pass checks the original report digest, all 38 manifests, request identities, model identity, normalized results and task order. For accepted studies it independently rechecks the child numerical evidence. It recreates the report from these checks and the original capture timings.

```bash
python -m benchmarks.planning --directory runs/paired \
  --replay-report runs/paired-results.json --report-sha256 "$REPORT_SHA" \
  --report runs/paired-replay.json
cmp runs/paired-results.json runs/paired-replay.json
```

## Reading the scores

- **Authorized verified completions** count accepted numerical replays with an independently permitted template, out of 14 executable tasks. **Intended verified completions** also require the preregistered preferred ID. The nine preference tasks supply a separate denominator for preference accuracy.
- **False accepts on review controls** count held requests that reached numerical acceptance, out of five. **Unauthorized acceptances across all tasks** additionally detect an ineligible template accepted on an otherwise feasible request. An eligible, numerically accepted choice that misses the description is counted as a preference error.
- **False rejects** count explicit rejection or referral of executable tasks. Provider or numerical failures reduce verified completion and have their own state counts. The code reports the number of local-model calls, provider errors, proposal rejections by reason and retries at the proposal and numerical stages.
- **Time** is the wall time of the entire workflow per request, including any model call, numerical study, internal verification and archive finalization. An additional independent digest-pinned replay is timed separately. Method order alternates by row. A single observation per method per task includes machine and model cache effects.
- **Cost** includes measured input and generated tokens. A missing counter stays unknown, even when other calls reported a value. Local inference has zero hosted API charges. Electricity consumption and hardware depreciation were unmeasured.

This is a fixed regression panel. Its denominators describe these requests and these reviewed templates. [Stage 4 results](RESULTS.md)

## Three repetitions per request with explicit proposal reuse

The E4 comparison exercised the same 19 fixed tasks three times per method in randomized order within each task/repeat block. Every eligible request ran a fresh native MFEM study with independent numerical checks. The bounded mode acquired a live preview for each of the nine ambiguous tasks and charged all nine to its total. Five singleton tasks used deterministic selection, and five review controls stopped at preflight. The live and bounded arms used the same Qwen2.5 1.5B GGUF model digest. [Benchmark source and offline audit](../benchmarks/planner_repeats.py) · [Measured data](../evidence/e4/planner-repeats.json.gz) · [Model prompts and responses](../evidence/e4/planner-model-records.tar.gz)

| Method | Verified numerical requests | Required reviews | Expected template matches | New inference calls | Total measured seconds |
|---|---:|---:|---:|---:|---:|
| Rules | 42/42 | 15/15 | 24/42 | 0 | 83.96 |
| Live model on eligible requests | 42/42 | 15/15 | 39/42 | 42 | 142.02 |
| Bounded with pinned previews | 42/42 | 15/15 | 39/42 | 9 | 100.76 |

The bounded total includes 14.68 s for its nine live previews and falls 29.1% below the always-live total on this panel. All three methods returned consistent template IDs in these three repetitions. The earlier independent Stage 4 captures did observe different live choices. Pinned preview reuse guarantees the same approved template only for the exact archived task, prompt, model settings and template digest. These measurements use one WSL2 host, a single pinned Python CPU and the local Ollama daemon. Other scientific jobs shared the host. The measured clock covers planning, numerical execution and its immediate evidence check. The separate post-run replay is outside that clock. [Results and source fingerprints](../evidence/e4/planner-audit.json)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$CPU" python -m benchmarks.planner_repeats \
  --directory "$RUN/planner" --report "$RUN/planner.json" --repeats 3 \
  --model qwen2.5:1.5b --model-digest 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b
sha256sum "$RUN/planner.json"
python -m benchmarks.planner_repeats --directory "$RUN/planner" \
  --report "$RUN/planner.json" --audit-sha256 "$REPORT_SHA"
```
