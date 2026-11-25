# Local model setup and inference records

The optional planner accepts a model tag and its complete 64-character manifest SHA-256. In this profile a trusted local Ollama daemon listens on `127.0.0.1:11434`. Windows and WSL2 can share the same daemon when WSL can reach this address. On other WSL network configurations, start Ollama inside the distribution. The inference client uses only the Python standard library. [Provider implementation](../sciagent/local_llm.py)

## Install and identify the model

Run the installation command once on the machine hosting Ollama. It downloads the public model weights. In the Linux environment used for the studies, check that the installed tag and its digest match the pinned experiment record.

```bash
ollama pull qwen2.5:1.5b
curl -fsS http://127.0.0.1:11434/api/version
curl -fsS http://127.0.0.1:11434/api/tags
```

The Stage 4 WSL2 run used Ollama 0.40.17 with `qwen2.5:1.5b` (GGUF Q4_K_M). Its manifest SHA-256 is `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`. The weight blob is `183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4`.

The provider checks the installed manifest digest, the GGUF metadata and the weight blob named by the local Modelfile before sending a prompt. After generation it checks both the running model digest and the installed manifest digest. It connects directly to the numeric loopback address and rejects HTTP redirects. The local host, model store and daemon are trusted inputs. [Installed model inventory](https://docs.ollama.com/api/tags) · [Running model inventory](https://docs.ollama.com/api/ps)

## Interpreting the measurements

Each invocation requests JSON text at temperature 0 with seed 42, a 4,096-token context and a 128-token generation limit. The two-minute keep-alive permits repeated requests to reuse a loaded model. The raw text and stop reason are retained for the deterministic candidate checks.

The inference record carries provider-reported input and output token counts and the nanosecond counters for loading, prompt evaluation, token evaluation and total model time. Client wall time includes metadata checks and transport. A missing usage counter is recorded as `null`. Local inference incurs a hosted API charge of USD 0. Measuring electricity or amortized hardware expense requires separate instrumentation. [Generation API](https://docs.ollama.com/api/generate)

## Adaptive finite-element adviser

In the adaptive benchmark, `ai_assisted` receives up to five candidates from the same model-based feasible shortlist as its paired `advisor_rules` control. The prompt supplies predicted L2 error, wall time and memory, the accuracy target, and the problem and hardware descriptors. Each response can nominate exactly one offered identifier. An abstention, truncated answer, unknown ID or provider error retains the deterministic fallback. Every executed candidate still runs the independent numerical checks. The receipt distinguishes a completed inference, a valid nomination and a nomination that changed the choice. It retains the raw response, model digest, sampling options, token counts and timings. [Adviser](../sciagent/adaptive_advisor.py) · [Attribution audit](../benchmarks/advisor_ablation.py)

```bash
python -m benchmarks.advisor_ablation "$RUN/adaptive_comparison.json" \
  --sha256 "$COMPARISON_SHA" --output "$RUN/advisor-ablation.json"
```

The audit checks all paired tasks and repetitions against the retained comparison digest. It counts changes relative to the leading feasible forecast in the **same** live invocation, alongside the first choice of the separate deterministic arm. Paired timing differences use only trials where both arms met the same accuracy target.

The 2026-11-25 single-host campaign ran seven held-out tasks three times each. The pinned model made 21 valid nominations and both members of all 21 pairs met their frozen accuracy targets. Each nomination matched its own leading feasible forecast. The model therefore changed zero choices. The median paired difference in total time was +2.277 seconds with the model. Ollama reported 15,495 input and 435 output tokens over these calls. This panel provides no evidence that adding live advice improved the bounded numerical decision. [Paired attribution](../evidence/e4/advisor-ablation.json) · [Six-arm numerical results](../evidence/e4/adaptive-comparison.json)
