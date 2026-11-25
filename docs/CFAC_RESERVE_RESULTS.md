# Reserving a checked RT1 trial when CFAC makes slow progress

The new policy recovered the historical signed-stripe failure in six of six attempts. On the three held-out load families it never triggered. Both CFAC policies delivered 30 independently accepted fields out of 36 held-out submissions. Classical stopping delivered all 36. The policy remains opt-in.

| Physical submissions | Original CFAC | CFAC with reserve | Classical stopping |
|---|---:|---:|---:|
| Historical cases, accepted / submitted | 30/36 | 36/36 | 36/36 |
| Held-out cases, accepted / submitted | 30/36 | 30/36 | 36/36 |
| All physical cases, accepted / submitted | 60/72 | 66/72 | 72/72 |
| Wall seconds spent on all physical attempts | 29.59 | 19.96 | 14.81 |

### Two different reasons for an unsuccessful stop

On the historical stripes at tolerance 1/9, the original policy reached the 128-action limit with a current-field upper bound of 0.115775. The new policy triggered on step 41, performed its primal sweep, checked the RT0 field and obtained an independently checked RT1 upper bound of **0.100483 on step 45**. Its median time to the returned field was 833.0 ms. The four reserved actions averaged 154.2 ms, including 123.2 ms for RT1 reconstruction. The triggering witness probe averaged another 34.7 ms. Classical stopping needed 30 actions, delivered a bound of 0.094017 and took a median 314.7 ms. The original policy's 2427.1 ms median measures time to failure.

On the held-out checkerboard at tolerance 1/12, both CFAC variants found the refined P1/RT0 pair **futile**. The reserve requires a **sufficient** pair and did not activate. After a primal sweep and a fine-mesh RT1 trial, the checked upper bound was 0.095718, above the target. Both policies stopped at action 16 with 112 actions left. Classical stopping used six primal sweeps on the refined mesh before its RT1 trial and delivered an upper bound of 0.080428 on action 26. A follow-up experiment can test whether an algebraic stopping check gives the primal field enough smoothing before RT1 reconstruction after a proof of RT0-pair futility.

### Paired timing by case

Times are medians in milliseconds for the submitted job. A dagger marks zero acceptances in six attempts. Every other timing cell represents six accepted jobs. The two ratios are medians of within-block wall-time ratios, with six jointly accepted pairs wherever a number appears. A dash denotes zero such pairs. The six new physical cases are labeled **new**. The final two rows are controls.

| Load and tolerance | Original CFAC | Reserve | Classical | Reserve / original | Reserve / classical |
|---|---:|---:|---:|---:|---:|
| Unit, 3/20 | 82.9 | 82.2 | 75.5 | 0.997 | 1.101 |
| Unit, 1/8 | 131.5 | 131.3 | 152.2 | 1.007 | 0.866 |
| Signed stripes, 1/6 | 214.5 | 216.9 | 321.0 | 0.998 | 0.672 |
| Signed stripes, 1/9 | 2427.1† | 833.0 | 314.7 | — | 2.658 |
| Localized, 1/5 | 109.6 | 109.1 | 95.0 | 1.029 | 1.154 |
| Localized, 3/20 | 441.5 | 443.4 | 326.3 | 0.992 | 1.349 |
| **New** checkerboard, 2/15 | 77.3 | 80.5 | 95.0 | 1.043 | 0.831 |
| **New** checkerboard, 1/12 | 345.3† | 340.2† | 294.6 | — | — |
| **New** diagonal band, 1/5 | 99.7 | 99.2 | 96.5 | 1.016 | 1.049 |
| **New** diagonal band, 2/15 | 550.7 | 539.0 | 321.5 | 0.988 | 1.671 |
| **New** corner dipole, 1/8 | 82.7 | 82.8 | 96.5 | 0.997 | 0.860 |
| **New** corner dipole, 1/12 | 381.7 | 388.4 | 270.2 | 1.008 | 1.436 |
| Zero load, 1/8 | 58.4 | 60.1 | 60.4 | 1.028 | 1.014 |
| One action, 1/8 | 57.7† | 56.6† | 59.8† | — | — |

All 252 submissions are accounted for. There were 216 acceptances, 24 budget stops and 12 stops with no eligible action. No worker errored or exceeded the deadline. Unsuccessful submissions consumed 19.68 s of 66.49 s total. Decision-time peak RSS ranged from 20.97 to 22.86 MiB, sampled just before encoding the live packet. Process peak RSS including export ranged from 21.14 to 47.66 MiB. Each case, policy and repetition retains its measured costs. [Raw measurements and 35 distinct journals](../evidence/cfac/reserve-study.json.gz) · [Paired ratios, costs and terminal outcomes](../evidence/cfac/reserve-summary.json)

Only the six runs on the diagnosed case activated the reserve, and all six accepted at the RT1 inspection. The other 13 cases followed the same sequence of actions, state digests and next decisions with and without the reserve. The ordinary study therefore supplies zero observations of a reserved trial missing its target. Regression tests exercise an RT1 miss, a charged restoration, budget exhaustion before restoration, a construction failure and a malformed candidate. [Controller tests](../tests/test_cfac_progress_reserve.py) · [Replay and accounting tests](../tests/test_cfac_reserve_study.py)

## How the controller spends the reserve

Pair sufficiency concerns the best attainable fields in the current spaces. The independent verifier checks the fields actually delivered. On a refined mesh, the new policy compares the positive excess of the certified RT0 squared upper bound over the squared tolerance at distinct tight probes that establish pair sufficiency. If two successive corrections each leave at least 90% of the previous excess, the controller can try the reserve once. At least four actions must remain. No timing or reference-solution data enter the decision.

The reserve buys one primal sweep, a fresh RT0 check and, if still needed, RT1 reconstruction and an independent RT1 check. A conforming P1 field that passes the intermediate check can return early. An RT1 miss may spend a further action to restore the post-sweep RT0 state. Subsequent corrections continue to use the shared global budget. This rule changes the allocation of work. The P1 space, PDE, energy tolerance and mathematical certificate keep their original definitions. [Frozen trigger and cases](CFAC_RESERVE_PROTOCOL.md) · [Controller](../cfac/controller.py)

The earlier 392-run receipt identifies where the historical signed case first qualified for this rule. At the second slow interval, step 41, the last correction closed about 6.26% of the excess and 87 actions remained. These observations informed the trigger. The checkerboard, diagonal-band and corner-dipole results evaluate its behavior on other loads. [Earlier result and replay](CFAC_STOPPING_RESULTS.md)

## Measurement boundary

One Linux process starts for each submission. Its measured time includes interpreter startup, setup, all controller decisions and receipt of the full live decision. A success delivers the problem, field, flux and independent certificate. The controller retains its action times and independent checks on unsuccessful paths. Ordinary failures retain the elapsed time to their terminal decision. A timeout or worker error is charged through termination and collection. Full journal export after delivery and offline replay are outside the live endpoint. Peak memory is measured both just before packet encoding and across the whole worker lifetime including journal export. [Runner and summary](../research/cfac/reserve_study.py)

## Reproduction

The design was committed at `128acdc2e2c644826e86272e0d53521cd8db81aa` before measurement. The runner checks its SHA-256, `da54e3ec3167470d3bf1daccd3ae90d016fb2a8cb9f1060c1a12cae862ccc498`. The implementation ran from a clean Linux-filesystem checkout of `1332a03c3d7bd9f816f2619d0f80d5ce00c67833`. The machine was an Intel Core Ultra 7 155H under WSL2 Ubuntu, Linux 6.6.87.2, CPython 3.12.3, pinned to logical CPU 2. Timing uses six blocks with rotated policy positions. All ratios in this report use measurements from this experiment.

From the branch containing the evidence, use a fresh destination and the retained SHA-256 of the compressed archive. Replay rebuilds each distinct policy trajectory, validates child-mesh loads, checks the field and certificate against the frozen problem, and checks the per-step costs and submission schedule.

```sh
out=$(mktemp -d)
digest=1f2571690667df2f8959e90889c5def3097b5352421d91087aa075c6ea3d99ab
python -m research.cfac.reserve_study --replay evidence/cfac/reserve-study.json.gz \
  --sha256 "$digest" --summary "$out/summary.json"
cmp "$out/summary.json" evidence/cfac/reserve-summary.json
```

To collect new measurements on Linux, select a CPU in the current affinity mask and use a clean checkout on a Linux filesystem. Each new submission independently measures wall time.

```sh
rev=1332a03c3d7bd9f816f2619d0f80d5ce00c67833
out=$(mktemp -d)
git worktree add --detach "$out/source" "$rev"
cd "$out/source"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
env LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c "$cpu" python -m research.cfac.reserve_study \
  --output "$out/reserve-study.json.gz" --revision "$rev"
```

The updated runner also replayed all 392 historical submissions. For the eight shared cases (including controls), both original policies reproduced the exact deterministic journal digests from that archive. The comparisons cover 16 policy/case combinations across the two runs. [Original raw archive and digest](CFAC_STOPPING_RESULTS.md#reproduction-and-verification)

WSL2 CPython 3.12.3 and Windows CPython 3.13.2 each replayed all 252 new submissions and generated byte-identical summary JSON. The full WSL2 test run used the existing serial and MPI MFEM builds after the timed campaign. The command `python -m pytest -q --require-mfem --require-mpi --junitxml=/root/mfem-cfac-reserve-l93xy4au/suite.xml` passed **649 tests in 189.59 s**, with zero failures, errors or skips. Twelve test cases were added in this stage. [Full test stdout](../evidence/cfac/reserve-pytest.txt) · [GitHub Actions on the implementation commit](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45645746999)
