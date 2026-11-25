# Primal stopping before refined RT1 reconstruction

On the strict checkerboard in the preceding study, both CFAC policies proved the refined P1/RT0 pair futile. They then reconstructed RT1 on a field with one fine-grid primal sweep and stopped after the checked upper bound missed the target. Classical stopping returned a verified field on that mesh. The proposed change uses the existing certified algebraic stopping test to decide how long to smooth the primal field before paying for the refined RT1 trial. [Diagnosed case](CFAC_RESERVE_RESULTS.md)

## Comparison

Compare the previous reserve policy (`cfac_reserve`), the opt-in policy with the new rule (`cfac_retry`), and classical stopping (`classical`). Each starts from identical data and uses the existing independent current-field checkers. The main endpoint is an accepted field on the historical checkerboard at tolerance 1/12 under the original 128-action limit. Report preservation of coverage on the other eleven historical physical cases, including the strict stripes. Evaluate separately on six fresh problems. Record the cost of additional stopping checks and sweeps even when the delivered field misses.

## Cases and measurement

The new policy retains the original coarse-mesh path, the preparatory primal sweep and the sufficient-pair reserve. Change only a refined-mesh `PAIR_FUTILE` decision that would ordinarily start RT1 reconstruction. With at least three actions left, spend one on `futile_stopping_check` using the existing certified criterion with fraction 1/2. It stops when the squared algebraic upper bound is at most one quarter of the squared discretization lower bound. A passing check sends the field for RT1 reconstruction and independent RT1 inspection. A failing check buys another primal sweep and check only if four actions remain after that check. Otherwise the policy spends the remaining opportunity on the best available field and records the budget-limited decision. With fewer than three actions at the original decision, follow the existing direct RT1 path. All these actions share the original budget. Only the independent field inspection can accept. A missed RT1 trial follows the existing terminal decision on the last mesh.

Freeze all inputs before timing. The three new families use a 3-by-3 mesh with two triangles per square and diffusion one. Within a square the split case uses the triangle ordering in `uniform_mesh`. The other two families assign the same load to both triangles. [Case definitions, full load vectors and budgets](../research/cfac/futile_design.json)

| New family | Load | Energy tolerances |
|---|---|---|
| Split triangles | 2 on the first triangle and −1 on the second in each square | 1/8 and 1/14 |
| Edge patch | Square rows (3, 3, 0), (0, −1, 0), (0, 0, 0) | 1/8 and 1/12 |
| Elbow | Square rows (2, 2, 2), (2, 0, 0), (2, 0, 0) | 3/20 and 1/12 |

Run the twelve historical cases, the six fresh cases and the original zero-load and one-action controls. Each physical submission gets 128 actions, at most one uniform refinement and a 30 s wall deadline. Use six interleaved blocks, three policies and one pinned Linux CPU, giving 360 fresh-process submissions. Each policy occupies each within-case order position twice. Count a success only after independent acceptance of the actual delivered field.

Report per-case coverage, submitted wall times including failures, peak RSS, the number of entries into the new decision path, stopping checks, additional sweeps, and final certificate outcomes. Report all three paired contrasts using within-block wall-time ratios only when both runs return a checked solution. Report incomplete pairs and their costs separately. Timing begins before process launch and ends when the parent receives the decision packet, including the field, signed load and certificate on success. Diagnostic journal export and offline replay follow delivery. Only frozen comparisons on this small exact-arithmetic CPU workload support performance statements. Retain unsuccessful and slower cases.
