# A budgeted RT1 trial after slow progress on a sufficient pair

The previous comparison identified one diagnosed failure. On signed stripes at tolerance 1/9, the refined P1/RT0 pair was proved sufficient, but the current fields did not meet the requested tolerance before CFAC used all 128 actions. The classical policy certified a field on the same refined mesh. We will compare an optional change in scheduling against both original policies. [Previous experiment](CFAC_STOPPING_RESULTS.md)

## Problems and controls

The study freezes the complete cellwise loads, tolerances, action budgets and labels before timing. All physical cases use constant unit diffusion, zero Dirichlet data and at most one uniform refinement. The six historical physical cases test the diagnosed failure and possible regressions on the other five cases. Six new cases use three new load patterns on a 3-by-3 square grid, with the same load on both triangles of each square. [Machine-readable design](../research/cfac/reserve_design.json)

| New family | Load on the nine squares, row by row | Energy tolerances |
|---|---|---|
| Checkerboard | (2, −1, 2), (−1, 2, −1), (2, −1, 2) | 2/15 and 1/12 |
| Diagonal band | (3, 1, 1), (1, 3, 1), (1, 1, 3) | 1/5 and 2/15 |
| Corner dipole | (3, 0, 0), (0, 0, 0), (0, 0, −3) | 1/8 and 1/12 |

The zero-load control must obtain independent acceptance. The one-action control must exhaust its budget after setup. Both controls are reported separately from physical coverage. Each physical submission receives 128 actions and a 30 s deadline. The exact-rational reference solver, initial field, flux constructors and independent checkers are shared.

## Comparison and endpoints

The new policy starts with the original CFAC schedule. On the refined mesh, collect the positive excess of the checked RT0 squared upper bound over the target, $X=U^2-\epsilon^2$, only at tight probes returning `PAIR_SUFFICIENT`. Compare successive distinct corrected states. A slow interval satisfies $10X_{\mathrm{new}}\ge 9X_{\mathrm{old}}$, including equality or an increase. A faster interval breaks the slow streak. A different verdict at a tight probe clears its history. Ordinary inspections and repeated observations of the same state do not supply an interval.

After two consecutive slow intervals, and with at least four unspent actions, try the reserve once on that mesh. Spend one action on primal smoothing and one on an ordinary RT0 inspection. Accept immediately if that inspection meets the original tolerance. Otherwise spend one action on patch RT1 reconstruction and one on independent RT1 inspection. A valid miss can spend the next action restoring the post-sweep RT0 state and resume ordinary CFAC, subject to the original budget. If reconstruction fails, the retained RT0 field is available for inspection. An invalid checked candidate stops the run. The trigger is a heuristic decision about spending work. The existing RT checkers alone control acceptance. The constants were selected using the historical failure and are frozen for all new loads.

Run original CFAC, CFAC with the reserve, and certified classical stopping in six complete interleaved blocks on one Linux CPU. For every case each policy occupies each order position twice. Fourteen cases × three policies × six blocks give 252 fresh submissions. Keep the historic 392-trial archive as originally published.

The primary outcomes are recovery of the diagnosed case, preservation of previously accepted physical cases, and per-case acceptance on the three new load families. Report new gains and losses side by side. A repeated observation on the same case tests timing variability and reproducibility. Each new family supplies two tolerance settings.

Measure elapsed time from before process launch to receipt of the independently checked field or unsuccessful terminal decision. Include every inspection, reconstruction, discarded trial and recovery action in the controller budget and action times. For each case report submitted times and the median of within-block wall-time ratios only on jointly accepted pairs, with their counts. Also report all-attempt cost by policy, peak RSS, triggers, successful reserved trials and misses, and whether a miss eventually reaches acceptance. An exhausted budget or lack of eligible actions is an unsuccessful submission. Preserve the original tolerances, complete trials and counterexamples regardless of the outcome.
