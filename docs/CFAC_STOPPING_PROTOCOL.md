# Frozen comparison of algebraic stopping and CFAC

The primary question is whether CFAC reduces elapsed time from submission to independently accepted energy accuracy on these fixed small CPU problems. A second question targets the earlier mesh result. On the unit load at tolerance 3/20, compare CFAC with its RT1 option enabled and disabled. At 1/8 the earlier RT1 trial failed to save a refinement. Retain both cases.

## Design before timing

The machine-readable design fixes seven policies, eight cases, seven complete interleaved blocks, the tolerances, action limits and per-trial wall limit. Each policy occupies each within-case order position once. Use one fresh Python process per trial and sequential timed runs on a single Linux CPU. This reference implementation uses rational arithmetic. Each timed process starts from the same frozen problem definition and zero field. Each eligible policy may use one uniform red refinement that preserves the source on every child. Keep all attempts, including failures and the one-action budget control. [Frozen design](../research/cfac/stopping_design.json)

- Two previously measured constant-load cases reproduce the separating and tighter tolerances from the RT1 experiment.
- Four held-out physical tasks combine a signed three-stripe source or a localized positive source with two tolerances. Both families use n=3. They were specified before measuring any comparison policy. These cases cover two load families, not independent draws from a population.
- A zero load checks early acceptance and a one-action allowance checks exhaustion before acceptance. Report controls separately from physical tasks.
- The fixed policy targets a free-node relative residual of one part per million. The classical policy seeks an algebraic energy upper bound at most one half of a discretization indicator. Freeze both factors across all cases and use the same primal sweep, flux selection and final independent acceptance tests in these two baselines.
- Compare full CFAC with RT0-only CFAC and with three single policy changes disabling coarse dual correction, the P2 witness, or floor-based scheduling respectively. Identify any remaining numerical checks and any fallback schedule in the implementation report.

## Endpoints

For each case and policy report successes out of seven, terminal reasons, the distribution of submitted wall times, peak resident memory, refinements and charged action counts. Archive the journals with action costs, including failed reconstruction or inspection and discarded work. The primary paired contrasts are full CFAC divided by classical stopping, classical divided by fixed stopping, and full CFAC divided by RT0-only CFAC. Use the median of within-block wall-time ratios for pairs in which both are independently accepted. Ratios below one favor the numerator. Retain failures in the success denominator and in total resources consumed. A failed stop time measures resources spent without an accepted solution. It cannot serve as a time to solution. Keep startup, process communication and returning the result inside submitted time. Record policy-internal wall time separately. Offline replay and archiving costs are separate from the live decision and must be identified.

Report the number of paired observations alongside ratios. Show the seven individual measurements or their range in the underlying evidence. No post-hoc stopping, rejection of timing outliers or tuning on held-out outcomes. The study can establish measured behavior on this machine and these cases. It cannot establish a platform-independent speedup, a general scheduling theorem or polynomial-degree robustness.
