# Matched accuracy and cost of algebraic stopping

On the six nonzero physical cases, classical stopping and fixed stopping each returned 42 independently accepted fields in 42 attempts. Full CFAC returned 35 in 42. CFAC delivered faster than classical stopping on the tighter unit load and the looser signed load. Classical stopping delivered faster on the other three jointly accepted cases and solved the tight signed case on all seven attempts. The comparison covers one WSL2 CPU and two held-out load families.

Summing all 42 physical submissions per policy, including failed work, gives 9.38 s for classical stopping, 12.54 s for fixed stopping, 24.66 s for CFAC and 25.66 s for RT0-only CFAC. The different acceptance counts matter when interpreting these totals.

## Measured time to a checked field

Times are medians in milliseconds over seven fresh processes per cell. A dagger marks a cell with **zero acceptances** and reports time until the unsuccessful stop. All other time cells had **seven acceptances**. Ratios are medians of within-block ratios, with seven jointly accepted pairs per reported value. A dash means zero jointly accepted pairs. The last two rows are controls.

| Load and energy tolerance | Fixed | Classical | CFAC | CFAC RT0 only | CFAC / classical | Classical / fixed | CFAC / RT0 only |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unit, 3/20 | 74.3 | 73.3 | 81.2 | 100.0 | 1.095 | 0.989 | 0.827 |
| Unit, 1/8 | 177.4 | 158.5 | 140.0 | 139.7 | 0.864 | 0.881 | 1.002 |
| Signed stripes, 1/6 | 470.7 | 327.6 | 217.4 | 190.8 | 0.670 | 0.699 | 1.162 |
| Signed stripes, 1/9 | 461.8 | 322.3 | 2533.2† | 2492.8† | — | 0.677 | — |
| Localized load, 1/5 | 101.7 | 97.3 | 112.9 | 113.1 | 1.170 | 1.034 | 0.981 |
| Localized load, 3/20 | 466.0 | 337.9 | 472.7 | 545.0 | 1.368 | 0.708 | 0.861 |
| Zero load, 1/8 | 59.0 | 64.3 | 59.6 | 64.4 | 0.984 | 1.062 | 1.035 |
| One-action budget, 1/8 | 62.9† | 63.4† | 57.3† | 63.3† | — | — | — |

At unit load and tolerance 3/20, CFAC's RT1 step avoided one refinement in every block and reduced paired elapsed time by 17.3% relative to RT0-only CFAC. The classical policy also used RT1 on the initial mesh and delivered the field sooner than full CFAC. At tolerance 1/8, CFAC paid for an unsuccessful RT1 trial and refined in all seven runs. Its paired median against RT0 only was 1.002. Across the two held-out load families, the benefit of an RT1 option depended on the case.

On the signed stripes at 1/9, classical stopping took 30 actions, used one refinement and obtained an RT1 certificate on the refined mesh. Full CFAC spent its entire allowance of 128 actions on every attempt. Each run included 42 RT0 inspections, 42 tight inspections with a witness and 34 local dual sweeps. The 294 witness probes together took 10.91 s and the 294 ordinary inspections took another 4.50 s. The baseline certificates show the target was reachable under the shared mesh and action limits. A budget stop itself gives no lower bound on attainable accuracy.

Forty of each run's 42 tight probes found that the optimal P1/RT0 pair could meet the tolerance. The last inspected current-field bound was 0.115775, still above 1/9. CFAC tried RT1 only on the starting mesh. Classical stopping reached a verified bound of 0.094017 on the refined mesh. This identifies a scheduling hypothesis for a fresh experiment. When current-field progress stalls despite repeated pair-sufficiency findings, reserve enough actions for primal work and a refined-mesh RT1 trial.

## What the scheduling changes cost

| Nonzero load and tolerance | CFAC / no coarse correction | CFAC / no P2 witness | CFAC / no floor schedule |
|---|---:|---:|---:|
| Unit, 3/20 | 0.976 | 1.039 | 1.054 |
| Unit, 1/8 | 1.099 | 1.090 | 0.895 |
| Signed stripes, 1/6 | 0.790 | 1.121 | 1.467 |
| Signed stripes, 1/9 | — | — | — |
| Localized load, 1/5 | 1.006 | 1.175 | — |
| Localized load, 3/20 | 0.870 | 1.270 | — |

Removing the P2 witness retained full CFAC's coverage on the six physical cases and reduced the paired median elapsed time on all five jointly accepted cases. The fallback schedule without floor-based decisions accepted three of the six cases. It stopped with no eligible action on the tight signed load and both localized-load cases. The coarse dual correction reduced the median on three of the five jointly accepted cases. Seven repetitions characterize variability on this machine; they do not supply additional independent load families.

Across all policies and controls, 294 of 392 attempts returned an independently accepted field, 77 exhausted their budgets and 21 ended with no eligible action. No worker errored or timed out. The 98 unsuccessful submissions consumed 75.485 s of the total 132.381 s of submitted wall time. Worker peak RSS before encoding the decision ranged from 20.80 to 22.69 MiB. The per-process peak including full journal export ranged from 20.97 to 69.70 MiB. On the zero load, CFAC's median internal controller time was 2.1 ms against 59.6 ms for the delivered field, illustrating the size of the startup and transfer costs on these small problems.

The archive holds the 392 measured trajectories with 46 distinct journal payloads. The per-case summary includes action counts, time distributions, memory, coverage, unsuccessful time and each paired ratio. [Raw timings and proof journals](../evidence/cfac/stopping-study.json.gz) · [Machine-readable results](../evidence/cfac/stopping-summary.json)

## Mathematical stopping rules

For each frozen P1 problem, the reduced system uses stiffness matrix $K$, load vector $b$, current coefficients $v$ and residual $r=b-Kv$ on free vertices. The fixed rule stops primal iteration when $\|r\|_2\le10^{-6}\|b\|_2$. If $b=0$, the rule requires $r=0$, including when signed element loads cancel at the free vertices.

The classical policy stops primal iteration if $A^+\le\gamma^2 D^-$ with $\gamma=1/2$. Here $A^+$ bounds the **squared algebraic error from above** and $D^-$ bounds the **squared Galerkin discretization error from below**. This implements the principle of balancing algebraic work with discretization accuracy. The particular tree lifting and bubble formulas below are the methods tested in this repository. The study does not benchmark the published implementation of Papež, Strakoš and Vohralík. [Related primary work](https://doi.org/10.1007/s00211-017-0915-5) · [Implemented stopping rules](../cfac/stopping.py)

**Upper bound.** With diagonal $D$ of $K$, take the direction $d=D^{-1}r$ and the exact trial length $\alpha=(r,d)/(d,Kd)$. The attained energy reduction is $g=\alpha(r,d)$. The remaining residual is $r'=r-\alpha Kd$. Exact completion of the square gives $\|v_*-v\|_K^2=g+(r',K^{-1}r')$. This trial is used to evaluate the bound. It does not change the solver iterate. For $r=0$ the algebraic error vanishes.

Route the remaining nodal residual weights along mesh edges to zero Dirichlet anchors. In a triangle $T$ containing the edge from child $j$ to parent $i$, add $s(x_j-x_i)/|T|$ to a piecewise constant vector $L$, where $s$ is the accumulated residual weight. Pairing with $\nabla w$ of a P1 function gives $s(w_j-w_i)$. The tree sum telescopes, so $(L,\nabla w)=r'^T w$ on the zero-trace P1 space. Cauchy–Schwarz gives $A^+=g+\sum_T |T|\,|L_T|^2/\kappa$. This broken vector field certifies discrete algebraic energy. The separate RT checker certifies accuracy for the PDE solution.

**Lower bound.** The cubic cell bubble $\psi_T=\lambda_0\lambda_1\lambda_2$, extended by zero, has zero trace. Its gradient is energy-orthogonal to every conforming P1 function for constant $\kappa$, since the gradient of each P1 test is constant on the cell. It is also orthogonal to the bubbles on other cells. For piecewise constant $f$, the residual pairing is $f_T|T|/60$ and $\int_T|\nabla\psi_T|^2=|T|\sum_i|\nabla\lambda_i|^2/180$. Optimizing over their disjoint span gives
$$D^-=\sum_T\frac{|T| f_T^2}{20\kappa\sum_i|\nabla\lambda_i|^2}\le\|u-v_*\|_\kappa^2.$$
The squaring is elementwise, so signed loads do not cancel in this indicator. Unit tests also compare the algebraic bound with exact discrete solves on a skew mesh, check thresholds at equality, and compare the discretization bound with a separately enclosed continuous constant-load energy. [Numerical controls](../tests/test_cfac_stopping.py)

## What the seven policies do

All policies start from the zero field on the identical P1 mesh with tree-equilibrated RT0 flux. Uniform refinement subdivides each triangle into four, preserves each parent load on its four children, prolongs the P1 field and retains the frozen tolerance and global budget. A stopping check consumes an action and never accepts a PDE field. Acceptance is possible only after the independent current-field RT0 or RT1 check. The two algebraic baselines use the same assembly, primal sweeps, flux choice and checkers.

| Policy ID | Scheduling choice |
|---|---|
| fixed | Check relative residual initially and after each Gauss–Seidel sweep. Once stopped, inspect tree RT0, try patch RT1 if needed, then refine if eligible. |
| classical | Same sequence with the certified algebraic/discretization balance test. |
| cfac | Inspect RT0 and floor evidence, choose primal or dual corrections, and allow a separately checked RT1 trial following a proof of pair futility. |
| cfac_rt0 | Use CFAC with the RT1 trial disabled. |
| no_coarse | Replace the coarse dual correction by a local dual sweep. |
| no_witness | Retain tight algebraic/floor inspection and omit its P2 residual witness. |
| no_floor | Inspect current-field certificates and use a predetermined primal, coarse dual, local dual, then RT1 schedule on each level. Removing the proof changes the schedule as well as the inspection cost. |

The two stopping baselines isolate the effect of changing the algebraic criterion. The CFAC comparisons include the cost and consequences of different action schedules. A cheaper action count alone cannot determine which policy finishes first. [Policy implementation](../cfac/baselines.py) · [Frozen protocol](CFAC_STOPPING_PROTOCOL.md)

## Measurement and audit boundary

The parent starts its clock before launching a fresh interpreter. A successful worker sends the accepted primal–flux state, frozen current problem, certificate observation and status. The parent stops its clock after decoding the whole packet. The worker then writes the complete diagnostic journal. Ordinary unsuccessful decisions deliver a terminal status. Timeouts and worker errors are charged through process termination and collection. Per-action times include unsuccessful trials and inspections, and the controller's own elapsed time includes the decision logic. Linux VmHWM is sampled immediately before encoding the live packet. A second peak from wait4 includes post-decision journal export. These are process memory observations, not finite-element storage estimates.

Each unique deterministic journal is stored once, with the timing of every action retained separately for each trial. Offline replay independently rechecks accepted fields, recomputes child triangles and inherited loads, checks budgets and terminal decisions, and reruns each distinct policy trajectory. This second audit is outside the time to receive a live, independently checked field. A compressed-file digest binds the archived times as well as the mathematical records. [Runner and replay](../research/cfac/stopping_study.py) · [Paired analysis](../research/cfac/stopping_analysis.py)

## Reproduction and verification

On 2026-11-25, an Intel Core Ultra 7 155H ran the frozen design under WSL2 Ubuntu, Linux 6.6.87.2 and CPython 3.12.3. All workers inherited affinity to logical CPU 2. The design was committed at `f5e82479b7b33d19da1150d03ef4b8bbb27852f4` before the measured implementation at `def4aac1e09ec0ecb4fce9d370c6d3583dcf03d1`. The design file SHA-256 is `4f854f649114f1bea571138dec24a5a5d4e3933ffdf552a87d4d5e7283f9aa05`.

From the research branch containing the evidence, replay the stored fields and regenerate the summary. Windows CPython 3.13.2 independently replayed all 392 submissions. The regenerated summary was byte-identical to the archived JSON.

```sh
receipt=evidence/cfac/stopping-study.json.gz
digest=dd68dfaad28f1c4d7e437563b7fe8555318563d59bc910a41bc123bf8679bbda
out=$(mktemp -d)
python -m research.cfac.stopping_study --replay "$receipt" --sha256 "$digest"
python -m research.cfac.stopping_analysis "$receipt" --sha256 "$digest" --output "$out/summary.json"
cmp "$out/summary.json" evidence/cfac/stopping-summary.json
```

To take new measurements on Linux, use an unused output path and a clean worktree on a local Linux filesystem. The following selects an available logical CPU. New measurements will have their own timestamps, timings and digest.

```sh
rev=def4aac1e09ec0ecb4fce9d370c6d3583dcf03d1
out=$(mktemp -d)
git worktree add --detach "$out/source" "$rev"
cd "$out/source"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
env LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c "$cpu" python -m research.cfac.stopping_study \
  --output "$out/stopping-study.json.gz" --revision "$rev"
```

After measurement, a separate run on the measured revision used the existing serial MFEM and MPI builds with both required-test switches. `python -m pytest -q --require-mfem --require-mpi --junitxml=/root/mfem-stopping-decmzo9o/suite.xml` passed 637 tests in 197.44 s, with zero failures, errors or skips. This includes 24 new tests covering the mathematical inequalities, terminal decisions, P0 transfer, policy variants, archived field verification, failed workers and paired analysis. [Full-suite stdout](../evidence/cfac/stopping-pytest.txt) · [GitHub Actions on the implementation commit](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45641713538)
