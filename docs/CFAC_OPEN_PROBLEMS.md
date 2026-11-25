# CFAC open problems and comparison protocol

## Which action to take after pair futility

A proof that the current P1/RT0 pair cannot attain the tolerance leaves two possibilities. The primal space may already be adequate while the dual space limits the certificate, or both spaces may require improvement. A positive lower bound on the primal discretization error can settle some cases. Comparing nested primal and dual spaces separately would settle more. A budget stop leaves the mathematical question open.

## A stronger dual baseline

The P1/RT1 vertex-patch construction and independent moment checker now work on the frozen rational contract. Exact tests cover patch stationarity, non-Galerkin iterates, affine-divergence modes and quartic integration. At fixed Galerkin fields, six tolerance rows on two mesh resolutions could certify before refining. The inexact-field control gave a wider RT1 bound than optimal RT0. The next proof question is whether an efficient, local algebraic residual lifting can retain quality for inexact iterates. Higher polynomial degrees and robustness in degree remain open. [RT1 experiments](CFAC_RT1_ENRICHMENT.md) · [Ern and Vohralík, SIAM Journal on Numerical Analysis (2015)](https://doi.org/10.1137/130950100)

## Classical algebraic stopping

The matched small-mesh study now compares a fixed relative residual with a certified algebraic/discretization stopping rule at a frozen fraction of one half. Both use the same primal sweep, refinement and independent field checkers. The study also measures CFAC, its RT0-only variant and three changes to its scheduling policy. The remaining comparison is against scalable native stopping and flux reconstruction on larger problems at equal verified accuracy. [Measured policies and outcomes](CFAC_STOPPING_RESULTS.md) · [Papež, Strakoš and Vohralík (2018)](https://doi.org/10.1007/s00211-017-0915-5)

## Matched-accuracy experiment

The original comparison found that CFAC spent 84 of 128 actions on inspections on the signed stripes at tolerance 1/9. A second frozen study tested a four-action reserve when the current refined pair is sufficient and progress is slow. It recovered the diagnosed case and retained the original policy's coverage on three new load families. [Original paired comparison](CFAC_STOPPING_RESULTS.md) · [Reserve study](CFAC_RESERVE_RESULTS.md)

The refined RT1 stopping study tested a rule for additional primal work after a refined P1/RT0 pair-futility verdict. It recovered the historical strict checkerboard and a held-out edge patch. It matched classical coverage on all 18 physical cases, with higher total cost. The historical sweep diagnostic shows that a valid RT1 field was available four sweeps before algebraic balance. Estimating the value of an earlier expensive RT1 inspection remains an open cost question. [Paired study and exact field diagnostics](CFAC_FUTILE_RESULTS.md)

## Proofs and representations to extend

- Develop scalable upper bounds on primal and dual algebraic energy. Compare their width and evaluation cost with exact discrete residual solves. Any spectral bound must act on a precisely identified norm and Neumann quotient.
- General non-polynomial data require a trusted description of the original source, a validated projection or enclosure of the cell means, and a certified oscillation integral. Point samples and the sine micro-check's floating quadrature leave these obligations open.
- Introduce independent enclosures for floating geometry, field uncertainty and RT moment uncertainty before accepting a tolerance using rounded computation. A certificate for an exactly represented binary field establishes a result for that field.
- Test whether estimates of future gains and inspection costs help on held-out problems. Refinement changes both discretization spaces. A ski-rental bound assumes a known purchase cost and successful purchase. A greedy covering bound assumes known separable nonincreasing marginal gains and fixed costs. Neither theorem supplies a competitive ratio for this prototype.
- Extend the native interface after defining an independently replayable RT export and verifying MFEM orientation against the rational oracle. Distributed solvers, general adaptive remeshing and hardware comparisons require their own evidence.
