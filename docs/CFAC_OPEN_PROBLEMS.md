# CFAC open problems and comparison protocol

## Which action to take after pair futility

A proof that the current P1/RT0 pair cannot attain the tolerance leaves two possibilities. The primal space may already be adequate while the dual space limits the certificate, or both spaces may require improvement. A positive lower bound on the primal discretization error can settle some cases. Comparing nested primal and dual spaces separately would settle more. A budget stop leaves the mathematical question open.

## A stronger dual baseline

Implement a degree-robust patch reconstruction using the mixed local Neumann problems of Ern and Vohralík. For P1 investigate RT1 with shared oriented face moments and cell moments. Specify the local compatibility conditions, boundary traces, Piola orientation, stable lifting for an inexact iterate and independently checked polynomial equilibrium. Verify that the checker can integrate the higher-degree gap exactly. Start with deterministic unit tests on one patch, then compare local RT0, globally optimized RT0 and patch RT1 at fixed primal fields. The inexpensive spanning-tree flux supplies feasibility but its quality depends on the tree. [Ern and Vohralík, SIAM Journal on Numerical Analysis (2015)](https://doi.org/10.1137/130950100)

## Classical algebraic stopping

Compare a fixed-tolerance solver with a solver that stops when its guaranteed algebraic error bound falls below a prescribed fraction of a discretization indicator. Implement a reconstruction-based algebraic bound and freeze its fraction before evaluation. Apply the same final certificate checker, mesh sequence, initial field and flux policy to both. Compare this baseline with CFAC with and without dual coarse correction, witnesses and floor inspection. The exact dense residual solves used as controls in this branch must be charged in full. [Papež, Strakoš and Vohralík, author's publication list](https://who.rocq.inria.fr/Martin.Vohralik/Pages/PublRevInt.html)

## Matched-accuracy experiment

Freeze held-out P0 load families, tolerances on both sides of observed floor transitions, one normalization of the energy norm, hardware, memory limits and solver budgets. Pair runs by problem and tolerance and interleave order. Include mesh generation, assembly, solving, witness probes, reconstruction, certification, refinement, discarded work, failures and timeouts in totals. Report counts of certificates issued, unresolved outcomes, peak memory and the distribution of paired time to verified solution. Report expensive inspections and unsuccessful runs individually. A weak RT0 baseline alone cannot establish a speed advantage. The current timing data measure only the cost of running the reference demonstration.

## Proofs and representations to extend

- Develop scalable upper bounds on primal and dual algebraic energy. Compare their width and evaluation cost with exact discrete residual solves. Any spectral bound must act on a precisely identified norm and Neumann quotient.
- General non-polynomial data require a trusted description of the original source, a validated projection or enclosure of the cell means, and a certified oscillation integral. Point samples and the sine micro-check's floating quadrature leave these obligations open.
- Introduce independent enclosures for floating geometry, field uncertainty and RT moment uncertainty before accepting a tolerance using rounded computation. A certificate for an exactly represented binary field establishes a result for that field.
- Test whether estimates of future gains and inspection costs help on held-out problems. Refinement changes both discretization spaces. A ski-rental bound assumes a known purchase cost and successful purchase. A greedy covering bound assumes known separable nonincreasing marginal gains and fixed costs. Neither theorem supplies a competitive ratio for this prototype.
- Extend the native interface after defining an independently replayable RT export and verifying MFEM orientation against the rational oracle. Distributed solvers, general adaptive remeshing and hardware comparisons require their own evidence.
