# Certifying a fixed P1 field with an RT1 flux

On two uniform meshes for the same unit-load Poisson problem, a vertex-patch RT1 flux certified a fixed P1 Galerkin field at tolerances below the **proven minimum P1/RT0 gap**. A uniform refinement with RT0 also worked at those tolerances. An inexact-field control and a separate two-triangle example show that local RT1 reconstruction can give a wider bound than optimal RT0. These experiments establish feasibility on the measured cases. Matched end-to-end performance and polynomial-degree robustness remain open.

## Fixed-field experiment

The load and diffusion are both one, with homogeneous Dirichlet data on the unit square. For each coarse mesh we froze 33 rational tolerances from 10/200 to 42/200 inclusive before running the RT1 constructor. We checked a tree-equilibrated RT0 flux, the best RT0 flux for the same P1 field, the patch RT1 flux for that field, and independent optimal P1/RT0 fields on the uniformly doubled mesh. The latter are discrete optimizers. No continuous truth is needed.

A target counts as an avoided refinement only when the certified coarse P1/RT0 pair floor exceeds the squared tolerance, the RT1 upper bound meets it with **the identical coarse primal field**, and the refined RT0 control also meets it. For Galerkin fields the optimal RT0 gap equals the proven pair floor. Magnitudes below are square roots of the exact squared quantities in the archive.

| Coarse P1 field | Tree RT0 | Optimal RT0 | Patch RT1 | Refined optimal RT0 | Tolerances meeting the three conditions |
|---|---:|---:|---:|---:|---|
| Galerkin, n=2 | 0.297560 | 0.161374 | 0.149141 | 0.094614 | 0.150, 0.155, 0.160 |
| Galerkin, n=3 | 0.248452 | 0.120442 | 0.108093 | 0.065485 | 0.110, 0.115, 0.120 |
| Zero iterate, n=2, unit load | 0.322749 | 0.204124 | 0.258001 | 0.094614 | None |
| Zero field, n=2, zero load | 0 | 0 | 0 | 0 | None. The coarse field already passes every target. |

For n=2 the squared RT0 floor is 5/192 and the patch RT1 squared upper bound is 245989/11059200. For n=3 the corresponding values are 47/3240 and 673/57600. Six of the 132 case–tolerance combinations met all three conditions. Adjacent thresholds reuse the same fields, and the evidence covers two Galerkin mesh resolutions for one PDE. A separate regression on n=1 checks an adverse case even at Galerkin optimality. At tolerance 41/200 the current RT0 field passes with squared bound 1/24 while the patch RT1 trial misses with 13/300. [Fields, digests and all 132 verdicts](../evidence/cfac/rt1-study.json) · [Additional adverse example](../tests/test_cfac_rt1_controller.py)

## Reconstruction

Each vertex with hat function $\phi_a$ defines a patch. Write $\kappa$ for the constant positive diffusion and $v$ for the current P1 field. On each patch minimize
$$\int_{\omega_a}\kappa^{-1}|q_a-\phi_a\kappa\nabla v|^2$$
in RT1 with a prescribed affine divergence. Set zero normal trace on each edge opposite the patch center. Boundary spokes on the physical Dirichlet boundary can carry flux. Minimization over this space and addition of the local fluxes follow the equilibrated vertex-patch approach. The implementation uses this particular trace convention even where an opposite edge lies on the physical boundary. [Ern and Vohralík, SIAM Journal on Numerical Analysis (2015)](https://doi.org/10.1137/130950100)

For a Galerkin field the local divergence is $g_a=\kappa\nabla v\cdot\nabla\phi_a-f\phi_a$. On an interior patch its integral vanishes by the P1 discrete equations. For an inexact field, compute the residual of each interior hat. A deterministic vertex tree routes these residuals to the Dirichlet boundary. Each transfer adds and subtracts the same integral in two vertex entries of one triangle, giving a constant correction to $g_a$ there. The sum on each triangle remains $-f$ and every interior star now has zero integral. The correction vanishes for an exact discrete solution. This algebraic lifting proves compatibility; its spatial distribution can affect certificate quality.

The constructive lift first realizes the integral of each corrected $g_a$ by RT0 fluxes on radial edges. The bubble $-(f/3)\phi_a(x)(x-p_a)$ adds its remaining zero-mean affine part. Its normal trace vanishes on every edge and $\operatorname{div}[\phi_a(x)(x-p_a)]=3\phi_a-1$. Minimize over divergence-free corrections using curls of the central hat and the quadratic bubbles of the radial edges. A stream for a divergence-free RT1 difference is continuous P2 up to an additive constant. Zero normal traces set the stream constant along the connected chain of opposite edges. After fixing that constant, the center vertex and the spoke midpoints supply precisely the free coefficients. The reduced patch solve is SPD. The degree-three rational quadrature integrates the normal equations exactly. Summing the patches gives an equilibrated flux regardless of the quality of the parent RT0 flux. Uniform degree-robust efficiency for the inexact lifting has not been established. [Reconstructor](../cfac/rt1_reconstruction.py) · [Independent moment and stationarity tests](../tests/test_cfac_rt1_reconstruction.py)

## Independent proof check

Each oriented edge carries the zeroth and first integral moments of its normal flux along its parameter $0\le t\le1$. Two more moments per triangle give the integral of each physical vector component. The checker reconstructs the linear normal density from its endpoint values, reversing both the normal sign and the parameter for the opposite local orientation. On each triangle a contravariant Piola pullback recovers the eight RT1 polynomial coefficients from these moments. Shared moments determine the full linear normal trace. The checker imports no code from the RT1 reconstruction. It rechecks the retained problem and parent, validates the moment layout, and binds its result to a separately retained candidate digest.

The checker integrates the degree-four squared mismatch with exact reference monomial moments. It checks all three coefficients of the affine divergence, including the two modes that a cell-average balance cannot detect. For an exactly equilibrated P0 source the quadratic radial part vanishes in the summed flux. Defective candidates can have a nonzero quadratic part, so the fourth-degree integration is still necessary.

For an affine original load the correct residual is $f+\operatorname{div}q$. A linear RT1 divergence can correlate with the within-cell load oscillation. The checker integrates the square of the **combined** residual directly and applies the unit-square Dirichlet correction from the foundation. With $q=(x^2,xy)$ and $f=3x$, summing the two separate squared norms gives 11. The squared norm of the full residual is 12. At tolerance 27/20 the former would yield a false acceptance. The regression checks that the verifier rejects it. RT1 acceptance certifies the original energy error of the current P1 field. The checker does not issue an RT1 space-floor verdict. [Independent checker](../cfac/rt1_checker.py) · [Analytical and adversarial tests](../tests/test_cfac_rt1_checker.py)

## Controller and cost

Enable the experimental policy with the keyword argument **allow_rt1=True** or with the demonstration switch **--rt1**. A proven primal-space failure goes directly to the available mesh refinement. A proven RT0 pair failure can instead trigger one preparatory primal sweep when positive primal algebraic error is proven, followed by a fresh inspection. If pair futility persists, construction and independent RT1 inspection each consume an action. The new state records the parent P1 field and every flux moment. A failed inspection establishes no RT1 floor. When a valid trial misses the target, the policy can spend its remaining refinement. A budget ending after construction cannot authorize acceptance.

The controller comparison was chosen after the fixed-field study to show both outcomes. It starts each policy from zero on n=2 with the same problem, tolerance, and 24-action allowance.

| Tolerance | RT0 policy | Policy allowing RT1 |
|---|---|---|
| 3/20 | Accepted in 13 actions with one refinement | Accepted in 8 actions on the original mesh |
| 1/8 | Accepted in 17 actions with one refinement | RT1 missed, then accepted after one refinement in 19 actions |

These action counts include checks, probes and corrections. Individual actions have unequal cost. Both journals retain measured times, including the unsuccessful enrichment trial. In the fixed-field study the same coefficients were reconstructed once per case, then checked at every tolerance. Neither design estimates a general wall-time speedup. [Controller journals](../evidence/cfac/rt1-policy.json) · [Controller regressions](../tests/test_cfac_rt1_controller.py)

## Replay

Run from the repository root using Python 3.12 or 3.13. Each output path must be new. The first command binds the study to the archived bytes and recomputes its mathematical observations without calling the patch solver. The controller replay regenerates its decisions and checks the fields and proof values. Timings vary across runs.

~~~bash
python -m research.cfac.rt1_study --replay evidence/cfac/rt1-study.json --sha256 1f4c1bd0ab000fdd10942384107491317463004cf374ecd65ddd8031c84b01b3
python -m research.cfac.controller_demo --rt1 --replay evidence/cfac/rt1-policy.json
mkdir -p runs/cfac
python -m research.cfac.rt1_study --output runs/cfac/rt1-study.json
python -m research.cfac.controller_demo --rt1 --output runs/cfac/rt1-policy.json
python -m pytest -q tests/test_cfac_rt1*.py
python -m pytest -q --require-mfem --require-mpi
~~~

The next comparison should freeze a classical algebraic stopping rule, then charge both methods for every primal iteration, reconstruction, inspection and failed trial at matched verified accuracy. Use additional signed and heterogeneous load families for that comparison. The present study tests low-order mathematics and finite controller behavior on small CPU problems.

## Tested revision

The implementation commit is `38f28d7105aa3df104c8352d07e944a7ebe8f520`, based on foundation commit `7ee28b181b5871dbbd4d5ac231eb6ee717b581bb`. A clean Linux export passed all **613 tests** in WSL2 in 182.74 s, including the existing real serial and MPI driver tests. The JUnit report counted zero skips, failures and errors. The four new RT1 test files account for 27 tests. Those 27 passed separately on native Windows CPython 3.13 using `pytest --noconftest` to bypass the repository's Linux-only legacy bootstrap. The Windows-generated study was independently regenerated in Linux with identical fields, bounds and verdicts across all 132 targets. Windows also replayed the new Linux policy journal and the original foundation journal. [Integrated test output](../evidence/cfac/rt1-pytest.txt) · [GitHub source-build CI](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45636361762)
