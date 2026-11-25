# Primal smoothing before refined RT1 certification

The optional `cfac_retry` policy recovered the strict historical checkerboard in all six blocks. It also recovered a strict held-out edge-patch load in six of six blocks. Every earlier physical case accepted by the preceding CFAC schedule remained accepted. On the 18 nonzero cases the new schedule and classical stopping covered the same 16. Classical stopping spent less total wall time. A separate global RT1 reference delivered independently accepted fields for the remaining two cases using the same meshes and tolerances. These are small exact-rational CPU problems.

| Physical submissions | Previous CFAC reserve | CFAC with algebraic stopping before RT1 | Classical stopping |
|---|---:|---:|---:|
| Twelve historical cases, accepted / submitted | 66/72 | 72/72 | 72/72 |
| Six fresh cases, accepted / submitted | 18/36 | 24/36 | 24/36 |
| All physical cases, accepted / submitted | 84/108 | 96/108 | 96/108 |
| Submitted seconds across all physical attempts | 30.47 | 31.89 | 24.43 |

### When the new rule participated

The rule ran on four cases, once per block. Each stopping check and each extra sweep was charged, as were construction and independent inspection of the refined RT1 candidate. In all 24 trials the algebraic test passed before reconstruction. Twelve of those fields achieved the requested energy accuracy. The 16 other cases, including both controls, followed identical sequences of actions, field digests and next decisions with both CFAC schedules.

| Case and tolerance | Extra sweeps / stopping checks per run | Previous CFAC bound | New RT1 bound | Previous / new actions | New outcome |
|---|---:|---:|---:|---:|---|
| Historical checkerboard, 1/12 | 7 / 8 | 0.0957184 | 0.0803500 | 16 / 31 | Accepted |
| New split triangles, 1/14 | 7 / 8 | 0.0861803 | 0.0718395 | 16 / 31 | Checked miss |
| New edge patch, 1/12 | 3 / 4 | 0.0979743 | 0.0745770 | 16 / 23 | Accepted |
| New elbow, 1/12 | 9 / 10 | 0.1378959 | 0.0853808 | 16 / 35 | Checked miss |

For the checkerboard, the eight algebraic checks averaged 38.8 ms total per submission, the seven extra sweeps 7.1 ms, construction 132.3 ms and the final inspection 17.2 ms. Classical stopping accepted the checkerboard on action 26 and the edge patch on action 22. All three policies reached a checked miss on the strict split and elbow cases. The reported bound is for the actual field and flux submitted. A single failed certificate does not establish that the tolerance is impossible on the mesh.

### Paired delivery times

The following table gives per-case median submission times in milliseconds. A dagger denotes zero acceptances in six attempts and measures elapsed time until the unsuccessful decision. Every other time cell represents six accepted fields. Each reported ratio is the median of six within-block ratios for jointly accepted submissions. A dash denotes zero eligible pairs. [Per-block ratios, costs, action counts and coverage](../evidence/cfac/futile-summary.json) · [Raw receipts](../evidence/cfac/futile-study.json.gz)

| Load and energy tolerance | Previous CFAC | New CFAC | Classical | New / previous | New / classical | Previous / classical |
|---|---:|---:|---:|---:|---:|---:|
| Unit, 3/20 | 82.1 | 81.0 | 76.7 | 0.976 | 1.090 | 1.098 |
| Unit, 1/8 | 126.3 | 127.8 | 151.1 | 1.009 | 0.847 | 0.836 |
| Signed stripes, 1/6 | 215.1 | 219.1 | 310.4 | 1.011 | 0.687 | 0.679 |
| Signed stripes, 1/9 | 852.9 | 838.2 | 309.5 | 0.995 | 2.713 | 2.749 |
| Localized, 1/5 | 111.4 | 113.8 | 99.5 | 1.023 | 1.136 | 1.112 |
| Localized, 3/20 | 446.6 | 442.8 | 327.5 | 0.992 | 1.362 | 1.385 |
| Checkerboard, 2/15 | 81.2 | 77.2 | 97.0 | 0.968 | 0.807 | 0.831 |
| Checkerboard, 1/12 | 351.1† | 403.4 | 296.7 | — | 1.384 | — |
| Diagonal, 1/5 | 102.8 | 102.8 | 99.2 | 0.974 | 1.030 | 1.040 |
| Diagonal, 2/15 | 552.3 | 565.9 | 323.4 | 1.028 | 1.750 | 1.710 |
| Dipole, 1/8 | 88.1 | 84.4 | 99.4 | 0.938 | 0.847 | 0.865 |
| Dipole, 1/12 | 401.6 | 400.2 | 280.1 | 1.020 | 1.442 | 1.433 |
| **New** split triangles, 1/8 | 87.4 | 88.5 | 99.2 | 1.031 | 0.916 | 0.882 |
| **New** split triangles, 1/14 | 347.2† | 411.6† | 299.8† | — | — | — |
| **New** edge patch, 1/8 | 260.2 | 263.7 | 277.1 | 1.013 | 0.953 | 0.941 |
| **New** edge patch, 1/12 | 343.2† | 371.6 | 280.1 | — | 1.353 | — |
| **New** elbow, 3/20 | 242.8 | 250.0 | 304.4 | 0.997 | 0.788 | 0.797 |
| **New** elbow, 1/12 | 344.5† | 420.4† | 309.1† | — | — | — |
| Zero-load control, 1/8 | 61.6 | 65.5 | 61.0 | 1.034 | 1.059 | 1.017 |
| One-action control, 1/8 | 59.6† | 58.6† | 58.4† | — | — | — |

All 360 submissions are accounted for. The 294 successes include 18 accepted zero-load controls. Of the 66 unsuccessful submissions, 48 ended with no eligible action and 18 exhausted the one-action control budget. No worker errored or timed out. The unsuccessful submissions consumed 18.25 s of the total 89.01 s. Worker peak RSS ranged from 21.31 to 22.52 MiB immediately before packet encoding and from 21.48 to 37.81 MiB when journal export is included. Six repetitions measure variation in execution time on one CPU. The held-out coverage comparison uses three load families and two tolerances per family.

## What changed

The strict checkerboard had exposed a limitation of the previous schedule. Its refined P1/RT0 space could not attain the requested certificate, and the first refined RT1 trial on its inexact P1 field missed. The new optional policy `cfac_retry` uses this specific decision to test whether more primal work is warranted before constructing that refined RT1 flux. Its name refers to the refined-mesh attempt following the coarse trial. The controller constructs at most one RT1 candidate on that path on the refined mesh.

The stopping check compares a certified squared upper bound on the algebraic error with one quarter of a squared lower bound on the discretization error. Until the test passes, the policy alternates a primal sweep with a fresh stopping check, protecting the last two actions for reconstruction and independent inspection. If the stopping test still fails near the budget limit, the verifier can accept a field whose actual upper bound meets the tolerance. A stopping decision or a proof about an approximation space cannot accept a field. The ordinary coarse refinement logic and the existing sufficient-pair reserve operate on their original paths. [Frozen decision rules](CFAC_FUTILE_PROTOCOL.md) · [Controller](../cfac/controller.py) · [Stopping bounds](../cfac/stopping.py)

## How much smoothing did the historical field need?

An offline diagnostic starts from the exact retained P1 parent of the previous policy's last RT1 trial on the strict checkerboard. It reconstructs and independently checks a new RT1 flux after each additional Gauss–Seidel sweep until the algebraic test passes. The first reconstructed candidate reproduces the retained digest. The energy bounds below are rounded for display. Acceptance decisions use exact squared bounds.

| Extra primal sweeps | RT1 error upper bound | $\sqrt{A^+/D^-}$ | Independently accepted at 1/12 |
|---:|---:|---:|---|
| 0 | 0.0957184 | 2.5505 | No |
| 1 | 0.0896842 | 2.1366 | No |
| 2 | 0.0851432 | 1.7376 | No |
| 3 | 0.0830239 | 1.3225 | Yes |
| 4 | 0.0817130 | 0.9961 | Yes |
| 5 | 0.0809672 | 0.7499 | Yes |
| 6 | 0.0805623 | 0.5650 | Yes |
| 7 | 0.0803500 | 0.4256 | Yes |

On this field the algebraic criterion crossed 0.5 four sweeps after an RT1 field could first be certified. The live policy sees only the candidate it chooses to reconstruct. Trying earlier candidates would incur their construction and inspection costs. The offline diagnostic does not enter any timed result. The gated trial reports its net change in gap relative to the recorded pre-smoothing RT0 state, so this change includes the sweeps and the richer flux. [Exact diagnostic output](../evidence/cfac/checkerboard-sweeps.json) · [Diagnostic code](../research/cfac/checkerboard_sweeps.py)

## Verified fields for the two remaining misses

After the frozen comparison, two offline checks examined the same refined meshes and original tolerances. Starting from the submitted retry fields, up to eight additional primal sweeps still produced a checked miss using the same patch flux reconstruction. On each mesh we then computed the exact **discrete** P1 Galerkin solution. Its free-node residual vanished. A conforming P2 residual witness gave a lower bound on its energy error. Checking the patch RT1 flux on that exact P1 field gave the upper bound in the middle column below. [Patch trials and independent witnesses](../evidence/cfac/futile-limits.json)

| Case | P2 witness lower bound | Patch RT1 upper bound on exact P1 field | Globally optimized RT1 upper bound on same P1 field | Tolerance |
|---|---:|---:|---:|---:|
| Split triangles, strict | 0.06244194 | 0.07149216 | **0.06612099** | 0.07142857 |
| Elbow, strict | 0.08033017 | 0.08547337 | **0.08169325** | 0.08333333 |

In both cases an independently checked field meets the original target on the original refined mesh. Holding the Galerkin P1 field fixed and changing the flux reconstruction sufficed. This supplies checked solutions for the two cases missed by all three timed policies. They are archived separately from the 360 submission outcomes. The reference calculation has access to exact discrete optimizers and supplies neither timing data nor decisions to the controller. [Materialized fields and proof results](../evidence/cfac/rt1-pair-reference.json) · [Offline reference and stationarity checks](../research/cfac/rt1_pair_reference.py)

For these constant-diffusion, P0-load problems, feasible corrections to the flux are divergence-free RT1 fields. They are cellwise affine because the linear part of their divergence fixes the quadratic radial term. On the simply connected unit square, their shared normal traces allow them to be represented by rotated gradients of continuous P2 stream functions, with one constant null mode. No normal flux is prescribed on the Dirichlet boundary. The reference fixes a single stream value and minimizes the flux energy in the other coordinates. The ordinary RT1 checker validates the archived moments and computes the squared gap. Separate exact residual checks establish Galerkin stationarity of the P1 field and vanishing residual in every stream direction. An equilibrium-preserving flux perturbation is orthogonal to gradients of conforming zero-trace P1 fields by integration by parts. The two stationarity conditions thus characterize the minimum of the convex gap in this discrete pair.

On each refined mesh the checks cover 25 free P1 coefficients and 168 independent stream coefficients. The certified squared minima are exactly 596345/136401408 for split triangles and 118724810261/17789718858912 for the elbow. The corresponding squared targets are 1/196 and 1/144. Archive replay validates equilibrium, stationarity and the original tolerance from the stored fields without solving again.

## Reproduction

The design was frozen at `5d7d920ace7e6497bf8792e5cb279db108838d62` and has SHA-256 `dc9dcf98a5bff71c82e0c96d4d5f0a8f3e5032010458a08b745f35aaab7352c1`. The measured implementation was `d54d6e80ff2b87b40b1fc8156a84492a12b569b6` in a clean Linux-filesystem checkout. Measurements ran on 2026-11-25 under WSL2 Ubuntu, Linux 6.6.87.2 and CPython 3.12.3 on logical CPU 2 of an Intel Core Ultra 7 155H.

From a revision containing the archive, replay the 360 timed submissions and regenerate the paired summary and the historical checkerboard diagnostic. Linux CPython 3.12.3 and Windows CPython 3.13.2 returned byte-identical summary and diagnostic JSON. Raw timing evidence includes the 44 distinct deterministic journals and per-submission action costs. The 28 problem/policy combinations shared with the reserve experiment reproduced their complete original deterministic journal digests.

```sh
out=$(mktemp -d)
digest=7e4145dec3fc8dfaac14846e7ae0179d873caf7eb31202d1bdd9ccdf533222a7
python -m research.cfac.futile_study --replay evidence/cfac/futile-study.json.gz \
  --sha256 "$digest" --summary "$out/summary.json"
cmp "$out/summary.json" evidence/cfac/futile-summary.json
python -m research.cfac.checkerboard_sweeps > "$out/sweeps.json"
cmp "$out/sweeps.json" evidence/cfac/checkerboard-sweeps.json
```

Verify the two supplemental fields directly. To regenerate the global reference and the per-sweep patch/witness data, use the same fresh output directory. Both operating systems reproduced the full global reference JSON, including the field coefficients and flux moments, byte for byte. The reference file's SHA-256 is `f15f77316f4f1d6fab1929dabea452d8ff61a9f25f66adc432a5ed03cce34bdc`.

```sh
python -m research.cfac.rt1_pair_reference --verify evidence/cfac/rt1-pair-reference.json
python -m research.cfac.rt1_pair_reference --output "$out/global-reference.json"
cmp "$out/global-reference.json" evidence/cfac/rt1-pair-reference.json
python -m research.cfac.futile_limits > "$out/patch-limits.json"
cmp "$out/patch-limits.json" evidence/cfac/futile-limits.json
```

To measure afresh on Linux, use an unused destination, an available CPU and the measured source revision. The runner requires a matching clean checkout and binds each job to the frozen inputs. A successful timed packet returns its problem, P1 field, flux and field certificate. The timer includes process launch, all numerical work, verification and packet delivery. The worker then exports its complete journal for offline audit.

```sh
rev=d54d6e80ff2b87b40b1fc8156a84492a12b569b6
out=$(mktemp -d)
git worktree add --detach "$out/source" "$rev"
cd "$out/source"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
env LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c "$cpu" python -m research.cfac.futile_study \
  --output "$out/futile-study.json.gz" --revision "$rev"
```

## Regression evidence

On the measured revision the full local suite passed 662 tests in 191.67 s. After adding the two offline field diagnostics, the final code revision `2b1f12b8f416372892d7cb82f008958f707b0f42` passed **668 tests in 187.18 s**, with zero failures, errors or skips. Both runs required serial MFEM and MPI. The final command was `python -m pytest -q --require-mfem --require-mpi --junitxml=/root/mfem-futile-m8ug0qbf/final-suite.xml`. The 19 new tests relative to the preceding stage cover stopping and budget boundaries, actual RT1 acceptance, state binding, study accounting, residual witnesses, global stationarity, and replay of the two archived solutions. The original 392-trial and 252-trial receipts also passed full replay. [Measured-revision test output](../evidence/cfac/futile-pytest.txt) · [Final test output](../evidence/cfac/futile-final-pytest.txt) · [GitHub CI on timed code](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45649625383) · [GitHub CI including offline reference](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45650463583)
