# CFAC experiments and replay

The sufficient-pair reserve study archives 252 submissions. It recovers the previously diagnosed budget failure, and retains the held-out checkerboard case where both CFAC variants stop at action 16 with work still available. [Measurements, raw journals and commands](CFAC_RESERVE_RESULTS.md)

The matched-accuracy stopping study archives 392 timed submissions, including 98 unsuccessful outcomes. Its report gives the digest and commands for replaying the checked fields and regenerating the case-level results. [Matched-accuracy evidence and reproduction](CFAC_STOPPING_RESULTS.md)

## Supplied sine experiment

The reviewed script was inspected and copied byte for byte. Its SHA-256 is `c11345887563330d06a90e9cb28ebbfd713ad0142c50c6277266fe0d7cf11eee`. WSL2 Ubuntu ran CPython 3.12.3, NumPy 2.5.5 and SciPy 1.18.1 with one BLAS thread. The script rounds values to six significant digits when writing stdout. The first run used the four requested resolutions. A separate optional run at 128 used an address-space limit of 2 GiB and a 90 s time limit.

| n | Interior P1 DOFs | Energy error | Local gap/error | Optimized gap/error | P2 witness/error | Relative local equilibrium defect | Sweeps to remove 90% of smooth dual slack |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 49 | 0.431798 | 1.15614 | 1.14599 | 0.994655 | 2.03e-15 | 9 |
| 16 | 225 | 0.217536 | 1.17192 | 1.15251 | 0.998642 | 8.51e-15 | 31 |
| 32 | 961 | 0.108975 | 1.19251 | 1.15415 | 0.999659 | 4.69e-14 | 121 |
| 64 | 3,969 | 0.0545137 | 1.23008 | 1.15456 | 0.999915 | 2.42e-13 | 479 |
| 128 | 16,129 | 0.0272601 | 1.30133 | 1.15467 | 0.999979 | 1.39e-12 | 1,912 |

The first four energy errors and smooth sweep counts reproduce Table 5.1 of the independent review at its displayed precision. Their largest reported relative projected-load identity discrepancy here is 1.45e-14. Floating defects vary with arithmetic and library build. The two output SHA-256 values cover different resolution sets. [Four-resolution stdout](../evidence/cfac/microcheck/stdout.jsonl) · [Environment](../evidence/cfac/microcheck/environment.json) · [Optional stdout](../evidence/cfac/microcheck128/stdout.jsonl) · [Optional environment](../evidence/cfac/microcheck128/environment.json)

The script assembles the sine load by quadrature, uses a direct floating Galerkin solve, equilibrates to numerical cell means and evaluates a quadratic bubble witness. The Duffy rule integrates polynomials through degree 14. The sine source and its products with the manufactured solution are non-polynomial. Its witness shift adds a P2 function to the field and evaluates the energy in the larger space. The observed shift identity applies to that enriched field. These measurements test identities and smooth-mode behavior. Neither this quadrature nor the observed equilibrium residual encloses the original PDE error.

## A failed compatibility assumption

At n=8 we perturbed the central P1 coefficient by exactly 1/64 immediately after the script's first sparse solve. Its other assembly and patch steps ran unchanged. The infinity norm of the primal residual rose from 1.17e-15 to 0.0625. The relative equilibrium defect rose from 2.03e-15 to 0.0519865 and the relative identity discrepancy reached 0.000373273. The new prototype routes the load through a dual spanning tree before inspecting the primal iterate. Its exact cell balances apply to arbitrary conforming P1 iterates. [Controlled perturbation](../evidence/cfac/inexact_patch.json)

## Exact floor and original-load controls

For an n=3 mesh with constant load 1, diffusion 2 and tolerance 1/10, we applied one primal and one local dual sweep. The following are squared energy quantities. [Exact input fields and output](../evidence/cfac/foundation.json)

| Quantity | Result |
|---|---:|
| Current gap | 123391/10616832 ≈ 0.01162221 |
| Primal algebraic error | 79/41472 ≈ 0.00190490 |
| Dual algebraic error | 130811/53084160 ≈ 0.00246422 |
| Exact optimal pair floor | 47/6480 ≈ 0.00725309 |
| Inexpensive floor interval | [0, 0.00829735] rounded outward for display |
| Primal discretization term, continuous series cutoff 31 | [0.0052260380, 0.0052275278] |
| Dual discretization term, same cutoff | [0.0020255586, 0.0020270485] |

The current gap equals the two algebraic terms plus the optimal floor exactly. Both floor intervals prove the pair theoretically sufficient for the squared target 0.01, while the current certificate exceeds it. The continuous discretization intervals use the same unknown series energy with opposite signs. Their exact sum follows by cancellation, without assigning either endpoint as ground truth.

For the original source $f=x$, the checked cell-mean projection has squared oscillation 1/162. Its original-PDE squared upper bound is 0.01299397. The independently frozen projected PDE gives 0.01018010 for the same field and flux and permits floor attribution. Perturbing one projected flux moment by 1/100 gives a raw gap of 0.00987784, below the frozen threshold 0.01. The independently computed squared divergence defect is 9/5000. The corrected squared upper bound is 0.01133339 and the field is not accepted. [Original, projected and perturbed controls](../evidence/cfac/foundation.json)

## Controller demonstration

Each run starts on the same n=2 triangulation with $a=f=1$. A setup, certificate inspection, witness inspection, solver sweep, coarse correction or uniform refinement costs one action. A witness inspection performs a tight floor assessment and an independent P2 residual test. The state, exact gains at successive same-pair inspections, reason, next decision and measured wall time are in the journal. Replay checks all deterministic values, including every field and gap. [Complete decision journals](../evidence/cfac/controller.json)

| Tolerance | Budget | Actions spent | Terminal outcome |
|---:|---:|---:|---|
| 1/4 | 32 | 6 | Accepted on the initial mesh after coarse and local dual corrections |
| 1/8 | 32 | 17 | Accepted after a pair-futility proof and one uniform refinement |
| 1/9 | 20 | 20 | Budget exhausted after a primal correction awaiting inspection |
| 1/100 | 20 | 6 | Initial primal-space futility, then pair futility on the refined mesh. No eligible action remained |
| 1/8 | 1 | 1 | Budget exhausted after setup |
| 1/8 | 0 | 0 | Budget exhausted before setup |

The last accepted squared bounds in the first two cases are 533/12288 and 2665463963/206158430208. The time fields are measured costs of this specific run. No end-to-end speed comparison was conducted.

Windows x86-64 CPython 3.13.2 reproduced all six WSL2 journals after excluding measured times. Its regenerated `foundation.json` was byte-identical to the Linux record, with SHA-256 `cfbe267e87c771147c5730c2914b4c48096621144dab49ed88761337f6c8e262`. The commands were `py -3.13 -m research.cfac.foundation_evidence TEMP/foundation.json` in a fresh temporary directory and `py -3.13 -m research.cfac.controller_demo --replay evidence/cfac/controller.json`. [Cross-platform replay receipt](../evidence/cfac/windows-replay.json)

## Commands

Use a fresh output directory. The two generators and the perturbation probe refuse to overwrite existing files. The checked-in Linux dependency lock supplies the versions used by the supplied numerical script. The CFAC mathematical core uses the Python standard library.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-challenge.lock
mkdir -p runs/cfac
python research/cfac/reproduce_microcheck.py runs/cfac/microcheck
timeout 90s prlimit --as=2147483648 -- python research/cfac/reproduce_microcheck.py runs/cfac/microcheck128 --extended
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/cfac/probe_inexact_patch.py runs/cfac/inexact_patch.json
python -m research.cfac.foundation_evidence runs/cfac/foundation.json
cmp runs/cfac/foundation.json evidence/cfac/foundation.json
python -m research.cfac.controller_demo --output runs/cfac/controller.json
python -m research.cfac.controller_demo --replay evidence/cfac/controller.json
python -m pytest -q tests/test_cfac*.py
python -m pytest -q --require-mfem --require-mpi
```

The archived measurements were generated with the following commands from the repository root in WSL2. Here `python` stands for `/root/mfem-stage1/venv/bin/python`. The separate evidence directories preserve the stdout of the unmodified script exactly.

```bash
python research/cfac/reproduce_microcheck.py evidence/cfac/microcheck
timeout 90s prlimit --as=2147483648 -- python research/cfac/reproduce_microcheck.py evidence/cfac/microcheck128 --extended
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/cfac/probe_inexact_patch.py evidence/cfac/inexact_patch.json
python -m research.cfac.foundation_evidence evidence/cfac/foundation.json
python -m research.cfac.controller_demo --output evidence/cfac/controller.json
python -m research.cfac.controller_demo --replay evidence/cfac/controller.json
```

For a fresh MFEM installation, follow the repository's serial and MPI build instructions. The starting-snapshot regression used the existing release builds in WSL2. During the research-tree rerun the kernel watchdog reported a soft lockup while starting a real MPI rank. The recorded CFAC-only run completed separately. [MPI build](MPI_BUILD.md) · [WSL watchdog excerpt](../evidence/cfac/wsl-interruption.txt) · [CFAC pytest output](../evidence/cfac/cfac-pytest.txt)

The hosted Python job passed 462 tests with 124 deselected by its marker filter. The serial MFEM source-build job passed 114 with 472 deselected and completed the field and benchmark replay steps. [Python and serial MFEM jobs](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45629018647)

The hosted MPI workflow built both drivers from source at implementation commit `7cb788f4b3a9ff827f13ecc91ef423eb0109dfc4` and ran `python -m pytest -q --require-mfem --require-mpi`. All 586 tests passed in 167.40 s with zero skips and zero deselections. The subsequent serial/parallel field comparison, replay and artifact upload also passed. [Integrated pytest log excerpt](../evidence/cfac/integrated-pytest.txt) · [Complete GitHub Actions run](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45629095372)
