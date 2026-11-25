# Resolution of the November 2026 project review

The original repository's main branch contains the E3 research history through `07dca40` and the E4 repairs. The rows below correspond to the sixteen findings in the external review. Times refer to the study named in each row and include only the phases identified there. Full raw archives and their SHA-256 values accompany the release. [Research baseline](https://github.com/rodriguezrobertbfrkx6857-sudo/project/commit/07dca409cb71d096c732f3a9825e88930c6ca79d) · [E4 evidence release](https://github.com/rodriguezrobertbfrkx6857-sudo/project/releases/tag/e4-verified-studies-20261125)

| # | Finding | Change and measured outcome |
|---:|---|---|
| 1 | MPI cost exceeded serial on small studies | A single-host, five-block paired comparison covered three meshes up to 524,288 triangles. Median verified time was 15.933 s with one MPI rank and 11.064 s with four. The ratio was 1.44× with the same hypre solver. The paired interval for the difference was −10.725 to −1.956 s. The selector chose serial on a separate smaller calibration. [MPI measurements](../evidence/e4/mpi-report.json) |
| 2 | Independent reference verification dominated runtime | Error integration and assembly now share a quadrature traversal. On the same three-level archive ending at 1,050,625 DOFs, four scored pairs gave median full verifier times of 64.993 and 53.178 s. Both versions accepted every level with identical L2 and H1 norms. Median worker RSS increased from 1,081.5 to 1,149.4 MiB. [Raw pairs, source and input digests](../evidence/e4/streaming-revision.json) |
| 3 | Preconditioner choice had little evidence | Ordinary search can compare GS, Jacobi and unpreconditioned CG in supported configurations. A 30-study matched search accepted all 15 Q2 runs, rejected all 15 Q1 runs on accuracy and replayed all 30 verdicts. [Search measurements](../evidence/e4/strategy-search.json) |
| 4 | Live template selection was expensive and could vary | Three repeated campaigns certified 42 numerical submissions and 15 review decisions per policy. Bounded reuse used nine model calls and retained 39/42 preferred templates, equal to always-live with 42 calls. Rules were fastest. This capture observed stable choices for all three policies. [Timing and replay](PLANNING_BENCHMARK.md) |
| 5 | Limited PDE and dimensional coverage | Added heat, wave and vector isotropic elasticity in 2D and 3D with tensor Q1/Q2 elements. All 12 studies converged on three meshes each and replayed. The largest transient study had 274,625 scalar DOFs. The largest elasticity study had 107,811 vector DOFs. Inputs select one of the reviewed manufactured operators. [Per-level errors and costs](../evidence/e4/field-families.json) |
| 6 | Fragmented research and stale clone instructions | The original main includes the upstream E3 commits. The README clones the original repository. The historical E3 instructions select the exact research commit before building. [README](../README.md) · [E3 reproduction](E3_PERFORMANCE_ADAPTIVE.md) |
| 7 | Direct reference solves exhausted memory and large complex cases were unmeasured | The residual-checked iterative reference passed at 1,640,961 P1 DOFs with 1,993.4 MiB verifier peak RSS under the 3 GiB address-space cap. A separate 37,249-DOF Q3 variable-coefficient mixed-boundary study used a distorted quadrilateral mesh. The 3D tensor study has its own independent operator and memory measurements. Direct LU retains its historical limit. [Large P1](../evidence/e4/p1-1280.json) · [Irregular Q3](../evidence/e4/irregular-q3.json) |
| 8 | Partial assembly could slow the entire solve | Search ranks independently qualified runs by total elapsed time with matched accuracy. On the measured Q2 sine workload the selected partial-Jacobi median was 1.181 s, compared with 1.371 s for full Jacobi. On a separate Linux host, the larger 263,169-DOF Q2 Gaussian favored full Jacobi in all three paired trials. Median verified times were 28.632 s for full and 35.618 s for partial. [Search and assembly results](ADAPTIVE_REPAIRS.md) · [Clean hosted measurement](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45616524974) |
| 9 | Geometry-based reuse failed on irregular or variable material | Per-element weighted gradient products apply to varying diffusivity and distorted quads. Against full local contraction, measured median kernel improvements were 2.44× for variable-coefficient triangles and 4.17× for distorted quads. The largest cellwise relative difference across five cases was 1.90e−15. The verifier also checks the actual refined cells against the pinned Gmsh source. [Method and measurements](GEOMETRY_KERNELS.md) |
| 10 | Coarse sharp-Gaussian quadrature disagreed | Native and independent quadrature now resolve both registered Gaussian peaks. Coarse Q3 sharp-Gaussian verification met the original self-report criterion. The verifier checks the reported integration rule. [Numerical regressions](../tests/test_adaptive_repairs.py) |
| 11 | AMR exhausted ten steps above the target | The revised Q2 marking fraction reached L2 error 2.86546e−5 in ten solved meshes with 2,209 true DOFs. The goal remained 4e−5 and the cap remained 4,000 DOFs. [AMR measurements](ADAPTIVE_REPAIRS.md) |
| 12 | Policies failed to qualify difficult tasks | The revised six policies each accepted 21/21 trials across seven tasks and three repeats. Five tasks were known from the prior panel, two were transfer cases. This is empirical coverage on one machine. [Trial-level comparison](../evidence/e4/adaptive-comparison.json) |
| 13 | No measured contribution from a real adaptive LLM | A pinned Qwen2.5 model made 21 valid live nominations. All matched its deterministic first choice on the same shortlist. Inference added a paired median of 2.277 s. [Attribution audit](../evidence/e4/advisor-ablation.json) |
| 14 | MPI, optimized verification and GPU lacked joint evidence | Real MPI/hypre fields passed independent streaming and SA-AMG checks; the calibrated rank choice was executed and replayed. The Windows Intel Arc separately recomputed L2 and H1 on the four MPI partitions at all three sizes. It rejected an altered-field control at each size. MFEM assembly and solution in these experiments ran on CPU. [MPI study](MPI.md) · [FP64 GPU receipt](../evidence/e4/gpu-mpi-norms.json) |
| 15 | Failed-baseline speedups, native-only memory and discarded fields | The E3 ratios against a 0/15 control are identified as elapsed-cost ratios with no equal-quality speedup. E4 includes a successful fixed baseline, qualified paired comparisons, process-tree memory estimates, raw files, and authenticated replays of 290 numerical verdicts. The predictor's 95% paired interval against fixed includes parity. [Protocol and limitations](ADAPTIVE_REPAIRS.md) · [Replay result](../evidence/e4/adaptive-audit.json) |
| 16 | Calibration statistics and checkout commands differed | The E3 report now matches the original stored training set: 32 valid measurements out of 36, with log10 RMSEs of 0.2525, 0.1030 and 0.0887 for error, time and memory. A regression checks those numbers against the JSON. [Historical report](E3_PERFORMANCE_ADAPTIVE.md) · [Data checks](../tests/test_e4_evidence.py) |

## Reproduce the added numerical studies

Build the pinned serial MFEM library and the three drivers using the README instructions. The public release contains the native field exports and the manifest digests. For example, a new triple of meshes for three-dimensional elasticity can be computed and replayed as follows.

```bash
python -m sciagent pde-study --family elasticity --dimension 3 --order 2 --levels 4 8 16 --directory runs/elasticity > elasticity.json
digest=$(python -c 'import json; print(json.load(open("elasticity.json"))["manifest_sha256"])')
python -m sciagent verify-pde runs/elasticity --manifest-sha256 "$digest"
python -m benchmarks.field_families --directory runs/field-matrix
python -m benchmarks.scalar_scale irregular-q3 --directory runs/irregular-q3 --output irregular-q3.json
python -m benchmarks.scalar_scale p1-1280 --directory runs/large-p1 --output large-p1.json
```

Heat uses Crank–Nicolson, wave uses the implicit midpoint rule for displacement and velocity, and elasticity uses the isotropic form with both Lamé parameters equal to one. All fields use homogeneous boundary values on the unit square or cube. The transient initial field is the product of one sine per axis, with zero initial wave velocity. An independent tensor eigensolve checks the half-time and final snapshots, including wave velocity. Elasticity verification applies an independently integrated vector weak operator to the exported displacement. All three require boundary agreement, finite complete fields, a continuous L2 error limit and convergence under mesh refinement. [Native implementation](../cpp/field_family.cpp) · [Independent mathematics](../verifier/field_family.py)

The Windows GPU audit takes a verified four-rank receipt and its original rank exports. Extract `mpi-qualified.tar.gz`; use the subdirectory containing the three `n128`, `n256` and `n512` folders from the four-rank warm-up. Its location is identified by the receipt's `label`. From the checkout in Windows PowerShell, run the following with `$rankDirectory` set to that directory.

```powershell
py -3.13 -m venv .venv-opencl
& ./.venv-opencl/Scripts/python -m pip install --only-binary=:all: --require-hashes -r requirements-opencl-win-py313.lock
$pin = (Get-FileHash -Algorithm SHA256 evidence/e4/gpu-mpi-input.json).Hash.ToLower()
& ./.venv-opencl/Scripts/python -m benchmarks.gpu_mpi_norms --directory $rankDirectory --receipt evidence/e4/gpu-mpi-input.json --receipt-sha256 $pin --output gpu-replay.json
```

## Submission-to-result timing

A second experiment timed twenty fresh submissions with the same controller and native binary. The large H1 workload had median endpoint times of 19.117 s with the E3 verifier and 18.586 s with E4. Its paired difference interval was −0.992 to +1.564 s. The smaller H2 workload also had an interval containing zero. Every run passed the same checks with the same numerical fingerprint within its workload. The intervals include zero for both full workflows. Row 2 measures only the verifier at a larger mesh size. [Fresh submissions, intervals and raw archive](VERIFIED_PERFORMANCE.md)

## Regression coverage and archive integrity

The inherited E3 checkout passed 463 tests with both serial and MPI binaries required. Both the local E4 suite and a separate clean GitHub-hosted serial/MPI build passed all 517 tests with zero skips. [Test counts and run identifiers](../evidence/e4/qualification.json) · [JUnit XML](../evidence/e4/pytest.xml.gz) · [Clean MPI workflow](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45614554981)

The E4 suite adds regressions for the new fields, solver choices, archive reuse and grid provenance. The canonical suite reproduced the terminal states, retry counts, frozen problem hashes and expected failure codes for all 31 original tasks. Across 68 accepted error values the largest relative difference from the original reports was 1.18e−13. A separate scikit-fem oracle verified all 42 Stage 2 combinations. The 284 scored challenge trials produced zero false accepts and zero false rejects on this panel. Running with profiling produced the same 284 verdicts. [Canonical results](../evidence/e4/canonical-regression.json) · [Stage 2 replay](../evidence/e4/stage2-replay.json) · [Verifier challenge](../evidence/e4/verifier-challenge.json.gz)

To download the complete evidence set, run the following in an empty directory. Digest checks authenticate the published bytes. Each study's numerical replay command and its external manifest digest establish which field checks passed. [Asset sizes, member counts and hashes](../evidence/e4/release-assets.json)

```bash
gh release download e4-verified-studies-20261125 --repo rodriguezrobertbfrkx6857-sudo/project
sha256sum -c SHA256SUMS
```

## Remaining experiments

The MPI interval describes one physical host with up to four ranks. The extra Gaussian assembly comparison used an independent Intel Xeon Linux runner. Those serial measurements add a second hardware setting for the assembly experiment. A two-node run would measure network communication and global reductions. An MFEM device build with a supported GPU backend would allow direct measurement of GPU assembly, solution and host/device transfers. The current GPU result is an independent field audit. The new PDE registry has fixed manufactured coefficients, boundary data and domains. User-defined weak forms, complex 3D geometries and coupled systems require additional independent references and convergence tests. The next performance experiment is a second-host replay of the same serial/MPI design with communication costs recorded per rank.
