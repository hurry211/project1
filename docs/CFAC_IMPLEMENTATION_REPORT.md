# CFAC foundation implementation report

The research branch adds an independent numerical acceptance contract for two-dimensional Poisson energy error and a proof-based decision about continuing on a fixed P1/RT0 pair. The runnable demonstration has no exact solution input. Its decisions and costs can be reproduced from the frozen problems. [Experiment results](CFAC_EXPERIMENTAL_EVIDENCE.md) · [Proofs and scope](CFAC_MATHEMATICAL_SPEC.md)

## Repository and inputs

The supplied research baseline was `ceb485cb9836c73ce846f62d49ca9591fcb83bfc`. We started from the later public main tip `fd2cadecc8d09f7c8128924aff23307fb50ab382`. The difference between these trees spans 180 files, 60,017 insertions and 267 deletions, including the E3/E4 numerical studies, MPI results and adaptive experiments. The review describes the earlier baseline. The dedicated branch `research/cfac-foundation` retains the later work. Its tested implementation commit is `7cb788f4b3a9ff827f13ecc91ef423eb0109dfc4`. The subsequent report commit adds documentation and test receipts.

We read the supplied independent CFAC review, its preceding certificate-driven brief and the digest-pinned numerical script. The sources of authority for each research result are the mathematical assumptions, code, retained input and replay data. The original MMS verification workflow, native solvers and all earlier acceptance tests run on this branch. The CFAC solver, checker and controller use an independent namespace and can run together without loading MFEM.

| Input document | SHA-256 of the reviewed file |
|---|---|
| `Certify_or_Refine_Independent_Research_Review.docx` | `62bdc10491101317df03fc4900d1c646d772815bf9e0fc78835e5098c8016043` |
| `Certificate_Driven_Scientific_Computing_Research_Review_EN.docx` | `ab39b08ef35f80fa295d1b09a3c3877acd2d7c8fccc5d3c51da518019b8cbd30` |

## What is established

| Category | Result and evidence |
|---|---|
| Established mathematical methods | Conforming P1 Galerkin projection, RT0 normal moments, the equilibrated hypercircle bound, Cauchy–Schwarz residual witnesses, stream-function minimization and orthogonal projections underlie this work. |
| Proven identities under the stated assumptions | The four-term error split, state-specific two-sided floor interval, witness-based primal futility test and projected-load cross term follow from the variational identities in the mathematical specification. A mesh edge path bound applies to the same pinned coordinates as the reduced dual system. |
| Numerically tested identities | The supplied sine experiment reproduces five mesh resolutions. Direct and energy-update evaluations of the optimized gap agree at the reported precision. The experiment uses floating quadrature and sparse solves. |
| Fully certified computations | Exact rational RT0 equilibrium and polynomial integration establish individual field bounds. An independently assembled discrete reference agrees exactly on the constant-load algebraic energies and optimal pair floor. A bounded Fourier series encloses each continuum discretization term. Two controller runs ended with independent acceptance. |
| New research hypotheses | A floor interval may help allocate work between solving, dual correction, enrichment and refinement. Its value relative to robust flux reconstruction and classical stopping at matched accuracy remains an experimental question. |
| Unverified assumptions and remaining work | Performance, broad usefulness of the policy, scalability of rational elimination and efficiency of RT1/RTp enrichment require further evidence. General non-polynomial original loads need validated data enclosures. [Comparison plan](CFAC_OPEN_PROBLEMS.md) |
| Failed experiments and negative controls | A controlled inexact iterate breaks the supplied local-patch compatibility. An altered flux makes the uncorrected gap cross the acceptance threshold in the wrong direction. An exact P2 witness produced outside P1 never changes the primal state. Budget exhaustion and an exhausted action menu were observed without acceptance. |

Native Windows repository-wide pytest collection encounters the inherited POSIX `resource` dependency. The CFAC-only suite and the inherited regression ran on WSL2 Ubuntu. The rational foundation also reproduced byte for byte on native Windows and the six controller journals replayed there. [Windows replay receipt](../evidence/cfac/windows-replay.json)

## Implementation and decisions

- `Problem` freezes the rational geometry, P0 cell order, original load when supplied, diffusion, tolerance and total action allowance. Input validation checks the caller-retained digest. The controller retains a fresh child digest before generating a refined flux. Its one authorized uniform refinement prolongs P1 values and is enabled for constant original P0 data.
- The independent checker validates the domain, reconstructs the oriented RT0 field, checks the primal trace, integrates the gap and residual and issues an upper bound for the original problem. Exact equilibrium is a separate flag relative to the projected load. An affine original supports an exactly verified data-oscillation correction. A projected or defective state cannot receive an original-PDE floor proof with unresolved residual.
- The primal solver uses Gauss–Seidel sweeps. An exact dual spanning-tree lifting needs only the cell loads. Local curl updates and a three-mode coarse stream correction preserve equilibrium. The evaluation harness owns direct discrete optimizers and the constant-load continuous reference. Production decisions never load either.
- The floor module assembles its own residual systems. It can use a Jacobi line-search and an anchored graph bound, or exact Schur elimination for tiny systems. The controller chooses an inspection before acting on a new state. Acceptance comes from that inspection. Its heuristic compares algebraic lower bounds and tries a coarse correction before local sweeps. The measured times are archived and have no role in the mathematical verdict.
- RT1 and RTp are extension protocols with oriented edge and interior moment layouts. Their documentation identifies the unisolvence, equilibrium, reconstruction and integration proofs a future checker must establish. The current controller supports P1/RT0.

## Exact changed-file inventory

The file names within each directory are listed individually. All numerical evidence is new to this branch.

| Directory | Files |
|---|---|
| Repository root | `.gitattributes`, `README.md` |
| `cfac/` | `__init__.py`, `spec.py`, `checker.py`, `prototype.py`, `evaluation.py`, `floor.py`, `controller.py`, `enrichment.py` |
| `research/cfac/` | `microcheck.py`, `reproduce_microcheck.py`, `probe_inexact_patch.py`, `foundation_evidence.py`, `controller_demo.py` |
| `tests/` | `test_cfac_checker.py`, `test_cfac_floor.py`, `test_cfac_prototype.py`, `test_cfac_enrichment.py`, `test_cfac_controller.py`, `test_cfac_microcheck.py` |
| `evidence/cfac/` | `foundation.json`, `inexact_patch.json`, `controller.json`, `cfac-pytest.txt`, `integrated-pytest.txt`, `wsl-interruption.txt`, `windows-replay.json` |
| `evidence/cfac/microcheck/` | `stdout.jsonl`, `stderr.txt`, `environment.json` |
| `evidence/cfac/microcheck128/` | `stdout.jsonl`, `stderr.txt`, `environment.json` |
| `docs/` | `CFAC_MATHEMATICAL_SPEC.md`, `CFAC_IMPLEMENTATION_REPORT.md`, `CFAC_EXPERIMENTAL_EVIDENCE.md`, `CFAC_OPEN_PROBLEMS.md` |

## Testing

GitHub Actions built real serial and MPI drivers and passed the integrated suite of **586 tests with zero skips and zero deselections in 167.40 seconds**. The WSL2 integrated rerun encountered a kernel watchdog lockup during MPI launch. The hosted workflow also completed its paired MPI field comparison and replay. [Hosted CI](https://github.com/rodriguezrobertbfrkx6857-sudo/project/actions/runs/45629095372) · [Local interruption](../evidence/cfac/wsl-interruption.txt)

The 69 CFAC tests comprise 21 checker, 16 floor, 13 prototype/evaluation, 4 enrichment, 11 controller and 4 reproduction/evidence cases. They include signed and discontinuous loads, malformed geometry, shared normals, nonzero equilibrium defects, stale state, smooth and oscillatory dual errors, P2 probes, exact equality and sub-float threshold changes, nonfinite values, complete trace replay and exhausted budgets. The checker also exercises seeded rational fields inside a property test. The 517 inherited tests passed on the starting snapshot before integration. [CFAC test output](../evidence/cfac/cfac-pytest.txt) · [Full integration output](../evidence/cfac/integrated-pytest.txt)

All research and test commands are in the replay section. The implementation revision and its parent can be verified independently with the following commands. [Commands and expected results](CFAC_EXPERIMENTAL_EVIDENCE.md#commands)

```bash
git log --oneline ceb485cb9836c73ce846f62d49ca9591fcb83bfc..fd2cadecc8d09f7c8128924aff23307fb50ab382
git diff --shortstat ceb485cb9836c73ce846f62d49ca9591fcb83bfc fd2cadecc8d09f7c8128924aff23307fb50ab382
git rev-parse 7cb788f4b3a9ff827f13ecc91ef423eb0109dfc4^
git log --format='%H %s' fd2cadecc8d09f7c8128924aff23307fb50ab382..research/cfac-foundation
```
