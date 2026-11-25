"""Run the Stage 1 verifier challenge on fresh or retained MFEM control archives."""

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from sciagent.controller import reverify
from sciagent.profiling import capture
from verifier import verifier

from .scoring import metrics
from .controls import ATTEMPT, ROOT, generate_controls, load_controls, sha
from .mutations import (ARCHIVE, CLEAN, NUMERICAL, numerical_mutation, read_json, renumber,
                        require, write_json)


def design(controls, seed):
    """Fixed coverage plus separate uniform IID draws with replacement."""
    names = list(controls)
    rng = random.Random(seed)
    trials = []
    for kind, families, count in (("numerical_fault", NUMERICAL, 64), ("numerical_clean", CLEAN, 32),
                                  ("archive_fault", ARCHIVE, 32), ("archive_clean", ("relocated",), 16)):
        for cohort, pairs in (("panel", [(c, f) for c in names for f in families]),
                              ("iid", [(rng.choice(names), rng.choice(families)) for _ in range(count)])):
            for c, f in pairs:
                ns = list(controls[c].levels)
                n = ns[0] if cohort == "panel" else rng.choice(ns)
                trials.append((cohort, kind, c, f, n, rng.getrandbits(32)))
    return trials


def result_checks(report, n):
    checks = list(report["study_checks"])
    for level in report["levels"]:
        if level["n"] == n:
            checks.extend(level["checks"])
    if report.get("accuracy_check"):
        checks.append(report["accuracy_check"])
    return {c["name"]: bool(c["passed"]) for c in checks}


def _verdict(check_profiling, verify, *args, **kwargs):
    baseline = verify(*args, **kwargs)
    if check_profiling:
        with capture():
            timed = verify(*args, **kwargs)
        # Each pair reads exactly the same directory, including corrupted inputs.
        require(json.dumps(timed) == json.dumps(baseline),
                "enabling the phase timers changed the full numerical or archive verdict")
    return baseline


def trial(row, controls, workspace, *, check_profiling=False):
    cohort, kind, case, family, n, seed = row
    control = controls[case]
    level_path = ATTEMPT / "levels" / f"n{n:04d}"
    rng = np.random.default_rng(seed)
    donors = [c for c in controls.values() if c is not control and n in c.levels]
    donor = donors[int(rng.integers(len(donors)))]
    source = donor.run if family == "substituted_archive" else control.run
    run = workspace / "run"
    # No links: mutations cannot touch the certified source or other trials.
    shutil.copytree(source, run)
    level = run / level_path
    clean = control.levels[n]
    if family == "original":
        witness = {"certified_n": n}
    elif family == "renumbered":
        witness = renumber(level, clean, rng)
    elif kind == "numerical_fault":
        witness = numerical_mutation(level, family, clean, control.validation, rng, donor.run / level_path)
    elif family == "substituted_archive":
        require(donor.digest != control.digest, "donor digest identical")
        witness = {"presented_manifest_sha256": donor.digest}
    elif family in ("relaxed_validation", "relaxed_retry_policy"):
        if family == "relaxed_validation":
            path = run / "frozen/validation.json"
            data = read_json(path)
            require(data["require_reference_solve"] is True, "control omitted reference solve")
            data["require_reference_solve"] = False
        else:
            path = run / "frozen/policy.json"
            data = read_json(path)
            data["max_attempts"] += 1
        old = sha(path)
        write_json(path, data)
        witness = {"original_sha256": old, "changed_sha256": sha(path)}
        require(old != witness["changed_sha256"], "frozen file unchanged")
    elif family == "relocated":
        witness = {"manifest_sha256": sha(run / "manifest.json")}
    else:
        raise ValueError(f"unknown challenge {family}")
    out = {"cohort": cohort, "kind": kind, "case": case, "family": family,
           "n": n, "draw_seed": seed, "witness": witness}
    try:
        if kind.startswith("archive"):
            report = _verdict(check_profiling, reverify, run / ATTEMPT, expected_digest=control.digest)
            checks = {"archive_integrity": report["integrity"]["passed"]}
        else:
            report = _verdict(check_profiling, verifier.verify_attempt, run / ATTEMPT, require_run_records=True)
            checks = result_checks(report, n)
        out.update(verdict=report["verdict"], codes=report["failure_codes"],
                   evaluated=sorted(checks), failed=sorted(k for k, passed in checks.items() if not passed))
    except Exception as exc:  # An unexpected verifier failure is separately scored and fails the command.
        out.update(verdict="ERROR", codes=[], evaluated=[], failed=[], error=f"{type(exc).__name__}: {exc}")
    return out


def checker_controls(controls, *, check_profiling=False):
    """Deliberately damage the production reference assembly on certified good inputs."""
    original = verifier.reference_solve
    records = []
    for mutation in ("negate_reference_volume_load", "scale_reference_stiffness"):
        def altered(mesh, basis, q, problem, coords, on_dir, u, f, m, **solve_options):
            if mutation == "negate_reference_volume_load":
                def load(x, y):
                    return -f(x, y)
            else:
                load = f
                q = replace(q, gphi=q.gphi * np.sqrt(1.25))
            return original(mesh, basis, q, problem, coords, on_dir, u, load, m, **solve_options)
        with patch.object(verifier, "reference_solve", altered):
            for case, control in controls.items():
                report = _verdict(check_profiling, verifier.verify_attempt, control.run / ATTEMPT,
                                  require_run_records=True)
                records.append({"case": case, "mutation": mutation, "verdict": report["verdict"],
                                "codes": report["failure_codes"],
                                "min_relative_reference_difference":
                                    min(lv["reference_rel_diff"] for lv in report["levels"])})
    return records


def run_campaign(directory, output, seed, *, check_profiling=False):
    controls = load_controls(directory)
    rows = []
    with TemporaryDirectory(prefix="poisson-challenge-") as base:
        for row in design(controls, seed):
            with TemporaryDirectory(dir=base) as scratch:
                rows.append(trial(row, controls, Path(scratch), check_profiling=check_profiling))
    checker = checker_controls(controls, check_profiling=check_profiling)
    scores = metrics(rows)
    report = {"schema_version": 1, "seed": seed, "verifier_version": verifier.VERIFIER_VERSION,
              "python": platform.python_version(), "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-fem")},
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for package in ("benchmarks", "verifier", "sciagent")
                                for p in sorted((ROOT / package).glob("*.py"))},
              "controls": {k: v.certificate for k, v in controls.items()}, "metrics": scores,
              "reference_checker_mutations": checker}
    if check_profiling:
        report["profiling_equivalence"] = {
            "scored_full_verdicts_equal": sum(row["verdict"] != "ERROR" for row in rows),
            "calibration_full_verdicts_equal": len(checker),
        }
    # Keep each individual trial on one line for diffing and streaming inspection.
    output.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(report, indent=2, allow_nan=False)[:-2]
    records = ",\n".join(json.dumps(r, allow_nan=False, separators=(",", ":")) for r in rows)
    with TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "report.json"
        staged.write_text(header + ',\n  "trials": [\n' + records + "\n  ]\n}\n")
        staged.replace(output)
    ok = all(r["verdict"] == ("REJECTED" if r["kind"].endswith("fault") else "ACCEPTED")
             and (bool(r["failed"]) == r["kind"].endswith("fault")) for r in rows)
    ok &= all(r["verdict"] == "REJECTED" and r["codes"] == ["REFERENCE_MISMATCH"] for r in checker)
    for cohort, groups in scores.items():
        print(cohort, {k: (g["n"], g["verdicts"]) for k, g in groups.items()})
    print(f"{len(checker)} reference checker controls; saved {output}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.add_mutually_exclusive_group(required=True)
    args.add_argument("--generate-controls", type=Path, help="create a new directory of authentic MFEM controls")
    args.add_argument("--controls", type=Path, help="reuse the archives bound by an existing index")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--check-profiling", action="store_true",
                        help="compare full verdicts with and without phase timing on the same archive")
    options = parser.parse_args()
    directory = options.generate_controls or options.controls
    if options.generate_controls:
        generate_controls(directory.resolve())
    ok = run_campaign(directory.resolve(), options.report, options.seed,
                      check_profiling=options.check_profiling)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
