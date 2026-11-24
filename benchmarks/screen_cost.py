"""Measure a native-metadata rejection screen alongside the full numerical verifier."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from sciagent.spec import ExecutionSpec
from verifier.verifier import LevelReport, _native_claims, verify_attempt


def screen(attempt, execution):
    """A failed necessary condition can reject a pilot. Every other outcome escalates."""
    for n in execution.mesh_levels:
        level = attempt / "levels" / f"n{n:04d}"
        try:
            data = json.loads((level / "result.json").read_bytes())
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict) or data.get("producer") != "poisson_mms":
            return []
        report = LevelReport(n, str(level))
        _native_claims(data, execution, report)
        failed = [check.code for check in report.checks if not check.passed]
        if failed:
            return failed
    return []


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def experiment(controls_dir, seed):
    from benchmarks.controls import ATTEMPT, load_controls
    from benchmarks.robustness import design, trial

    controls = load_controls(controls_dir)
    rows = []
    for row in design(controls, seed):
        cohort, kind, case, family, n, draw_seed = row
        if not kind.startswith("numerical"):
            continue
        with TemporaryDirectory(prefix="mfem-screen-") as temp:
            directory = Path(temp)
            original = trial(row, controls, directory)
            attempt = directory / "run" / ATTEMPT
            # Match the submitted, authorized execution. Read it independently
            # of the native result whose metadata the screen is inspecting.
            execution = ExecutionSpec.model_validate_json((attempt / "execution.json").read_bytes())
            def full():
                return verify_attempt(attempt, require_run_records=True)["verdict"]
            if len(rows) % 2:
                verdict, full_s = timed(full)
                failed, screen_s = timed(screen, attempt, execution)
            else:
                failed, screen_s = timed(screen, attempt, execution)
                verdict, full_s = timed(full)
            if verdict != original["verdict"] or verdict != ("REJECTED" if kind == "numerical_fault" else "ACCEPTED"):
                raise ValueError(f"full control verdict changed on {row}")
            rows.append({"cohort": cohort, "family": family, "case": case, "n": n, "draw_seed": draw_seed,
                         "verdict": verdict, "screen_codes": failed, "screen_seconds": screen_s,
                         "full_seconds": full_s})
    escalated = [r for r in rows if not r["screen_codes"]]
    false_rejects = sum(r["verdict"] == "ACCEPTED" and bool(r["screen_codes"]) for r in rows)
    full = sum(r["full_seconds"] for r in rows)
    cheap = sum(r["screen_seconds"] for r in rows)
    conditional = cheap + sum(r["full_seconds"] for r in escalated)
    return {"seed": seed, "scope": "historical numerical cases on warm local archives",
            "full_verdict_counts": dict(Counter(r["verdict"] for r in rows)),
            "early_rejections_by_family": dict(Counter(r["family"] for r in rows if r["screen_codes"])),
            "clean_inputs_rejected": false_rejects, "escalations": len(escalated),
            "escalated_faults": sum(r["verdict"] == "REJECTED" for r in escalated),
            "full_seconds": full, "screen_seconds_all": cheap,
            "projected_screen_plus_escalated_full_seconds": conditional, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    result = experiment(args.controls, args.seed)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if result["clean_inputs_rejected"]:
        raise SystemExit(2)
