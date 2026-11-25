"""Paired end-to-end timing and a bounded two-level pilot for the serial E1 study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


ARMS = ("gs", "jacobi", "gs_repeat")
SOLVERS = ("cg_gs", "cg_jacobi")
CALIBRATION = ("C1", "C2")
HELD_OUT = ("H1", "H2")
MIN_BLOCKS = 30


def _seconds(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError("expected a finite duration in seconds")
    return value


def _group(records, allowed):
    groups = defaultdict(dict)
    for r in records:
        if r["workload"] not in allowed or r["warmup"] is True:
            continue
        if r["warmup"] is not False or type(r["eligible"]) is not bool:
            raise ValueError("invalid warmup or eligibility flag")
        if r["arm"] not in ARMS or type(r["block"]) is not int:
            raise ValueError("unknown arm or block")
        if r["eligible"] and r["state"] != "ACCEPTED":
            raise ValueError("only accepted studies can qualify")
        _seconds(r["elapsed_seconds"], positive=True)
        arms = groups[r["workload"], r["block"]]
        if r["arm"] in arms:
            raise ValueError("duplicate arm within a paired block")
        arms[r["arm"]] = r
    return groups


def _interval(differences):
    """Resample whole paired differences. The RNG is local and repeatable."""
    if len(differences) < 2:
        return None
    rng = np.random.default_rng(20261124)
    sample = rng.choice(differences, size=(4096, len(differences)), replace=True)
    return np.quantile(np.median(sample, axis=1), [0.025, 0.975]).tolist()


def _margin(baseline_seconds):
    return max(0.020, 0.05 * baseline_seconds)


def _diagnostics(rows):
    """All durations in seconds. Each row is one fresh worker with one pinned core."""
    names = set()
    metrics = defaultdict(list)
    for r in rows:
        p = r.get("profile")
        if p is None:
            continue
        phases = p["phase_seconds"]
        names.update(phases)
        total = r["elapsed_seconds"]
        verify = sum(v for k, v in phases.items() if k.startswith("verify."))
        metrics["verification_share_of_external"].append(verify / total)
        metrics["worker_peak_rss_kib"].append(p["worker_peak_rss_kib"])
        metrics["external_outside_capture_seconds"].append(total - p["elapsed_seconds"])
        native = p["native_levels"]
        for name in ("setup_seconds", "solve_seconds", "error_seconds", "export_seconds"):
            if native and all(l.get(name) is not None for l in native):
                phase = sum(l[name] for l in native)
                metrics["native_" + name].append(phase)
                if name == "solve_seconds":
                    metrics["native_solve_share_of_external"].append(phase / total)
        rss = [l["peak_rss_kib"] for l in native if l.get("peak_rss_kib") is not None]
        if rss:
            metrics["largest_native_process_peak_rss_kib"].append(max(rss))
        iterations = [l["iterations"] for l in r.get("levels", []) if l.get("iterations") is not None]
        if iterations:
            metrics["total_cg_iterations"].append(sum(iterations))
    timed = [r for r in rows if r.get("profile") is not None]
    return {"profiled_trials": len(timed),
            "median_phase_seconds": {k: median(r["profile"]["phase_seconds"].get(k, 0) for r in timed)
                                     for k in sorted(names)},
            "medians": {k: median(v) for k, v in metrics.items()}}


def _paired(blocks):
    good = [b for b in blocks if set(b) == set(ARMS) and all(r["eligible"] for r in b.values())]
    sums = {}
    for arm in ARMS:
        rows = [b[arm] for b in blocks if arm in b]
        sums[arm] = {
            "submitted": len(rows), "qualified": sum(r["eligible"] for r in rows),
            "elapsed_seconds_all_attempts": sum(r["elapsed_seconds"] for r in rows),
            "elapsed_seconds_failed_attempts": sum(r["elapsed_seconds"] for r in rows if not r["eligible"]),
            "timing_and_memory_on_qualified": _diagnostics([r for r in rows if r["eligible"]]),
        }
    complete = sum(set(b) == set(ARMS) for b in blocks)
    noise_sensitivity = {"assessed_blocks": len(good), "median_absolute_aa_seconds": None,
                         "p90_absolute_aa_seconds": None, "absolute_aa_above_margin_count": 0,
                         "absolute_aa_above_margin_fraction": None, "median_absolute_aa_within_margin": None}
    result = {"observed_blocks": len(blocks), "qualified_complete_blocks": len(good),
              "missing_arm_blocks": len(blocks) - complete, "complete_but_ineligible_blocks": complete - len(good),
              "all_observed_blocks_complete_and_qualified": bool(blocks) and len(good) == len(blocks),
              "absolute_aa_sensitivity": noise_sensitivity,
              "arms": sums, "result": ("failed_or_incomplete_blocks"
                                       if len(good) < len(blocks) else "insufficient_blocks")}
    result["conservative_result"] = result["result"]
    if not good:
        return result
    gs = [b["gs"]["elapsed_seconds"] for b in good]
    jacobi = [b["jacobi"]["elapsed_seconds"] for b in good]
    aa = [b["gs_repeat"]["elapsed_seconds"] - b["gs"]["elapsed_seconds"] for b in good]
    delta = [j - g for j, g in zip(jacobi, gs, strict=True)]
    margin = _margin(median(gs))
    magnitudes = [abs(x) for x in aa]
    exceed = sum(value > margin for value in magnitudes)
    noise_sensitivity.update(median_absolute_aa_seconds=median(magnitudes),
                             p90_absolute_aa_seconds=float(np.quantile(magnitudes, 0.90)),
                             absolute_aa_above_margin_count=exceed,
                             absolute_aa_above_margin_fraction=exceed / len(aa),
                             median_absolute_aa_within_margin=median(magnitudes) <= margin)
    interval, noise = _interval(delta), _interval(aa)
    result.update({
        "median_baseline_seconds": median(gs), "median_jacobi_seconds": median(jacobi),
        "jacobi_to_gs_ratio_of_medians": median(jacobi) / median(gs),
        "median_paired_jacobi_minus_gs_seconds": median(delta),
        "paired_bootstrap_95pct_seconds": interval,
        "median_paired_aa_seconds": median(aa), "aa_bootstrap_95pct_seconds": noise,
        "practical_margin_seconds": margin,
    })
    if len(good) == len(blocks) and len(good) >= MIN_BLOCKS:
        if not (-margin <= noise[0] and noise[1] <= margin):
            result["result"] = "noise_unresolved"
        elif interval[1] < -margin:
            result["result"] = "jacobi_materially_faster"
        elif interval[0] > margin:
            result["result"] = "gs_materially_faster"
        elif -margin <= interval[0] and interval[1] <= margin:
            result["result"] = "within_margin"
        else:
            result["result"] = "inconclusive"
    result["conservative_result"] = result["result"]
    if (result["result"] not in {"failed_or_incomplete_blocks", "insufficient_blocks", "noise_unresolved"}
            and not noise_sensitivity["median_absolute_aa_within_margin"]):
        result["conservative_result"] = "absolute_aa_noise_unresolved"
    return result


def _pilot_forecast(levels, target_n):
    """Fit iteration growth on two levels. A diagonal or sparse sweep costs O(n²)."""
    if len(levels) != 2:
        raise ValueError("each pilot requires exactly two levels")
    small, large = levels
    for level in levels:
        if type(level["n"]) is not int or level["n"] < 1:
            raise ValueError("invalid mesh size")
        if type(level["iterations"]) is not int or level["iterations"] <= 0:
            raise ValueError("invalid iteration count")
        native = _seconds(level["native_wall_seconds"], positive=True)
        if _seconds(level["solve_seconds"], positive=True) > native:
            raise ValueError("solve duration exceeds native wall time")
    if large["n"] != 2 * small["n"] or type(target_n) is not int or target_n < large["n"]:
        raise ValueError("expected a dyadic refinement sequence")
    iteration_growth = large["iterations"] / small["iterations"]
    n = large["n"]
    seconds = large["solve_seconds"]
    total = small["solve_seconds"] + seconds
    while n < target_n:
        n *= 2
        seconds *= 4 * iteration_growth
        total += seconds
    if n != target_n or not math.isfinite(total):
        raise ValueError("target is outside the dyadic pilot sequence")
    return total


def fit_policy(records):
    """Calibrate on C1 and C2 only. Evidence below the design size stays descriptive."""
    groups = _group(records, CALIBRATION)
    factors = {s: [] for s in SOLVERS}
    training = {}
    conservative = {}
    baseline = {}
    wins = False
    for w in CALIBRATION:
        blocks = [b for (name, _), b in groups.items() if name == w]
        paired = _paired(blocks)
        training[w] = paired["result"]
        conservative[w] = paired["conservative_result"]
        sizes = {b["gs"]["levels"][-1]["n"] for b in blocks if
                 set(b) == set(ARMS) and all(r["eligible"] for r in b.values())}
        if len(sizes) > 1:
            raise ValueError("inconsistent workload sizes")
        if sizes:
            baseline[w] = {"n": sizes.pop(), "seconds": paired["median_baseline_seconds"]}
        for b in blocks:
            if set(b) != set(ARMS) or not all(r["eligible"] for r in b.values()):
                continue
            for arm, solver in zip(ARMS, SOLVERS):
                levels = b[arm]["levels"]
                if len(levels) < 3:
                    raise ValueError("qualified E1 study lacks levels")
                forecast = _pilot_forecast(levels[:2], levels[-1]["n"])
                factors[solver].append(sum(_seconds(l["solve_seconds"]) for l in levels) / forecast)
        wins |= paired["result"] == "jacobi_materially_faster"
    resolved = all(r in {"within_margin", "jacobi_materially_faster", "gs_materially_faster"}
                   for r in conservative.values())
    return {
        "version": 1, "enabled": resolved and wins, "calibration": training, "baseline": baseline,
        "calibration_with_absolute_aa_veto": conservative,
        "reason": ("calibrated_jacobi_advantage" if resolved and wins else
                   "unresolved_absolute_aa_noise" if "absolute_aa_noise_unresolved" in conservative.values() else
                   "no_material_calibration_advantage" if resolved else "unresolved_calibration"),
        "solve_scale": {s: median(v) if v else 1.0 for s, v in factors.items()},
    }


def choose_solver(policy, pilots, *, target_n, authorized_solvers, hardware_ok, pilot_elapsed_seconds=None):
    """Select from authorized E1 arms using pre-decision observations only.

    The caller checks the scalar SPD P1 full-assembly case, one-core execution
    envelope, and resource limits. The numerical controller still decides acceptance.
    """
    allowed = set(authorized_solvers)
    if allowed - set(SOLVERS):
        raise ValueError("unknown E1 solver")
    cost = sum(_seconds(l["native_wall_seconds"]) for levels in pilots.values() for l in levels)
    if pilot_elapsed_seconds is not None:
        if _seconds(pilot_elapsed_seconds) < cost:
            raise ValueError("external pilot time must include all measured native calls")
        cost = pilot_elapsed_seconds
    result = {"solver": None, "pilot_seconds": cost, "forecast_solver": None,
              "projected_solve_seconds": None, "reason": "ineligible"}
    if hardware_ok is not True or not allowed:
        return result
    default = next(s for s in SOLVERS if s in allowed)
    result.update(solver=default, reason="single_authorized_solver")
    if len(allowed) == 1:
        return result
    if set(pilots) != set(SOLVERS):
        result["reason"] = policy["reason"] if not policy["enabled"] else "missing_pilot"
        return result
    try:
        raw = {s: _pilot_forecast(pilots[s], target_n) for s in SOLVERS}
    except (KeyError, ValueError):
        result["reason"] = "unusable_pilot"
        return result
    predicted = {s: raw[s] * _seconds(policy["solve_scale"][s], positive=True) for s in SOLVERS}
    best = min(SOLVERS, key=predicted.get)
    result.update(projected_solve_seconds=predicted, forecast_solver=best)
    gain = predicted["cg_gs"] - predicted["cg_jacobi"]
    if not policy["enabled"]:
        result["reason"] = policy["reason"]
    else:
        c1, c2 = (policy["baseline"][w] for w in CALIBRATION)
        if c1["n"] >= c2["n"]:
            raise ValueError("expected two increasing calibration sizes")
        growth = math.log(c2["seconds"] / c1["seconds"]) / math.log(c2["n"] / c1["n"])
        baseline = c2["seconds"] * (target_n / c2["n"]) ** max(0.0, growth)
        result["projected_baseline_workflow_seconds"] = baseline
        if best == "cg_jacobi" and gain > cost + _margin(baseline):
            result.update(solver=best, reason="forecast_gain_exceeds_pilot_cost")
        else:
            result["reason"] = "pilot_cost_or_default_advantage"
    return result


def _evaluate(blocks, policy, target_n):
    """Replay decisions from prefix observations. No holdout outcome enters choose_solver."""
    forecasts = Counter()
    choices = Counter()
    charges = []
    for b in blocks:
        if "gs" not in b:
            continue
        trials = {s: b[a] for s, a in zip(SOLVERS, ARMS) if a in b}
        pilots = {s: r["levels"][:2] for s, r in trials.items()
                  if len(r.get("levels", [])) >= 2 and all(
                      l.get("native_wall_seconds") is not None for l in r["levels"][:2])}
        if policy["enabled"] and set(pilots) != set(SOLVERS):
            # The cost of an unsuccessful standalone probe cannot be recovered from a
            # missing prefix. Report the observation as unscored, never as a free win.
            continue
        decision = choose_solver(policy, pilots, target_n=target_n,
                                 authorized_solvers=SOLVERS, hardware_ok=True)
        forecasts[decision["forecast_solver"] or "unavailable"] += 1
        # A deployed policy would skip pilots entirely if calibration kept the default.
        cost = decision["pilot_seconds"] if policy["enabled"] else 0.0
        selected = trials[decision["solver"]]
        choices[decision["solver"]] += 1
        total = cost + selected["elapsed_seconds"]
        good = [r["elapsed_seconds"] for r in trials.values() if r["eligible"]]
        charges.append({
            "eligible": selected["eligible"], "total": total, "pilot": cost,
            "complete_observations": set(pilots) == set(SOLVERS),
            "probe_both_prefixes": decision["pilot_seconds"],
            "vs_gs": total - trials["cg_gs"]["elapsed_seconds"] if
            selected["eligible"] and trials["cg_gs"]["eligible"] else None,
            "vs_jacobi": total - trials["cg_jacobi"]["elapsed_seconds"] if
            selected["eligible"] and "cg_jacobi" in trials and trials["cg_jacobi"]["eligible"] else None,
            "vs_oracle": total - min(good) if selected["eligible"] and len(trials) == 2 else None,
            "forecast_correct": (trials[decision["forecast_solver"]]["elapsed_seconds"] == min(good))
            if len(good) == 2 and decision["forecast_solver"] is not None else None,
        })
    excess = [r["vs_oracle"] for r in charges if r["vs_oracle"] is not None]
    relative = [r["vs_gs"] for r in charges if r["vs_gs"] is not None]
    jacobi = [r["vs_jacobi"] for r in charges if r["vs_jacobi"] is not None]
    hits = [r["forecast_correct"] for r in charges if r["forecast_correct"] is not None]
    return {
        "mode": "retrospective prefix replay, native prefix wall times only",
        "blocks_with_complete_observations": sum(r["complete_observations"] for r in charges),
        "policy_actions_scored": len(charges), "blocks_unscored": len(blocks) - len(charges),
        "verified_selected": sum(r["eligible"] for r in charges), "selected_solvers": dict(choices),
        "forecast_solvers": dict(forecasts), "workflow_winner_forecast_hits": sum(hits),
        "workflow_winner_forecast_trials": len(hits),
        "elapsed_seconds_including_selected_failures_and_charged_pilots": sum(r["total"] for r in charges),
        "pilot_native_wall_seconds_charged": sum(r["pilot"] for r in charges),
        "probe_both_native_prefixes_seconds_if_always_probed": sum(r["probe_both_prefixes"] for r in charges),
        "median_excess_seconds_vs_fixed_gs": median(relative) if relative else None,
        "median_excess_seconds_vs_fixed_jacobi": median(jacobi) if jacobi else None,
        "median_excess_seconds_vs_fastest_qualified_arm_in_block": median(excess) if excess else None,
        "matched_default_comparisons": len(relative), "matched_oracle_comparisons": len(excess),
    }


def analyze(campaign):
    records = campaign["records"]
    names = (*CALIBRATION, *HELD_OUT)
    if any(r["workload"] not in names for r in records):
        raise ValueError("unknown E1 workload")
    groups = _group(records, names)
    policy = fit_policy(records)
    result = {}
    for w in names:
        blocks = [b for (name, _), b in groups.items() if name == w]
        result[w] = _paired(blocks)
        if w in HELD_OUT and w in campaign["workloads"]:
            target_n = campaign["workloads"][w]["levels"][-1]
            result[w]["pilot_selection"] = _evaluate(blocks, policy, target_n)
    return {
        "campaign_status": campaign.get("status"), "minimum_blocks_for_decision": MIN_BLOCKS,
        "uncertainty": "per-workload paired percentile bootstrap, 4096 block resamples, 95% marginal intervals",
        "noise_interpretation": ("result uses the original signed A/A median interval. "
                                 "conservative_result adds a post hoc veto when the median absolute A/A "
                                 "difference exceeds the same margin. This sensitivity check can only withhold "
                                 "an equivalence claim or a switch; it cannot establish a new advantage."),
        "policy": policy, "workloads": result,
        "submitted_elapsed_seconds": sum(_seconds(r["elapsed_seconds"]) for r in records),
        "warmup_count": sum(r["warmup"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("analyze", help="evaluate a recorded E1 campaign")
    score.add_argument("campaign", type=Path)
    select = commands.add_parser("choose", help="advise using calibration and prospective pilot observations")
    select.add_argument("campaign", type=Path, help="campaign with calibration records")
    select.add_argument("request", type=Path,
                        help="JSON with pilots, target_n, authorized_solvers, hardware_ok and optional pilot_elapsed_seconds")
    parser.add_argument("--output", type=Path, help="write a new report instead of stdout")
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_bytes())
    if args.command == "analyze":
        result = analyze(campaign)
    else:
        request = json.loads(args.request.read_bytes())
        result = choose_solver(fit_policy(campaign["records"]), **request)
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
