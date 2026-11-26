"""Render measured CFAC timing ratios, recovery, and attribution scaling."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(root, path):
    return json.loads((root / path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    root = args.evidence
    plt.rcParams["svg.hashsalt"] = "cfac-sparse-20261126"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")

    ax = axes[0]
    recovery = read(root, "fixed-primal-rt1.json")["results"]
    for case, row in recovery.items():
        entry = next(r for r in row["rt1_candidate_comparisons"] if r["policy"] == "predicted")
        values = [float(Fraction(s))**.5 for s in entry["upper_history_sq"]]
        tolerance = float(Fraction(row["epsilon_sq"]))**.5
        ax.plot(entry["inspection_steps"], [v/tolerance for v in values], "o-", label=case.replace("_tight", ""))
    ax.axhline(1, color="black", linewidth=1, linestyle="--")
    ax.set(yscale="log", xlabel="RT1 PCG iteration", ylabel="Certified upper bound / target",
           title="Fixed primal recovery")
    ax.legend(frameon=False)

    ax = axes[1]
    summary = read(root, "screen-summary.json")["contrasts"]
    for i, (policy, row) in enumerate(summary.items()):
        center = row["ratio_of_paired_times_median"]
        lo, hi = row["nested_percentile_95pct"]
        ax.errorbar(center, i, xerr=[[center-lo], [hi-center]], fmt="o", capsize=3)
    ax.axvline(1, color="black", linewidth=1, linestyle="--")
    ax.axvline(.9, color="gray", linewidth=1, linestyle=":")
    ax.set(yticks=list(range(len(summary))), yticklabels=list(summary),
           xlabel="Screened CFAC / baseline elapsed", title="Matched accepted trials")

    ax = axes[2]
    scale = [read(root, f"scale-{n}.json")["results"] for n in [32, 64, 100, 160, 320]]
    vertices = [row["free_p1_dofs"] for row in scale]
    for key, label in (("floor_seconds", "Certified attribution"), ("native_proposal_seconds", "MFEM primal proposal"),
                       ("sparse_p1_setup_seconds", "Python P1 assembly and lift")):
        ax.loglog(vertices, [row[key] for row in scale], "o-", label=label)
    ax.set(xlabel="Free P1 degrees of freedom", ylabel="Elapsed seconds", title="Scaling diagnostic")
    ax.legend(frameon=False, fontsize=8)
    image = root / "measurements.svg"
    fig.savefig(image, metadata={"Date": None})
    image.write_text("\n".join(line.rstrip() for line in image.read_text(encoding="utf-8").splitlines()) + "\n",
                     encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
