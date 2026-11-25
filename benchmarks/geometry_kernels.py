"""Compare element stiffness kernels on five native MFEM meshes with identical quadrature."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import time

import numpy as np

from benchmarks.e3_complex_meshes import run_repeated_period_degradation
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.mfem_runner import solver_binary
from verifier import expr, verifier
from verifier.fem import (build_local_basis, eval_monomials, evaluate_quadrature_slice,
                          load_mesh_data, reference_quadrature)


def compare_matrices(reference, actual):
    """Check each local matrix using its own scale, including small coefficients."""
    if actual.shape != reference.shape or reference.ndim != 3 or reference.shape[1] != reference.shape[2]:
        raise ValueError("element stiffness dimensions differ")
    scale = np.maximum(np.max(np.abs(reference), axis=(1, 2)), np.finfo(float).tiny)
    per_element = np.max(np.abs(actual - reference), axis=(1, 2)) / scale
    worst = float(per_element.max())
    if not np.isfinite(worst) or worst > 64 * np.finfo(float).eps:
        raise ValueError(f"element stiffness parity failed, max relative difference {worst}")
    return worst


def measure(level, scenario, blocks, rng):
    mesh = load_mesh_data(level)
    basis = build_local_basis(mesh, 1)
    xi, eta, weights = reference_quadrature(mesh.geom, 6)
    monom = eval_monomials(basis.exps, xi, eta)
    raw = json.loads((level / "result.json").read_bytes())
    coefficient = raw.get("case_diffusivity", "1" if raw["case_equation"] == verifier.POISSON else None)
    if coefficient != scenario["kappa"] or raw["solver_converged"] is not True:
        raise ValueError("native coefficient or convergence contradicts measurement design")
    kappa = None if scenario["kappa"] == "1" else expr.compile_expr(scenario["kappa"])
    equation = raw["case_equation"]
    size = max(1, 32768 // len(xi))
    elements = len(mesh.verts)
    firsts = [i % 2 for i in range(blocks)]
    rng.shuffle(firsts)
    rows = []
    hashes = None
    max_error = 0.0
    for block, first in enumerate([rng.randrange(2), *firsts], start=-1):
        order = ("full", "stream") if first == 0 else ("stream", "full")
        times = dict.fromkeys(order, 0.0)
        sums = {arm: hashlib.sha256() for arm in order}
        for start in range(0, elements, size):
            q = evaluate_quadrature_slice(mesh, basis, xi, eta, weights, start, min(elements, start + size),
                                          monom_cache=monom)
            matrices = {}
            for arm in order:
                before = time.perf_counter()
                if arm == "stream":
                    values = verifier._stream_stiffness(q, kappa, equation)
                else:
                    values = verifier._stiffness_block(q.gphi, q.wdet, q.X[..., 0], q.X[..., 1], kappa, equation)
                times[arm] += time.perf_counter() - before
                matrices[arm] = values
            max_error = max(max_error, compare_matrices(matrices["full"], matrices["stream"]))
            for arm in order:
                sums[arm].update(np.ascontiguousarray(matrices[arm]).data)
        current = {arm: d.hexdigest() for arm, d in sums.items()}
        if hashes is not None and current != hashes:
            raise ValueError("element matrices changed between paired blocks")
        hashes = current
        rows.append({"block": block, "order": order, "seconds": times})
    times = {arm: median(r["seconds"][arm] for r in rows[1:]) for arm in ("full", "stream")}
    return {"name": scenario["scenario"], "cells": elements, "geom": mesh.geom, "kappa": scenario["kappa"],
            "quadrature_points_per_element": len(xi), "detected_geometry_period": scenario["detected_period"],
            "median_seconds": times, "ratio_of_medians": times["full"] / times["stream"],
            "maximum_elementwise_relative_difference": max_error, "matrix_bytes_sha256": hashes,
            "native_inventory_sha256": artifact_hashes(level), "samples": rows}


def capture(directory, blocks, binary):
    if blocks < 4 or blocks % 2:
        raise ValueError("use an even number of at least four scored pairs")
    directory.mkdir(parents=True, exist_ok=False)
    scenarios = run_repeated_period_degradation(directory, binary)
    rng = random.Random(20261125)
    return {"seed": 20261125, "warmup_pairs": 1, "scored_pairs": blocks,
            "arm_order": "balanced first position, randomized within each scenario",
            "scope": "element stiffness only; quadrature construction, sparse assembly and solve excluded",
            "binary_sha256": sha256_file(binary), "kernel_module_sha256": sha256_file(Path(verifier.__file__)),
            "affinity": sorted(os.sched_getaffinity(0)), "numpy": np.__version__,
            "threads": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "cases": [measure(directory / ("period_" + case["scenario"]), case, blocks, rng) for case in scenarios]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--binary", type=Path, default=solver_binary())
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    data = capture(args.directory, args.blocks, args.binary)
    with args.report.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
