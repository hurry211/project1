"""Evaluation only: perturb one primal coefficient in the reviewed patch experiment."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import microcheck


def experiment(n=8):
    original_solve = microcheck.spla.spsolve
    records = []
    for offset in (0.0, 1.0 / 64):
        first_solve = {}

        def solve(matrix, rhs):
            coefficients = original_solve(matrix, rhs)
            if not first_solve:
                # The first sparse solve in the digest-pinned script is the P1 Dirichlet solve.
                middle = len(rhs) // 2
                coefficients[middle] += offset
                first_solve.update(index=middle, residual_inf=float(np.max(np.abs(rhs - matrix @ coefficients))))
            return coefficients

        with patch.object(microcheck.spla, "spsolve", solve):
            result = microcheck.run(n)
        records.append({"n": n, "offset": offset, **first_solve,
                        "relative_equilibrium_defect": float(result["eq_res"]),
                        "relative_identity_residual": float(result["ident_res"])})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    path = parser.parse_args().output
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(experiment(), out, indent=2)
        out.write("\n")
