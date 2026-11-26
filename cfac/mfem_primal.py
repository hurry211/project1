"""MFEM serial native P1 PCG with a vertex-ordered, restartable field packet."""

from pathlib import Path
import subprocess

import numpy as np

from .checker import _geometry, edges, validate_problem
from .sparse_linear import dyadic_field


def native_pcg(problem, trusted_digest, initial, binary, output, *, iterations, tolerance=1e-10, timeout=30):
    """Use MFEM's native sparse operator and return a candidate with residual telemetry."""
    validate_problem(problem, trusted_digest)
    mesh = problem.mesh
    if problem.original_affine is not None or len(initial) != len(mesh.vertices):
        raise ValueError("MFEM PCG needs original P0 data and one value per vertex")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    domain, data, field = (output / name for name in ("input.mesh", "input.txt", "field.txt"))
    _, _, boundary = _geometry(mesh)
    edge_list = edges(mesh)
    with domain.open("x", encoding="ascii", newline="\n") as out:
        print("MFEM mesh v1.0\n\ndimension\n2\n\nelements", len(mesh.triangles), sep="\n", file=out)
        for attr, cell in enumerate(mesh.triangles, 1):
            print(attr, 2, *cell, file=out)
        print("\nboundary", len(boundary), sep="\n", file=out)
        for i in sorted(boundary):
            print(1, 1, *edge_list[i], file=out)
        print("\nvertices", len(mesh.vertices), 2, sep="\n", file=out)
        for x, y in mesh.vertices:
            print(repr(float(x)), repr(float(y)), file=out)
    with data.open("x", encoding="ascii", newline="\n") as out:
        for values in (problem.loads, initial):
            for value in values:
                print(repr(float(value)), file=out)
    cmd = [str(Path(binary).resolve()), str(domain), str(data), str(float(problem.diffusion)),
           str(tolerance), str(iterations), str(field)]
    with (output / "native.log").open("x") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    with field.open() as out:
        steps, residual = out.readline().split()
        values = np.loadtxt(out, ndmin=2)
    native_residual = float(residual)
    if (values.shape != (len(mesh.vertices), 3) or not np.isfinite(values).all() or not np.isfinite(native_residual)
            or not np.array_equal(values[:, :2], np.array(mesh.vertices, dtype=float))):
        raise ValueError("MFEM exported unexpected vertices or values")
    return dyadic_field(values[:, 2]), {"iterations": int(steps), "mfem_residual_l2": native_residual}
