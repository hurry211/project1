"""Independent verification of interface-aligned Q1 high-contrast diffusion.

The analytical solution and Q1 reconstruction here do not call MFEM.
The exact piecewise linear field is representable on the aligned mesh;
there is no generic smooth-solution convergence-rate requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys

from sciagent.mfem_runner import solver_binary

FIELD_TOL = 1e-8
FLUX_TOL = 1e-5
ENERGY_TOL = 1e-7


def flux(kappa):
    return 2.0 / (1.0 + 1.0 / kappa)


def exact(x, kappa):
    q = flux(kappa)
    return q*x if x <= .5 else q*.5 + q*(x-.5)/kappa


def verify_field(path, n, kappa):
    """Require the complete aligned Cartesian Q1 field and evaluate physics."""
    if type(n) is not int or n < 2 or n > 128 or n % 2 or not 1 <= kappa <= 1e6:
        raise ValueError("invalid benchmark parameters")
    cells, global_values = {}, {}
    for line in Path(path).read_text().splitlines():
        e, x, y, value = line.split()
        e, x, y, value = int(e), float(x), float(y), float(value)
        if not 0 <= e < n*n or not all(map(math.isfinite, (x,y,value))):
            raise ValueError("invalid cell or nonfinite field")
        i, j = round(n*x), round(n*y)
        if not (0 <= i <= n and 0 <= j <= n
                and abs(x-i/n) <= 1e-12 and abs(y-j/n) <= 1e-12):
            raise ValueError("non-Cartesian node")
        node = (i,j)
        c = cells.setdefault(e,{})
        if node in c:
            raise ValueError("duplicate local node")
        c[node] = value
        if node in global_values and not math.isclose(value, global_values[node], rel_tol=0, abs_tol=1e-12):
            raise ValueError("inconsistent shared DOF")
        global_values[node] = value
    if len(cells) != n*n or len(global_values) != (n+1)**2:
        raise ValueError("incomplete mesh")
    seen, energy = set(), []
    q = flux(kappa)
    flux_error = 0.0
    normal_error = 0.0
    gp = (0.5-0.5/math.sqrt(3),0.5+0.5/math.sqrt(3))
    for c in cells.values():
        i, j = min(x for x,y in c), min(y for x,y in c)
        corners = {(i,j),(i+1,j),(i,j+1),(i+1,j+1)}
        if set(c) != corners or (i,j) in seen:
            raise ValueError("nonpartitioned cell topology")
        seen.add((i,j))
        v00,v10,v01,v11 = (c[i,j],c[i+1,j],c[i,j+1],c[i+1,j+1])
        a = 1.0 if i < n//2 else kappa
        # Q1 derivatives are affine in the transverse coordinate; extrema
        # occur at endpoints, so these checks also cover the interface trace.
        dx = [n*(v10-v00),n*(v11-v01)]
        dy = [n*(v01-v00),n*(v11-v10)]
        flux_error = max(flux_error, *(abs(a*d-q)/q for d in dx))
        normal_error = max(normal_error, *(abs(a*d)/q for d in dy))
        for s in gp:
            for t in gp:
                gx = dx[0]*(1-t)+dx[1]*t
                gy = dy[0]*(1-s)+dy[1]*s
                energy.append(a*(gx*gx+gy*gy)/(4*n*n))
    field_error = max(abs(value-exact(i/n,kappa)) for (i,j),value in global_values.items())
    bc_error = max(abs(value-i/n) for (i,j),value in global_values.items() if i in (0,n))
    energy_value = math.fsum(energy)
    energy_error = abs(energy_value-q)/q
    accepted = (field_error <= FIELD_TOL and bc_error <= FIELD_TOL
                and max(flux_error,normal_error) <= FLUX_TOL and energy_error <= ENERGY_TOL)
    return {"accepted": accepted, "max_nodal_error": field_error, "dirichlet_error": bc_error,
            "max_relative_x_flux_error": flux_error, "max_relative_y_flux_error": normal_error,
            "energy": energy_value, "exact_energy": q, "relative_energy_error": energy_error,
            "thresholds": {"field":FIELD_TOL,"flux":FLUX_TOL,"energy":ENERGY_TOL}}


def run(output, n, kappa, timeout=120, memory_mb=3072):
    if type(n) is not int or n<2 or n>128 or n%2 or not 1<=kappa<=1e6:
        raise ValueError("n must be even 2..128; kappa must be 1..1e6")
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    binary = Path(os.environ.get("LAYERED_DIFFUSION_BIN", str(solver_binary().with_name("layered_diffusion")))).resolve()
    args = [str(binary),"--n",str(n),"--kappa",str(kappa),"--out",str(output)]
    request = {"n":n,"kappa":kappa,"argv":args,
               "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}
    (output/"request.json").write_text(json.dumps(request,indent=2))
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(memory_mb*1024**2,)*2)
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, preexec_fn=limits)
    (output/"stdout.txt").write_text(p.stdout)
    (output/"stderr.txt").write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(f"native solver failed: {p.returncode}")
    native = json.loads((output/"result.json").read_text())
    if native["n"]!=n or native["kappa"]!=kappa or native["dofs"]!=(n+1)**2:
        raise ValueError("native metadata mismatch")
    field = output/"field.txt"
    result = verify_field(field,n,kappa)
    result["field_sha256"] = hashlib.sha256(field.read_bytes()).hexdigest()
    result["native"] = native
    (output/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--n",type=int,default=16)
    p.add_argument("--kappa",type=float,default=100)
    a=p.parse_args()
    result=run(a.output,a.n,a.kappa)
    print(json.dumps(result,indent=2))
    return 0 if result["accepted"] else 1


if __name__=="__main__":
    sys.exit(main())
