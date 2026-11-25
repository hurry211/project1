"""Adapter around the real MFEM executable ``poisson_mms``.

Responsibilities: build the solver, bind a ProblemSpec to a compiled
manufactured case (exact textual match), and invoke one mesh level under
explicit time and memory limits, preserving stdout/stderr/exit code.
"""

from __future__ import annotations

import json
import os
import resource
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import profiling
from .environment import DEFAULT_SOLVER_BIN, PROJECT_ROOT, mfem_install_dir
from .spec import ANISOTROPIC, LAYERED, POISSON, ExecutionSpec, ProblemSpec, compact_expression

EXIT_CODES = {
    0: "OK", 2: "INVALID_ARGUMENTS", 3: "BOUNDARY_COVERAGE", 4: "UNSUPPORTED_COMBINATION",
    5: "SOLVER_NOT_CONVERGED", 6: "NON_FINITE_SOLUTION", 7: "ARTIFACT_WRITE_ERROR",
}


class BuildError(RuntimeError):
    pass


# This adapter runs scalar H1 diffusion with boundary data from the exact
# solution. ProblemSpec supplies the equation, domain, and assignment of sides.
_POISSON_FORM = {
    "space": "scalar_H1",
    "bilinear_integrator": "DiffusionIntegrator",
    "dirichlet_trace": "u_exact",
    "neumann_trace": "grad(u_exact).n",
}


def solver_binary() -> Path:
    return Path(os.environ.get("POISSON_MMS_BIN", DEFAULT_SOLVER_BIN))


def build_solver(log_dir: Path | None = None) -> dict:
    """Configure + build cpp/ with CMake against the pinned MFEM install."""
    build_dir = PROJECT_ROOT / "build" / "cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    mfem_cmake = mfem_install_dir() / "lib" / "cmake" / "mfem"
    cmds = [
        ["cmake", "-S", str(PROJECT_ROOT / "cpp"), "-B", str(build_dir),
         f"-DMFEM_DIR={mfem_cmake}", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build_dir), "-j", os.environ.get("JOBS", "2")],
    ]
    record = {"commands": [], "ok": True}
    for cmd in cmds:
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        entry = {"cmd": shlex.join(cmd), "exit_code": p.returncode, "seconds": round(time.time() - t0, 3)}
        record["commands"].append(entry)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            idx = len(record["commands"])
            (log_dir / f"build_step{idx}.stdout.txt").write_text(p.stdout)
            (log_dir / f"build_step{idx}.stderr.txt").write_text(p.stderr)
        if p.returncode != 0:
            record["ok"] = False
            record["error_tail"] = (p.stderr or p.stdout)[-2000:]
            break
    return record


def _require_text_fields(record, keys, name: str):
    if not isinstance(record, dict) or not all(
        isinstance(record.get(k), str) and record[k].strip() for k in keys
    ):
        raise ValueError(f"invalid {name}")


def _validate_registry(catalog):
    if not isinstance(catalog, dict) or not isinstance(catalog.get("cases"), list):
        raise ValueError("expected a cases array")
    version = catalog.get("registry_version")
    if "registry_version" in catalog or "formulation" in catalog:
        if type(version) is not int or version not in (1, 2):
            raise ValueError("unsupported case registry version")
        _require_text_fields(catalog.get("formulation"), ("equation", "domain", *_POISSON_FORM), "formulation")

    ids, signatures = set(), set()
    for case in catalog["cases"]:
        _require_text_fields(case, ("id", "exact_solution", "source"), "manufactured case entry")
        if "formulation" in case:
            if version != 2:
                raise ValueError("case-specific formulations require registry version 2")
            _require_text_fields(case["formulation"], ("equation", "domain", "kappa", *_POISSON_FORM),
                                 "case-specific formulation")
        signature = (compact_expression(case["exact_solution"]), compact_expression(case["source"]))
        if case["id"] in ids or signature in signatures:
            raise ValueError("duplicate case identifier or formula signature")
        ids.add(case["id"])
        signatures.add(signature)


def list_cases(binary: Path | None = None, *, include_e3: bool = False) -> dict:
    binary = binary or solver_binary()
    if not binary.is_file():
        raise BuildError(f"solver binary not found: {binary}")
    try:
        arguments = [str(binary), "--list-cases"]
        if include_e3:
            arguments.append("--include-e3")
        p = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"cannot execute {binary}: {exc}") from exc
    if p.returncode != 0:
        raise BuildError(f"--list-cases failed: {p.stderr}")
    try:
        cases = json.loads(p.stdout)
        _validate_registry(cases)
        return cases
    except ValueError as exc:
        raise BuildError(f"invalid --list-cases output: {exc}") from exc


def bind_case(problem: ProblemSpec, cases: dict) -> str | None:
    """Return a unique compatible case from a current or historical catalog."""
    try:
        _validate_registry(cases)
    except ValueError:
        return None
    expected = {"equation": problem.equation, "domain": problem.domain, **_POISSON_FORM}
    variable = problem.equation == "-div(kappa*grad(u)) = f"
    if variable:
        expected["neumann_trace"] = "kappa*grad(u_exact).n"
    coefficient = problem.kappa
    if problem.equation == ANISOTROPIC:
        coefficient, expected["neumann_trace"] = "diag(1000,1)", "(A*grad(u_exact)).n"
    elif problem.equation == LAYERED:
        coefficient, expected["neumann_trace"] = "1 (x<0.5), 10000 (x>=0.5)", "a(x)*grad(u_exact).n"
    for c in cases["cases"]:
        if "equation" in c and c["equation"] != problem.equation:
            continue
        if variable and "formulation" not in c:
            continue  # Historical solvers implement only unit diffusion.
        form = c.get("formulation", cases.get("formulation"))
        if form is None and c.get("equation", POISSON) != problem.equation:
            continue
        if form is not None and (
            any(form[key] != value for key, value in expected.items())
            or compact_expression(form.get("kappa", "1")) != compact_expression(coefficient)
        ):
            continue
        if (compact_expression(c["exact_solution"]) == compact_expression(problem.exact_solution)
                and compact_expression(c["source"]) == compact_expression(problem.source)):
            return c["id"]
    return None


@dataclass
class LevelRun:
    n: int
    level_dir: str
    command: list[str]
    exit_code: int | None
    exit_meaning: str
    wall_seconds: float
    timed_out: bool
    stdout_file: str
    stderr_file: str


def _limits(mem_mb: int):
    def apply():
        lim = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    return apply


def run_level(problem: ProblemSpec, execution: ExecutionSpec, case_id: str, n: int,
              level_dir: Path, timeout_s: float, mem_mb: int, binary: Path | None = None, *,
              mesh_path: Path | None = None, refinements: int = 0) -> LevelRun:
    binary = binary or solver_binary()
    level_dir.mkdir(parents=True, exist_ok=False)  # never overwrite evidence
    cmd = [
        str(binary), "--case", case_id, "--mesh-type", execution.mesh_type,
        "--n", str(n), "--order", str(execution.polynomial_order),
        "--dirichlet", ",".join(map(str, problem.dirichlet_attributes())) or "none",
        "--neumann", ",".join(map(str, problem.neumann_attributes())) or "none",
        "--assembly", execution.assembly, "--solver", execution.solver,
        "--rel-tol", repr(execution.rel_tol), "--max-iter", str(execution.max_iter),
        "--inject-fault", execution.inject_fault, "--out", str(level_dir),
    ]
    if execution.mesh_topology != "cartesian":
        cmd.extend(["--mesh-topology", execution.mesh_topology])
    if mesh_path is not None:
        cmd.extend(["--mesh", str(mesh_path), "--mesh-refinements", str(refinements)])
    (level_dir / "command.json").write_text(json.dumps({"argv": cmd, "shell": shlex.join(cmd)}, indent=2))
    t0 = time.monotonic()
    timed_out = False
    try:
        env = {**os.environ, "MFEM_STUDY_PROFILE": "1"} if profiling.enabled() else None
        with profiling.phase("native.process"):
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                               preexec_fn=_limits(mem_mb), env=env)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = None
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    wall = time.monotonic() - t0
    profiling.record_native(n, wall, err)
    (level_dir / "stdout.txt").write_text(out)
    (level_dir / "stderr.txt").write_text(err)
    meaning = "TIMEOUT" if timed_out else EXIT_CODES.get(code, f"UNKNOWN_EXIT_{code}")
    lr = LevelRun(n=n, level_dir=str(level_dir), command=cmd, exit_code=code, exit_meaning=meaning,
                  wall_seconds=round(wall, 4), timed_out=timed_out,
                  stdout_file=str(level_dir / "stdout.txt"), stderr_file=str(level_dir / "stderr.txt"))
    (level_dir / "run_record.json").write_text(json.dumps(asdict(lr), indent=2))
    return lr


def amr_binary() -> Path:
    return solver_binary().parent / "poisson_amr"


def run_amr(
    case_id: str,
    mesh_type: str,
    order: int,
    out_dir: Path,
    *,
    n: int = 4,
    assembly: str = "full",
    solver: str = "cg_gs",
    dirichlet: str = "1,2,3,4",
    neumann: str = "none",
    mesh_path: Path | None = None,
    max_steps: int = 7,
    max_dofs: int = 40_000,
    theta: float = 0.7,
    rel_tol: float = 1e-12,
    max_iter: int = 20_000,
    inject_fault: str = "none",
    timeout_s: float = 300.0,
    mem_mb: int = 3072,
    binary: Path | None = None,
) -> dict:
    binary = binary or amr_binary()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cmd = [
        str(binary), "--case", case_id, "--mesh-type", mesh_type,
        "--n", str(n), "--order", str(order),
        "--dirichlet", dirichlet, "--neumann", neumann,
        "--assembly", assembly, "--solver", solver,
        "--rel-tol", repr(rel_tol), "--max-iter", str(max_iter),
        "--max-steps", str(max_steps), "--max-dofs", str(max_dofs),
        "--theta", repr(theta), "--inject-fault", inject_fault,
        "--out", str(out_dir),
    ]
    if mesh_path is not None:
        cmd.extend(["--mesh", str(mesh_path)])
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, preexec_fn=_limits(mem_mb))
    wall = time.monotonic() - t0
    (out_dir / "stdout.txt").write_text(p.stdout)
    (out_dir / "stderr.txt").write_text(p.stderr)
    summary_path = out_dir / "amr_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return {
        "exit_code": p.returncode,
        "wall_seconds": round(wall, 4),
        "summary": summary,
    }
