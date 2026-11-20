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

from .environment import DEFAULT_SOLVER_BIN, PROJECT_ROOT, mfem_install_dir
from .spec import ExecutionSpec, ProblemSpec

EXIT_CODES = {
    0: "OK", 2: "INVALID_ARGUMENTS", 3: "BOUNDARY_COVERAGE", 4: "UNSUPPORTED_COMBINATION",
    5: "SOLVER_NOT_CONVERGED", 6: "NON_FINITE_SOLUTION",
}


class BuildError(RuntimeError):
    pass


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
        ["cmake", "--build", str(build_dir), "-j", "2"],
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


def list_cases(binary: Path | None = None) -> dict:
    binary = binary or solver_binary()
    if not binary.is_file():
        raise BuildError(f"solver binary not found: {binary}")
    p = subprocess.run([str(binary), "--list-cases"], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise BuildError(f"--list-cases failed: {p.stderr}")
    return json.loads(p.stdout)


def _norm(s: str) -> str:
    return "".join(s.split())


def bind_case(problem: ProblemSpec, cases: dict) -> str | None:
    """Return the compiled case id implementing exactly this problem, if any."""
    for c in cases["cases"]:
        if _norm(c["exact_solution"]) == _norm(problem.exact_solution) and \
           _norm(c["source"]) == _norm(problem.source):
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
              level_dir: Path, timeout_s: float, mem_mb: int, binary: Path | None = None) -> LevelRun:
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
    (level_dir / "command.json").write_text(json.dumps({"argv": cmd, "shell": shlex.join(cmd)}, indent=2))
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           preexec_fn=_limits(mem_mb))
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = None
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    wall = time.time() - t0
    (level_dir / "stdout.txt").write_text(out)
    (level_dir / "stderr.txt").write_text(err)
    meaning = "TIMEOUT" if timed_out else EXIT_CODES.get(code, f"UNKNOWN_EXIT_{code}")
    lr = LevelRun(n=n, level_dir=str(level_dir), command=cmd, exit_code=code, exit_meaning=meaning,
                  wall_seconds=round(wall, 4), timed_out=timed_out,
                  stdout_file=str(level_dir / "stdout.txt"), stderr_file=str(level_dir / "stderr.txt"))
    (level_dir / "run_record.json").write_text(json.dumps(asdict(lr), indent=2))
    return lr
