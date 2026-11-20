"""Environment probe: records facts, never guesses.

Every field is either measured by running a command / reading a file, or set
to ``None`` with the reason recorded in ``notes``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MFEM_INSTALL = PROJECT_ROOT / "third_party" / "mfem-install"
DEFAULT_SOLVER_BIN = PROJECT_ROOT / "build" / "cpp" / "poisson_mms"

MFEM_FLAGS = (
    "MFEM_USE_MPI", "MFEM_USE_METIS", "MFEM_USE_METIS_5", "MFEM_USE_SUITESPARSE",
    "MFEM_USE_LAPACK", "MFEM_USE_OPENMP", "MFEM_USE_CUDA", "MFEM_USE_HIP",
    "MFEM_USE_RAJA", "MFEM_USE_OCCA", "MFEM_USE_PETSC", "MFEM_USE_ZLIB",
    "MFEM_USE_DOUBLE", "MFEM_USE_SINGLE",
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mfem_install_dir() -> Path:
    return Path(os.environ.get("MFEM_INSTALL", DEFAULT_MFEM_INSTALL))


def probe_mfem(install: Path | None = None) -> dict:
    install = install or mfem_install_dir()
    info: dict = {"install_dir": str(install), "found": False, "config": {}, "notes": []}
    cfg = install / "include" / "mfem" / "config" / "_config.hpp"
    if not cfg.is_file():
        info["notes"].append(f"MFEM config header not found at {cfg}")
        return info
    text = cfg.read_text()
    info["found"] = True
    for flag in MFEM_FLAGS:
        info["config"][flag] = bool(re.search(rf"^#define {flag}\b", text, re.M))
    m = re.search(r'#define MFEM_VERSION_STRING "([^"]+)"', text)
    info["version"] = m.group(1) if m else None
    m = re.search(r"#define MFEM_GIT_STRING \"([^\"]*)\"", text)
    info["git"] = m.group(1) if m else None
    lib = install / "lib" / "libmfem.a"
    info["library"] = str(lib) if lib.is_file() else None
    info["library_sha256"] = sha256_file(lib)
    return info


def probe(solver_bin: Path | None = None) -> dict:
    solver_bin = solver_bin or Path(os.environ.get("POISSON_MMS_BIN", DEFAULT_SOLVER_BIN))
    mem_total_kb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
    except OSError:
        pass
    pkgs = {}
    for mod in ("numpy", "scipy", "pydantic", "pytest"):
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - record absence, do not fail
            pkgs[mod] = None
    mpirun = shutil.which("mpirun")
    env = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_total_mb": mem_total_kb // 1024 if mem_total_kb else None,
        "python": sys.version.split()[0],
        "python_packages": pkgs,
        "cxx": _run(["g++", "--version"]),
        "cmake": _run(["cmake", "--version"]),
        "make": _run(["make", "--version"]),
        "mpirun_path": mpirun,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "mfem": probe_mfem(),
        "solver_binary": str(solver_bin),
        "solver_binary_sha256": sha256_file(solver_bin),
        "notes": [],
    }
    if mpirun and not env["mfem"]["config"].get("MFEM_USE_MPI"):
        env["notes"].append("mpirun exists but MFEM was built without MPI; MPI path unavailable")
    if not mpirun:
        env["notes"].append("no mpirun on PATH; MPI path unavailable")
    if not env["nvidia_smi_path"]:
        env["notes"].append("no nvidia-smi on PATH; no CUDA device detected")
    return env
