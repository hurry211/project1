import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sciagent.mfem_runner import solver_binary  # noqa: E402

HAVE_SOLVER = solver_binary().is_file() and os.access(solver_binary(), os.X_OK)
requires_mfem = pytest.mark.skipif(not HAVE_SOLVER, reason="poisson_mms binary not built (run scripts/build_solver.sh)")
MPI_SOLVER = Path(os.getenv("POISSON_MPI_BIN", ROOT / "build" / "mpi" / "poisson_mpi"))
HAVE_MPI = MPI_SOLVER.is_file() and os.access(MPI_SOLVER, os.X_OK) and shutil.which("mpiexec") is not None
requires_mpi = pytest.mark.skipif(not HAVE_MPI, reason="MPI driver or launcher unavailable (see docs/MPI.md)")


def pytest_addoption(parser):
    parser.addoption("--require-mpi", action="store_true", help="require the MPI executable and launcher")
    parser.addoption("--require-mfem", action="store_true",
                     help="fail before collection if the real solver cannot be executed")


def pytest_configure(config):
    if config.getoption("--require-mpi") and not HAVE_MPI:
        raise pytest.UsageError(f"--require-mpi needs an executable {MPI_SOLVER} and mpiexec")
    if config.getoption("--require-mfem") and not HAVE_SOLVER:
        raise pytest.UsageError(f"--require-mfem needs an executable solver at {solver_binary()}")


@pytest.fixture
def specs_dir() -> Path:
    return ROOT / "specs"
