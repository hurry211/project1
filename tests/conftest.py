import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sciagent.mfem_runner import solver_binary  # noqa: E402

HAVE_SOLVER = solver_binary().is_file() and os.access(solver_binary(), os.X_OK)
requires_mfem = pytest.mark.skipif(not HAVE_SOLVER, reason="poisson_mms binary not built (run scripts/build_solver.sh)")


def pytest_addoption(parser):
    parser.addoption("--require-mfem", action="store_true",
                     help="fail before collection if the real solver cannot be executed")


def pytest_configure(config):
    if config.getoption("--require-mfem") and not HAVE_SOLVER:
        raise pytest.UsageError(f"--require-mfem needs an executable solver at {solver_binary()}")


@pytest.fixture
def specs_dir() -> Path:
    return ROOT / "specs"
