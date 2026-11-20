import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sciagent.mfem_runner import solver_binary  # noqa: E402

HAVE_SOLVER = solver_binary().is_file()
requires_mfem = pytest.mark.skipif(not HAVE_SOLVER, reason="poisson_mms binary not built (run scripts/build_solver.sh)")


@pytest.fixture
def specs_dir() -> Path:
    return ROOT / "specs"
