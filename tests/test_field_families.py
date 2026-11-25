import numpy as np
import pytest

from conftest import requires_mfem
from sciagent import field_study
from verifier.field_family import verify_level


@pytest.mark.parametrize("family", ["heat", "wave", "elasticity"])
@pytest.mark.mfem
@requires_mfem
def test_real_three_dimensional_fields_and_independent_negative_controls(tmp_path, family):
    cfg = field_study.request(family, 3, 2, [2, 4, 8])
    root = tmp_path / family
    outcome = field_study.capture(root, cfg)
    assert outcome["verdict"] == "ACCEPTED"
    assert field_study.replay(root, outcome["manifest_sha256"])["verdict"] == "ACCEPTED"
    level = root / "n0008/field.tsv"
    field = np.loadtxt(level)
    i = np.argmin(np.sum((field[:, :3] - 0.5)**2, axis=1))
    # A wrong half-time snapshot must fail even with an intact final field.
    field[i, 3] += 0.01
    np.savetxt(level, field, fmt="%.17g")
    report = verify_level(level, family, 3, 8, 2, 32)
    assert not report["passed"] and report["independent_error"] > report["independent_limit"]
    with pytest.raises(ValueError, match="inventory"):
        field_study.replay(root, outcome["manifest_sha256"])
    field[0, :3] = field[1, :3]
    np.savetxt(level, field, fmt="%.17g")
    with pytest.raises(ValueError, match="lattice"):
        verify_level(level, family, 3, 8, 2, 32)


@pytest.mark.parametrize("args", [("wave", 4, 1, [2, 4, 8]), ("heat", 3, 3, [2, 4, 8]),
                                  ("unknown", 3, 2, [2, 4, 8]), ("heat", 3, 2, [2, 6, 8])])
def test_registry_declines_unreviewed_studies(args):
    with pytest.raises(ValueError):
        field_study.request(*args)
