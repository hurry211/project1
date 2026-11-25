import json
import random

import pytest

from benchmarks.verified_revision import numerical_digest, ordered_pairs
from sciagent.evidence import sha256_file


def test_order_balances_first_position_and_pairs_revisions():
    schedule = list(ordered_pairs(12, random.Random(17)))
    assert len(schedule) == 12
    assert schedule.count(["baseline", "candidate"]) == schedule.count(["candidate", "baseline"]) == 6
    assert schedule == list(ordered_pairs(12, random.Random(17)))


def write_sample(folder, *, timing, residual):
    level = folder / "levels/n0002"
    level.mkdir(parents=True)
    result = level / "result.json"
    result.write_text(json.dumps({"time_setup_ms": timing, "time_solve_ms": 3, "time_error_ms": 2,
                                 "relative_residual": residual}))
    data = {"levels": [{"n": 2, "level_dir": str(level), "checks": [{"passed": True}]}],
            "input_sha256": {"levels/n0002/result.json": sha256_file(result),
                             "levels/n0002/dof_values.txt": "pinned-field"}}
    (folder / "verdict.json").write_text(json.dumps(data))
    return folder


def test_fresh_study_comparison_checks_native_science_and_retains_field_pins(tmp_path):
    first = write_sample(tmp_path / "a", timing=10, residual=1e-12)
    second = write_sample(tmp_path / "b", timing=4, residual=1e-12)
    third = write_sample(tmp_path / "c", timing=10, residual=1e-8)
    assert numerical_digest(first) == numerical_digest(second)
    assert numerical_digest(first) != numerical_digest(third)
    (second / "levels/n0002/result.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        numerical_digest(second)
    path = first / "verdict.json"
    verdict = json.loads(path.read_text())
    original = numerical_digest(first)
    verdict["input_sha256"]["levels/n0002/dof_values.txt"] = "different-field"
    path.write_text(json.dumps(verdict))
    assert numerical_digest(first) != original
