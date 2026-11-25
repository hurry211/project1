"""Validate the fixed-field comparisons, retained inputs and serialized evidence."""
from fractions import Fraction as Q
import json
from pathlib import Path

import pytest

from research.cfac.rt1_study import GRID, replay


STUDY = Path(__file__).resolve().parents[1] / "evidence" / "cfac" / "rt1-study.json"


def load_study():
    return json.loads(STUDY.read_bytes())


def test_all_archived_thresholds_recheck_without_running_the_patch_producer():
    data = load_study()
    replay(data)
    for case in data["cases"].values():
        assert [Q(row["epsilon"]) for row in case["rows"]] == list(GRID)
        assert case["squared_energies"]["rt1_original_residual_sq"] == "0"


@pytest.mark.parametrize("name,rt0_floor,rt1_upper,accepted_grid", [
    ("primary", Q(5, 192), Q(245989, 11059200), [Q(3, 20), Q(31, 200), Q(4, 25)]),
    ("replication", Q(47, 3240), Q(673, 57600), [Q(11, 100), Q(23, 200), Q(3, 25)]),
])
def test_rt1_succeeds_below_proved_rt0_floor(name, rt0_floor, rt1_upper, accepted_grid):
    case = load_study()["cases"][name]
    bounds = case["squared_energies"]
    assert Q(bounds["rt0_pair_floor_sq"]) == rt0_floor
    assert Q(bounds["patch_rt1_upper_sq"]) == rt1_upper
    successes = [row for row in case["rows"] if row["avoids_refinement"]]
    assert [Q(row["epsilon"]) for row in successes] == accepted_grid
    for row in successes:
        assert row["rt0_floor_verdict"] == "PAIR_FUTILE"
        assert rt1_upper <= Q(row["epsilon"])**2 < rt0_floor
        assert row["accepted"] == {"patch_rt1": True, "refined_optimal_rt0": True,
                                   "tree": False, "optimal_rt0": False}


def test_inexact_and_null_controls():
    cases = load_study()["cases"]
    for name in ("inexact", "null"):
        assert not any(row["avoids_refinement"] for row in cases[name]["rows"])
    values = cases["inexact"]["squared_energies"]
    assert Q(values["primal_alg_sq"]) > 0
    assert Q(values["patch_rt1_upper_sq"]) > Q(values["optimal_rt0_upper_sq"])
    assert all(all(row["accepted"].values()) for row in cases["null"]["rows"])


def test_changed_physical_moment_cannot_replay_saved_verdicts():
    data = load_study()
    moments = data["cases"]["primary"]["fields"]["rt1_cell"]
    moments[0][0] = str(Q(moments[0][0]) + Q(1, 97))
    with pytest.raises(ValueError, match="missing exact-equilibrium"):
        replay(data)


def test_threshold_records_cannot_be_reordered():
    data = load_study()
    data["cases"]["primary"]["rows"].reverse()
    with pytest.raises(ValueError, match="mismatched rows"):
        replay(data)
