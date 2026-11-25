"""Compare counterfactual RT1 fields with the separate algebraic predicate."""
from fractions import Fraction

from research.cfac.checkerboard_sweeps import inspect_sweeps


def test_checkerboard_miss_and_balance_have_different_stopping_points():
    diagnostic = inspect_sweeps()
    fields = diagnostic["fields"]
    assert fields[0]["state_digest"] == diagnostic["initial_parent_digest"]
    assert len({f["state_digest"] for f in fields}) == len(fields)
    assert [f["extra_sweeps"] for f in fields if f["rt1_accepted"]] == [3, 4, 5, 6, 7]
    assert [f["extra_sweeps"] for f in fields if f["balanced"]] == [7]
    for field in fields:
        assert field["rt1_accepted"] == (Fraction(field["rt1_upper_sq"]) <= Fraction(diagnostic["epsilon"])**2)
        assert field["balanced"] == (4 * Fraction(field["algebraic_upper_sq"])
                                     <= Fraction(field["discretization_lower_sq"]))
