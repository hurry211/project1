"""Stage 2: comparison logic and native smoke; original capabilities stay frozen."""
import pytest

from benchmarks.performance import compare_pair, run
from conftest import requires_mfem


def trial(assembly, seconds):
    return {"assembly": assembly, "verdict": "ACCEPTED",
            "end_to_end_seconds": seconds,
            "levels": [{"n": 8, "dofs": 81, "l2": .01, "h1": .1}]}


def test_pair_speedup():
    assert compare_pair(trial("full", 4), trial("partial", 2)) == 2


def test_pair_rejects_different_error():
    a, b = trial("full", 4), trial("partial", 2)
    b["levels"][0]["l2"] *= 2
    with pytest.raises(ValueError, match="verified l2"):
        compare_pair(a, b)


def test_pair_rejects_unverified():
    a, b = trial("full", 4), trial("partial", 2)
    b["verdict"] = "REJECTED"
    with pytest.raises(ValueError, match="unverified"):
        compare_pair(a, b)


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("order", [1, 2])
def test_native_fair_comparison(tmp_path, order):
    import json
    output = tmp_path / "comparison"
    run(output, 1, order, [8, 16, 32], 120, 3072)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "COMPLETE"
    assert len(report["pairs"]) == 1
    for mode in ("full", "partial"):
        assert report["pairs"][0][mode]["verdict"] == "ACCEPTED"
