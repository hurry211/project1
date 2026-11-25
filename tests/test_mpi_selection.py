from copy import deepcopy

import pytest

from benchmarks.mpi_scaling import choose_arm, schedule


def panel(repetitions=5):
    times = {"serial": 10.0, "mpi1": 6.5, "mpi2": 5.0, "mpi4": 4.8, "mpi4_tsv": 9.0}
    return [{"arm": arm, "block": block, "verified_seconds": times[arm], "verdict": "ACCEPTED"}
            for block, arm in schedule(repetitions, 73)]


def test_only_qualified_paired_mpi_times_support_switching():
    observed = panel()
    assert choose_arm(observed) == "mpi4"
    noisy = deepcopy(observed)
    for row in noisy:
        if row["block"] >= 0 and row["arm"] != "serial":
            row["verified_seconds"] = 9.7
    assert choose_arm(noisy) == "serial"
    assert choose_arm(panel(3)) == "serial"
    observed[0]["verdict"] = "REJECTED"
    with pytest.raises(ValueError, match="ineligible"):
        choose_arm(observed)


def test_cli_fails_closed_before_running_without_calibration(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from sciagent.__main__ import main
    destination = tmp_path / "selected"
    monkeypatch.setattr(sys, "argv", ["sciagent", "select-mpi", "--calibration", str(tmp_path),
                                      "--report-sha256", "0"*64, "--directory", str(destination)])
    assert main() == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "REJECTED"
    assert not destination.exists()
