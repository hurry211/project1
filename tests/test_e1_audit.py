import argparse
from copy import deepcopy
import json
import os

import pytest

from benchmarks.e1 import block_schedule, campaign, workload_definitions
from benchmarks.e1_audit import audit, validate_schedule
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary


def test_audit_requires_seeded_slots_and_complete_counts():
    workloads = {"C1": workload_definitions("compact")["C1"]}
    cfg = dict(workloads=workloads, schema_version=1, scale="compact", seed=18, warmups=0, repetitions=2)
    rows = [{"workload": name, "block": b, "arm": a, "warmup": False, "code": "OK", "eligible": True,
             "elapsed_seconds": 1., "directory": f"{name}_block{b:03d}_{a}"}
            for name, b, a in block_schedule(workloads, 2, 0, 18)]
    data = dict(workloads=workloads, status="COMPLETE", records=rows, completed_trials=6,
                scheduled_trials=6, stopped_arms=[])
    validate_schedule(data, cfg)
    missing = dict(data, records=rows[:-1], completed_trials=5)
    with pytest.raises(ValueError, match="incomplete schedule"):
        validate_schedule(missing, cfg)
    validate_schedule(dict(missing, status="BUDGET_EXHAUSTED"), cfg)
    crossing = deepcopy(data)
    crossing["records"][-1] = crossing["records"][0]
    with pytest.raises(ValueError, match="assignment"):
        validate_schedule(crossing, cfg)
    crossing["records"][-1] = dict(rows[-1], directory=rows[0]["directory"])
    with pytest.raises(ValueError, match="assignment"):
        validate_schedule(crossing, cfg)


@pytest.mark.mfem
@pytest.mark.skipif(not solver_binary().is_file(), reason="requires serial MFEM")
def test_all_worker_receipts_and_inventories_checked_before_representative_replay(tmp_path):
    root = tmp_path / "e1"
    args = argparse.Namespace(directory=root, scale="compact", workloads=["C1"], cpu=min(os.sched_getaffinity(0)),
                              seed=18, repetitions=2, warmups=0, study_timeout=180., wall_budget=300.,
                              min_free_gib=0., max_output_gib=1., compact_archives=True)
    data = campaign(args)
    pin = sha256_file(root / "campaign.json")
    valid = audit(root, pin)
    assert valid["inventories_checked"] == 6 and valid["replayed"] == 3
    secondary = data["records"][-1]
    assert secondary["block"] == 1
    trial = root / secondary["directory"]
    archive = trial / secondary["archive"]
    worker = json.loads((trial / "worker.json").read_bytes())
    worker["profile"]["worker_peak_rss_kib"] += 1
    for path, replacement, error in [
            (trial / "worker.json", json.dumps(worker).encode(), "worker receipt"),
            (trial / "worker-clocks.json", b'{}', "external clocks"),
            (archive / "manifest.json", b'{}', "SHA-256"),
            (archive / "attempts/attempt_01/levels/n0016/dof_values.txt", b'0\n', "inventory"),
    ]:
        original = path.read_bytes()
        # Replace the path so a shared hardlink in a representative trial stays unchanged.
        staged = path.with_name("changed.payload")
        staged.write_bytes(replacement)
        staged.replace(path)
        with pytest.raises(ValueError, match=error):
            audit(root, pin)
        path.write_bytes(original)
    assert audit(root, pin, all_trials=True)["replayed"] == 6
