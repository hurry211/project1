import argparse
import json
import os

import pytest

from benchmarks import e1
from benchmarks.verified_revision import capture
from benchmarks.verified_revision_audit import audit
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary


@pytest.mark.mfem
@pytest.mark.skipif(not solver_binary().is_file(), reason="requires serial MFEM")
def test_relocated_pair_rechecks_complete_verdict_clocks_and_all_inventories(tmp_path):
    root = tmp_path / "capture"
    capture(argparse.Namespace(directory=root, scale="compact", workloads=["C1"], seed=1, blocks=2,
                               baseline=e1.ROOT, candidate=e1.ROOT, cpu=min(os.sched_getaffinity(0))))
    design_pin = sha256_file(root / "design.json")
    report_pin = sha256_file(root / "report.json")
    root = root.rename(tmp_path / "moved")
    verified = audit(root, design_pin, report_pin)
    assert verified["inventories_checked"] == 6
    assert {r["directory"] for r in verified["replayed"]} == {"C1_000_baseline", "C1_000_candidate"}
    raw = json.loads((root / "report.json").read_bytes())
    trial = root / raw["records"][-1]["directory"]
    archive = trial / raw["records"][-1]["archive"]
    receipt = json.loads((trial / "worker.json").read_bytes())
    receipt["levels"][0]["l2_error"] = 1.0
    for path, payload, error in [
        (trial / "worker.json", json.dumps(receipt).encode(), "receipts disagree"),
        (trial / "worker-clocks.json", b"{}", "endpoint timing"),
        (archive / "attempts/attempt_01/levels/n0016/dof_values.txt", b"field", "manifest inventory"),
    ]:
        original = path.read_bytes()
        # Replace this name without modifying any other archive sharing the bytes.
        staged = tmp_path / "replacement"
        staged.write_bytes(payload)
        staged.replace(path)
        with pytest.raises(ValueError, match=error):
            audit(root, design_pin, report_pin)
        path.write_bytes(original)
    raw["scores"]["C1"]["median_paired_delta_seconds"] += 1
    (root / "report.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest mismatch"):
        audit(root, design_pin, report_pin)
    with pytest.raises(ValueError, match="paired summary differs"):
        audit(root, design_pin, sha256_file(root / "report.json"))
