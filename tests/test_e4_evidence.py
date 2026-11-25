"""Keep the public numerical claims and GPU replay inputs tied to their receipts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.gpu_mpi_norms import agrees, rank_data, triangle_rule

ROOT = Path(__file__).resolve().parents[1]


def test_historical_calibration_report_matches_published_data():
    report = json.loads((ROOT / "evidence/e3/adaptive_comparison.json").read_bytes())
    text = (ROOT / "docs/E3_PERFORMANCE_ADAPTIVE.md").read_text(encoding="utf-8")
    stats = report["predictor"]["train_stats"]
    assert f'{stats["n_samples"]} valid samples from {len(report["calibration_observations"])} observations' in text
    for key in ("log10_l2_rmse", "log10_time_rmse", "log10_mem_rmse"):
        assert f"{key}={stats[key]:.4f}" in text


def test_gpu_report_is_bound_to_real_rank_receipt_and_kernel():
    saved = json.loads((ROOT / "evidence/e4/gpu-mpi-norms.json").read_bytes())
    receipt = (ROOT / "evidence/e4/gpu-mpi-input.json").read_bytes()
    assert hashlib.sha256(receipt).hexdigest() == saved["mpi_receipt_sha256"]
    assert hashlib.sha256((ROOT / "benchmarks/mpi_field_norms.cl").read_bytes()).hexdigest() == saved["kernel_sha256"]
    for level, original in zip(saved["levels"], json.loads(receipt)["levels"], strict=True):
        assert level["n"] == original["n"]
        assert level["triangles"] == original["global_elements"]
        assert agrees(level["gpu_norms"], [original["l2_error"], original["h1_semi_error"]])
        assert not agrees(level["wrong_field_control_norms"], level["cpu_receipt_norms"])


def test_gpu_rank_input_digest_and_quadrature(tmp_path):
    xi, eta, w = triangle_rule().T
    assert w.sum() == pytest.approx(0.5)
    assert (w*xi*eta).sum() == pytest.approx(1/24)
    level = tmp_path / "n1"
    level.mkdir()
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype="<f8")
    cells = np.array([[0, 1, 2]], dtype="<i4")
    inventory = {}
    for name, array in (("nodes.0.f64", nodes), ("cells.0.i32", cells)):
        (level / name).write_bytes(array.tobytes())
        inventory[f"n1/{name}"] = hashlib.sha256(array.tobytes()).hexdigest()
    actual_nodes, actual_cells = rank_data(tmp_path, 1, 0, inventory)
    np.testing.assert_array_equal(actual_nodes, nodes)
    np.testing.assert_array_equal(actual_cells, cells)
    nodes[2, 2] = 1
    (level / "nodes.0.f64").write_bytes(nodes.tobytes())
    with pytest.raises(ValueError, match="digest"):
        rank_data(tmp_path, 1, 0, inventory)
