"""Source mesh contracts and replay independence for the Gmsh import path."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from conftest import ROOT, requires_mfem
from sciagent import controller
from sciagent.gmsh import ARCHIVED_MESH
from verifier.verifier import verify_attempt


TRIANGLE = (ROOT / "specs/stage2/unit_square_tri.msh").read_bytes()
ATTRIBUTES = {"bottom": 11, "right": 22, "top": 33, "left": 44}


def write_study(directory, data, *, sha256=None, levels=(4, 8, 16)):
    path = directory / "source.msh"
    path.write_bytes(data)
    spec = json.loads((ROOT / "specs/poisson_sin_sin_p1_tri.json").read_text())
    spec["schema_version"] = "0.3"
    for bc in spec["problem"]["boundaries"]:
        bc["attribute"] = ATTRIBUTES[bc["label"]]
    spec["execution"]["mesh_levels"] = levels
    spec["external_mesh"] = {"file": path.name, "sha256": sha256 or hashlib.sha256(data).hexdigest()}
    request = directory / "request.json"
    request.write_text(json.dumps(spec))
    return request


@pytest.mark.parametrize("old,new,reason", [
    (b"2.2 0 8", b"4.1 0 8", "2.2 ASCII"),
    (b"2.2 0 8", b"2.2 1 8", "2.2 ASCII"),
    (b"$Nodes\n4\n", b"$Nodes\n4 \n", "integer"),
    (b"$EndNodes", b"$EndNodes\x0c", "control character"),
    (b'11 "bottom"', b'22 "bottom"', "duplicate or out-of-range"),
    (b'11 "bottom"', b'11 "south"', "boundary names and tags"),
    (b"4 0 1 0", b"4 0 0.9 0", "regular input lattice"),
    (b"4 0 1 0", b"4 0 1 1", "planar unit square"),
    (b"4 0 1 0", b"4 0 0 0", "one Cartesian vertex"),
    (b"4 1 2 44 4 4 1", b"4 1 2 11 4 4 1", "incorrectly tagged"),
    (b"4 1 2 44 4 4 1", b"4 1 2 11 4 1 2", "duplicate"),
    (b"4 1 2 44 4 4 1", b"4 1 2 44 4 3 1", "interior"),
    (b"6 2 2 1 1 1 3 4", b"6 2 2 1 1 1 2 3", "complementary triangles"),
    (b"6 2 2 1 1 1 3 4", b"6 2 2 1 1 1 4 3", "inverted"),
    (b"6 2 2 1 1 1 3 4", b"6 2 2 9 1 1 3 4", "physical surface"),
    (b"6 2 2 1 1 1 3 4", b"6 9 2 1 1 1 3 4", "homogeneous linear"),
    (b"6 2 2 1 1 1 3 4", b"6 2 2 1 1 1 3 9", "unknown element node"),
    (b"6 2 2 1 1 1 3 4", b"5 2 2 1 1 1 3 4", "unique element id"),
    (b"$EndElements", b"$EndElements\n$Periodic", "extra Gmsh sections"),
])
def test_gmsh_rejected_before_probe(tmp_path, monkeypatch, old, new, reason):
    data = TRIANGLE.replace(old, new)
    assert data != TRIANGLE
    request = write_study(tmp_path, data)
    monkeypatch.setattr(controller.environment, "probe", lambda: pytest.fail("environment probed before preflight"))
    manifest = controller.run_study(request, tmp_path / "runs")
    assert manifest["final_code"] == "INVALID_EXTERNAL_MESH"
    assert reason in manifest["final_reason"]
    assert not manifest["attempts"]


@pytest.mark.mfem
@requires_mfem
def test_checkerboard_coarse_mesh_with_crlf(tmp_path):
    # Sparse node labels on a 2x2 base. The diagonals alternate by coarse cell.
    nodes = [f"{10 + i + 3*j} {i/2} {j/2} 0" for j in range(3) for i in range(3)]
    elements = []
    for i in range(2):
        for attr, a, b in [(11, 10+i, 11+i), (22, 12+3*i, 15+3*i),
                           (33, 17+i, 16+i), (44, 13+3*i, 10+3*i)]:
            elements.append(f"1 2 {attr} 1 {a} {b}")
    for j in range(2):
        for i in range(2):
            sw = 10 + i + 3*j
            se, nw, ne = sw + 1, sw + 3, sw + 4
            triangles = [(sw, se, ne), (sw, ne, nw)] if i == j else [(sw, se, nw), (se, ne, nw)]
            elements.extend(f"2 2 1 1 {a} {b} {c}" for a, b, c in triangles)
    rows = ["$Nodes", "9", *nodes, "$EndNodes", "$Elements", str(len(elements)),
            *(f"{k} {row}" for k, row in enumerate(elements, 1)), "$EndElements", ""]
    header = TRIANGLE.split(b"$Nodes")[0].replace(b'"domain"', b'"sample surface"')
    data = (header + "\n".join(rows).encode()).replace(b"\n", b"\r\n")
    request = write_study(tmp_path, data, levels=(8, 16, 32))
    manifest = controller.run_study(request, tmp_path / "runs")
    assert manifest["final_state"] == "ACCEPTED", manifest["final_reason"]
    assert manifest["external_mesh"]["base_n"] == 2
    run = Path(manifest["run_dir"])
    assert (run / ARCHIVED_MESH).read_bytes() == data
    assert controller.reverify(run / "attempts/attempt_01")["verdict"] == "ACCEPTED"


@pytest.mark.parametrize("digest,levels,reason", [
    ("0" * 64, [4, 8, 16], "SHA-256"),
    (None, [3, 6, 12], "uniform refinement"),
])
def test_gmsh_pinning_and_refinement_plan(tmp_path, digest, levels, reason):
    request = write_study(tmp_path, TRIANGLE, sha256=digest, levels=levels)
    manifest = controller.run_study(request, tmp_path / "runs")
    assert (manifest["final_state"], manifest["final_code"]) == ("INVALID_SPEC", "INVALID_EXTERNAL_MESH")
    assert reason in manifest["final_reason"]
    assert not manifest["attempts"]


@pytest.mark.mfem
@requires_mfem
def test_opposite_triangle_diagonal_and_relocated_replay(tmp_path):
    # Keep geometry, groups, n and the exact solution fixed; change connectivity alone.
    alternative = TRIANGLE.replace(b"5 2 2 1 1 1 2 3", b"5 2 2 1 1 1 2 4").replace(
        b"6 2 2 1 1 1 3 4", b"6 2 2 1 1 2 3 4")
    request = write_study(tmp_path, alternative)
    manifest = controller.run_study(request, tmp_path / "runs")
    assert manifest["final_state"] == "ACCEPTED", manifest
    original = Path(manifest["run_dir"])
    run = shutil.copytree(original, tmp_path / "relocated")
    attempt = run / "attempts/attempt_01"
    (tmp_path / "source.msh").unlink()
    assert (run / ARCHIVED_MESH).read_bytes() == alternative
    digest = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    report = controller.reverify(attempt, digest)
    assert report["verdict"] == "ACCEPTED"
    assert report["input_sha256"][ARCHIVED_MESH] == hashlib.sha256(alternative).hexdigest()
    for level in report["levels"]:
        assert any(c["name"] == "external_mesh_refinement" and c["passed"] for c in level["checks"])
        cmd = json.loads((attempt / "levels" / f"n{level['n']:04d}" / "command.json").read_text())["argv"]
        assert Path(cmd[cmd.index("--mesh") + 1]) == original / ARCHIVED_MESH
    (run / ARCHIVED_MESH).write_bytes(TRIANGLE)
    assert controller.reverify(attempt, digest)["failure_codes"] == ["EVIDENCE_INTEGRITY"]

    # In the numerical replay alone, a valid but different mesh request must also fail.
    archived_request = run / "request/study_spec.json"
    swapped = json.loads(archived_request.read_text())
    swapped["external_mesh"]["sha256"] = hashlib.sha256(TRIANGLE).hexdigest()
    archived_request.write_text(json.dumps(swapped))
    numerical = verify_attempt(attempt, require_run_records=True)
    assert numerical["failure_codes"] == ["MESH_MISMATCH"]
    assert all([c["name"] for c in lv["checks"] if not c["passed"]] == ["external_mesh_refinement"]
               for lv in numerical["levels"])
