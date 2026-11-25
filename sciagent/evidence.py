"""Append-only run evidence store.

Layout of one run directory::

    runs/<run_id>/
      request/study_spec.json        original file as submitted
      frozen/problem.json            frozen ProblemSpec (hash in manifest)
      frozen/validation.json         frozen ValidationSpec
      frozen/policy.json             frozen RetryPolicy
      environment.json               environment probe
      events.jsonl                   append-only state transitions
      attempts/attempt_01/
        execution.json               ExecutionSpec actually used
        levels/n0008/...             raw solver artifacts + logs
        verdict.json                 independent verifier output
      manifest.json                  final summary (written once at the end)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

from .spec import StudySpec, sha256_of
from .gmsh import ARCHIVED_MESH, ARCHIVED_MFEM_MESH, archived_mesh_path


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    """Hash regular files by portable names. Never follow links in an archive."""
    hashes = {}
    for path in sorted(run_dir.rglob("*")):
        name = path.relative_to(run_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"unexpected symbolic link: {name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"expected a regular file: {name}")
        if name not in {"manifest.json", "manifest.sha256"}:
            hashes[name] = sha256_file(path)
    return hashes


def inspect_manifest(run: Path, expected_digest: str | None = None) -> tuple[dict, dict, set[str]]:
    """Read the manifest and compare its complete inventory when available."""
    actual = artifact_hashes(run)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_file(manifest_path)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("manifest.json differs from the supplied SHA-256")
    sidecar = run / "manifest.sha256"
    if "evidence_version" in manifest:
        if manifest["evidence_version"] != 1 or sidecar.read_text().strip() != f"{digest}  manifest.json":
            raise ValueError("unsupported evidence version or manifest checksum mismatch")
        expected = manifest["artifact_sha256"]
        if not isinstance(expected, dict):
            raise ValueError("invalid artifact inventory")
        changed = {p for p in actual.keys() | expected.keys() if actual.get(p) != expected.get(p)}
    elif sidecar.exists() or "artifact_sha256" in manifest:
        raise ValueError("missing evidence inventory")
    else:
        changed = set()
    return manifest, actual, changed


def check_archive(attempt: Path, expected_digest: str | None) -> tuple[str, dict, list[str]]:
    """Check a completed run before using its files for a numerical replay."""
    run = attempt.parent.parent
    manifest, actual, changed = inspect_manifest(run, expected_digest)
    summaries = manifest["attempts"]
    numbers = [a["attempt"] for a in summaries]
    if numbers != list(range(1, len(summaries) + 1)):
        raise ValueError("attempts in manifest.json are out of sequence")
    names = [f"attempt_{n:02d}" for n in numbers]
    if attempt.parent.name != "attempts" or attempt.name not in names:
        raise ValueError("attempt is not listed in manifest.json")
    stored = json.loads((attempt / "verdict.json").read_text())
    if not isinstance(stored, dict):
        raise ValueError("verdict.json must contain an object")
    mode = "sha256-inventory" if "evidence_version" in manifest else "legacy-input-hashes"
    if mode == "legacy-input-hashes" and stored.get("verifier_version") != "0.1.0":
        raise ValueError("missing evidence inventory")
    request = StudySpec.model_validate_json((run / "request/study_spec.json").read_text())
    expected_mesh = archived_mesh_path(request.external_mesh)
    for mesh_rel in (ARCHIVED_MESH, ARCHIVED_MFEM_MESH):
        exp_sha = request.external_mesh.sha256 if mesh_rel == expected_mesh else None
        if actual.get(mesh_rel) != exp_sha:
            changed.add(mesh_rel)
    for name in ("problem", "validation", "policy"):
        path = f"frozen/{name}.json"
        if actual.get(path) != manifest["frozen_sha256"][name] or actual.get(path) != sha256_of(getattr(request, name)):
            changed.add(path)
    index = names.index(attempt.name)
    if len(summaries) > request.policy.max_attempts or any(a["verdict"] != "RETRY_ELIGIBLE" for a in summaries[:index]):
        raise ValueError("attempt exceeds the original retry policy")
    if request.execution.mesh_topology == "amr":
        steps = request.execution.amr_max_steps
        dofs = request.execution.amr_max_dofs
        for _ in range(index):
            steps = min(20, steps + 2)
            dofs = min(request.policy.max_dofs_per_level, dofs * 2)
        expected_execution = request.execution.model_copy(update={"amr_max_steps": steps, "amr_max_dofs": dofs})
    else:
        levels = list(request.execution.mesh_levels)
        for _ in range(index):
            levels.append(2 * levels[-1])
        expected_execution = request.execution.model_copy(update={"mesh_levels": tuple(levels)})
    execution_path = f"attempts/{attempt.name}/execution.json"
    if actual.get(execution_path) != sha256_of(expected_execution):
        changed.add(execution_path)
    if summaries[index]["verdict"] != stored.get("verdict"):
        changed.add(f"attempts/{attempt.name}/verdict.json")
    return mode, stored, sorted(changed)


class EvidenceStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._events = self.run_dir / "events.jsonl"

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def write_once(self, rel: str, content: str) -> Path:
        p = self.run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return p

    def write_json_once(self, rel: str, obj) -> Path:
        return self.write_once(rel, json.dumps(obj, indent=2, sort_keys=False) + "\n")

    def write_bytes_once(self, rel: str, data: bytes) -> Path:
        path = self.run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return path

    def finish_manifest(self, manifest: dict) -> None:
        manifest["evidence_version"] = 1
        manifest["artifact_sha256"] = artifact_hashes(self.run_dir)
        path = self.write_json_once("manifest.json", manifest)
        self.write_once("manifest.sha256", f"{sha256_file(path)}  manifest.json\n")

    def event(self, state: str, **data) -> dict:
        ev = {"t": utc_now(), "state": state, **data}
        with self._events.open("a") as f:
            f.write(json.dumps(ev) + "\n")
        return ev
