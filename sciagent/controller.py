"""Deterministic controller: explicit state machine with a bounded retry.

States:
    RECEIVED -> SPEC_VALIDATED -> ENVIRONMENT_CHECKED -> READY
      -> (RUNNING -> VERIFYING -> {ACCEPTED | RETRY_ALLOWED -> RUNNING ...})
    terminal: ACCEPTED, FAILED, BLOCKED, BUDGET_EXHAUSTED, INVALID_SPEC

The only corrective action is ``append_refined_level`` (add mesh level 2n),
taken only when the verifier returns RETRY_ELIGIBLE (accuracy target missed
while every other check passed). The ProblemSpec, ValidationSpec and
RetryPolicy are frozen and hash-checked before every attempt.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
from pathlib import Path

from pydantic import ValidationError

from verifier.verifier import ACCEPTED, RETRY_ELIGIBLE, check_mms_consistency, verify_attempt

from . import capability, environment, mfem_runner
from .evidence import EvidenceStore, check_archive, sha256_file, utc_now
from .gmsh import ARCHIVED_MESH, inspect_gmsh, read_mesh
from .spec import ExecutionSpec, StudySpec, canonical_json, sha256_of

TERMINAL = {"ACCEPTED", "FAILED", "BLOCKED", "BUDGET_EXHAUSTED", "INVALID_SPEC"}
STRUCTURAL_EXITS = {"INVALID_ARGUMENTS", "BOUNDARY_COVERAGE", "UNSUPPORTED_COMBINATION", "ARTIFACT_WRITE_ERROR", "TIMEOUT"}


def _new_run_dir(runs_dir: Path, label: str) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(runs_dir) / f"{stamp}_{label}"


def run_study(spec_path: str | Path, runs_dir: str | Path = "runs") -> dict:
    spec_path = Path(spec_path)
    raw = spec_path.read_text()
    label = spec_path.stem[:60]
    store = EvidenceStore(_new_run_dir(Path(runs_dir), label))
    store.write_once("request/study_spec.json", raw)
    store.event("RECEIVED", spec_file=str(spec_path), sha256=hashlib.sha256(raw.encode()).hexdigest())
    started = time.monotonic()
    manifest: dict = {"run_dir": str(store.run_dir), "spec_file": str(spec_path),
                      "started_utc": utc_now(), "attempts": []}

    def finish(state: str, code: str, reason: str) -> dict:
        store.event(state, code=code, reason=reason)
        manifest.update({"final_state": state, "final_code": code, "final_reason": reason,
                         "finished_utc": utc_now(), "total_wall_seconds": round(time.monotonic() - started, 3)})
        store.finish_manifest(manifest)
        return manifest

    # ---- SPEC_VALIDATED -----------------------------------------------------
    try:
        study = StudySpec.model_validate_json(raw)
    except ValidationError as exc:
        errs = [{"loc": list(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()]
        manifest["spec_errors"] = errs
        return finish("INVALID_SPEC", "INVALID_SPEC", f"{len(errs)} schema error(s): {errs[0]['msg']}")
    mms = check_mms_consistency(study.problem, study.validation.mms_consistency_rel_tol)
    if not mms.passed:
        return finish("INVALID_SPEC", mms.code, f"{mms.name}: {mms.detail} (value={mms.value})")

    mesh = None
    if study.external_mesh is not None:
        try:
            data = read_mesh(spec_path.parent / study.external_mesh.file, study.external_mesh)
            mesh = inspect_gmsh(data, study.problem, study.execution)
        except (OSError, ValueError) as exc:
            return finish("INVALID_SPEC", "INVALID_EXTERNAL_MESH", str(exc))
        store.write_bytes_once(ARCHIVED_MESH, data)
        manifest["external_mesh"] = {"sha256": study.external_mesh.sha256, "base_n": mesh.subdivisions}

    frozen = {"problem": study.problem, "validation": study.validation, "policy": study.policy}
    hashes = {}
    for name, model in frozen.items():
        store.write_once(f"frozen/{name}.json", canonical_json(model))
        hashes[name] = sha256_of(model)
    manifest["frozen_sha256"] = hashes
    store.event("SPEC_VALIDATED", frozen_sha256=hashes)

    # ---- ENVIRONMENT_CHECKED --------------------------------------------------
    env = environment.probe()
    store.write_json_once("environment.json", env)
    manifest["environment"] = {"mfem_version": env["mfem"].get("version"), "mfem_git": env["mfem"].get("git"),
                               "solver_binary_sha256": env["solver_binary_sha256"], "cxx": env["cxx"]}
    store.event("ENVIRONMENT_CHECKED", mfem_found=env["mfem"]["found"],
                solver_binary=env["solver_binary_sha256"] is not None)
    if not env["mfem"]["found"]:
        return finish("BLOCKED", "MFEM_NOT_FOUND", "; ".join(env["mfem"]["notes"]))
    if env["solver_binary_sha256"] is None:
        return finish("BLOCKED", "BUILD_MISSING", f"solver binary not found at {env['solver_binary']}")

    # ---- READY: capability + case binding -------------------------------------
    try:
        cases = mfem_runner.list_cases()
    except mfem_runner.BuildError as exc:
        return finish("BLOCKED", "BUILD_FAILURE", str(exc))
    store.write_json_once("case_registry.json", cases)
    case_id = mfem_runner.bind_case(study.problem, cases)
    mesh_source = "gmsh22" if study.external_mesh is not None else "cartesian"
    decision = capability.evaluate(study.execution, env, case_id=case_id, mesh_source=mesh_source, catalog=cases)
    manifest["capability"] = decision.__dict__
    if decision.status != "tested":
        state = "BLOCKED" if decision.code in {"UNSUPPORTED_BACKEND", "NOT_VERIFIED_BUILD"} else "FAILED"
        return finish(state, decision.code, decision.reason)
    if case_id is None:
        return finish("FAILED", "UNSUPPORTED_PROBLEM",
                      "the compiled catalog has no matching formulation and manufactured expression pair")
    manifest["bound_case"] = case_id
    store.event("READY", case=case_id, capability=decision.code)

    # ---- attempt loop ---------------------------------------------------------
    pol = study.policy
    execution = study.execution
    for attempt in range(1, pol.max_attempts + 1):
        if execution.estimated_dofs(execution.mesh_levels[-1]) > pol.max_dofs_per_level:
            return finish("BUDGET_EXHAUSTED", "RESOURCE_LIMIT", "requested mesh exceeds max_dofs_per_level")
        adir = f"attempts/attempt_{attempt:02d}"
        store.write_once(f"{adir}/execution.json", canonical_json(execution))
        pinned = {store.path("frozen", f"{name}.json"): h for name, h in hashes.items()}
        pinned[store.path(adir, "execution.json")] = sha256_of(execution)
        if mesh is not None:
            pinned[store.path(ARCHIVED_MESH)] = study.external_mesh.sha256
        if any(sha256_file(p) != h for p, h in pinned.items()):
            return finish("FAILED", "SPEC_TAMPERED", "frozen inputs changed before solver execution")
        store.event("RUNNING", attempt=attempt, mesh_levels=list(execution.mesh_levels),
                    execution_sha256=sha256_of(execution))
        level_runs = []
        for n in execution.mesh_levels:
            elapsed = time.monotonic() - started
            remaining = pol.max_total_wall_seconds - elapsed
            if remaining <= 0:
                manifest["attempts"].append({"attempt": attempt, "levels": level_runs, "verdict": None})
                return finish("BUDGET_EXHAUSTED", "RESOURCE_LIMIT", f"wall budget exceeded ({elapsed:.1f}s)")
            mesh_args = {"mesh_path": store.path(ARCHIVED_MESH), "refinements": mesh.refinements(n)} if mesh else {}
            lr = mfem_runner.run_level(study.problem, execution, case_id, n,
                                       store.path(adir, "levels", f"n{n:04d}"),
                                       timeout_s=min(pol.per_level_timeout_seconds, remaining),
                                       mem_mb=pol.per_level_memory_mb, **mesh_args)
            level_runs.append({"n": n, "exit_code": lr.exit_code, "exit_meaning": lr.exit_meaning,
                               "wall_seconds": lr.wall_seconds})
            store.event("LEVEL_DONE", attempt=attempt, n=n, exit=lr.exit_meaning, seconds=lr.wall_seconds)
            if lr.exit_meaning in STRUCTURAL_EXITS:
                break
        store.event("VERIFYING", attempt=attempt)
        if any(sha256_file(p) != h for p, h in pinned.items()):
            return finish("FAILED", "SPEC_TAMPERED", "frozen inputs changed during solver execution")
        verdict = verify_attempt(store.path(adir), require_run_records=True)
        store.write_json_once(f"{adir}/verdict.json", verdict)
        summary = {
            "attempt": attempt, "mesh_levels": list(execution.mesh_levels), "levels": level_runs,
            "verdict": verdict["verdict"], "failure_codes": verdict["failure_codes"],
            "errors": [{"n": lv["n"], "l2": lv["l2_error"], "h1_semi": lv["h1_semi_error"],
                        "reference_rel_diff": lv["reference_rel_diff"]} for lv in verdict["levels"]],
            "orders": verdict["convergence"]["pairs"],
        }
        manifest["attempts"].append(summary)
        store.event("VERDICT", attempt=attempt, verdict=verdict["verdict"], codes=verdict["failure_codes"])

        if verdict["verdict"] == ACCEPTED:
            return finish("ACCEPTED", "OK", verdict["statement"])
        if verdict["verdict"] != RETRY_ELIGIBLE:
            exits = {lr["exit_meaning"] for lr in level_runs} - {"OK"}
            code = "SOLVER_DIVERGED" if "SOLVER_NOT_CONVERGED" in exits else "VALIDATION_FAILED"
            return finish("FAILED", code, f"verifier codes {verdict['failure_codes']}; solver exits {sorted(exits)}")
        # RETRY_ELIGIBLE: only the accuracy target failed.
        if attempt == pol.max_attempts:
            return finish("BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED",
                          f"accuracy target not met after {attempt} attempt(s) (max_attempts={pol.max_attempts})")
        n_next = 2 * execution.mesh_levels[-1]
        if execution.estimated_dofs(n_next) > pol.max_dofs_per_level:
            return finish("BUDGET_EXHAUSTED", "RESOURCE_LIMIT",
                          f"next level n={n_next} would need ~{execution.estimated_dofs(n_next)} dofs "
                          f"> max_dofs_per_level={pol.max_dofs_per_level}")
        new_exec = ExecutionSpec.model_validate({**execution.model_dump(), "mesh_levels": [*execution.mesh_levels, n_next]})
        store.event("RETRY_ALLOWED", attempt=attempt, action=pol.action, old_levels=list(execution.mesh_levels),
                    new_levels=list(new_exec.mesh_levels),
                    unchanged=["problem", "validation", "policy"], reason="ACCURACY_NOT_MET")
        execution = new_exec
    return finish("BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED", "attempt loop ended")  # pragma: no cover


def reverify(attempt_dir: str | Path, expected_digest: str | None = None) -> dict:
    """Check the completed archive and recompute the selected numerical verdict."""
    attempt = Path(attempt_dir)
    out = {"levels": [], "convergence": {"pairs": []}, "failure_codes": []}
    try:
        mode, saved, changed = check_archive(attempt, expected_digest)
        # Retain numerical diagnostics for damaged level files. A changed spec
        # or archive record must never authorize a different verification job.
        level_prefix = f"attempts/{attempt.name}/levels/"
        if any(not p.startswith(level_prefix) for p in changed):
            raise ValueError(f"changed archive files: {', '.join(changed)}")
        out = verify_attempt(attempt, require_run_records=True)
        if changed:
            raise ValueError(f"changed level files: {', '.join(changed)}")
        if saved["input_sha256"] != out["input_sha256"]:
            raise ValueError("stored verifier input hashes differ from the replay")
        if saved["verdict"] != out["verdict"]:
            raise ValueError("stored verdict differs from numerical replay")
        out["integrity"] = {"mode": mode, "passed": True}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.update(verdict="REJECTED", statement="archive verification failed",
                   failure_codes=sorted(set(out["failure_codes"]) | {"EVIDENCE_INTEGRITY"}),
                   integrity={"passed": False, "detail": str(exc)})
    return out


def summarize(manifest: dict) -> str:
    return json.dumps({k: manifest.get(k) for k in ("run_dir", "final_state", "final_code", "final_reason")}, indent=2)
