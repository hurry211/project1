"""Diagnostic replay using physical per-element Jacobians and the same verified nodal basis.

From the repository root: PYTHONPATH=. python evidence/scaling-stage/geometry_control.py ATTEMPT MANIFEST_SHA256
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def worker(attempt, memory_mb):
    import resource
    limit = int(memory_mb) * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    from dataclasses import replace
    from unittest.mock import patch
    from verifier import verifier

    quadrature = verifier.element_quadrature

    def per_element(mesh, basis, points):
        return quadrature(mesh, replace(basis, translated=False), points)

    with patch.object(verifier, "element_quadrature", per_element):
        result = verifier.verify_attempt(attempt, require_run_records=True)
    print(json.dumps(result, allow_nan=False))


def report(attempt, digest):
    from sciagent.evidence import check_archive
    from sciagent.spec import RetryPolicy, VERIFIER_MEMORY_MB, VERIFIER_TIMEOUT_SECONDS

    mode, saved, changed = check_archive(attempt, digest)
    if mode != "sha256-inventory" or changed or saved["verdict"] != "ACCEPTED":
        raise ValueError("requires a complete, externally pinned, accepted archive")
    policy = RetryPolicy.model_validate_json((attempt.parent.parent / "frozen/policy.json").read_text())
    memory = policy.verifier_memory_mb or VERIFIER_MEMORY_MB
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(attempt), str(memory)],
                          text=True, capture_output=True, env=env, check=True,
                          timeout=policy.verifier_timeout_seconds or VERIFIER_TIMEOUT_SECONDS)
    try:
        control = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(f"invalid diagnostic output: {proc.stdout[:240]!r}, stderr: {proc.stderr[-400:]!r}") from exc
    if check_archive(attempt, digest)[2] or saved["input_sha256"] != control["input_sha256"]:
        raise ValueError("archive or verifier inputs changed")
    metrics = ("n", "l2_error", "h1_semi_error", "reference_rel_diff")
    return {"archive": attempt.parent.parent.name, "manifest_sha256": digest, "memory_mb": memory,
            "control": "evaluate quadrature separately at physical vertices, retain verified basis coefficients",
            "verdict": control["verdict"], "failure_codes": control["failure_codes"],
            "levels": [{k: lv[k] for k in metrics} for lv in control["levels"]]}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 3:
        print(json.dumps(report(Path(sys.argv[1]).resolve(), sys.argv[2]), indent=2))
    else:
        raise SystemExit(__doc__)
