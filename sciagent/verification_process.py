"""Supervise an independent verification job under its own wall and memory limits."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .spec import VERIFIER_MEMORY_MB, VERIFIER_TIMEOUT_SECONDS, RetryPolicy


def verify_bounded(attempt: Path, policy: RetryPolicy, *, remaining: float | None = None,
                   reused: dict | None = None) -> tuple[dict, dict]:
    timeout = policy.verifier_timeout_seconds or VERIFIER_TIMEOUT_SECONDS
    if remaining is not None:
        timeout = min(timeout, remaining)
    memory = policy.verifier_memory_mb or VERIFIER_MEMORY_MB
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    start = time.monotonic()
    metrics = {"timeout_seconds": timeout, "memory_mb": memory, "peak_rss_kib": None}
    command = [sys.executable, "-m", "sciagent.verification_worker", str(attempt), str(memory)]
    try:
        proc = subprocess.run(command, input=json.dumps(reused or {}), capture_output=True,
                              text=True, timeout=timeout, env=env)
        metrics["exit_code"] = proc.returncode
        metrics["stderr"] = proc.stderr[-4000:]
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            metrics["peak_rss_kib"] = data["peak_rss_kib"]
            return data["verdict"], metrics
        code = "VERIFIER_MEMORY_LIMIT" if proc.returncode == 2 else "VERIFIER_PROCESS_FAILED"
    except subprocess.TimeoutExpired:
        code = "VERIFIER_TIMEOUT"
    except (OSError, ValueError, KeyError) as exc:
        code = "VERIFIER_PROCESS_FAILED"
        metrics["stderr"] = str(exc)
    finally:
        metrics["wall_seconds"] = time.monotonic() - start
    return {"verdict": "REJECTED", "statement": "bounded verification did not complete",
            "failure_codes": [code], "levels": [], "convergence": {"pairs": []},
            "input_sha256": {}}, metrics
