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
import json
from pathlib import Path


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


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
        if p.exists():
            raise FileExistsError(f"evidence file already exists (append-only store): {p}")
        p.write_text(content)
        return p

    def write_json_once(self, rel: str, obj) -> Path:
        return self.write_once(rel, json.dumps(obj, indent=2, sort_keys=False) + "\n")

    def event(self, state: str, **data) -> dict:
        ev = {"t": utc_now(), "state": state, **data}
        with self._events.open("a") as f:
            f.write(json.dumps(ev) + "\n")
        return ev
