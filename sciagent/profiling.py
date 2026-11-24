"""Optional serial wall-time accounting, separate from numerical evidence."""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar


_active = ContextVar("study_profile", default=None)


class StudyProfile:
    def __init__(self):
        self.started = self.last = time.monotonic()
        self.name = "controller.other"
        self.seconds = {}
        self.levels = []
        self.elapsed = None
        self.peak = None

    def switch(self, name):
        now = time.monotonic()
        self.seconds[self.name] = self.seconds.get(self.name, 0.0) + (now - self.last)
        self.name, self.last = name, now

    def summary(self):
        if self.elapsed is None:
            raise RuntimeError("finish the capture before reading its summary")
        return {"elapsed_seconds": self.elapsed, "phase_seconds": dict(self.seconds),
                "worker_peak_rss_kib": self.peak, "native_levels": list(self.levels)}


@contextmanager
def capture():
    """Capture exclusive Python wall-time buckets in the calling context."""
    profile = StudyProfile()
    token = _active.set(profile)
    try:
        yield profile
    finally:
        profile.switch(None)
        profile.elapsed = profile.last - profile.started
        _active.reset(token)
        if sys.platform.startswith("linux"):
            import resource
            profile.peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def enabled():
    return _active.get() is not None


def mark(name):
    """Move the active timer within a phase. Exiting the phase restores its parent."""
    profile = _active.get()
    if profile is not None:
        profile.switch(name)


@contextmanager
def phase(name):
    profile = _active.get()
    if profile is None:
        yield
        return
    parent = profile.name
    profile.switch(name)
    try:
        yield
    finally:
        profile.switch(parent)


def record_native(n, elapsed, stderr):
    """Older binaries or early native exits can leave detail counters unavailable."""
    profile = _active.get()
    if profile is None:
        return
    measurements = dict.fromkeys(("peak_rss_kib", "setup_seconds", "solve_seconds",
                                  "error_seconds", "export_seconds"))
    for line in stderr.splitlines():
        if line.startswith("MFEM_PROFILE_V1 "):
            try:
                data = json.loads(line.removeprefix("MFEM_PROFILE_V1 "))
                for key in measurements:
                    value = data.get(key) if isinstance(data, dict) else None
                    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                        measurements[key] = value
            except (ValueError, OverflowError):
                pass
    profile.levels.append({"n": n, "process_seconds": elapsed, **measurements})
