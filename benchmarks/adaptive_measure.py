"""Linux process-tree wall and memory measurements for adaptive trials."""
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import time


def _resident_kib(pid):
    """Sum resident memory in the observed Linux process tree (shared pages may appear twice)."""
    try:
        fields = Path(f"/proc/{pid}/statm").read_text().split()
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except (OSError, IndexError):
        return 0
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") // 1024 + sum(_resident_kib(c) for c in children)


def _run(receipt, memory_mb, function, args, kwargs):
    os.setsid()
    limit = memory_mb * 1024**2
    inherited = resource.getrlimit(resource.RLIMIT_AS)
    limit = min([limit] + [v for v in inherited if v != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result = function(*args, **kwargs)
    result["worker_high_water_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["native_high_water_mib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    receipt.write_text(json.dumps(result, allow_nan=False))


def measured(receipt, policy, function, *args, **kwargs):
    if receipt.exists():
        raise FileExistsError(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    process = multiprocessing.get_context("fork").Process(
        target=_run, args=(receipt, policy.per_level_memory_mb, function, args, kwargs))
    process.start()
    peak = 0
    timed_out = False
    try:
        while process.is_alive():
            peak = max(peak, _resident_kib(process.pid))
            if time.monotonic() - start > policy.max_total_wall_seconds:
                timed_out = True
                break
            process.join(0.01)
    finally:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
        process.join()
    elapsed = time.monotonic() - start
    if timed_out or process.exitcode != 0:
        code = "WALL_BUDGET_EXHAUSTED" if timed_out else "WORKER_FAILED"
        result = {"ok": False, "accepted": False, "failure_codes": [code], "exit_code": process.exitcode,
                  "wall_seconds": elapsed, "decision_seconds": 0., "pilot_seconds": 0., "attempts": [],
                  "worker_high_water_mib": None, "native_high_water_mib": None,
                  "l2_error": None, "finest_l2_error": None, "timing_detail_available": False}
    else:
        result = json.loads(receipt.read_bytes())
    result["sampled_tree_peak_mib"] = peak / 1024
    result["peak_rss_mb"] = max(peak / 1024, result["worker_high_water_mib"] or 0., result["native_high_water_mib"] or 0.)
    result["external_wall_seconds"] = elapsed
    receipt.write_text(json.dumps(result, allow_nan=False))
    return result
