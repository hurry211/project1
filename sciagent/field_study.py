"""Run and replay the reviewed tensor-grid heat, wave and elasticity benchmarks."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .environment import PROJECT_ROOT
from .evidence import EvidenceStore, inspect_manifest, sha256_file
from .mfem_runner import _limits

FAMILIES = ("heat", "wave", "elasticity")
SCHEME = {"heat": "Crank-Nicolson", "wave": "implicit midpoint in displacement and velocity", "elasticity": "static Galerkin"}
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def request(family, dimension, order, levels):
    levels = list(levels)
    if family not in FAMILIES or type(dimension) is not int or dimension not in (2, 3) or type(order) is not int or order not in (1, 2):
        raise ValueError("unreviewed operator, dimension or element order")
    if (len(levels) != 3 or any(type(n) is not int or not 2 <= n <= 32 for n in levels)
            or any(b != 2*a for a, b in zip(levels, levels[1:]))):
        raise ValueError("supply three successive mesh doublings between 2 and 32")
    return {"schema": 1, "family": family, "dimension": dimension, "order": order, "levels": levels,
            "scheme": SCHEME[family], "domain": "unit hypercube", "trace": "homogeneous Dirichlet",
            "end_time": None if family == "elasticity" else 1/32, "steps_per_cell": None if family == "elasticity" else 4,
            "memory_mib": 3072, "process_timeout_seconds": 300}


def assess(records, order):
    errs = [r["relative_l2"] for r in records]
    if not all(math.isfinite(x) and x > 0 for x in errs):
        return {"verdict": "REJECTED", "l2_orders": [], "reason": "invalid continuous error"}
    orders = [math.log2(a/b) for a, b in zip(errs, errs[1:])]
    ok = all(r["passed"] for r in records) and all(v > 0 for v in orders) and orders[-1] >= order+1-0.5
    return {"verdict": "ACCEPTED" if ok else "REJECTED", "l2_orders": orders, "minimum_finest_order": order+0.5}


def process(command, cfg, *, peak_file=None):
    if peak_file is not None:
        command = ["/usr/bin/time", "-f", "%M", "-o", str(peak_file), *command]
    start = time.monotonic()
    with subprocess.Popen(command, env={**os.environ, **THREADS}, preexec_fn=_limits(cfg["memory_mib"]),
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=cfg["process_timeout_seconds"])
        except subprocess.TimeoutExpired:
            # The GNU time wrapper and its numerical child share this new group.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        if proc.returncode:
            raise ValueError(f"process exited {proc.returncode}: {stderr[-2000:]}")
    return stdout, time.monotonic()-start


def independent(field, cfg, n, peak_file=None):
    out, elapsed = process([sys.executable, "-m", "verifier.field_family", str(field), cfg["family"],
                            str(cfg["dimension"]), str(n), str(cfg["order"]), str(4*n)], cfg, peak_file=peak_file)
    return json.loads(out), elapsed


def capture(directory, cfg):
    binary = Path(os.environ.get("FIELD_FAMILY_BIN", PROJECT_ROOT / "build/cpp/field_family")).resolve(strict=True)
    directory = Path(directory).resolve()
    store = EvidenceStore(directory)
    store.write_json_once("request.json", cfg)
    records, costs = [], []
    start = time.monotonic()
    for n in cfg["levels"]:
        field = directory / f"n{n:04d}/field.tsv"
        field.parent.mkdir()
        native_peak = field.with_name("native-rss-kib.txt")
        verify_peak = field.with_name("verifier-rss-kib.txt")
        _, native_seconds = process([str(binary), cfg["family"], str(cfg["dimension"]), str(n), str(cfg["order"]),
                                     str(4*n), str(field)], cfg, peak_file=native_peak)
        record, verify_seconds = independent(field, cfg, n, verify_peak)
        records.append(record)
        costs.append({"n": n, "native_seconds": native_seconds, "native_peak_rss_kib": int(native_peak.read_text()),
                      "verifier_seconds": verify_seconds, "verifier_peak_rss_kib": int(verify_peak.read_text())})
    outcome = {**assess(records, cfg["order"]), "levels": records}
    store.write_json_once("verdict.json", outcome)
    store.finish_manifest({"schema": "field-family-v1", "native_sha256": sha256_file(binary),
                           "request_sha256": sha256_file(directory / "request.json"), "costs": costs,
                           "elapsed_before_archive_seconds": time.monotonic()-start, "verdict": outcome["verdict"]})
    return {**outcome, "manifest_sha256": sha256_file(directory / "manifest.json"), "costs": costs}


def replay(directory, digest):
    directory = Path(directory).resolve()
    manifest, _, changed = inspect_manifest(directory, digest)
    if changed or manifest.get("schema") != "field-family-v1":
        raise ValueError("field study inventory mismatch")
    cfg = json.loads((directory / "request.json").read_bytes())
    expected = request(cfg["family"], cfg["dimension"], cfg["order"], cfg["levels"])
    if expected != cfg or manifest["request_sha256"] != sha256_file(directory / "request.json"):
        raise ValueError("field study request differs")
    levels = [independent(directory / f"n{n:04d}/field.tsv", cfg, n)[0] for n in cfg["levels"]]
    actual = {**assess(levels, cfg["order"]), "levels": levels}
    saved = json.loads((directory / "verdict.json").read_bytes())
    # A different BLAS implementation can round the independent diagnostics
    # differently. Recompute every gate and compare numbers at roundoff scale.
    if (actual["verdict"] != saved["verdict"] or actual["verdict"] != manifest["verdict"]
            or len(saved["levels"]) != len(levels) or set(saved) != set(actual)):
        raise ValueError("recomputed field outcome differs")
    for old, new in zip(saved["levels"], levels, strict=True):
        if old.keys() != new.keys() or any(
            not math.isclose(old[k], v, rel_tol=1e-10, abs_tol=1e-13)
            if k in ("relative_l2", "independent_error", "boundary_linf") else old[k] != v for k, v in new.items()
        ):
            raise ValueError("recomputed level differs")
    if (saved.get("minimum_finest_order") != actual.get("minimum_finest_order")
            or len(saved["l2_orders"]) != len(actual["l2_orders"])
            or any(not math.isclose(a, b, abs_tol=1e-10) for a, b in zip(saved["l2_orders"], actual["l2_orders"]))):
        raise ValueError("recomputed convergence differs")
    return {**actual, "manifest_sha256": digest}
