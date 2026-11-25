"""Paired serial/MPI experiment for the reviewed sine P1 triangle benchmark."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
from pathlib import Path

from . import capability, controller, environment
from .evidence import EvidenceStore, inspect_manifest, sha256_file
from .spec import StudySpec, sha256_of

LEVELS = (8, 16, 32, 64)
RANKS = (1, 2, 4)
SERIAL_SPEC = environment.PROJECT_ROOT / "specs" / "poisson_sin_sin_p1_tri.json"
SERIAL_SPEC_SHA256 = "63b28f4d02857036e4e3d5032706b0a39122505415607c06dee6c9361afbf303"
KIND = "mpi_poisson_p1_comparison_v1"
PACKAGES = {"openmpi-bin": "4.1.6-7ubuntu2", "libopenmpi3t64": "4.1.6-7ubuntu2",
            "libhypre-2.28.0": "2.28.0-8build2", "libmetis5": "5.1.0.dfsg-7build3"}
LIBRARIES = {"libmpi.so.": "libmpi.so.40", "libHYPRE-": "libHYPRE-2.28.0.so", "libmetis.so.": "libmetis.so.5"}


class ExperimentError(ValueError):
    """The paired experiment cannot be certified."""


def read(path: Path):
    return json.loads(path.read_text())


def reviewed_configuration() -> dict:
    return {"case": "sin_sin", "mesh": "generated Cartesian unit square", "element": "P1 triangle",
            "boundary": "four Dirichlet sides", "assembly": "full", "backend": "mpi_cpu",
            "solver": "MPI CG with HypreBoomerAMG", "ranks": list(RANKS), "levels": list(LEVELS),
            "rel_tol": 1e-12, "max_iter": 20_000, "mfem_version": capability.MFEM_VERSION,
            "mfem_commit": capability.MFEM_COMMIT, "openmpi": "4.1.6", "hypre": "2.28.0",
            "metis": "5.1.0", "entrypoint": "mpi-study", "evidence": "evidence/stage3/results.json"}


def plan(repetitions: int) -> dict:
    if type(repetitions) is not int or not 3 <= repetitions <= 15:
        raise ExperimentError("use 3 to 15 repetitions")
    return {"schema_version": 1, "serial_study_sha256": SERIAL_SPEC_SHA256,
            "configuration": reviewed_configuration(), "repetitions": repetitions,
            "strong_n": 64, "weak_n_by_ranks": {"1": 32, "4": 64}}


def _check_profile(build):
    flags, native, toolchain = build["mfem"], build["native"], build["toolchain"]
    revision = flags.get("git") or ""
    required = {"MFEM_USE_MPI": True, "MFEM_USE_METIS": True, "MFEM_USE_METIS_5": True,
                "MFEM_USE_DOUBLE": True, "MFEM_USE_SINGLE": False, "MFEM_USE_CUDA": False,
                "MFEM_USE_HIP": False, "MFEM_USE_OPENMP": False}
    expected = {"mfem_version": capability.MFEM_VERSION, "mfem_git": revision, "mfem_use_mpi": True,
                "mfem_use_metis": True, "mfem_use_metis_5": True, "mfem_use_double": True,
                "hypre_version": 22800, "case": "sin_sin", "solver": "cg_boomeramg"}
    if (flags.get("version") != capability.MFEM_VERSION or
            not (revision == capability.MFEM_COMMIT or revision.endswith("-g" + capability.MFEM_COMMIT))
            or any(flags["config"].get(k) is not v for k, v in required.items())
            or not flags.get("library_sha256") or any(native.get(k) != v for k, v in expected.items())):
        raise ExperimentError("MPI executable or installation differs from the reviewed build")
    if (toolchain.get("schema_version") != 1 or toolchain.get("mfem_commit") != capability.MFEM_COMMIT
            or toolchain.get("mfem_version") != capability.MFEM_VERSION
            or toolchain.get("library_sha256") != flags["library_sha256"]
            or any(toolchain.get("packages", {}).get(k) != v for k, v in PACKAGES.items())
            or build.get("runtime_packages") != PACKAGES):
        raise ExperimentError("MPI installation differs from its pinned toolchain provenance")
    libs = build["runtime_libraries"]
    if (build["launcher_version"] != "mpiexec (OpenRTE) 4.1.6" or set(libs) != set(LIBRARIES)
            or len(set(build["cpu_affinity"])) < 4 or build["binding"] != "none"):
        raise ExperimentError("the launcher, available CPUs or runtime libraries differ from the reviewed profile")
    digests = [build["binary_sha256"], build["launcher_sha256"], flags["library_sha256"], toolchain["config_sha256"]]
    if not all(re.fullmatch("[0-9a-f]{64}", h) for h in digests + [lib["sha256"] for lib in libs.values()]):
        raise ExperimentError("incomplete MPI build digests")


def _probe():
    third_party = Path(os.getenv("MPI_THIRD_PARTY", environment.PROJECT_ROOT / "third_party" / "mpi"))
    install = Path(os.getenv("MFEM_MPI_INSTALL", third_party / "mfem-mpi-install"))
    binary = Path(os.getenv("POISSON_MPI_BIN", environment.PROJECT_ROOT / "build" / "mpi" / "poisson_mpi")).resolve()
    launcher = shutil.which("mpiexec")
    if not launcher or not binary.is_file():
        raise ExperimentError("mpiexec or poisson_mpi is missing; see the MPI build instructions")
    if not Path(launcher).samefile("/usr/bin/orterun"):
        raise ExperimentError("mpiexec does not resolve to the reviewed Open MPI launcher")
    flags = environment.probe_mfem(install)
    native = json.loads(subprocess.check_output([str(binary), "--build-info"], text=True, timeout=30))
    launcher_version = subprocess.check_output([launcher, "--version"], text=True, timeout=15).splitlines()[0]
    toolchain = read(install / "share/mfem/mpi-toolchain.json")
    if toolchain.get("config_sha256") != sha256_file(install / "include/mfem/config/_config.hpp"):
        raise ExperimentError("MPI configuration header differs from the installed toolchain record")
    libraries = {}
    linked = subprocess.check_output(["ldd", str(binary)], text=True, timeout=15)
    for soname, library in LIBRARIES.items():
        paths = []
        for line in linked.splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[0].startswith(soname) and fields[1] == "=>":
                paths.append(fields[2])
        if len(paths) != 1 or not Path(paths[0]).samefile(Path("/usr/lib/x86_64-linux-gnu") / library):
            raise ExperimentError(f"cannot identify the linked {soname} runtime")
        libraries[soname] = {"path": paths[0], "sha256": sha256_file(Path(paths[0]))}
    versions = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\\n", *PACKAGES],
                                       text=True, timeout=15)
    build = {"mfem": flags, "native": native, "binary": str(binary), "binary_sha256": sha256_file(binary),
            "launcher": launcher, "launcher_version": launcher_version, "launcher_sha256": sha256_file(Path(launcher)),
            "toolchain": toolchain, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "runtime_packages": dict(line.split("=", 1) for line in versions.splitlines()),
            "runtime_libraries": libraries, "native_threads": {"OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1},
            "verifier_thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "binding": "none"}
    _check_profile(build)
    return build


def _mpi_argv(build: dict, ranks: int, n: int, out: str) -> list[str]:
    return [build["launcher"], "--host", "localhost:4", "--bind-to", "none", "-n", str(ranks), build["binary"],
            "--n", str(n), "--rel-tol", "1e-12", "--max-iter", "20000", "--out", out]


def _kill_launch_session(session_id: int):
    # Open MPI gives each local rank a distinct process group inside the
    # session created by Popen. A pidfd keeps the signal tied to that process.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) != session_id:
                continue
            fd = os.pidfd_open(pid)
            try:
                if os.getsid(pid) == session_id:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            finally:
                os.close(fd)
        except ProcessLookupError:
            continue


def _launch(directory: Path, build: dict, ranks: int, n: int, timeout: float, *,
            diagnostic: str | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    argv = _mpi_argv(build, ranks, n, str(directory.resolve()))
    if diagnostic == "null":
        argv = argv[:argv.index(build["binary"]) + 1] + ["--null-launch"]
    elif diagnostic == "compact":
        argv.extend(("--phase-timing", "--compact-partition"))
    elif diagnostic == "phases":
        argv.append("--phase-timing")
    elif diagnostic is not None:
        raise ValueError("unknown MPI diagnostic")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if os.geteuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    started = time.perf_counter()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()  # Give mpiexec time to stop and reap its ranks.
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            _kill_launch_session(proc.pid)
            stdout, stderr = proc.communicate(timeout=3)
        finally:
            # A launcher can also exit successfully while ranks remain.
            _kill_launch_session(proc.pid)
    elapsed = time.perf_counter() - started
    (directory / "stdout.txt").write_bytes(stdout)
    (directory / "stderr.txt").write_bytes(stderr)
    record = {"argv": argv, "exit_code": proc.returncode, "timed_out": timed_out, "launch_seconds": elapsed}
    (directory / "process.json").write_text(json.dumps(record, indent=2) + "\n")
    if timed_out or proc.returncode:
        raise ExperimentError(f"MPI n={n}, ranks={ranks} failed with exit {proc.returncode}; see {directory}/stderr.txt")
    return record


def _serial_levels(root: Path, digest: str):
    manifests = list((root / "serial").glob("*/manifest.json"))
    if len(manifests) != 1 or sha256_file(manifests[0]) != digest:
        raise ExperimentError("expected one digest-matched serial control")
    run = manifests[0].parent
    serial = read(manifests[0])
    study = StudySpec.model_validate_json((run / "request/study_spec.json").read_text())
    if (sha256_of(study) != SERIAL_SPEC_SHA256 or serial.get("final_state") != "ACCEPTED"
            or serial.get("bound_case") != "sin_sin" or len(serial["attempts"]) != 1):
        raise ExperimentError("the serial control differs from the accepted reviewed study")
    attempt = run / "attempts/attempt_01"
    verdict = controller.reverify(attempt, expected_digest=digest)
    if verdict["verdict"] != "ACCEPTED" or not verdict["integrity"]["passed"]:
        raise ExperimentError("serial reference replay failed")
    return attempt / "levels"


def _checked_sample(root, label, ranks, n, serial, build):
    from verifier.mpi import verify_mpi_level

    directory = root / label
    started = time.perf_counter()
    if build is not None:
        process = _launch(directory, build, ranks, n, timeout=120)
    else:
        process = read(directory / "process.json")
        if type(process.get("exit_code")) is not int or process["exit_code"] != 0 or process.get("timed_out") is not False:
            raise ExperimentError(f"failed archived launcher in {label}")
        argv = process.get("argv")
        if (not isinstance(argv, list) or not argv or not isinstance(argv[-1], str)
                or not Path(argv[-1]).is_absolute() or not Path(argv[-1]).as_posix().endswith("/" + label)
                or argv != _mpi_argv(read(root / "mpi_environment.json"), ranks, n, argv[-1])):
            raise ExperimentError(f"launcher arguments differ from the reviewed sample in {label}")
    metrics = verify_mpi_level(directory, serial / f"n{n:04d}", n=n, ranks=ranks)
    if build is not None:
        # Includes the launch, file export and independent verification of this sample.
        measured = {"workflow_seconds": time.perf_counter() - started}
        (directory / "measurement.json").write_text(json.dumps(measured) + "\n")
    else:
        measured = read(directory / "measurement.json")
    raw = read(directory / "result.json")
    timings = {"workflow_seconds": measured["workflow_seconds"], "launch_seconds": process["launch_seconds"],
               "solve_seconds_max": raw["solve_seconds_max"]}
    if (not all(type(t) in (float, int) and math.isfinite(t) and t > 0 for t in timings.values())
            or timings["workflow_seconds"] < timings["launch_seconds"]
            or timings["launch_seconds"] < timings["solve_seconds_max"]):
        raise ExperimentError(f"invalid sample timing in {label}")
    return metrics, timings


def _experiment(root: Path, serial: Path, repetitions: int, build: dict | None) -> dict:
    from verifier.mpi import verify_mpi_orders

    screens = {}
    for ranks in RANKS:
        levels = []
        for n in LEVELS:
            values, _ = _checked_sample(root, f"screen/r{ranks}_n{n}", ranks, n, serial, build)
            levels.append(values)
        screens[str(ranks)] = {"levels": levels, "orders": verify_mpi_orders(levels)}

    # Enter the timing phase only after every rank count passes the convergence study.
    conditions = [(1, 64), (2, 64), (4, 64), (1, 32)]
    samples = {f"r{r}_n{n}": [] for r, n in conditions}
    for i in range(repetitions):
        offset = i % len(conditions)
        for ranks, n in conditions[offset:] + conditions[:offset]:
            key = f"r{ranks}_n{n}"
            _, times = _checked_sample(root, f"timing/repeat_{i + 1:02d}/{key}", ranks, n, serial, build)
            samples[key].append(times)
    medians = {key: {k: statistics.median(row[k] for row in rows) for k in rows[0]} for key, rows in samples.items()}
    strong = {}
    for ranks in RANKS:
        strong[str(ranks)] = {k: medians["r1_n64"][k] / t for k, t in medians[f"r{ranks}_n64"].items()}
    weak = {k: medians["r1_n32"][k] / t for k, t in medians["r4_n64"].items()}
    return {"screens": screens, "timing_samples": samples, "median_seconds": medians,
            "strong_speedup": strong, "weak_efficiency_1_to_4": weak}


def run(directory: Path, repetitions: int = 3) -> dict:
    request = plan(repetitions)
    started = time.perf_counter()
    store = EvidenceStore(directory.resolve())
    store.write_json_once("plan.json", request)
    manifest = {"kind": KIND, "state": "FAILED"}
    try:
        build = _probe()
        store.write_json_once("mpi_environment.json", build)
        baseline = controller.run_study(SERIAL_SPEC, store.path("serial"))
        digest = sha256_file(Path(baseline["run_dir"]) / "manifest.json")
        manifest["serial_manifest_sha256"] = digest
        levels = _serial_levels(store.run_dir, digest)
        result = _experiment(store.run_dir, levels, repetitions, build)
        store.write_json_once("results.json", result)
        manifest["state"] = "ACCEPTED"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        manifest["reason"] = str(exc)
    manifest["seconds_through_verification"] = time.perf_counter() - started
    store.finish_manifest(manifest)
    return {"state": manifest["state"], "reason": manifest.get("reason"), "directory": str(store.run_dir),
            "manifest_sha256": sha256_file(store.path("manifest.json")),
            "wall_seconds_including_inventory": time.perf_counter() - started}


def _equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equivalent(x, y) for x, y in zip(a, b))
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13)
    return type(a) is type(b) and a == b


def replay(directory: Path, expected_digest: str | None) -> dict:
    directory = directory.resolve()
    manifest, _, changed = inspect_manifest(directory, expected_digest)
    if (changed or manifest.get("kind") != KIND or manifest.get("evidence_version") != 1
            or manifest.get("state") != "ACCEPTED"):
        raise ExperimentError(f"archive is incomplete, changed or rejected: {sorted(changed)}")
    request = read(directory / "plan.json")
    repetitions = request["repetitions"]
    if request != plan(repetitions):
        raise ExperimentError("archived plan differs from the reviewed protocol")
    _check_profile(read(directory / "mpi_environment.json"))
    levels = _serial_levels(directory, manifest["serial_manifest_sha256"])
    result = _experiment(directory, levels, repetitions, build=None)
    if not _equivalent(result, read(directory / "results.json")):
        raise ExperimentError("archived report differs from independently recomputed results")
    return {"state": "ACCEPTED", "ranks": list(RANKS), "screened_levels": len(LEVELS) * len(RANKS),
            "verified_timing_samples": 4 * repetitions, "manifest_sha256": sha256_file(directory / "manifest.json")}
