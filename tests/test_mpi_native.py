"""Process contracts of the MPI P1 Poisson driver."""

import json
import os
import subprocess

import pytest

from conftest import ROOT, requires_mpi

pytestmark = [pytest.mark.mpi, requires_mpi]


def driver():
    return os.environ.get("POISSON_MPI_BIN", str(ROOT / "build/mpi/poisson_mpi"))


def run_level(directory, ranks, *extra):
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               OMPI_MCA_rmaps_base_oversubscribe="1")
    return subprocess.run([env.get("MPIEXEC", "mpiexec"), "-n", str(ranks), driver(),
                           "--out", str(directory), "--n", "16", *extra],
                          capture_output=True, text=True, timeout=60, env=env)


def test_build_info_comes_from_parallel_library():
    info = json.loads(subprocess.check_output([driver(), "--build-info"], text=True, timeout=15))
    assert info["mfem_version"] == "4.10.0"
    assert info["mfem_git"].endswith("d964264cdb9a13e94a201b6c236c7721e0c8765f")
    assert info["hypre_version"] > 0
    assert all(info[key] is True for key in ("mfem_use_mpi", "mfem_use_double", "mfem_use_metis", "mfem_use_metis_5"))


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_all_ranks_export_owned_cells(tmp_path, ranks):
    process = run_level(tmp_path, ranks)
    assert process.returncode == 0, process.stderr
    record = json.loads((tmp_path / "result.json").read_text())
    assert (record["n"], record["ranks"], record["global_dofs"]) == (16, ranks, 17**2)
    assert record["solver_converged"] and 0 < record["solver_iterations"] <= record["max_iter"]
    counts = [len((tmp_path / f"cells.{rank}.tsv").read_text().splitlines()) for rank in range(ranks)]
    assert min(counts) > 0 and sum(counts) == record["global_elements"] == 2*16**2
    assert sum(len((tmp_path / f"boundary.{rank}.tsv").read_text().splitlines()) for rank in range(ranks)) == 4*16


def test_nonempty_directory_is_preserved(tmp_path):
    sentinel = tmp_path / "result.json"
    sentinel.write_bytes(b"previous study")
    assert run_level(tmp_path, 2).returncode != 0
    assert list(tmp_path.iterdir()) == [sentinel] and sentinel.read_bytes() == b"previous study"


def test_opt_in_rank_clocks_preserve_the_verified_field(tmp_path):
    from benchmarks.mpi_phases import PHASES, cpu_set
    from sciagent import mpi_study
    base, timed = tmp_path / "ordinary", tmp_path / "profiled"
    base.mkdir()
    timed.mkdir()
    assert run_level(base, 2).returncode == 0
    assert run_level(timed, 2, "--phase-timing").returncode == 0
    assert list(base.glob("phases.*")) == []
    for rank in (0, 1):
        phase = json.loads((timed / f"phases.{rank}.json").read_text())
        assert phase["rank"] == rank and cpu_set(phase["cpus_allowed_list"])
        assert all(phase[k] >= 0 for k in PHASES)
        assert sum(phase[k] for k in PHASES) == pytest.approx(phase["measured_window_seconds"])
        for stream in ("cells", "boundary"):
            assert (timed / f"{stream}.{rank}.tsv").read_bytes() == (base / f"{stream}.{rank}.tsv").read_bytes()
    build = {"launcher": os.environ.get("MPIEXEC", "mpiexec"), "binary": driver()}
    null = mpi_study._launch(tmp_path / "null", build, 2, 0, 30, diagnostic="null")
    assert null["exit_code"] == 0 and null["argv"][-1] == "--null-launch"
    assert not (tmp_path / "null/result.json").exists()


def test_iteration_limit_fails_collectively(tmp_path):
    process = run_level(tmp_path, 2, "--max-iter", "1")
    assert process.returncode == 5, process.stderr
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["solver_converged"] is False and record["solver_iterations"] == 1
    assert record["solver_final_norm"] > record["rel_tol"] * record["solver_initial_norm"]


def test_compact_cartesian_tiles_match_full_mpi_field(tmp_path):
    import numpy as np
    from verifier.mpi import _merge

    legacy, compact = tmp_path / "metis", tmp_path / "tiles"
    legacy.mkdir()
    compact.mkdir()
    assert run_level(legacy, 4).returncode == 0
    assert run_level(compact, 4, "--compact-partition").returncode == 0
    old_field, _ = _merge(legacy, 16, 4)
    new_field, counts = _merge(compact, 16, 4, compact=True)
    assert counts == [128] * 4
    np.testing.assert_allclose(old_field, new_field, atol=1e-10, rtol=0)
