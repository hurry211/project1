"""Open MPI starts local ranks in separate process groups."""
import json
import shutil
import time
from pathlib import Path

import pytest

from conftest import requires_mpi
from sciagent.mpi_study import ExperimentError, _launch


@pytest.mark.mpi
@requires_mpi
@pytest.mark.parametrize("stall_launcher", [False, True], ids=["graceful", "forced"])
def test_timeout_stops_all_local_ranks(tmp_path, stall_launcher, monkeypatch):
    worker = tmp_path / "sleeping_rank"
    worker.write_text("""#!/usr/bin/env python3
import os, signal, sys, time
from pathlib import Path

directory = Path(sys.argv[sys.argv.index("--out") + 1])
rank = os.environ["OMPI_COMM_WORLD_RANK"]
(directory / f"pid.{rank}").write_text(f"{os.getpid()} {os.getsid(0)}")
if rank == "0" and os.environ["STALL_LAUNCHER"] == "1":
    while not (directory / "pid.1").exists():
        time.sleep(.01)
    os.kill(os.getsid(0), signal.SIGSTOP)
time.sleep(30)
""")
    worker.chmod(0o755)
    directory = tmp_path / "out"
    monkeypatch.setenv("STALL_LAUNCHER", str(int(stall_launcher)))
    with pytest.raises(ExperimentError, match="MPI .* failed with exit"):
        _launch(directory, {"launcher": shutil.which("mpiexec"), "binary": str(worker)},
                ranks=2, n=8, timeout=3)
    assert json.loads((directory / "process.json").read_text())["timed_out"] is True
    for rank in range(2):
        pid, sid = map(int, (directory / f"pid.{rank}").read_text().split())
        # A killed orphan may briefly wait for PID 1 to reap it.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{pid}/stat").read_text().split()
            except FileNotFoundError:
                break
            if int(status[5]) != sid or status[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail(f"timed-out MPI rank {rank} is still running")
