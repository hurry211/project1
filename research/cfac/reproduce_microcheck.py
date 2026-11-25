"""Reproduce the supplied identity experiment in a fresh evidence directory."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import scipy


SCRIPT = Path(__file__).with_name("microcheck.py")
EXPECTED = "c11345887563330d06a90e9cb28ebbfd713ad0142c50c6277266fe0d7cf11eee"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--extended", action="store_true", help="run the optional n=128 case separately")
    args = parser.parse_args()
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    if digest != EXPECTED:
        parser.error("input script SHA-256 mismatch")
    args.directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    resolutions = [128] if args.extended else [8, 16, 32, 64]
    command = [sys.executable, str(SCRIPT), *map(str, resolutions)]
    result = subprocess.run(command, capture_output=True, env=env, check=False)
    (args.directory / "stdout.jsonl").write_bytes(result.stdout)
    (args.directory / "stderr.txt").write_bytes(result.stderr)
    metadata = {
        "purpose": "floating-point identity experiment", "script_sha256": digest,
        "command": ["python", "research/cfac/microcheck.py", *command[2:]],
        "returncode": result.returncode, "platform": platform.platform(),
        "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
        "threads": {k: env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }
    path = args.directory / "environment.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    result.check_returncode()
    if [json.loads(line)["n"] for line in result.stdout.splitlines()] != resolutions:
        raise ValueError("missing output resolution")


if __name__ == "__main__":
    main()
