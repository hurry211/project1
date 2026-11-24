"""POSIX worker for the independent verifier. Set the limit before importing SciPy."""

import json
import resource
import sys
from pathlib import Path


def peak_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def main() -> int:
    limit = int(sys.argv[2]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        from verifier.verifier import verify_attempt

        reused = {int(n): entry for n, entry in json.load(sys.stdin).items()}
        verdict = verify_attempt(Path(sys.argv[1]), require_run_records=True, reused_levels=reused)
        print(json.dumps({"verdict": verdict, "peak_rss_kib": peak_rss_kib()}))
        return 0
    except MemoryError:
        return 2


if __name__ == "__main__":
    sys.exit(main())
