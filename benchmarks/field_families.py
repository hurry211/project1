"""Capture the manufactured PDE coverage matrix and replay every accepted field study."""
import argparse
import json
from pathlib import Path

from sciagent import field_study


def capture(root):
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for family in field_study.FAMILIES:
        for dimension in (2, 3):
            for order in (1, 2):
                levels = [8, 16, 32] if (family, dimension, order) == ("heat", 3, 2) else [4, 8, 16]
                cfg = field_study.request(family, dimension, order, levels)
                path = root / f"{family}-{dimension}d-q{order}"
                native = field_study.capture(path, cfg)
                check = field_study.replay(path, native["manifest_sha256"])
                rows.append({"directory": path.name, "request": cfg, "capture": native, "replay": check["verdict"]})
                print(path.name, native["verdict"], check["verdict"], flush=True)
    report = {"schema": 1, "studies": rows}
    (root / "matrix.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return all(row["replay"] == row["capture"]["verdict"] == "ACCEPTED" for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    raise SystemExit(0 if capture(parser.parse_args().directory) else 1)
