#!/usr/bin/env bash
# Run every example and negative spec through the controller, keeping all
# evidence under runs/, and print a one-line summary per study.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUNS="${RUNS_DIR:-runs}"
for spec in specs/*.json specs/negative/*.json; do
  out="$(python3 -m sciagent run "$spec" --runs-dir "$RUNS" 2>&1)"
  state="$(echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["final_state"], d["final_code"])' 2>/dev/null || echo "ERROR")"
  printf '%-55s %s\n' "$spec" "$state"
done
