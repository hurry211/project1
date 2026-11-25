"""Paired Linux fresh-process measurements and platform-independent proof replay."""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import select
import signal
import subprocess
import sys
import tempfile
from time import perf_counter_ns

from cfac.checker import check, validate_problem
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.prototype import uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.spec import Mesh, Problem


DESIGN_PATH = Path(__file__).with_name("stopping_design.json")
DESIGN_SHA = "4f854f649114f1bea571138dec24a5a5d4e3933ffdf552a87d4d5e7283f9aa05"
DESIGN_COMMIT = "f5e82479b7b33d19da1150d03ef4b8bbb27852f4"
SCHEMA = "cfac-stopping-measurements-v1"
TERMINAL = {"ACCEPTED", "NO_ELIGIBLE_ACTION", "INVALID_INPUT", "UNDETERMINED", "BUDGET_EXHAUSTED"}


@dataclass(frozen=True)
class StudyProtocol:
    design_path: Path
    design_sha: str
    design_commit: str
    schema: str
    worker_module: str


STOPPING = StudyProtocol(DESIGN_PATH, DESIGN_SHA, DESIGN_COMMIT, SCHEMA, "research.cfac.stopping_study")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def design(protocol=STOPPING):
    raw = protocol.design_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == protocol.design_sha, "frozen design digest differs")
    return json.loads(raw)


def schedule(spec):
    cases, policies = spec["cases"], spec["policies"]
    for block in range(spec["blocks"]):
        for j in range(len(cases)):
            case_index = (block + j) % len(cases)
            for position in range(len(policies)):
                policy = policies[(position + block + case_index) % len(policies)]
                yield cases[case_index], block, position, policy


def problem_for(case):
    return Problem(uniform_mesh(case["n"]), case["loads"], case["diffusion"],
                   case["epsilon"], case["max_actions"])


def delivery(journal):
    packet = {"status": journal["status"], "accepted": journal["accepted"]}
    if journal["accepted"]:
        event = journal["trace"][-1]
        state = event["after"]
        packet.update(state=state, certificate=event["observation"],
                      problem=journal["problems"][state["problem_digest"]])
    return packet


def worker(case_id, policy, destination, protocol=STOPPING):
    from cfac.baselines import BALANCE_FRACTION, RELATIVE_TOLERANCE, run_policy
    spec = design(protocol)
    require(BALANCE_FRACTION == Fraction(spec["classical_fraction"])
            and RELATIVE_TOLERANCE == Fraction(spec["fixed_relative"]), "stopping parameters differ from protocol")
    case = next(c for c in spec["cases"] if c["id"] == case_id)
    require(policy in spec["policies"], "unknown policy")
    problem = problem_for(case)
    result = run_policy(problem, problem.digest(), policy=policy)
    peak = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:"))
    # The timed return includes the independently accepted field and proof material.
    # The complete diagnostic journal is exported after delivery.
    packet = json.dumps([delivery(result), peak], separators=(",", ":")).encode() + b"\n"
    sys.stdout.buffer.write(packet)
    sys.stdout.buffer.flush()
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))


def journal_hash(journal):
    return hashlib.sha256(json.dumps(journal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _action_totals(journal, costs):
    counts, times = Counter(), Counter()
    for event, cost in zip(journal["trace"], costs, strict=True):
        require(type(cost) is int and cost >= 0, "invalid action time")
        counts[event["action"]] += 1
        times[event["action"]] += cost
    return dict(counts), dict(times)


def _receive(pipe, deadline):
    data = bytearray()
    while True:
        remaining = (deadline - perf_counter_ns()) / 1e9
        if remaining <= 0 or not select.select([pipe], [], [], remaining)[0]:
            raise TimeoutError("deadline expired before complete delivery")
        chunk = os.read(pipe.fileno(), 65536)
        if not chunk:
            raise ValueError("worker exited before returning a complete decision")
        data.extend(chunk)
        require(len(data) <= 64 * 1024**2, "worker decision exceeds delivery limit")
        if data.endswith(b"\n"):
            packet, peak = json.loads(data)
            require(packet["status"] in TERMINAL and type(packet["accepted"]) is bool and
                    type(peak) is int and peak > 0, "invalid worker decision")
            if perf_counter_ns() >= deadline:
                raise TimeoutError("deadline expired decoding the decision")
            return packet, peak, len(data)


def collect(case, block, position, policy, timeout, journals, directory, protocol=STOPPING):
    destination = directory / "worker.json"
    command = [sys.executable, "-m", protocol.worker_module, "--worker", case["id"], policy,
               "--journal", str(destination)]
    row = dict(case=case["id"], block=block, position=position, policy=policy,
               accepted=False, status="WORKER_ERROR", controller_ns=None, peak_rss_kib=None,
               decision_peak_rss_kib=None, actions_used=None, action_counts={}, action_ns={},
               journal_id=None, step_ns=[], returncode=None, error=None, delivery_bytes=0)
    with tempfile.TemporaryFile() as stderr:
        start = perf_counter_ns()
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        except OSError as exc:
            row.update(elapsed_ns=perf_counter_ns()-start, error=str(exc))
            return row
        try:
            try:
                packet, peak, size = _receive(proc.stdout, start + int(timeout*1e9))
                row.update(status=packet["status"], accepted=packet["accepted"],
                           decision_peak_rss_kib=peak, delivery_bytes=size)
            except TimeoutError as exc:
                row.update(status="TIMEOUT", error=str(exc))
                os.kill(proc.pid, signal.SIGKILL)
            except (ValueError, KeyError, TypeError) as exc:
                row["error"] = f"missing or malformed worker decision: {exc}"
                os.kill(proc.pid, signal.SIGKILL)
            elapsed = perf_counter_ns()-start
            # The independent decision was returned. Bound the separate raw evidence export.
            fd = os.pidfd_open(proc.pid)
            try:
                if not select.select([fd], [], [], max(timeout, 30))[0]:
                    os.kill(proc.pid, signal.SIGKILL)
                    row.update(accepted=False, status="WORKER_ERROR", error="journal export timed out")
            finally:
                os.close(fd)
        finally:
            _, wait_status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(wait_status)
            proc.stdout.close()
        row.update(returncode=proc.returncode, peak_rss_kib=usage.ru_maxrss)
        if not proc.returncode and not row["error"]:
            try:
                journal = json.loads(destination.read_bytes())
                require(delivery(journal) == packet,
                        "returned decision differs from journal")
                costs = [e.pop("cost_ns") for e in journal["trace"]]
                counts, times = _action_totals(journal, costs)
                inner = journal.pop("elapsed_ns")
                key = journal_hash(journal)
                journals.setdefault(key, journal)
                row.update(journal_id=key, controller_ns=inner, actions_used=len(costs),
                           action_counts=counts, action_ns=times, step_ns=costs, elapsed_ns=elapsed)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["error"] = f"worker journal unavailable: {exc}"
        if proc.returncode or row["error"]:
            row.update(status="TIMEOUT" if row["status"] == "TIMEOUT" else "WORKER_ERROR",
                       accepted=False, elapsed_ns=perf_counter_ns()-start)
            stderr.seek(0)
            row["error"] = (row["error"] or "worker exited unsuccessfully") + " " + stderr.read(4096).decode(errors="replace")
    destination.unlink(missing_ok=True)
    return row


def _child_contract(parent, child):
    # Independently compare the physical loaded triangles, including all four children of
    # each parent. This comparison does not depend on child vertex numbering or cell order.
    require(child.diffusion == parent.diffusion and child.epsilon == parent.epsilon
            and child.max_actions == parent.max_actions and child.original_affine is None,
            "refinement changed the frozen PDE, threshold or action budget")
    expected = Counter()
    for triangle, load in zip(parent.mesh.triangles, parent.loads):
        a, b, c = (parent.mesh.vertices[v] for v in triangle)
        ab, bc, ca = (tuple((x+y)/2 for x,y in zip(i,j)) for i,j in ((a,b),(b,c),(c,a)))
        for t in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)):
            expected[(tuple(sorted(t)), load)] += 1
    actual = Counter((tuple(sorted(child.mesh.vertices[v] for v in t)), f)
                     for t, f in zip(child.mesh.triangles, child.loads))
    require(actual == expected, "refined mesh or cellwise load differs from original contract")


def _certificate(p, state):
    anchor = p.digest()
    if state.get("space") == "P1/RT1":
        parent = state["parent"]
        require(parent["problem_digest"] == anchor and state["order"] == 1, "wrong RT1 parent")
        req = EnrichmentRequest(p, anchor, parent["state_digest"], parent["primal"], parent["flux"], 1)
        moments = RTMoments(anchor, parent["state_digest"], 1, state["face_moments"], state["cell_moments"])
        cert = RT1Checker().verify(req, moments, state["state_digest"])
    else:
        cert = check(p, anchor, state["primal"], state["flux"])
    require(cert.status == "CERTIFIED" and cert.state_digest == state["state_digest"],
            "unbound or invalid archived field")
    return cert


def validate_journal(case, journal):
    original = problem_for(case)
    anchor = original.digest()
    require(journal["initial_digest"] == anchor and journal["budget"] == original.max_actions,
            "journal initial problem differs from frozen design")
    problems = {}
    for digest, s in journal["problems"].items():
        p = Problem(Mesh(s["vertices"], s["triangles"]), s["loads"], s["diffusion"],
                    s["epsilon"], s["max_actions"], s["original_affine"])
        validate_problem(p, digest)
        if digest != anchor:
            _child_contract(original, p)
        problems[digest] = p
    require(len(problems) <= 2, "too many refinement levels")
    trace = journal["trace"]
    require(journal["actions_used"] == len(trace) <= original.max_actions, "action budget mismatch")
    before, next_action = None, "setup"
    for i, event in enumerate(trace):
        require(event["step"] == i+1 and event["before"] == before and event["action"] == next_action,
                "disconnected action trace")
        after = event["after"]
        if event["action"] == "rt1_reconstruct" and after and after.get("space") == "P1/RT1":
            require(after["parent"] == before, "RT1 reconstruction changed its parent field")
        if event["action"] == "restore_rt0":
            require(before and before.get("space") == "P1/RT1" and after == before["parent"],
                    "RT0 restoration differs from the enriched field's parent")
        if after:
            label = after["problem_digest"]
            require(label in problems, "unknown state problem")
            if before is None:
                require(label == anchor, "first state uses a different problem")
            elif label != before["problem_digest"]:
                require(event["action"] == "refine" and before["problem_digest"] == anchor,
                        "problem changed outside the permitted refinement")
        obs = event["observation"]
        if obs and "accepted" in obs:
            require(before == after and after is not None, "inspection changed its field")
            cert = _certificate(problems[after["problem_digest"]], after)
            require(obs["state_digest"] == cert.state_digest and obs["accepted"] is cert.accepted
                    and obs["upper_sq"] == str(cert.upper_sq), "archived certificate differs from independent check")
        if obs and "state_digest" in obs:
            require(after is not None and obs["state_digest"] == after["state_digest"], "stale observation")
        next_action, before = event["next_decision"], after
    status = journal["status"]
    require(status in TERMINAL and journal["accepted"] is (status == "ACCEPTED"), "invalid terminal status")
    if next_action in TERMINAL:
        require(status == next_action, "terminal outcome contradicts last decision")
    else:
        require(status == "BUDGET_EXHAUSTED" and len(trace) == original.max_actions, "premature budget stop")
    if status == "ACCEPTED":
        require(trace and trace[-1]["action"] in ("inspect", "rt1_inspect", "witness_probe", "algebraic_probe")
                and trace[-1]["observation"]["accepted"] is True, "missing independently accepted final field")


def _replay_trials(receipt, spec, protocol=STOPPING):
    from cfac.baselines import run_policy
    require(receipt["schema"] == protocol.schema and receipt["design_sha256"] == protocol.design_sha and
            receipt["design_commit"] == protocol.design_commit, "receipt or design identity mismatch")
    require(re.fullmatch("[0-9a-f]{40}", receipt["revision"]), "invalid code revision")
    checked = set()
    for (case, block, pos, policy), row in zip(schedule(spec), receipt["trials"], strict=True):
        require((row["case"],row["block"],row["position"],row["policy"]) ==
                (case["id"],block,pos,policy), "incomplete or reordered schedule")
        require(type(row["elapsed_ns"]) is int and row["elapsed_ns"] > 0, "invalid submission wall time")
        if row["status"] in {"TIMEOUT", "WORKER_ERROR"}:
            require(row["accepted"] is False and row["journal_id"] is None and row["error"]
                    and row["controller_ns"] is None and row["actions_used"] is None
                    and row["step_ns"] == [] and not row["action_counts"] and not row["action_ns"],
                    "failed trial was credited as completed computation")
            if row["status"] == "TIMEOUT":
                require(row["elapsed_ns"] >= spec["timeout_s"]*10**9, "uncharged timeout")
            continue
        key = row["journal_id"]
        journal = receipt["journals"][key]
        if (case["id"], policy, key) not in checked:
            require(journal_hash(journal) == key, "journal digest mismatch")
            validate_journal(case, journal)
            p = problem_for(case)
            regenerated = run_policy(p, p.digest(), policy=policy)
            del regenerated["elapsed_ns"]
            for action in regenerated["trace"]:
                del action["cost_ns"]
            require(journal_hash(regenerated) == key, "policy replay differs from archived trajectory")
            checked.add((case["id"], policy, key))
        counts, times = _action_totals(journal, row["step_ns"])
        require(counts == row["action_counts"] and times == row["action_ns"] and
                sum(counts.values()) == row["actions_used"] == journal["actions_used"], "action totals mismatch")
        require(row["status"] == journal["status"] and row["accepted"] is journal["accepted"]
                and row["returncode"] == 0 and row["error"] is None, "journal verdict mismatch")
        size = len(json.dumps([delivery(journal), row["decision_peak_rss_kib"]],
                              separators=(",", ":")).encode()) + 1
        require(row["delivery_bytes"] == size, "delivered field or certificate size mismatch")
        require(0 <= sum(times.values()) <= row["controller_ns"] <= row["elapsed_ns"] and
                row["decision_peak_rss_kib"] > 0 and row["peak_rss_kib"] > 0, "invalid resource measurement")
    require({key for _, _, key in checked} == set(receipt["journals"]), "unreferenced journal")


def replay(receipt, protocol=STOPPING):
    _replay_trials(receipt, design(protocol), protocol)


def verify_checkout(revision, root):
    """Bind the declared measurement revision to the running policy and runner sources."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    hint = "use a clean Git checkout of the declared implementation commit"
    try:
        require(Path(git("rev-parse", "--show-toplevel")).resolve() == root.resolve(), hint)
        require(git("rev-parse", "HEAD") == revision, hint)
        changes = git("status", "--porcelain=v1", "--untracked-files=all", "--", "cfac", "research/cfac")
        require(not changes, f"{hint}; changes in measured source directories: {changes}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(hint) from exc


def measure(revision, protocol=STOPPING):
    require(sys.platform == "linux" and hasattr(os, "pidfd_open"), "fresh-worker measurement requires Linux pidfds")
    require(len(os.sched_getaffinity(0)) == 1, "pin the experiment to one available CPU using taskset -c")
    require(re.fullmatch("[0-9a-f]{40}", revision), "supply the full implementation commit")
    verify_checkout(revision, protocol.design_path.parents[2])
    spec = design(protocol)
    cpu = next((l.partition(":")[2].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.machine())
    data = {"schema": protocol.schema, "design_sha256": protocol.design_sha, "design_commit": protocol.design_commit,
            "revision": revision, "environment": {"started_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(), "cpu": cpu,
            "affinity": sorted(os.sched_getaffinity(0)),
            "elapsed_ns": "Parent pre-spawn to decoded status. Acceptance includes field, problem and certificate. TIMEOUT and WORKER_ERROR include worker reap.",
            "peak_rss_kib": "Linux per-worker wait4 ru_maxrss including journal export.",
            "decision_peak_rss_kib": "Linux worker VmHWM immediately before encoding the live decision packet. Offline audit excluded."},
            "journals": {}, "trials": []}
    with tempfile.TemporaryDirectory() as td:
        for case, block, pos, policy in schedule(spec):
            row = collect(case, block, pos, policy, spec["timeout_s"], data["journals"], Path(td), protocol)
            data["trials"].append(row)
    data["environment"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    replay(data, protocol)
    return data


def main(protocol=STOPPING, *, summarize_receipt=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--worker", nargs=2, metavar=("CASE", "POLICY"), help=argparse.SUPPRESS)
    parser.add_argument("--revision")
    parser.add_argument("--sha256", help="externally retained digest of the compressed receipt")
    parser.add_argument("--journal", type=Path, help=argparse.SUPPRESS)
    if summarize_receipt:
        parser.add_argument("--summary", type=Path, help="write paired contrasts after replaying the entire receipt")
    args = parser.parse_args()
    if getattr(args, "summary", None) and not args.replay:
        parser.error("--summary requires --replay and --sha256")
    if args.worker:
        worker(*args.worker, args.journal, protocol)
    elif args.replay:
        if summarize_receipt and not args.sha256:
            parser.error("--replay requires the externally retained --sha256")
        raw = args.replay.read_bytes()
        if args.sha256:
            require(hashlib.sha256(raw).hexdigest() == args.sha256, "receipt digest mismatch")
        data = json.loads(gzip.decompress(raw))
        replay(data, protocol)
        print("Replayed", len(data["trials"]), "scheduled trials")
        if getattr(args, "summary", None):
            result = summarize_receipt(data, args.sha256)
            with args.summary.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
                output.write("\n")
    else:
        if not args.revision or args.output.exists():
            parser.error("supply --revision and a new output path")
        data = measure(args.revision, protocol)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as td:
            complete = Path(td) / "receipt.gz"
            with gzip.open(complete, "wt", encoding="utf-8", compresslevel=6) as out:
                json.dump(data, out, separators=(",", ":"))
            os.link(complete, args.output)
        print(len(data["trials"]), "trials archived in", args.output)


if __name__ == "__main__":
    main()
