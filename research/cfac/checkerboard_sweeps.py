"""Recheck RT1 after each extra sweep from the archived checkerboard miss."""
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path

from cfac.checker import check
from cfac.enrichment import EnrichmentRequest
from cfac.prototype import ReferenceSystem
from cfac.rt1_checker import RT1Checker
from cfac.rt1_reconstruction import RT1Reconstructor
from cfac.spec import Mesh, Problem
from cfac.stopping import inspect_stopping
from . import reserve_study, stopping_study


ARCHIVE = Path(__file__).resolve().parents[2] / "evidence/cfac/reserve-study.json.gz"
ARCHIVE_SHA = "1f2571690667df2f8959e90889c5def3097b5352421d91087aa075c6ea3d99ab"


def inspect_sweeps():
    raw = ARCHIVE.read_bytes()
    if sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ValueError("historical reserve archive digest differs")
    receipt = json.loads(gzip.decompress(raw))
    case = next(c for c in stopping_study.design(reserve_study.PROTOCOL)["cases"]
                if c["id"] == "checker_tight")
    trial = next(t for t in receipt["trials"] if t["case"] == case["id"] and t["policy"] == "cfac_reserve")
    journal = receipt["journals"][trial["journal_id"]]
    stopping_study.validate_journal(case, journal)
    archived_inspection = journal["trace"][-1]
    state = archived_inspection["after"]["parent"]
    anchor = state["problem_digest"]
    data = journal["problems"][anchor]
    problem = Problem(Mesh(data["vertices"], data["triangles"]), data["loads"], data["diffusion"],
                      data["epsilon"], data["max_actions"], data["original_affine"])
    system = ReferenceSystem(problem)
    primal, flux = state["primal"], state["flux"]
    results = []
    for step in range(problem.max_actions):
        parent = check(problem, anchor, primal, flux)
        request = EnrichmentRequest(problem, anchor, parent.state_digest, primal, flux, 1)
        candidate = RT1Reconstructor().reconstruct(request)
        candidate_digest = candidate.digest()
        cert = RT1Checker().verify(request, candidate, candidate_digest)
        if cert.status != "CERTIFIED" or cert.state_digest != candidate_digest:
            raise ValueError("could not certify the reconstructed RT1 field")
        if step == 0 and candidate_digest != archived_inspection["after"]["state_digest"]:
            raise ValueError("first reconstruction differs from the historical trial")
        balance = inspect_stopping(problem, anchor, primal, "classical", fraction=Fraction(1, 2))
        results.append({"extra_sweeps": step, "state_digest": parent.state_digest,
                        "algebraic_upper_sq": str(balance.algebraic_upper_sq),
                        "discretization_lower_sq": str(balance.discretization_lower_sq),
                        "balanced": balance.stop, "rt1_upper_sq": str(cert.upper_sq),
                        "rt1_accepted": cert.accepted})
        if balance.stop:
            break
        primal = system.primal_step(primal)
    return {"historical_archive_sha256": ARCHIVE_SHA, "initial_parent_digest": state["state_digest"],
            "epsilon": str(problem.epsilon), "fields": results}


if __name__ == "__main__":
    print(json.dumps(inspect_sweeps(), indent=2))
