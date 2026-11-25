"""Higher-order RT evidence format and interfaces shared by producers and checkers."""

from dataclasses import dataclass
from typing import Protocol

from .checker import Certificate, check, edges
from .spec import Problem, digest_parts, ordered, rational


def _coefficients(values):
    return tuple(rational(value) for value in ordered(values))


def _order(p):
    if type(p) is not int or p < 1:
        raise ValueError("RT enrichment requires an integer order p >= 1")
    return p


@dataclass(frozen=True, slots=True)
class EnrichmentRequest:
    """An action anchored to the controller's independently retained current state."""

    problem: Problem
    problem_digest: str
    parent_state_digest: str
    primal: tuple
    rt0_flux: tuple
    order: int

    def __post_init__(self):
        _order(self.order)
        object.__setattr__(self, "primal", _coefficients(self.primal))
        object.__setattr__(self, "rt0_flux", _coefficients(self.rt0_flux))
        parent = check(self.problem, self.problem_digest, self.primal, self.rt0_flux)
        if parent.status != "CERTIFIED" or parent.state_digest != self.parent_state_digest:
            raise ValueError("invalid or stale RT0 parent state")


@dataclass(frozen=True, slots=True)
class RTMoments:
    """Physical moment evidence with one shared row per global oriented face.

    Face (i,j), i<j, uses the checker's right unit normal and t from i to j.
    Column k stores integral_e (q.n) t^k ds for k=0,...,p.
    Cell rows follow the frozen triangle order. For each physical component
    x then y, store integral_K q_d lambda_1^a lambda_2^b dx. Enumerate
    total degree m=0,...,p-1 and then a=0,...,m with b=m-a.
    """

    problem_digest: str
    parent_state_digest: str
    order: int
    face_moments: tuple
    cell_moments: tuple

    def __post_init__(self):
        _order(self.order)
        for name in ("face_moments", "cell_moments"):
            rows = tuple(_coefficients(row) for row in ordered(getattr(self, name)))
            object.__setattr__(self, name, rows)

    def digest(self):
        return digest_parts([
            "CFAC/physical-RT-moments/v1", self.problem_digest, self.parent_state_digest,
            self.order, [[str(x) for x in row] for row in self.face_moments],
            [[str(x) for x in row] for row in self.cell_moments],
        ])

    def validate_layout(self, request):
        """Check parent, polynomial order and counts. Field validity requires an RTp checker."""
        if (self.problem_digest, self.parent_state_digest, self.order) != (
                request.problem_digest, request.parent_state_digest, request.order):
            raise ValueError("moments belong to another state, problem or order")
        mesh = request.problem.mesh
        p = self.order
        if (len(self.face_moments) != len(edges(mesh))
                or any(len(row) != p + 1 for row in self.face_moments)):
            raise ValueError("incomplete global face-normal moments")
        if (len(self.cell_moments) != len(mesh.triangles)
                or any(len(row) != p * (p + 1) for row in self.cell_moments)):
            raise ValueError("incomplete cell-vector moments")


class RTpReconstructor(Protocol):
    """Future local or p-robust method producing an independently checkable proposal.

    Implementations must track local compatibility or lift an inexact primal
    residual. Claims of p-robustness also need a stability argument.
    """

    def reconstruct(self, request: EnrichmentRequest) -> RTMoments: ...


class RTpChecker(Protocol):
    """Future checker for an enriched current-field upper bound.

    Verify the retained digests and RT polynomial unisolvence on the checked
    geometry. Establish full normal-trace continuity and all divergence modes,
    enclose the weighted gap and any original-load residual, and bind the
    verdict to the enriched state. Floor tests require a separate derivation
    for its new dual space. The reference implementation supports order one.
    """

    def verify(self, request: EnrichmentRequest, candidate: RTMoments,
               retained_candidate_digest: str) -> Certificate: ...
