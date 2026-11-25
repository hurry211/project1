"""Independent exact RT1 certificate on the existing P1 unit-square contract."""

from dataclasses import dataclass
from fractions import Fraction as Q
from math import factorial

from .checker import Certificate, _geometry, _grad, _load_values, _sqrt_upper, check
from .enrichment import EnrichmentRequest, RTMoments


# Reference monomials 1, r, s, r^2, rs, s^2. Integrating their products
# gives the degree-four mass matrix on (0,0), (1,0), (0,1).
_POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
_MASS = tuple(tuple(Q(factorial(i + k) * factorial(j + l), factorial(i + j + k + l + 2))
                    for k, l in _POWERS) for i, j in _POWERS)


def _square_integral(coeff):
    """Integral of the square of an affine or quadratic reference polynomial."""
    return sum(x * y * _MASS[i][j] for i, x in enumerate(coeff) if x
               for j, y in enumerate(coeff) if y)


def _outward_endpoints(cell, moment_rows):
    """Invert the two face moments of the linear normal density q.n ds/dt."""
    result = []
    for edge, sign in zip(cell.edge_ids, cell.signs):
        mean, first = moment_rows[edge]
        start, end = 4 * mean - 6 * first, 6 * first - 2 * mean
        # Reversing the edge changes both the normal and the parameter.
        result.append((start, end) if sign == 1 else (-end, -start))
    return result


def _field_on_cell(cell, faces, physical_moments):
    """Invert eight unisolvent moments using a contravariant Piola pullback.

    qhat=(a+b*r+c*s+r*(h*r+i*s), d+e*r+g*s+s*(h*r+i*s)).
    Local face endpoint pairs follow 1->2, 2->0 and 0->1.
    """
    (x0, y0), (x1, y1), (x2, y2) = cell.points
    xr, xs, yr, ys = x1 - x0, x2 - x0, y1 - y0, y2 - y0
    det = cell.twice_area
    diag, left, bottom = _outward_endpoints(cell, faces)
    a, d = -left[1], -bottom[0]
    c, e = -left[0] - a, -bottom[1] - d
    b_plus_h = diag[0] - a - d - e
    g_plus_i = diag[1] - a - d - c

    # The physical vector integral equals J times the reference vector integral.
    mx, my = physical_moments
    rx = (ys * mx - xs * my) / det - a / 2 - (b_plus_h + c) / 6
    ry = (xr * my - yr * mx) / det - d / 2 - (e + g_plus_i) / 6
    # The remaining cell-moment block is [[-2,1],[1,-2]]/24.
    h, i = -8 * (2 * rx + ry), -8 * (rx + 2 * ry)
    b, g = b_plus_h - h, g_plus_i - i
    qhat_x = (a, b, c, h, i, 0)
    qhat_y = (d, e, g, 0, h, i)
    qx = tuple((xr * u + xs * v) / det for u, v in zip(qhat_x, qhat_y))
    qy = tuple((yr * u + ys * v) / det for u, v in zip(qhat_x, qhat_y))
    divergence = ((b + g) / det, 3 * h / det, 3 * i / det)
    return qx, qy, divergence


@dataclass(frozen=True)
class RT1Certificate(Certificate):
    """Exact squared L2 norm of the original load plus the reconstructed divergence."""

    original_residual_sq: Q | None


class RT1Checker:
    """Verify one enriched flux independently of the procedure that generated it."""

    def verify(self, request, candidate, retained_candidate_digest):
        try:
            if type(request) is not EnrichmentRequest or type(candidate) is not RTMoments:
                raise ValueError("expected a frozen request and RT moment evidence")
            if request.order != 1:
                raise ValueError("only RT1 certification is available")
            candidate.validate_layout(request)
            if candidate.digest() != retained_candidate_digest:
                raise ValueError("retained RT1 digest mismatch")
            p = request.problem
            parent = check(p, request.problem_digest, request.primal, request.rt0_flux)
            if parent.status != "CERTIFIED" or parent.state_digest != request.parent_state_digest:
                raise ValueError("stale or invalid parent state")

            cells, _, _ = _geometry(p.mesh)
            gap = defect = original_defect = oscillation = Q()
            for idx, cell in enumerate(cells):
                qx, qy, div = _field_on_cell(cell, candidate.face_moments, candidate.cell_moments[idx])
                for component, grad in zip((qx, qy), _grad(cell, request.primal)):
                    mismatch = (component[0] - p.diffusion * grad, *component[1:])
                    gap += cell.twice_area * _square_integral(mismatch) / p.diffusion

                f0, f1, f2 = _load_values(p, cell, idx)
                fr, fs = f1 - f0, f2 - f0
                fh = p.loads[idx]
                defect += cell.twice_area * _square_integral((div[0] + fh, div[1], div[2]))
                oscillation += cell.twice_area * _square_integral((f0 - fh, fr, fs))
                # The linear divergence can correlate with the source oscillation.
                original_defect += cell.twice_area * _square_integral((div[0] + f0, div[1] + fr, div[2] + fs))

            correction = original_defect / (18 * p.diffusion)
            upper = gap + correction + 2 * _sqrt_upper(gap * correction)
            return RT1Certificate("CERTIFIED", upper <= p.epsilon**2, defect == 0, gap,
                                  upper, retained_candidate_digest, defect, oscillation,
                                  "validated RT1 original-PDE energy upper bound", original_defect)
        except (TypeError, ValueError, OverflowError) as exc:
            return RT1Certificate("ESTIMATED", False, False, None, None, None, None, None, str(exc), None)
