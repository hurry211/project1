"""Evaluation-only references. Solvers, certifiers and controllers must not import this module."""

from dataclasses import dataclass
from fractions import Fraction

from .prototype import _dot, _matvec, _solve_spd


def optimal_fields(system):
    """Exact rational *discrete* optimizers in the fixed P1/RT0 pair."""
    free = system.free_nodes
    matrix = [[system.primal_matrix[i][j] for j in free] for i in free]
    primal = list(system.zero_primal())
    for i, x in zip(free, _solve_spd(matrix, [system.load[i] for i in free]), strict=True):
        primal[i] = x
    # Stream constants have zero curl. Eliminate one value exactly.
    rhs = system.stream_rhs(system.feasible_flux)
    matrix = [row[1:] for row in system.stream_matrix[1:]]
    stream = (Fraction(0),) + _solve_spd(matrix, rhs[1:])
    return tuple(primal), system.stream_correction(system.feasible_flux, stream)


def primal_energy(system, primal):
    return _dot(primal, _matvec(system.primal_matrix, primal))


def flux_energy(system, flux):
    """Integrate ||q||^2_(a^-1) via barycentric mass moments at the vertices."""
    total = Fraction(0)
    for cell in system.cells:
        p = [system.vertices[i] for i in cell.nodes]
        normals = [sign * flux[edge] for edge, sign in cell.faces]
        values = [
            tuple(sum(v * (vertex[d] - point[d]) for v, point in zip(normals, p))
                  / (2 * cell.area) for d in range(2))
            for vertex in p
        ]
        sum_q = tuple(map(sum, zip(*values)))
        # Integral of lambda_i lambda_j is |K| (1 + delta_ij) / 12.
        moment = _dot(sum_q, sum_q) + sum(_dot(q, q) for q in values)
        total += cell.area * moment / (12 * system.problem.diffusion)
    return total


def _atan_interval(x, terms):
    terms_sum = sum((-1)**k * x**(2*k + 1) / (2*k + 1) for k in range(terms))
    other = terms_sum + (-1)**terms * x**(2*terms + 1) / (2*terms + 1)
    return min(terms_sum, other), max(terms_sum, other)


def square_energy_interval(load, diffusion, cutoff):
    """Continuous constant-load energy using the sine series and a tail bound.

    Odd modes contribute 64 f^2 / (a pi^6 m^2 n^2 (m^2+n^2)).
    Modes outside the cutoff box contribute to the dimensionless sum at
    most 2 sum_{m>N} m^-4 sum_{n>=1} n^-2 <= 4/(3 N^3).
    """
    if type(cutoff) is not int or cutoff < 1 or diffusion <= 0:
        raise ValueError("expected a positive cutoff and diffusion")
    a_lo, a_hi = _atan_interval(Fraction(1, 5), 12)
    b_lo, b_hi = _atan_interval(Fraction(1, 239), 12)
    pi_lo, pi_hi = 16 * a_lo - 4 * b_hi, 16 * a_hi - 4 * b_lo
    partial = sum(Fraction(1, m*m * n*n * (m*m + n*n))
                  for m in range(1, cutoff + 1, 2) for n in range(1, cutoff + 1, 2))
    tail = Fraction(4, 3 * cutoff**3)
    scale = 64 * Fraction(load)**2 / Fraction(diffusion)
    return scale * partial / pi_hi**6, scale * (partial + tail) / pi_lo**6


@dataclass(frozen=True)
class FourTerms:
    primal_discretization_sq: tuple
    primal_algebraic_sq: Fraction
    dual_discretization_sq: tuple
    dual_algebraic_sq: Fraction
    pair_floor_sq: Fraction


def four_terms(system, primal, flux, cutoff=15):
    """Split both discretization floors using an enclosure of continuous energy.

    The two floors share the same continuous energy with opposite signs.
    Their sum therefore has an exact rational value.
    """
    p = system.problem
    if len(set(p.loads)) != 1 or getattr(p, "original_affine", None) is not None:
        raise ValueError("the continuous energy reference requires a constant original load")
    if len(primal) != len(system.vertices) or any(
            primal[i] for i, (x, y) in enumerate(system.vertices) if x in (0, 1) or y in (0, 1)):
        raise ValueError("expected a conforming homogeneous primal field")
    if len(flux) != len(system.edges) or any(
            sum(sign * flux[e] for e, sign in cell.faces) != -cell.area * f
            for cell, f in zip(system.cells, p.loads, strict=True)):
        raise ValueError("four-term split requires exact equilibrium")
    primal_star, flux_star = optimal_fields(system)
    p_opt = primal_energy(system, primal_star)
    d_opt = flux_energy(system, flux_star)
    lo, hi = square_energy_interval(p.loads[0], p.diffusion, cutoff)
    primal_diff = tuple(x - y for x, y in zip(primal, primal_star, strict=True))
    dual_diff = tuple(x - y for x, y in zip(flux, flux_star, strict=True))
    return FourTerms((max(0, lo - p_opt), hi - p_opt), primal_energy(system, primal_diff),
                     (max(0, d_opt - hi), d_opt - lo), flux_energy(system, dual_diff),
                     d_opt - p_opt)
