"""Restricted expression evaluator and exact AD derivatives for manufactured solutions.

Only the variables ``x``, ``y``, and L-shape polar angle ``theta_lshape``,
the constant ``pi``/``e``, numeric literals, + - * / **, unary minus, and the
functions sin, cos, tan, exp, log, sqrt, sinh, cosh, tanh are accepted. Anything
else (attribute access, unknown names, arbitrary calls, comprehensions, ...)
raises ``ExpressionError``.

Provides both 4th-order central finite-difference operators (historical
check_mms_consistency & unit-square path) and exact AST forward-mode dual
gradients (essential for re-entrant L-shape and sub-millimeter AMR cells).
"""

from __future__ import annotations

import ast
from typing import Callable

import numpy as np

_FUNCS: dict[str, Callable] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "sqrt": np.sqrt, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}
_CONSTS = {"pi": np.pi, "e": np.e}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}


class ExpressionError(ValueError):
    pass


def _theta_lshape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = x - 0.5
    dy = y - 0.5
    phi = np.arctan2(dy, dx)
    phi = np.where(phi < 0.5 * np.pi - 1e-12, phi + 2.0 * np.pi, phi)
    return phi - 0.5 * np.pi


def _theta_lshape_dual(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = x - 0.5
    dy = y - 0.5
    r2 = dx * dx + dy * dy
    safe_r2 = np.maximum(r2, 1e-30)
    theta = _theta_lshape(x, y)
    dtx = np.where(r2 < 1e-30, 0.0, -dy / safe_r2)
    dty = np.where(r2 < 1e-30, 0.0, dx / safe_r2)
    return theta, dtx, dty


def _compile_node(node: ast.AST) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if isinstance(node, ast.Expression):
        return _compile_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        try:
            v = float(node.value)
        except OverflowError as exc:
            raise ExpressionError("numeric literal exceeds floating-point range") from exc
        if not np.isfinite(v):
            raise ExpressionError("numeric literal must be finite")
        return lambda x, y: np.full_like(x, v, dtype=float)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x, y: x
        if node.id == "y":
            return lambda x, y: y
        if node.id == "theta_lshape":
            return _theta_lshape
        if node.id in _CONSTS:
            c = _CONSTS[node.id]
            return lambda x, y: np.full_like(x, c, dtype=float)
        raise ExpressionError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        op = _BINOPS[type(node.op)]
        lf, rf = _compile_node(node.left), _compile_node(node.right)
        return lambda x, y: op(lf(x, y), rf(x, y))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        f = _compile_node(node.operand)
        if isinstance(node.op, ast.USub):
            return lambda x, y: -f(x, y)
        return f
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ExpressionError("only whitelisted functions may be called")
        if len(node.args) != 1 or node.keywords:
            raise ExpressionError("functions take exactly one positional argument")
        fn = _FUNCS[node.func.id]
        a = _compile_node(node.args[0])
        return lambda x, y: fn(a(x, y))
    raise ExpressionError(f"unsupported syntax: {type(node).__name__}")


def _compile_dual_node(node: ast.AST) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if isinstance(node, ast.Expression):
        return _compile_dual_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        try:
            v = float(node.value)
        except OverflowError as exc:
            raise ExpressionError("numeric literal exceeds floating-point range") from exc
        if not np.isfinite(v):
            raise ExpressionError("numeric literal must be finite")
        return lambda x, y: (np.full_like(x, v, dtype=float), np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float))
    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x, y: (x, np.ones_like(x, dtype=float), np.zeros_like(x, dtype=float))
        if node.id == "y":
            return lambda x, y: (y, np.zeros_like(x, dtype=float), np.ones_like(x, dtype=float))
        if node.id == "theta_lshape":
            return _theta_lshape_dual
        if node.id in _CONSTS:
            c = _CONSTS[node.id]
            return lambda x, y: (np.full_like(x, c, dtype=float), np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float))
        raise ExpressionError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        lf, rf = _compile_dual_node(node.left), _compile_dual_node(node.right)
        if isinstance(node.op, ast.Add):
            return lambda x, y: tuple(a + b for a, b in zip(lf(x, y), rf(x, y)))  # type: ignore[return-value]
        if isinstance(node.op, ast.Sub):
            return lambda x, y: tuple(a - b for a, b in zip(lf(x, y), rf(x, y)))  # type: ignore[return-value]
        if isinstance(node.op, ast.Mult):
            def _mul(x, y):
                u, ux, uy = lf(x, y)
                v, vx, vy = rf(x, y)
                return u * v, ux * v + u * vx, uy * v + u * vy
            return _mul
        if isinstance(node.op, ast.Div):
            def _div(x, y):
                u, ux, uy = lf(x, y)
                v, vx, vy = rf(x, y)
                inv2 = 1.0 / (v * v)
                return u / v, (ux * v - u * vx) * inv2, (uy * v - u * vy) * inv2
            return _div
        # Power u**v: evaluate constant exponent cleanly when possible (handles r2**(1/3) at r=0)
        def _pow(x, y):
            u, ux, uy = lf(x, y)
            v, vx, vy = rf(x, y)
            val = np.power(u, v)
            const_exp = np.all(vx == 0.0) and np.all(vy == 0.0)
            if const_exp:
                scale = np.where((u == 0.0) & (v >= 1.0), 0.0, v * np.power(np.maximum(u, 1e-300), v - 1.0))
                scale = np.where((u == 0.0) & (v < 1.0), 0.0, scale) if np.any((u == 0.0) & (ux == 0.0) & (uy == 0.0)) else np.where(u == 0.0, 0.0, v * np.power(u, v - 1.0))
                return val, scale * ux, scale * uy
            lgu = np.log(np.maximum(u, 1e-300))
            return val, val * (vx * lgu + v * ux / np.maximum(u, 1e-300)), val * (vy * lgu + v * uy / np.maximum(u, 1e-300))
        return _pow
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        f = _compile_dual_node(node.operand)
        if isinstance(node.op, ast.USub):
            def _neg(x, y):
                u, ux, uy = f(x, y)
                return -u, -ux, -uy
            return _neg
        return f
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ExpressionError("only whitelisted functions may be called")
        if len(node.args) != 1 or node.keywords:
            raise ExpressionError("functions take exactly one positional argument")
        name = node.func.id
        af = _compile_dual_node(node.args[0])
        def _call(x, y):
            u, ux, uy = af(x, y)
            if name == "sin":
                v, dv = np.sin(u), np.cos(u)
            elif name == "cos":
                v, dv = np.cos(u), -np.sin(u)
            elif name == "tan":
                c = np.cos(u)
                v, dv = np.tan(u), 1.0 / (c * c)
            elif name == "exp":
                v = np.exp(u); dv = v
            elif name == "log":
                v, dv = np.log(u), 1.0 / u
            elif name == "sqrt":
                v = np.sqrt(u)
                dv = np.where(v == 0.0, 0.0, 0.5 / np.maximum(v, 1e-300))
            elif name == "sinh":
                v, dv = np.sinh(u), np.cosh(u)
            elif name == "cosh":
                v, dv = np.cosh(u), np.sinh(u)
            else:  # tanh
                t = np.tanh(u)
                v, dv = t, 1.0 - t * t
            return v, dv * ux, dv * uy
        return _call
    raise ExpressionError(f"unsupported syntax: {type(node).__name__}")


def compile_expr(text: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in '{text}': {exc}") from exc
    f = _compile_node(tree)

    def wrapped(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(all="ignore"):
            out = f(x, y)
        return np.broadcast_to(out, np.broadcast(x, y).shape).astype(float)

    return wrapped


def compile_exact_gradient(text: str) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Compile forward-mode exact AST gradient (grad_x, grad_y) directly from expression text."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in '{text}': {exc}") from exc
    df = _compile_dual_node(tree)

    def wrapped(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shp = np.broadcast(x, y).shape
        xb = np.broadcast_to(x, shp).astype(float)
        yb = np.broadcast_to(y, shp).astype(float)
        with np.errstate(all="ignore"):
            _, gx, gy = df(xb, yb)
        return np.broadcast_to(gx, shp).astype(float), np.broadcast_to(gy, shp).astype(float)

    return wrapped


# 4th-order central difference weights.
_D1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
_D2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
_OFF = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def gradient(f, x, y, h: float = 1e-3):
    """4th-order central-difference gradient; truncation O(h^4)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    gx = sum(w * f(x + o * h, y) for w, o in zip(_D1, _OFF) if w != 0.0) / h
    gy = sum(w * f(x, y + o * h) for w, o in zip(_D1, _OFF) if w != 0.0) / h
    return gx, gy


def neg_laplacian(f, x, y, h: float = 1e-3, *, diffusion_x: float = 1.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dxx = sum(w * f(x + o * h, y) for w, o in zip(_D2, _OFF)) / h**2
    dyy = sum(w * f(x, y + o * h) for w, o in zip(_D2, _OFF)) / h**2
    return -(diffusion_x * dxx + dyy)
