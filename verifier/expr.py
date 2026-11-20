"""Restricted expression evaluator for manufactured solutions.

Only the variables ``x`` and ``y``, the constant ``pi``/``e``, numeric
literals, + - * / **, unary minus, and the functions sin, cos, tan, exp, log,
sqrt, sinh, cosh, tanh are accepted. Anything else (attribute access, names,
calls to other functions, comprehensions, ...) raises ``ExpressionError``.

Derivatives are computed with high-order central finite differences; this is
independent of any analytic gradient compiled into the solver.
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


def _compile_node(node: ast.AST) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if isinstance(node, ast.Expression):
        return _compile_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        v = float(node.value)
        return lambda x, y: np.full_like(x, v, dtype=float)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return lambda x, y: x
        if node.id == "y":
            return lambda x, y: y
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


def neg_laplacian(f, x, y, h: float = 1e-3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dxx = sum(w * f(x + o * h, y) for w, o in zip(_D2, _OFF)) / h**2
    dyy = sum(w * f(x, y + o * h) for w, o in zip(_D2, _OFF)) / h**2
    return -(dxx + dyy)
