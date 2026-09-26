"""Scalar root-finding solvers, each instrumented for convergence analysis.

The three methods bracket-free/bracket-based here span the classic orders of
convergence:

* ``bisection``  -- linear, order 1, guaranteed for continuous ``f`` on a bracket
* ``secant``     -- superlinear, order phi = (1 + sqrt(5)) / 2 ~ 1.618
* ``newton_raphson`` -- quadratic, order 2, needs ``f'``

Reference problem: ``f(x) = x*exp(x) - 2 = 0`` has the closed-form root
``W(2) ~ 1.1461932206205820``, which the test suite uses to verify that the
measured orders match the theory.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .convergence import Result

Scalar = Callable[[float], float]


def _finish(
    x: float,
    history: list[float],
    f: Scalar | None,
    tol: float,
) -> Result:
    residual = abs(f(x)) if f is not None else None
    return Result(
        x=x,
        history=history,
        converged=residual is not None and residual <= tol,
        iterations=len(history) - 1,
        residual=residual,
    )


def bisection(
    f: Scalar,
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Result:
    """Bisect a sign-changing bracket ``[a, b]``.

    Halves the bracket each step, so the error is bounded by ``(b - a) / 2**n``
    after ``n`` steps regardless of the function's smoothness. Robust but slow.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` share a sign, or ``a >= b``.
    """
    if a >= b:
        raise ValueError(f"require a < b, got a={a}, b={b}")
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f does not change sign on [{a}, {b}]: f(a)={fa:.6g}, f(b)={fb:.6g}"
        )

    history = [0.5 * (a + b)]
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        history.append(m)
        fm = f(m)
        if abs(fm) <= tol:
            return _finish(m, history, f, tol)
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return _finish(history[-1], history, f, tol)


def secant(
    f: Scalar,
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Result:
    """Secant method: replace Newton's derivative with a finite difference.

    Uses points ``(x0, f(x0))`` and ``(x1, f(x1))`` to build the linear
    interpolant, then takes its root. Superlinear with order phi ~ 1.618, and
    needs no derivative, at the cost of two function evaluations per step.
    """
    f0, f1 = f(x0), f(x1)
    if f1 == f0:
        raise ValueError("f(x0) == f(x1); secant slope is undefined")

    history = [x0, x1]
    for _ in range(max_iter):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        f2 = f(x2)
        if abs(f2) <= tol:
            return _finish(x2, history, f, tol)
        x0, f0, x1, f1 = x1, f1, x2, f2
    return _finish(history[-1], history, f, tol)


def newton_raphson(
    f: Scalar,
    df: Scalar,
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Result:
    """Newton-Raphson: ``x_{n+1} = x_n - f(x_n) / f'(x_n)``.

    Locally quadratic, so the number of correct digits roughly doubles per
    step. Requires an analytic derivative; ``df`` is not approximated.

    Raises
    ------
    ValueError
        If the derivative vanishes at the current iterate.
    """
    history = [x0]
    x = x0
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        if abs(fx) <= tol:
            return _finish(x, history, f, tol)
        if dfx == 0:
            raise ValueError(f"f'(x) = 0 at x={x!r}; Newton iteration undefined")
        x = x - fx / dfx
        history.append(x)
    return _finish(x, history, f, tol)
