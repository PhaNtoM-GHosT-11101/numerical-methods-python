"""Fixed-step ODE integrators for ``y' = f(t, y)``.

Orders of accuracy: explicit Euler is 1, Heun's method (explicit trapezoid) is 2,
classic RK4 is 4. The error constant differs between them, so at equal step size
RK4 is roughly ``O(h**3)`` more accurate than Euler.

The package targets scalar ``y``; the same loops extend componentwise to
vector states by letting ``f`` return an array and reducing with ``+``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .convergence import Result

ODE = Callable[[float, float], float]


def _integrate(
    method: Callable[[float, float], float],
    f: ODE,
    y0: float,
    t0: float,
    t1: float,
    n: int,
    method_name: str,
) -> Result:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    h = (t1 - t0) / n
    t, y = float(t0), float(y0)
    grid = [t]
    states = [y]

    for _ in range(n):
        y = method(f, t, y, h)
        t += h
        grid.append(t)
        states.append(y)

    return Result(
        x=states[-1],
        history=states,
        converged=False,
        iterations=n,
        residual=None,
    )


def euler(f: ODE, y0: float, t0: float, t1: float, n: int = 100) -> Result:
    """Explicit Euler: ``y_{k+1} = y_k + h f(t_k, y_k)``. Order 1.

    Cheapest possible step; error accumulates linearly in ``n``.
    """
    return _integrate(
        lambda f, t, y, h: y + h * f(t, y), f, y0, t0, t1, n, "euler"
    )


def heun(f: ODE, y0: float, t0: float, t1: float, n: int = 100) -> Result:
    """Heun / explicit trapezoid: predict with Euler, correct with the average
    of the slopes at both ends. Order 2, and stable far beyond Euler's range.
    """
    return _integrate(
        lambda f, t, y, h: y + (h / 2) * (f(t, y) + f(t + h, y + h * f(t, y))),
        f,
        y0,
        t0,
        t1,
        n,
        "heun",
    )


def rk4(f: ODE, y0: float, t0: float, t1: float, n: int = 100) -> Result:
    """Classic fourth-order Runge-Kutta. Four evaluations of ``f`` per step.

    For a smooth problem the global error is ``O(h**4)``, so halving ``h``
    should cut the error by roughly 16 -- the check the test suite asserts.
    """
    return _integrate(
        lambda f, t, y, h: y
        + (h / 6)
        * (
            f(t, y)
            + 2 * f(t + h / 2, y + (h / 2) * f(t, y))
            + 2 * f(t + h / 2, y + (h / 2) * f(t + h / 2, y + (h / 2) * f(t, y)))
            + f(t + h, y + h * f(t + h / 2, y + (h / 2) * f(t + h / 2, y + (h / 2) * f(t, y))))
        ),
        f,
        y0,
        t0,
        t1,
        n,
        "rk4",
    )
