"""Iteration history bookkeeping and empirical convergence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Result:
    """Outcome of an iterative solver.

    Attributes
    ----------
    x:
        Final approximation.
    history:
        Every iterate, including the initial guess at index 0.
    converged:
        Whether the solver's stopping criterion was satisfied.
    iterations:
        Number of steps taken, i.e. ``len(history) - 1``.
    residual:
        ``abs(f(x))`` at the returned point, when ``f`` was supplied.
    """

    x: float
    history: list[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    residual: float | None = None

    def errors(self, exact: float) -> np.ndarray:
        """Absolute error at each iterate against a known exact value."""
        return np.abs(np.asarray(self.history) - exact)

    def step_sizes(self) -> np.ndarray:
        """Successive step magnitudes ``|x_n - x_{n-1}|``."""
        h = np.asarray(self.history)
        return np.abs(np.diff(h))


def empirical_order(
    errors: np.ndarray,
    lo: int = 0,
    hi: int | None = None,
    atol: float = 0.0,
) -> float:
    """Estimate the order of convergence ``p`` from an error sequence.

    Uses the standard log-ratio estimator

        p_n = log(e_{n+1} / e_n) / log(e_n / e_{n-1})

    Parameters
    ----------
    errors:
        Strictly decreasing positive errors ``e_1, e_2, ...``.
    lo, hi:
        Slice bounds over which to average ``p``, used to skip the transient
        region where the asymptotic regime has not yet been reached.
    atol:
        Absolute floor below which an error is treated as numerical noise and
        discarded. A double-precision solver cannot beat ``eps * |x|``, so its
        final iterate sits *at* that floor rather than on the asymptotic trend;
        including it biases the mean order downwards. Pass the solver's
        attainable accuracy here when analysing a real run.

    Returns
    -------
    float
        Mean estimated order, or ``nan`` if there are too few usable points.
    """
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if atol > 0:
        e = e[e > atol]
    if e.size < 3:
        return float("nan")

    # Strip the stagnating *tail* only. Iterative solvers flatten out at the
    # float noise floor, where ratios become 1.0 by construction and would
    # drag the mean toward zero. The head must be kept even when the sequence
    # is not monotone there (bisection's error oscillates within a shrinking
    # envelope, and truncating at the first increase would discard everything).
    k = e.size
    while k > 1 and e[k - 1] >= e[k - 2]:
        k -= 1
    e = e[:k]
    if e.size < 3:
        return float("nan")

    hi = e.size if hi is None else min(hi, e.size)
    orders = []
    # Start at n=1: a three-point difference needs e[n-1], and n=0 would reach
    # it by negative indexing, pairing the largest error with the smallest.
    for n in range(max(lo, 1), hi - 1):
        num = np.log(e[n + 1] / e[n])
        den = np.log(e[n] / e[n - 1])
        if den == 0 or not np.isfinite(num) or not np.isfinite(den):
            continue
        orders.append(num / den)
    return float(np.mean(orders)) if orders else float("nan")


def richardson_extrapolation(fine: float, coarse: float, p: float) -> float:
    """One step of Richardson extrapolation for a method of known order ``p``.

    If ``coarse = e + C h^p`` and ``fine = e + C (h/2)^p`` approximate the same
    limit ``e``, the combination

        e ~ (2^p * fine - coarse) / (2^p - 1)

    cancels the leading error term exactly, because ``2^p * C (h/2)^p = C h^p``.
    The weight ``2^p`` (not ``p``) is what makes the leading term vanish; the
    ``p``-weighted form is only correct in the first-order limit.

    Parameters
    ----------
    fine:
        Approximation produced with the halved step size.
    coarse:
        Approximation produced with the full step size.
    p:
        Known order of convergence, typically 1, 2, 3 or 4.

    Returns
    -------
    float
        The accelerated estimate of the limit.
    """
    if p <= 0:
        raise ValueError(f"order must be positive, got {p}")
    ratio = 2.0**p
    return (ratio * fine - coarse) / (ratio - 1.0)


def make_counter(f: Callable[[float], float], tol: float, max_iter: int) -> Callable[[float], bool]:
    """Build a predicate reporting whether ``|f(x)| <= tol``.

    Exposed so solvers and callers agree on a single stopping rule.
    """

    def done(x: float) -> bool:
        if not np.isfinite(x) or abs(f(x)) > tol:
            return False
        return True

    done.max_iter = max_iter  # type: ignore[attr-defined]
    return done
