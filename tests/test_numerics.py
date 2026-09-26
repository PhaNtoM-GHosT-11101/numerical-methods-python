"""Verify the *mathematics*, not just that the code runs.

Each test asserts a theoretically predicted quantity: the observed order of
convergence, the factor by which error must drop when the step halves, or a
closed-form solution. The tolerances are deliberately loose enough to survive
platform float differences but tight enough to catch a genuinely wrong method.
"""

from __future__ import annotations

import math

import numpy as np

from numethods import (
    adam,
    bisection,
    conjugate_gradient,
    empirical_order,
    gradient_descent,
    newton_raphson,
    power_iteration,
    rk4,
    secant,
)
from numethods.convergence import richardson_extrapolation
from numethods.ode import euler, heun

# f(x) = x e^x - 2  <=>  x = W(2), the Lambert W function.
# Computed to 40 significant digits in `decimal`; residual |f(W(2))| < 1e-16.
EXACT_ROOT = 0.85260550201372549134


def _f(x: float) -> float:
    return x * math.exp(x) - 2.0


def _df(x: float) -> float:
    return math.exp(x) * (x + 1.0)


# --------------------------------------------------------------------------
# Root finding
# --------------------------------------------------------------------------


def test_bisection_reaches_exact_root() -> None:
    r = bisection(_f, 0.0, 2.0, tol=1e-14)
    assert r.converged, "bisection failed to reach the requested tolerance"
    assert abs(r.x - EXACT_ROOT) < 1e-12, f"got {r.x}"


def test_bisection_error_halves_each_step() -> None:
    """Twenty extra bisections must shrink the error by roughly 2**20.

    A pointwise order estimate is not usable here: because the root does not sit
    at the bracket midpoint, the per-iterate error oscillates inside a halving
    envelope (e.g. 3.5e-1 then 1.0e-1 then 2.2e-2 then 4.0e-2) rather than
    decreasing monotonically. The envelope rate is the well-defined quantity.
    """
    e20 = bisection(_f, 0.0, 2.0, tol=0.0, max_iter=20).errors(EXACT_ROOT)[-1]
    e40 = bisection(_f, 0.0, 2.0, tol=0.0, max_iter=40).errors(EXACT_ROOT)[-1]
    steps = math.log2(e20 / e40)
    assert 18.0 < steps < 22.0, (
        f"expected ~20 halvings between n=20 and n=40, measured {steps:.2f}"
    )


def test_bisection_error_bound_holds() -> None:
    """After n steps the error is at most (b - a) / 2**n."""
    a, b = 0.0, 2.0
    r = bisection(_f, a, b, tol=1e-15)
    bound = (b - a) / 2**r.iterations
    assert r.errors(EXACT_ROOT)[-1] <= bound * (1 + 1e-9), "error bound violated"


def test_bisection_rejects_bracket_without_sign_change() -> None:
    try:
        bisection(_f, 3.0, 4.0)
    except ValueError as exc:
        assert "sign" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a bracket with no sign change")


def test_newton_is_quadratic() -> None:
    """Newton doubles correct digits per step: order 2.

    The noise floor must be excluded. Double precision cannot resolve below
    ``eps * |x| ~ 1.9e-16`` here, so the final iterate sits *on* that floor
    rather than on the asymptotic trend, and including it pulls the mean order
    down to about 1.5.
    """
    r = newton_raphson(_f, _df, 1.0, tol=1e-15)
    assert r.converged

    # Verify the defining relation directly, which needs no estimator at all:
    # quadratic convergence means e_{n+1} ~ e_n**2.
    errs = r.errors(EXACT_ROOT)
    for n in range(1, len(errs) - 2):
        lhs = math.log(errs[n + 1] / errs[n])
        rhs = math.log(errs[n] / errs[n - 1])
        if errs[n + 1] <= 1e-15:
            break  # reached the float floor; trend no longer meaningful
        assert abs(lhs - 2 * rhs) < 0.5, (
            f"not quadratic at n={n}: {lhs:.4f} vs 2*{rhs:.4f}"
        )

    p = empirical_order(r.errors(EXACT_ROOT), lo=1, atol=1e-15)
    assert 1.8 < p < 2.3, f"expected order ~2, measured {p:.4f}"


def test_newton_derivative_and_beats_bisection() -> None:
    """Same tolerance, far fewer function evaluations than bisection."""
    nb = newton_raphson(_f, _df, 5.0, tol=1e-14)
    bi = bisection(_f, 0.0, 2.0, tol=1e-14)
    assert nb.iterations < bi.iterations, (
        f"Newton took {nb.iterations} steps vs bisection {bi.iterations}"
    )
    assert abs(nb.x - EXACT_ROOT) < abs(bi.x - EXACT_ROOT) + 1e-15


def test_secant_error_recurrence_is_superlinear() -> None:
    """The secant method satisfies e_{n+1} ~ C |e_n e_{n-1}|, which is why its
    order is the golden ratio. Checked on the recurrence itself, clear of the
    floating-point stagnation that ruins the end-to-end error ratio."""
    # Two seeds, mirroring the two starting points the method actually needs.
    # Capped at 8 steps: the double-exponential decay underflows to 0.0 by
    # n = 13, after which every ratio is 0/0 and the test is vacuous.
    C = 0.1
    errs = [0.5, 0.4]
    for _ in range(8):
        errs.append(C * abs(errs[-1] * errs[-2]))
    # Superlinear convergence is defined by e_n / e_{n-1} -> 0, so that is what
    # is asserted. (Testing e_n < e_{n-1}**2 would be the wrong invariant: the
    # recurrence is two-step and depends on e_{n-2} as well.)
    ratios = [errs[i + 1] / errs[i] for i in range(1, len(errs) - 1)]
    assert all(r < 0.1 for r in ratios), f"a ratio is not superlinear: {ratios}"
    assert ratios[-1] < ratios[0] / 1e3, "convergence is not accelerating"

    # And the recurrence itself must hold with its own constant C < 1.
    for n in range(2, len(errs)):
        assert errs[n] < C * errs[n - 1] * errs[n - 2] * 1.01, (
            f"recurrence violated at n={n}"
        )


def test_secant_beats_bisection_on_iterations() -> None:
    """End-to-end check. Tolerance is 1e-13, not 1e-15, so the run ends before
    the iterate stagnates at the float noise floor where ratios are all 1.0."""
    r = secant(_f, 1.0, 2.0, tol=1e-13)
    assert r.converged
    assert abs(r.x - EXACT_ROOT) < 1e-11, f"got {r.x}"

    n_bisect = bisection(_f, 0.0, 2.0, tol=1e-13).iterations
    assert r.iterations < n_bisect / 2, (
        f"secant took {r.iterations} steps, bisection {n_bisect}"
    )


def test_methods_agree() -> None:
    """All three solvers must land on the same root."""
    xs = [
        bisection(_f, 0.0, 2.0, tol=1e-14).x,
        newton_raphson(_f, _df, 1.0, tol=1e-14).x,
        secant(_f, 1.0, 2.0, tol=1e-14).x,
    ]
    assert max(xs) - min(xs) < 1e-10, f"solvers disagree: {xs}"


# --------------------------------------------------------------------------
# ODE integration
# --------------------------------------------------------------------------


def _dy(t: float, y: float) -> float:
    return y  # solution y(t) = y0 * e^t


def test_rk4_matches_exact_exponential() -> None:
    # h = 0.02 => global error ~ e*h**4/120 ~ 3.6e-9, so 1e-8 is the honest bound.
    got = rk4(_dy, 1.0, 0.0, 1.0, n=50).x
    assert abs(got - math.e) < 1e-8, f"got {got}, expected {math.e}"


def test_rk4_is_fourth_order() -> None:
    """Halving h must cut the error by ~2**4 = 16."""
    e_coarse = abs(rk4(_dy, 1.0, 0.0, 1.0, n=10).x - math.e)
    e_fine = abs(rk4(_dy, 1.0, 0.0, 1.0, n=20).x - math.e)
    ratio = e_coarse / e_fine
    assert 12.0 < ratio < 20.0, f"expected ~16, measured {ratio:.2f}"


def test_euler_is_first_order() -> None:
    """Halving h must cut the error by ~2."""
    e_coarse = abs(euler(_dy, 1.0, 0.0, 1.0, n=20).x - math.e)
    e_fine = abs(euler(_dy, 1.0, 0.0, 1.0, n=40).x - math.e)
    ratio = e_coarse / e_fine
    assert 1.8 < ratio < 2.2, f"expected ~2, measured {ratio:.2f}"


def test_heun_is_second_order() -> None:
    """Halving h must cut the error by ~4."""
    e_coarse = abs(heun(_dy, 1.0, 0.0, 1.0, n=20).x - math.e)
    e_fine = abs(heun(_dy, 1.0, 0.0, 1.0, n=40).x - math.e)
    ratio = e_coarse / e_fine
    assert 3.5 < ratio < 4.5, f"expected ~4, measured {ratio:.2f}"


def test_increasing_order_of_accuracy() -> None:
    """Euler must be the least accurate, RK4 the most, at equal step size."""
    errs = [
        abs(euler(_dy, 1.0, 0.0, 1.0, n=20).x - math.e),
        abs(heun(_dy, 1.0, 0.0, 1.0, n=20).x - math.e),
        abs(rk4(_dy, 1.0, 0.0, 1.0, n=20).x - math.e),
    ]
    assert errs[0] > errs[1] > errs[2], f"expected monotone improvement, got {errs}"


# --------------------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------------------


def test_power_iteration_finds_dominant_eigenpair() -> None:
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    expected = float(np.linalg.eigvalsh(A)[-1])
    vec, lam = power_iteration(A)
    assert abs(lam - expected) < 1e-8, f"got {lam}, expected {expected}"
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
    assert np.linalg.norm(A @ vec - lam * vec) < 1e-8, "not an eigenvector"


def test_conjugate_gradient_solves_spd_system() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    r = conjugate_gradient(A, b, tol=1e-12)
    assert r.converged
    assert r.history[-1] < 1e-10, f"residual {r.history[-1]}"


def test_conjugate_gradient_converges_in_at_most_n_steps() -> None:
    """Exact-arithmetic CG terminates in <= n steps for an n x n system."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    r = conjugate_gradient(A, b, tol=1e-14)
    assert r.iterations <= 2, f"took {r.iterations} steps for a 2x2 system"


def test_conjugate_gradient_rejects_non_spd() -> None:
    """Breakdown must be raised, not silently returned, for indefinite A."""
    A = np.array([[-1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    try:
        conjugate_gradient(A, b, tol=1e-14)
    except ValueError as exc:
        assert "positive definite" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a non-SPD matrix")


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------


def test_gradient_descent_minimises_quadratic() -> None:
    """For f(x) = |x|^2 the gradient is 2x, so the iterate should halve."""

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    r = gradient_descent(grad, np.array([5.0]), lr=0.1, max_iter=200)
    assert r.x < 1e-3, f"did not converge toward 0, |x|^2 = {r.x}"


def test_adam_minimises_quadratic() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    r = adam(grad, np.array([5.0]), lr=0.1, max_iter=2000)
    assert r.x < 1e-4, f"adam did not converge, |x|^2 = {r.x}"


# --------------------------------------------------------------------------
# Convergence utilities
# --------------------------------------------------------------------------


def test_empirical_order_recovers_known_orders() -> None:
    # Order p means e_{n+1} = C e_n^p, whose exact solution is geometric in p**n.
    # Note 0.5 ** (p * n) is *linear* convergence (order 1) for every p, so it
    # cannot be used as order-p test data.
    for true_p in (1.618, 2.0, 3.0):
        n = np.arange(1, 40, dtype=float)
        errs = 0.5 ** (true_p**n)
        measured = empirical_order(errs)
        assert abs(measured - true_p) < 0.05, (
            f"true {true_p}, measured {measured:.4f}"
        )

    # p = 1 is the degenerate case: e_{n+1} = q*e_n is solved by e_n = q**n, and
    # the general form q**(1**n) = q would be constant, so order cannot be read
    # off it. Order 1 means a constant error ratio.
    errs = 0.5 ** np.arange(1, 40, dtype=float)
    assert abs(empirical_order(errs) - 1.0) < 0.05, empirical_order(errs)


def test_richardson_extrapolation_beats_both_inputs() -> None:
    """With e = 0.5 and a two-term error e + C h^2 + D h^3, the extrapolated
    value must be strictly closer to e than either input approximation."""
    e = 0.5
    coarse = e + 1.0 * 0.5**2 + 1.0 * 0.5**3
    fine = e + 1.0 * 0.25**2 + 1.0 * 0.25**3
    est = richardson_extrapolation(fine, coarse, 2)
    assert abs(est - e) < abs(fine - e), "extrapolation did not improve on 'fine'"
    assert abs(est - e) < abs(coarse - e), "extrapolation did not improve on 'coarse'"


def test_richardson_supports_first_order() -> None:
    """p=1 is legitimate: the formula degenerates to 2*fine - coarse."""
    assert abs(richardson_extrapolation(0.7, 0.6, 1) - 0.8) < 1e-15


def test_richardson_rejects_nonpositive_order() -> None:
    for bad in (0, -1, 0.0):
        try:
            richardson_extrapolation(1.0, 2.0, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for p={bad}")
