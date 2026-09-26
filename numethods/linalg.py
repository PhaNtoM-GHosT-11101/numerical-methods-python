"""Dense linear-algebra routines, written out so the mechanics are visible.

Implemented in terms of NumPy primitives but without calling the corresponding
``numpy.linalg`` solver, to make the iteration structure explicit.
"""

from __future__ import annotations

import numpy as np

from .convergence import Result


def power_iteration(A: np.ndarray, n_iter: int = 1000, tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Estimate the dominant eigenpair of ``A`` by repeated multiplication.

    ``x_{k+1} = A x_k / ||A x_k||`` converges to the eigenvector of the largest
    magnitude eigenvalue; the Rayleigh quotient ``x^T A x / x^T x`` gives the
    eigenvalue once converged.

    Returns
    -------
    (eigenvector, eigenvalue)
        Unit-norm eigenvector and its Rayleigh quotient.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)

    for _ in range(n_iter):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            raise ValueError("A maps the current vector to zero")
        x_new = y / norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    lam = float(x @ (A @ x) / (x @ x))
    return x, lam


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> Result:
    """Solve ``A x = b`` for symmetric positive-definite ``A`` via CG.

    Builds mutually orthogonal search directions from the Krylov space
    ``span(b, A b, A^2 b, ...)``. Converges in at most ``n`` steps in exact
    arithmetic; in floating point the measured residual is tracked in
    ``history``.

    Raises
    ------
    ValueError
        If ``A`` is not square, ``b`` is not a vector of matching length, or the
        breakdown condition ``p^T A p <= 0`` is hit (which signals that ``A``
        is not positive definite).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b length must match A dimension")

    n = A.shape[0]
    max_iter = n if max_iter is None else max_iter
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)
    history = [float(np.linalg.norm(r))]

    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0:
            raise ValueError(
                "breakdown: p^T A p <= 0, so A is not positive definite"
            )
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        history.append(float(np.sqrt(rs_new)))
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return Result(
        x=float(x @ x),
        history=history,
        converged=bool(np.sqrt(rs_new) < tol),
        iterations=len(history) - 1,
    )
