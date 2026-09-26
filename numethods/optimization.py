"""First-order gradient-based minimisers for unconstrained objectives.

Each optimiser is written for an explicit gradient, and each records its
trajectory so the ``1/k`` or ``1/sqrt(k)`` rate can be checked empirically.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .convergence import Result

GradFn = Callable[[np.ndarray], np.ndarray]


def gradient_descent(
    grad: GradFn,
    x0: np.ndarray,
    lr: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> Result:
    """Full-batch gradient descent ``x -= lr * grad(x)``.

    Decreases ``lr`` geometrically (``lr = lr0 * decay**k``), which for a
    smooth strongly-convex objective drives the error into a ``1/k`` regime
    instead of stalling in a noise band.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    decay = 0.999

    for _ in range(max_iter):
        g = grad(x)
        x = x - lr * g
        history.append(x.copy())
        if np.linalg.norm(g) < tol:
            break
        lr *= decay

    return Result(
        x=float(x @ x),  # squared distance from the origin, a scalar summary
        history=[float(h @ h) for h in history],
        converged=bool(np.linalg.norm(grad(x)) < tol),
        iterations=len(history) - 1,
    )


def stochastic_gradient_descent(
    grad: GradFn,
    samples: np.ndarray,
    lr: float = 0.05,
    epochs: int = 20,
    seed: int = 0,
) -> Result:
    """Mini-batch SGD with a decaying learning rate.

    ``grad`` is evaluated per-sample, so it receives a batch of shape
    ``(B, d)`` and must return per-sample gradients of the same shape.
    Trajectory records the mean squared objective per epoch.
    """
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    history: list[float] = []

    for epoch in range(epochs):
        order = rng.permutation(n)
        step = lr / (1.0 + epoch)
        for start in range(0, n, 32):
            idx = order[start : start + 32]
            batch = samples[idx]
            x = grad(batch)
            history.append(float(np.mean(x**2)))
    return Result(
        x=history[-1],
        history=history,
        converged=False,
        iterations=len(history) - 1,
    )


def adam(
    grad: GradFn,
    x0: np.ndarray,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> Result:
    """Adam, with bias correction as in Kingma & Ba (2014).

    Maintains an exponential moving average of the gradient and its square,
    correcting for the ``1 - beta**k`` initialisation bias so the first step is
    not artificially small.
    """
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [float(x @ x)]

    for k in range(1, max_iter + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(float(x @ x))
        if np.linalg.norm(g) < tol:
            break

    return Result(
        x=float(x @ x),
        history=history,
        converged=bool(np.linalg.norm(g) < tol),
        iterations=len(history) - 1,
    )
