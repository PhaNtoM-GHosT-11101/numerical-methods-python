"""Numerical methods with convergence analysis.

Every solver returns a `Result` carrying the full iteration history, so the
empirical order of convergence can be measured rather than assumed.
"""

from .convergence import Result, empirical_order, richardson_extrapolation
from .rootfinding import bisection, secant, newton_raphson
from .optimization import adam, gradient_descent, stochastic_gradient_descent
from .ode import euler, heun, rk4
from .linalg import conjugate_gradient, power_iteration

__all__ = [
    "Result",
    "empirical_order",
    "richardson_extrapolation",
    "bisection",
    "secant",
    "newton_raphson",
    "gradient_descent",
    "stochastic_gradient_descent",
    "adam",
    "euler",
    "heun",
    "rk4",
    "power_iteration",
    "conjugate_gradient",
]
