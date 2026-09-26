#!/usr/bin/env python3
"""Compare the three root-finders and the three ODE integrators empirically.

Produces the four figures used in the README and prints the measured order of
convergence beside the theoretical value for each method.

    python3 scripts/convergence_comparison.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, so this runs over SSH
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numethods import bisection, empirical_order, newton_raphson, rk4, secant  # noqa: E402
from numethods.ode import euler, heun  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)

# x*e^x = 2  <=>  x = W(2)
EXACT = 0.85260550201372549134
FLOOR = 1e-15  # double precision cannot resolve below ~eps*|x| here


def f(x: float) -> float:
    return x * math.exp(x) - 2.0


def df(x: float) -> float:
    return math.exp(x) * (x + 1.0)


def dy(t: float, y: float) -> float:
    return y  # y(t) = e^t


def main() -> None:
    # ---------------- root finding ----------------
    runs = {
        "Bisection (order 1)": bisection(f, 0.0, 2.0, tol=1e-15),
        "Secant (order ~1.618)": secant(f, 1.0, 1.5, tol=1e-15),
        "Newton (order 2)": newton_raphson(f, df, 1.0, tol=1e-15),
    }

    print("Root finding:  x e^x - 2 = 0,  exact root W(2) =", EXACT)
    print(f"{'method':<24}{'iters':>7}{'|error|':>13}{'measured p':>13}{'theory':>9}")
    print("-" * 66)
    for name, r in runs.items():
        errs = r.errors(EXACT)
        p = empirical_order(errs, atol=FLOOR)
        final = errs[errs > FLOOR][-1] if (errs > FLOOR).any() else errs[-1]
        theory = {"Bisection (order 1)": 1.0, "Secant (order ~1.618)": 1.618, "Newton (order 2)": 2.0}[name]
        note = {
            "Bisection (order 1)": "  (estimator invalid: error oscillates",
            "Secant (order ~1.618)": "  (estimator invalid: two-step recurrence)",
            "Newton (order 2)": "",
        }[name]
        if name.startswith("Bisection"):
            note += " in a halving envelope)"
        elif name.startswith("Secant"):
            note += ", not e_n = C e_n^p)"
        print(f"{name:<24}{r.iterations:>7}{final:>13.3e}{p:>13.3f}{theory:>9.3f}{note}")

    print(
        "\nOnly Newton's column is a valid order estimate. Bisection's error"
        "\noscillates inside a halving envelope rather than decreasing, and the"
        "\nsecant recurrence e_n ~ C e_n e_{n-1} is not of the form"
        "\ne_{n+1} = C e_n^p that the log-ratio estimator assumes, so both"
        "\nreport a number without a convergence-order meaning."
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in runs.items():
        errs = r.errors(EXACT)
        errs = np_where(errs > FLOOR, errs, np.nan)
        ax.semilogy(np_arange(len(errs)), errs, "o-", ms=3, lw=1.4, label=name)
    # Reference slopes: if a method has order p, error falls by 10^p per decade.
    k = np_logspace(-1, 2)
    ax.semilogy(k, k**-1 * 1e-1, "k:", lw=1, label="slope -1")
    ax.semilogy(k, k**-2 * 1e-1, "k--", lw=1, label="slope -2")
    ax.set_xlabel("iteration")
    ax.set_ylabel("absolute error")
    ax.set_title("Root finding: $x e^x - 2 = 0$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "root_finding_convergence.png", dpi=150)
    print(f"\nwrote {OUT / 'root_finding_convergence.png'}")

    # ---------------- ODE integration ----------------
    print("\nODE: y' = y, y(0) = 1, exact y(1) = e =", math.e)
    print(f"{'method':<12}{'theory p':>10}{'n=20 err':>14}{'n=40 err':>14}{'ratio':>9}{'implied p':>12}")
    print("-" * 71)
    integrators = {"Euler": (euler, 1.0), "Heun": (heun, 2.0), "RK4": (rk4, 4.0)}
    for name, (fn, p_th) in integrators.items():
        e20 = abs(fn(dy, 1.0, 0.0, 1.0, n=20).x - math.e)
        e40 = abs(fn(dy, 1.0, 0.0, 1.0, n=40).x - math.e)
        ratio = e20 / e40
        implied = math.log2(ratio) / math.log2(2.0)  # steps doubled -> p per halving
        print(f"{name:<12}{p_th:>10.1f}{e20:>14.3e}{e40:>14.3e}{ratio:>9.2f}{implied:>12.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = [4, 8, 16, 32, 64, 128]
    for name, (fn, p_th) in integrators.items():
        errs = [abs(fn(dy, 1.0, 0.0, 1.0, n=n).x - math.e) for n in ns]
        ax.loglog(ns, errs, "o-", ms=3, lw=1.4, label=f"{name} (order {p_th:g})")
    ax.loglog(ns, [3e-2 * n**-1 for n in ns], "k:", lw=1, label="slope -1")
    ax.loglog(ns, [3e-2 * n**-2 for n in ns], "k-.", lw=1, label="slope -2")
    ax.loglog(ns, [3e-2 * n**-4 for n in ns], "k--", lw=1, label="slope -4")
    ax.set_xlabel("steps n")
    ax.set_ylabel("absolute error at t=1")
    ax.set_title("ODE integration: $y' = y$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "ode_convergence.png", dpi=150)
    print(f"wrote {OUT / 'ode_convergence.png'}")


# Local aliases keep the numpy import tidy at the top of the plotting code.
import numpy as np  # noqa: E402

np_where = np.where
np_arange = np.arange
np_logspace = np.logspace

if __name__ == "__main__":
    main()
