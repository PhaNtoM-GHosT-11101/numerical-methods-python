#!/usr/bin/env python3
"""Zero-dependency test runner, so the suite runs without pytest installed.

    python3 tests/run_tests.py        # run everything
    python3 tests/run_tests.py -v     # list each test as it runs

Under pytest the same functions are collected normally; this script exists so a
clone with only numpy is enough to verify the maths.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_numerics as suite  # noqa: E402


def main() -> int:
    verbose = "-v" in sys.argv
    tests = [(n, f) for n, f in vars(suite).items() if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda kv: kv[1].__code__.co_firstlineno)

    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed.append((name, traceback.format_exc()))
            print(f"FAIL  {name}")
        else:
            ms = (time.perf_counter() - start) * 1e3
            if verbose:
                doc = (fn.__doc__ or "").strip().splitlines()
                print(f"ok    {name}  ({ms:.1f}ms)  {doc[0] if doc else ''}")
            else:
                print(".", end="", flush=True)

    if not verbose:
        print()

    dt = time.perf_counter() - t0
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed in {dt * 1e3:.0f}ms")

    for name, tb in failed:
        print(f"\n{'=' * 70}\nFAILED: {name}\n{'=' * 70}\n{tb}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
