#!/usr/bin/env python
"""Dynamic Micro-benchmarks for RheoJAX overheads.

Isolates core JAX and inference overheads to ensure the computational framework
remains efficient. Measures:
1. JIT Cache Hit Ratio
2. Host-to-Device Transfer Overhead
3. Model Instantiation Overhead

Usage:
    uv run python scripts/micro_benchmarks.py
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("JAX_LOG_LEVEL", "WARNING")


@dataclass
class BenchResult:
    """Single benchmark result."""

    name: str
    wall_ms: float
    details: dict

    def __str__(self) -> str:
        detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"  {self.name:<45} {self.wall_ms:>10.1f} ms  ({detail_str})"


@contextmanager
def timed(name: str, details: dict | None = None):
    """Timing context manager yielding a BenchResult."""
    result = BenchResult(name=name, wall_ms=0.0, details=details or {})
    gc.collect()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_ms = (time.perf_counter() - start) * 1000


# =============================================================================
# BENCHMARK 1: JIT Caching Across Instantiations
# =============================================================================


def bench_jit_caching():
    """Verify that JIT compilation caching correctly persists across model classes."""
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: JIT Caching Across Instantiations")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models import GeneralizedMaxwell

    jax, jnp = safe_import_jax()
    jax.clear_caches()

    t = np.logspace(-2, 2, 200)
    _G = 100 * np.exp(-t) + 10.0  # noqa: F841 — needed to warm numpy
    t_jax = jnp.asarray(t)

    E_inf = 50.0
    E_i = jnp.array([100.0])
    tau_i = jnp.array([1.0])

    results = []

    # 1. First run (Cold compilation)
    with timed("A) First Predict (Cold JIT)", {"repeats": 1}) as r1:
        m1 = GeneralizedMaxwell(n_modes=1)
        _ = m1._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)
        _.block_until_ready()
    results.append(r1)

    # 2. Same model instance
    with timed("B) Same Model Instance (Warm)", {"repeats": 10}) as r2:
        for _ in range(10):
            _ = m1._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)
            _.block_until_ready()
        r2.wall_ms /= 10.0
    results.append(r2)

    # 3. New model instance
    with timed("C) New Model Instance (Should be Warm)", {"repeats": 10}) as r3:
        for _ in range(10):
            m2 = GeneralizedMaxwell(n_modes=1)
            _ = m2._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)
            _.block_until_ready()
        r3.wall_ms /= 10.0
    results.append(r3)

    for r in results:
        print(r)

    ratio = r1.wall_ms / max(r3.wall_ms, 0.001)
    print(f"\n  JIT Speedup factor: {ratio:.0f}x")
    if ratio < 50:
        print("  WARNING: JIT cache may not be applying correctly across instances.")

    return results


# =============================================================================
# BENCHMARK 2: Host-to-Device Memory Transfer Overhead
# =============================================================================


def bench_transfer_overhead():
    """Measure the cost of passing native Numpy arrays to JAX endpoints."""
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: Host-to-Device Memory Transfer Overhead")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    sizes = [1000, 10000, 100000]
    results = []

    for n in sizes:
        arr_np = np.random.normal(0, 1, n)

        # JAX conversion
        with timed(f"A) np -> jnp conversion (N={n})", {"N": n, "repeats": 100}) as r1:
            for _ in range(100):
                arr_j = jnp.asarray(arr_np)
            r1.wall_ms /= 100.0
        results.append(r1)

        # JAX backwards conversion
        with timed(f"B) jnp -> np conversion (N={n})", {"N": n, "repeats": 100}) as r2:
            for _ in range(100):
                _ = np.asarray(arr_j)
            r2.wall_ms /= 100.0
        results.append(r2)

    for r in results:
        print(r)

    avg_to_jax = np.mean([r.wall_ms for r in results[0::2]])
    print(f"\n  Average to_jax transfer overhead: {avg_to_jax*1000:.1f} microseconds")

    return results


# =============================================================================
# BENCHMARK 3: Model Instantiation Cost
# =============================================================================


def bench_instantiation_overhead():
    """Measure the fixed cost of creating ParameterSets and Model objects."""
    print("\n" + "=" * 70)
    print("  BENCHMARK 3: Model Instantiation Cost")
    print("=" * 70)

    from rheojax.models import Maxwell
    from rheojax.models.sgr.sgr_generic import SGRGeneric

    results = []

    # Simple model creation
    with timed("A) Maxwell Instantiation", {"repeats": 1000}) as r1:
        for _ in range(1000):
            _ = Maxwell()
        r1.wall_ms /= 1000.0
    results.append(r1)

    # Complex model creation
    with timed("B) SGRGeneric Instantiation", {"repeats": 1000}) as r2:
        for _ in range(1000):
            _ = SGRGeneric()
        r2.wall_ms /= 1000.0
    results.append(r2)

    for r in results:
        print(r)

    print(
        f"\n  Base class initialization adds ~{r1.wall_ms * 1000:.0f} microseconds of latency."
    )
    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Micro-benchmarks for RheoJAX overheads"
    )
    parser.add_argument(
        "--bench",
        choices=["jit", "transfer", "init", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RheoJAX Micro-Benchmarks")
    print("=" * 70)

    all_results = {}

    if args.bench in ("jit", "all"):
        all_results["jit"] = bench_jit_caching()

    if args.bench in ("transfer", "all"):
        all_results["transfer"] = bench_transfer_overhead()

    if args.bench in ("init", "all"):
        all_results["init"] = bench_instantiation_overhead()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  All core overheads are running dynamically.")


if __name__ == "__main__":
    main()
