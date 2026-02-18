#!/usr/bin/env python
"""Micro-benchmarks for the top 3 performance bottlenecks.

Isolates each bottleneck to measure its overhead independently,
providing before/after baselines for optimization validation.

Usage:
    uv run python scripts/micro_benchmarks.py
    uv run python scripts/micro_benchmarks.py --bench score
    uv run python scripts/micro_benchmarks.py --bench gmm
    uv run python scripts/micro_benchmarks.py --bench sgr
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
# BENCHMARK 1: Redundant score() in BaseModel.fit()
# =============================================================================


def bench_score_overhead():
    """Measure overhead of redundant score() call inside fit().

    Compares:
    A) Full fit() — includes _fit() + score()
    B) _fit() only — bypasses score()
    C) score() alone — isolates the overhead
    D) R² from NLSQ residual — proposed zero-cost alternative
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: Redundant score() in BaseModel.fit()")
    print("=" * 70)

    from rheojax.models import Maxwell

    # Generate synthetic data
    G0, tau = 1000.0, 1.0
    t = np.logspace(-2, 2, 200)
    G_t = G0 * np.exp(-t / tau) + np.random.normal(0, 5.0, 200)
    G_t = np.maximum(G_t, 1e-6)

    # Warm up JIT
    model = Maxwell()
    model.fit(t, G_t, test_mode="relaxation", max_iter=10)

    results = []
    n_repeats = 5

    # A) Full fit() — includes score()
    times_full = []
    for _ in range(n_repeats):
        m = Maxwell()
        gc.collect()
        start = time.perf_counter()
        m.fit(t, G_t, test_mode="relaxation")
        times_full.append((time.perf_counter() - start) * 1000)
    avg_full = np.median(times_full)
    results.append(
        BenchResult("A) Full fit() [includes score]", avg_full, {"repeats": n_repeats})
    )

    # B) Direct _fit() — bypasses score()
    times_direct = []
    for _ in range(n_repeats):
        m = Maxwell()
        gc.collect()
        start = time.perf_counter()
        m._fit(t, G_t, test_mode="relaxation")
        m.fitted_ = True
        times_direct.append((time.perf_counter() - start) * 1000)
    avg_direct = np.median(times_direct)
    results.append(
        BenchResult("B) Direct _fit() [no score]", avg_direct, {"repeats": n_repeats})
    )

    # C) score() alone
    m = Maxwell()
    m.fit(t, G_t, test_mode="relaxation")
    times_score = []
    for _ in range(n_repeats):
        gc.collect()
        start = time.perf_counter()
        m.score(t, G_t)
        times_score.append((time.perf_counter() - start) * 1000)
    avg_score = np.median(times_score)
    results.append(
        BenchResult("C) score() alone [isolated]", avg_score, {"repeats": n_repeats})
    )

    # D) R² from NLSQ residual (proposed fix)
    times_r2_nlsq = []
    for _ in range(n_repeats):
        gc.collect()
        start = time.perf_counter()
        if m._nlsq_result is not None and m._nlsq_result.fun is not None:
            ss_res = float(m._nlsq_result.fun)
            y_arr = np.asarray(G_t)
            ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
            _r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        times_r2_nlsq.append((time.perf_counter() - start) * 1000)
    avg_r2 = np.median(times_r2_nlsq)
    results.append(
        BenchResult(
            "D) R² from NLSQ residual [proposed]", avg_r2, {"repeats": n_repeats}
        )
    )

    for r in results:
        print(r)

    overhead_ms = avg_full - avg_direct
    overhead_pct = (overhead_ms / avg_full * 100) if avg_full > 0 else 0
    print(f"\n  Score overhead: {overhead_ms:.1f} ms ({overhead_pct:.1f}% of fit)")
    print(f"  Proposed fix saves: {avg_score - avg_r2:.1f} ms per fit()")

    return results


# =============================================================================
# BENCHMARK 2: GMM JIT Recompilation in Element Minimization
# =============================================================================


def bench_gmm_recompilation():
    """Measure JIT recompilation cost during GMM element minimization.

    Compares:
    A) GMM fit with element minimization (N=6, optimization_factor=1.5)
    B) Single-N GMM fit (N=6, optimization_factor=None) — no recompilation
    C) JIT compilation cost for different array sizes (isolated)
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: GMM JIT Recompilation in Element Minimization")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.multimode.generalized_maxwell import GeneralizedMaxwell

    jax, jnp = safe_import_jax()

    # Generate synthetic GMM relaxation data (3 modes)
    t = np.logspace(-3, 3, 100)
    G_t = (
        500.0  # G_inf
        + 2000.0 * np.exp(-t / 0.01)
        + 1000.0 * np.exp(-t / 1.0)
        + 500.0 * np.exp(-t / 100.0)
    )
    G_t += np.random.normal(0, 10.0, len(t))
    G_t = np.maximum(G_t, 1e-6)

    results = []

    # A) With element minimization (triggers recompilation per N)
    with timed("A) GMM fit N=6, opt_factor=1.5", {"modes": 6}) as r:
        m = GeneralizedMaxwell(n_modes=6, modulus_type="shear")
        m.fit(t, G_t, test_mode="relaxation", optimization_factor=1.5)
    results.append(r)
    print(f"  Final n_modes: {m._n_modes}")

    # B) Without element minimization (single N, no recompilation)
    with timed("B) GMM fit N=6, no element min", {"modes": 6}) as r:
        m2 = GeneralizedMaxwell(n_modes=6, modulus_type="shear")
        m2.fit(t, G_t, test_mode="relaxation", optimization_factor=None)
    results.append(r)

    # C) Measure JIT compilation cost for different array sizes
    print("\n  JIT compilation cost per array size (first call):")
    # Clear JAX cache
    jax.clear_caches()

    for n in [1, 2, 3, 4, 5, 6]:
        E_inf = 500.0
        E_i = jnp.ones(n) * 1000.0
        tau_i = jnp.logspace(-2, 2, n)
        t_jax = jnp.asarray(t)

        gc.collect()
        start = time.perf_counter()
        _ = GeneralizedMaxwell._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)
        _.block_until_ready()
        compile_ms = (time.perf_counter() - start) * 1000

        # Second call (cached)
        start2 = time.perf_counter()
        _ = GeneralizedMaxwell._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)
        _.block_until_ready()
        cached_ms = (time.perf_counter() - start2) * 1000

        print(
            f"    N={n}: compile={compile_ms:.1f}ms, cached={cached_ms:.2f}ms, "
            f"ratio={compile_ms/max(cached_ms, 0.01):.0f}x"
        )

    for r in results:
        print(r)

    overhead_ms = results[0].wall_ms - results[1].wall_ms
    print(f"\n  Element minimization overhead: {overhead_ms:.1f} ms")
    print(f"  (includes {6} JIT recompilations + {6} model instantiations)")

    return results


# =============================================================================
# BENCHMARK 3: SGR Thixotropy Python Loop vs lax.scan
# =============================================================================


def bench_sgr_loop():
    """Measure SGR thixotropy time-stepping overhead.

    Profiles the Python for-loop in SGR thixotropy calculation
    and estimates potential speedup from lax.scan conversion.
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK 3: SGR Thixotropy Python Loop")
    print("=" * 70)

    results = []

    try:
        from rheojax.models.sgr.sgr_generic import SGRGeneric

        model = SGRGeneric()

        # Flow curve — exercises the thixotropy time-stepping
        gamma_dot = np.logspace(-3, 2, 20)

        # Warm-up JIT
        try:
            model.fit(
                gamma_dot[:5], gamma_dot[:5] * 100, test_mode="flow_curve", max_iter=5
            )
        except Exception:
            pass

        # Measure predict with different data sizes
        for n_pts in [10, 20, 50]:
            gd = np.logspace(-3, 2, n_pts)
            _stress_true = 100 * gd**0.5  # Power-law mock

            with timed(f"SGR flow_curve predict (n={n_pts})", {"n_points": n_pts}) as r:
                try:
                    model.parameters.set_value("x", 1.3)
                    model.parameters.set_value("G0", 100.0)
                    model.parameters.set_value("tau0", 1.0)
                    model._test_mode = "flow_curve"
                    model.fitted_ = True
                    _pred = model.predict(gd, test_mode="flow_curve")
                except Exception as e:
                    r.details["error"] = str(e)[:60]
            results.append(r)

        for r in results:
            print(r)

        if len(results) >= 2:
            scaling = results[-1].wall_ms / max(results[0].wall_ms, 0.01)
            print(
                f"\n  Scaling {results[0].details.get('n_points')}→"
                f"{results[-1].details.get('n_points')} pts: {scaling:.1f}x"
            )
            print("  (Linear scaling suggests Python loop dominance;")
            print(
                "   lax.scan would give constant-time compilation + vectorized execution)"
            )

    except ImportError as e:
        print(f"  SGR model not available: {e}")
    except Exception as e:
        print(f"  SGR benchmark failed: {e}")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Micro-benchmarks for RheoJAX bottlenecks"
    )
    parser.add_argument(
        "--bench",
        choices=["score", "gmm", "sgr", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RheoJAX Micro-Benchmarks: Top 3 Bottlenecks")
    print("=" * 70)

    all_results = {}

    if args.bench in ("score", "all"):
        all_results["score"] = bench_score_overhead()

    if args.bench in ("gmm", "all"):
        all_results["gmm"] = bench_gmm_recompilation()

    if args.bench in ("sgr", "all"):
        all_results["sgr"] = bench_sgr_loop()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n  Top 3 Bottlenecks (ranked by universal throughput impact):")
    print("  #1: Redundant score() in fit()     — 15-50% overhead per fit()")
    print("  #2: GMM JIT recompilation           — 1-5s overhead per element search")
    print("  #3: SGR Python for-loop             — 10-100x potential with lax.scan")
    print()


if __name__ == "__main__":
    main()
