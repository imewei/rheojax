#!/usr/bin/env python
"""Comprehensive baseline profiler for RheoJAX performance optimization.

Measures: CPU hot paths, memory allocation, JIT cold/warm, import breakdown,
GeneralizedMaxwell (n_modes=10), and BaseModel overhead.

Usage:
    uv run python scripts/profile_comprehensive.py
"""

from __future__ import annotations

import cProfile
import gc
import os
import pstats
import sys
import time
import tracemalloc
from io import StringIO

import numpy as np

os.environ.setdefault("JAX_LOG_LEVEL", "WARNING")

# ---- Timing utility --------------------------------------------------------


def timed(label):
    """Simple timing context manager."""

    class Timer:
        def __init__(self):
            self.elapsed = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.start

    return Timer()


# ---- Section 1: Import Time Breakdown --------------------------------------


def profile_imports():
    print("\n" + "=" * 70)
    print("  SECTION 1: IMPORT TIME BREAKDOWN")
    print("=" * 70)

    # Already imported by script startup, measure from subprocess
    import subprocess

    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import rheojax"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = result.stderr.strip().split("\n")

    # Parse top-level cumulative imports
    imports = []
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                try:
                    self_us = int(parts[0].replace("import time:", "").strip())
                    cum_us = int(parts[1].strip())
                    pkg = parts[2].strip()
                    imports.append((cum_us, self_us, pkg))
                except ValueError:
                    continue

    imports.sort(key=lambda x: x[0], reverse=True)
    print("\n  Top 20 imports by cumulative time:")
    print(f"  {'Package':<50} {'Cumul (ms)':>10} {'Self (ms)':>10}")
    print("  " + "-" * 72)
    for cum, self_t, pkg in imports[:20]:
        print(f"  {pkg:<50} {cum/1000:>10.1f} {self_t/1000:>10.1f}")

    total_ms = imports[0][0] / 1000 if imports else 0
    print(f"\n  Total import time: {total_ms:.0f} ms")

    # Breakdown by category
    categories = {
        "scipy": 0,
        "jax": 0,
        "diffrax": 0,
        "equinox": 0,
        "nlsq": 0,
        "numpy": 0,
        "rheojax": 0,
        "other": 0,
    }
    for cum, _self_t, pkg in imports:
        matched = False
        for cat in ["scipy", "diffrax", "equinox", "nlsq", "numpy", "rheojax"]:
            if cat in pkg:
                categories[cat] = max(categories[cat], cum / 1000)
                matched = True
                break
        if not matched and "jax" in pkg:
            categories["jax"] = max(categories["jax"], cum / 1000)

    print("\n  Import time by category (peak cumulative):")
    for cat, ms in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if ms > 0:
            bar = "#" * int(ms / total_ms * 40) if total_ms > 0 else ""
            print(f"    {cat:<15} {ms:>8.0f} ms  {bar}")

    return total_ms


# ---- Section 2: JIT Cold vs Warm -------------------------------------------


def profile_jit_overhead():
    print("\n" + "=" * 70)
    print("  SECTION 2: JIT COMPILATION OVERHEAD (cold vs warm)")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell, Zener
    from rheojax.models.flow import PowerLaw

    np.random.seed(42)

    configs = [
        (
            "Maxwell (relaxation)",
            Maxwell,
            "relaxation",
            lambda: (
                np.logspace(-2, 2, 200),
                1000 * np.exp(-np.logspace(-2, 2, 200)) + np.random.normal(0, 5, 200),
            ),
        ),
        (
            "Maxwell (oscillation)",
            Maxwell,
            "oscillation",
            lambda: (
                np.logspace(-2, 2, 200),
                (
                    1000
                    * (np.logspace(-2, 2, 200)) ** 2
                    / (1 + (np.logspace(-2, 2, 200)) ** 2)
                    + 1j
                    * 1000
                    * np.logspace(-2, 2, 200)
                    / (1 + (np.logspace(-2, 2, 200)) ** 2)
                ),
            ),
        ),
        (
            "Zener (relaxation)",
            Zener,
            "relaxation",
            lambda: (
                np.logspace(-2, 2, 200),
                500
                + 800 * np.exp(-np.logspace(-2, 2, 200) / 0.25)
                + np.random.normal(0, 5, 200),
            ),
        ),
        (
            "PowerLaw (flow_curve)",
            PowerLaw,
            "flow_curve",
            lambda: (
                np.logspace(-2, 2, 150),
                50 * np.logspace(-2, 2, 150) ** 0.5 + np.random.normal(0, 0.5, 150),
            ),
        ),
    ]

    print(
        f"\n  {'Model':<30} {'Cold (ms)':>10} {'Warm (ms)':>10} {'Ratio':>8} {'JIT overhead':>12}"
    )
    print("  " + "-" * 74)

    for label, model_cls, test_mode, data_fn in configs:
        # Clear JAX caches
        jax.clear_caches()
        gc.collect()

        X, y = data_fn()

        # Cold fit
        m1 = model_cls()
        with timed("cold") as t_cold:
            m1.fit(X, y, test_mode=test_mode)

        # Warm fit
        m2 = model_cls()
        with timed("warm") as t_warm:
            m2.fit(X, y, test_mode=test_mode)

        cold_ms = t_cold.elapsed * 1000
        warm_ms = t_warm.elapsed * 1000
        ratio = cold_ms / warm_ms if warm_ms > 0 else float("inf")
        jit_pct = (1 - warm_ms / cold_ms) * 100 if cold_ms > 0 else 0

        print(
            f"  {label:<30} {cold_ms:>10.1f} {warm_ms:>10.1f} {ratio:>7.1f}x {jit_pct:>10.0f}%"
        )

    # Predict cold vs warm
    print("\n  predict() cold vs warm:")
    print(f"  {'Model':<30} {'Cold (ms)':>10} {'Warm (ms)':>10} {'Ratio':>8}")
    print("  " + "-" * 62)

    for label, model_cls, test_mode, data_fn in configs:
        jax.clear_caches()
        gc.collect()

        X, y = data_fn()
        m = model_cls()
        m.fit(X, y, test_mode=test_mode)

        # Cold predict (new JIT trace)
        jax.clear_caches()
        with timed("cold") as t_cold:
            m.predict(X)

        # Warm predict
        with timed("warm") as t_warm:
            m.predict(X)

        cold_ms = t_cold.elapsed * 1000
        warm_ms = t_warm.elapsed * 1000
        ratio = cold_ms / warm_ms if warm_ms > 0 else float("inf")
        print(f"  {label:<30} {cold_ms:>10.1f} {warm_ms:>10.1f} {ratio:>7.1f}x")


# ---- Section 3: Memory Profiling -------------------------------------------


def profile_memory():
    print("\n" + "=" * 70)
    print("  SECTION 3: MEMORY ALLOCATION")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell

    np.random.seed(42)
    t = np.logspace(-2, 2, 200)
    G_data = 1000 * np.exp(-t) + np.random.normal(0, 5, 200)

    # Measure Maxwell fit
    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    model = Maxwell()
    model.fit(t, G_data, test_mode="relaxation")

    snap_after = tracemalloc.take_snapshot()
    top_stats = snap_after.compare_to(snap_before, "lineno")

    print("\n  Top 15 memory allocations during Maxwell.fit():")
    print(f"  {'File:Line':<65} {'Size (KB)':>10}")
    print("  " + "-" * 77)
    for stat in top_stats[:15]:
        filepath = str(stat.traceback)
        # Shorten path
        if "rheojax/" in filepath:
            filepath = "rheojax/" + filepath.split("rheojax/")[-1]
        elif "site-packages/" in filepath:
            filepath = ".../" + filepath.split("site-packages/")[-1]
        filepath = filepath[:63]
        size_kb = stat.size_diff / 1024
        print(f"  {filepath:<65} {size_kb:>10.1f}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"\n  Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"  Peak memory:    {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()

    # GMM memory
    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    try:
        from rheojax.models import GeneralizedMaxwell

        gmm = GeneralizedMaxwell(n_modes=10)
        gmm.fit(
            t, np.maximum(G_data, 1e-6), test_mode="relaxation", optimization_factor=1.5
        )
        snap_after = tracemalloc.take_snapshot()
        top_stats = snap_after.compare_to(snap_before, "lineno")

        current, peak = tracemalloc.get_traced_memory()
        print("\n  GeneralizedMaxwell(n_modes=10) memory:")
        print(f"    Current: {current / 1024 / 1024:.1f} MB")
        print(f"    Peak:    {peak / 1024 / 1024:.1f} MB")
        print(f"    Optimal modes found: {gmm._n_modes}")
    except Exception as e:
        print(f"\n  GMM profiling error: {e}")

    tracemalloc.stop()
    return peak


# ---- Section 4: CPU Profiling (cProfile) ------------------------------------


def profile_cpu():
    print("\n" + "=" * 70)
    print("  SECTION 4: CPU PROFILING (cProfile)")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell

    np.random.seed(42)
    t = np.logspace(-2, 2, 200)
    G_data = 1000 * np.exp(-t) + np.random.normal(0, 5, 200)

    # Warm JIT first
    m_warm = Maxwell()
    m_warm.fit(t, G_data, test_mode="relaxation")

    # Profile warm fit
    model = Maxwell()
    profiler = cProfile.Profile()
    profiler.enable()
    model.fit(t, G_data, test_mode="relaxation")
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print("\n  --- Maxwell warm fit: Top 30 by cumulative time ---")
    print(stream.getvalue())

    # Profile cold fit
    jax.clear_caches()
    gc.collect()
    model2 = Maxwell()
    profiler2 = cProfile.Profile()
    profiler2.enable()
    model2.fit(t, G_data, test_mode="relaxation")
    profiler2.disable()

    stream2 = StringIO()
    stats2 = pstats.Stats(profiler2, stream=stream2).sort_stats("cumulative")
    stats2.print_stats(30)
    print("\n  --- Maxwell cold fit: Top 30 by cumulative time ---")
    print(stream2.getvalue())


# ---- Section 5: BaseModel.fit() Overhead Analysis --------------------------


def profile_basemodel_overhead():
    print("\n" + "=" * 70)
    print("  SECTION 5: BaseModel.fit() OVERHEAD ANALYSIS")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell

    np.random.seed(42)
    t = np.logspace(-2, 2, 200)
    G_data = 1000 * np.exp(-t) + np.random.normal(0, 5, 200)

    # Warm JIT
    m = Maxwell()
    m.fit(t, G_data, test_mode="relaxation")

    # Profile individual phases within fit()
    model = Maxwell()

    # Phase: Parameter setup
    with timed("param_setup") as _t_params:
        # This happens in _fit -> _setup_parameters
        pass

    # Phase: Data validation / conversion
    with timed("data_prep") as t_data:
        _X = np.asarray(t, dtype=np.float64)
        _y = np.asarray(G_data, dtype=np.float64)

    # Phase: Build model_function
    with timed("model_fn") as t_mfn:
        _model_fn = model.model_function

    # Phase: NLSQ call only

    with timed("nlsq_only") as t_nlsq:
        # Call the actual optimizer
        model2 = Maxwell()
        model2.fit(t, G_data, test_mode="relaxation")

    # Phase: Post-fit (score, state update)
    with timed("score") as t_score:
        model2.score(t, G_data)

    # Phase: predict
    with timed("predict") as t_pred:
        model2.predict(t)

    print("\n  BaseModel.fit() phase breakdown (warm, ms):")
    print(f"    Data preparation:    {t_data.elapsed*1000:>8.2f}")
    print(f"    model_function attr: {t_mfn.elapsed*1000:>8.2f}")
    print(f"    Full fit() call:     {t_nlsq.elapsed*1000:>8.2f}")
    print(f"    score() post-fit:    {t_score.elapsed*1000:>8.2f}")
    print(f"    predict() post-fit:  {t_pred.elapsed*1000:>8.2f}")

    # Check if fit() calls score() internally
    print("\n  Checking fit() internal calls...")
    model3 = Maxwell()
    original_score = model3.__class__.score

    call_log = []

    def patched_score(self, *args, **kwargs):
        call_log.append("score_called")
        return original_score(self, *args, **kwargs)

    model3.__class__.score = patched_score
    model3.fit(t, G_data, test_mode="relaxation")
    model3.__class__.score = original_score

    if call_log:
        print(f"    fit() internally calls score(): YES ({len(call_log)} times)")
    else:
        print("    fit() internally calls score(): NO")


# ---- Section 6: GeneralizedMaxwell Profiling --------------------------------


def profile_gmm():
    print("\n" + "=" * 70)
    print("  SECTION 6: GeneralizedMaxwell PROFILING")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    try:
        from rheojax.models import GeneralizedMaxwell
    except ImportError:
        print("  GeneralizedMaxwell not available, skipping.")
        return

    np.random.seed(42)
    t = np.logspace(-2, 2, 200)

    # Multi-mode relaxation data
    G_data = (
        500 * np.exp(-t / 0.1)
        + 300 * np.exp(-t / 1.0)
        + 200 * np.exp(-t / 10.0)
        + np.random.normal(0, 3, 200)
    )
    G_data = np.maximum(G_data, 1e-6)

    # Profile GMM with different n_modes
    for n_modes in [3, 5, 10]:
        jax.clear_caches()
        gc.collect()

        gmm = GeneralizedMaxwell(n_modes=n_modes)

        with timed("cold") as t_cold:
            gmm.fit(t, G_data, test_mode="relaxation", optimization_factor=1.5)

        gmm2 = GeneralizedMaxwell(n_modes=n_modes)
        with timed("warm") as t_warm:
            gmm2.fit(t, G_data, test_mode="relaxation", optimization_factor=1.5)

        print(f"\n  GMM n_modes={n_modes}:")
        print(f"    Cold fit:    {t_cold.elapsed*1000:>8.0f} ms")
        print(f"    Warm fit:    {t_warm.elapsed*1000:>8.0f} ms")
        print(
            f"    JIT overhead: {(t_cold.elapsed - t_warm.elapsed)*1000:>8.0f} ms ({(1 - t_warm.elapsed/t_cold.elapsed)*100:.0f}%)"
        )
        print(f"    Optimal modes: {gmm._n_modes}")

    # cProfile on warm GMM fit
    print("\n  --- GMM(n_modes=10) warm fit cProfile ---")
    gmm_warm = GeneralizedMaxwell(n_modes=10)
    gmm_warm.fit(t, G_data, test_mode="relaxation", optimization_factor=1.5)

    gmm_prof = GeneralizedMaxwell(n_modes=10)
    profiler = cProfile.Profile()
    profiler.enable()
    gmm_prof.fit(t, G_data, test_mode="relaxation", optimization_factor=1.5)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())


# ---- Section 7: I/O Path Analysis ------------------------------------------


def profile_io():
    print("\n" + "=" * 70)
    print("  SECTION 7: I/O PATH ANALYSIS")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    # Test RheoData creation
    from rheojax.core.data import RheoData

    np.random.seed(42)
    sizes = [100, 1000, 10000, 100000]

    print("\n  RheoData creation time:")
    print(f"  {'N points':<15} {'Create (ms)':>12} {'to_jax (ms)':>12}")
    print("  " + "-" * 42)

    for n in sizes:
        X = np.linspace(0.01, 100, n)
        y = np.random.normal(100, 10, n)

        with timed("create") as t_create:
            for _ in range(100):
                _rd = RheoData(x=X, y=y, metadata={"test_mode": "relaxation"})

        with timed("to_jax") as t_jax:
            for _ in range(100):
                _x_jax = jnp.asarray(X)
                _y_jax = jnp.asarray(y)

        print(f"  {n:<15} {t_create.elapsed*10:>12.3f} {t_jax.elapsed*10:>12.3f}")

    # Test CSV reader if available
    try:
        import csv
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["time", "stress"])
            for i in range(1000):
                writer.writerow([i * 0.01, np.random.normal(100, 10)])
            fname = f.name

        from rheojax.io import load_csv

        with timed("csv_load") as t_csv:
            for _ in range(10):
                _data = load_csv(fname, x_col="time", y_col="stress")

        print(
            f"\n  CSV reader (1000 rows, 10 iterations): {t_csv.elapsed*100:.1f} ms/read"
        )
        os.unlink(fname)
    except Exception as e:
        print(f"\n  CSV reader test skipped: {e}")


# ---- Section 8: Bayesian Inference Profiling --------------------------------


def profile_bayesian():
    print("\n" + "=" * 70)
    print("  SECTION 8: BAYESIAN INFERENCE (NUTS)")
    print("=" * 70)

    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()
    from rheojax.models import Maxwell

    np.random.seed(42)
    t = np.logspace(-2, 2, 100)
    G_data = 1000 * np.exp(-t) + np.random.normal(0, 10, 100)

    model = Maxwell()
    model.fit(t, G_data, test_mode="relaxation")

    gc.collect()
    jax.clear_caches()

    # Cold Bayesian
    with timed("bayes_cold") as t_cold:
        result = model.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            seed=42,
        )

    gc.collect()

    # Warm Bayesian (NUTS kernel already compiled)
    model2 = Maxwell()
    model2.fit(t, G_data, test_mode="relaxation")

    with timed("bayes_warm") as t_warm:
        _result2 = model2.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            seed=42,
        )

    print("\n  NUTS (50 warmup + 100 samples, 1 chain):")
    print(f"    Cold (includes JIT): {t_cold.elapsed*1000:>8.0f} ms")
    print(f"    Warm (cached JIT):   {t_warm.elapsed*1000:>8.0f} ms")
    print(
        f"    JIT overhead:        {(t_cold.elapsed - t_warm.elapsed)*1000:>8.0f} ms ({(1 - t_warm.elapsed/t_cold.elapsed)*100:.0f}%)"
    )

    # ArviZ conversion time
    with timed("arviz") as t_arviz:
        _idata = result.to_inference_data()
    print(f"    ArviZ conversion:    {t_arviz.elapsed*1000:>8.0f} ms")

    # Diagnostics computation
    with timed("diag") as t_diag:
        _diag = result.diagnostics
    print(f"    Diagnostics access:  {t_diag.elapsed*1000:>8.0f} ms")

    # Multi-chain overhead
    gc.collect()
    jax.clear_caches()

    model3 = Maxwell()
    model3.fit(t, G_data, test_mode="relaxation")

    with timed("bayes_4chain") as t_4chain:
        _result3 = model3.fit_bayesian(
            t,
            G_data,
            test_mode="relaxation",
            num_warmup=50,
            num_samples=100,
            num_chains=4,
            seed=42,
        )

    print("\n  Multi-chain (50+100, 4 chains):")
    print(f"    Total time:          {t_4chain.elapsed*1000:>8.0f} ms")
    print(f"    Per-chain (avg):     {t_4chain.elapsed*1000/4:>8.0f} ms")
    print(f"    vs 1-chain cold:     {t_4chain.elapsed/t_cold.elapsed:.1f}x")


# ---- Main ------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  RHEOJAX COMPREHENSIVE BASELINE PROFILING")
    print(f"  Python {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print("=" * 70)

    total_start = time.perf_counter()

    import_ms = profile_imports()
    profile_jit_overhead()
    profile_memory()
    profile_cpu()
    profile_basemodel_overhead()
    profile_gmm()
    profile_io()
    profile_bayesian()

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total profiling time: {total_elapsed:.0f}s")
    print(f"  Import overhead: {import_ms:.0f} ms")
    print("\n  Key findings will be reported in Task #1 output.")
    print("=" * 70)


if __name__ == "__main__":
    main()
