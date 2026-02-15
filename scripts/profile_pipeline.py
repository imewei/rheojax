#!/usr/bin/env python
"""Pipeline bottleneck profiler for RheoJAX.

Profiles the full NLSQ -> Bayesian pipeline across multiple model types,
measuring wall-clock time per phase and identifying bottlenecks.

Usage:
    uv run python scripts/profile_pipeline.py
    uv run python scripts/profile_pipeline.py --model maxwell --skip-bayesian
    uv run python scripts/profile_pipeline.py --verbose
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib
import os
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import numpy as np

# -- Timer utilities -----------------------------------------------------------


@dataclass
class TimingResult:
    """Stores timing for a single phase."""

    name: str
    wall_seconds: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_ms(self) -> float:
        return self.wall_seconds * 1000

    def __str__(self) -> str:
        detail_str = ""
        if self.details:
            parts = [f"{k}={v}" for k, v in self.details.items()]
            detail_str = f"  ({', '.join(parts)})"
        return f"  {self.name:<35} {self.wall_ms:>10.1f} ms{detail_str}"


@contextmanager
def timed_phase(name: str, details: dict[str, Any] | None = None):
    """Context manager that yields a TimingResult after execution."""
    result = TimingResult(name=name, wall_seconds=0.0, details=details or {})
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_seconds = time.perf_counter() - start


# -- Synthetic data generators -------------------------------------------------


def generate_maxwell_relaxation(n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell relaxation: G(t) = G0 * exp(-t/tau)."""
    G0, tau = 1000.0, 1.0
    t = np.logspace(-2, 2, n_points)
    G_t = G0 * np.exp(-t / tau) + np.random.normal(0, 5.0, n_points)
    return t, np.maximum(G_t, 1e-6)


def generate_maxwell_oscillation(n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell oscillation: G*(omega) = G' + iG''."""
    G0, tau = 1000.0, 1.0
    omega = np.logspace(-2, 2, n_points)
    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)
    noise = np.random.normal(0, 3.0, n_points)
    G_star = (G_prime + noise) + 1j * (G_double_prime + noise)
    return omega, G_star


def generate_zener_relaxation(n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Zener (SLS) relaxation: G(t) = Ge + Gm * exp(-t/tau)."""
    Ge, Gm, eta = 500.0, 800.0, 200.0
    tau = eta / Gm
    t = np.logspace(-2, 2, n_points)
    G_t = Ge + Gm * np.exp(-t / tau) + np.random.normal(0, 5.0, n_points)
    return t, np.maximum(G_t, 1e-6)


def generate_power_law_flow(n_points: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Power-law flow curve: sigma = K * gamma_dot^n."""
    K, n_pl = 50.0, 0.5
    gamma_dot = np.logspace(-2, 2, n_points)
    sigma = K * gamma_dot**n_pl + np.random.normal(0, 0.5, n_points)
    return gamma_dot, np.maximum(sigma, 1e-6)


# -- Model loading (no exec/eval) ---------------------------------------------


def _load_model_class(model_key: str):
    """Import model class by name without exec/eval."""
    class_map = {
        "maxwell": ("rheojax.models", "Maxwell"),
        "maxwell_osc": ("rheojax.models", "Maxwell"),
        "zener": ("rheojax.models", "Zener"),
        "power_law": ("rheojax.models", "PowerLaw"),
    }
    module_path, class_name = class_map[model_key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# -- Model configurations -----------------------------------------------------

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "maxwell": {
        "data_fn": generate_maxwell_relaxation,
        "test_mode": "relaxation",
        "label": "Maxwell (analytical, 2 params)",
    },
    "maxwell_osc": {
        "data_fn": generate_maxwell_oscillation,
        "test_mode": "oscillation",
        "label": "Maxwell oscillation (complex, 2 params)",
    },
    "zener": {
        "data_fn": generate_zener_relaxation,
        "test_mode": "relaxation",
        "label": "Zener (analytical, 3 params)",
    },
    "power_law": {
        "data_fn": generate_power_law_flow,
        "test_mode": "flow_curve",
        "label": "PowerLaw flow (analytical, 2 params)",
    },
}


# -- Core profiling engine -----------------------------------------------------


def profile_model(
    model_key: str,
    skip_bayesian: bool = False,
    num_warmup: int = 100,
    num_samples: int = 200,
    num_chains: int = 1,
    verbose: bool = False,
) -> list[TimingResult]:
    """Profile the full pipeline for a single model."""
    config = MODEL_CONFIGS[model_key]
    timings: list[TimingResult] = []

    print(f"\n{'='*70}")
    print(f"  Profiling: {config['label']}")
    print(f"{'='*70}")

    # -- Phase 0: Import -------------------------------------------------------
    with timed_phase("Import (safe_import_jax + model)") as t:
        from rheojax.core.jax_config import safe_import_jax

        safe_import_jax()
        model_cls = _load_model_class(model_key)
    timings.append(t)

    # -- Phase 1: Data generation ----------------------------------------------
    with timed_phase("Data generation", {"n_points": 200}) as t:
        np.random.seed(42)
        X, y = config["data_fn"](n_points=200)
    timings.append(t)

    # -- Phase 2: Model instantiation ------------------------------------------
    with timed_phase("Model instantiation") as t:
        model = model_cls()
    timings.append(t)

    # -- Phase 3: NLSQ fit (first call -- includes JIT) -----------------------
    with timed_phase("NLSQ fit (cold -- includes JIT)", {"test_mode": config["test_mode"]}) as t:
        model.fit(X, y, test_mode=config["test_mode"])
    timings.append(t)

    # -- Phase 4: Predict (post-fit) -------------------------------------------
    with timed_phase("predict() (post-fit)") as t:
        _y_pred = model.predict(X)
    timings.append(t)

    # -- Phase 5: Score (calls predict internally) -----------------------------
    with timed_phase("score() (includes predict)") as t:
        r2 = model.score(X, y)
        t.details["R2"] = f"{r2:.4f}" if r2 and not np.isnan(r2) else "NaN"
    timings.append(t)

    # -- Phase 6: NLSQ fit (warm -- JIT cached) --------------------------------
    model2 = model_cls()
    with timed_phase("NLSQ fit (warm -- JIT cached)") as t:
        model2.fit(X, y, test_mode=config["test_mode"])
    timings.append(t)

    # -- Phase 7: predict repeated (redundancy check) --------------------------
    with timed_phase("predict() x5 (no caching check)") as t:
        for _ in range(5):
            _ = model.predict(X)
        t.details["avg_ms"] = f"{t.wall_seconds / 5 * 1000:.1f}"
    timings.append(t)

    # -- Phase 8: Bayesian inference -------------------------------------------
    if not skip_bayesian:
        gc.collect()
        try:
            import jax as _jax

            _jax.clear_caches()
        except Exception:
            pass

        with timed_phase(
            "fit_bayesian() (NUTS)",
            {
                "warmup": num_warmup,
                "samples": num_samples,
                "chains": num_chains,
            },
        ) as t:
            try:
                bayes_result = model.fit_bayesian(
                    X,
                    y,
                    test_mode=config["test_mode"],
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=42,
                )
                t.details["divergences"] = bayes_result.diagnostics.get("num_divergences", "?")
            except Exception as e:
                t.details["error"] = str(e)[:60]
        timings.append(t)

        # -- Phase 9: Diagnostics access ----------------------------------------
        if "error" not in t.details:
            with timed_phase("Diagnostics access (R-hat, ESS)") as t_diag:
                diag = bayes_result.diagnostics
                t_diag.details["r_hat_keys"] = len([k for k in diag if "r_hat" in k.lower()])
            timings.append(t_diag)

            # -- Phase 10: ArviZ conversion -------------------------------------
            with timed_phase("ArviZ InferenceData conversion") as t_arviz:
                try:
                    idata = bayes_result.to_inference_data()
                    t_arviz.details["groups"] = list(idata.groups()) if idata else []
                except Exception as e:
                    t_arviz.details["error"] = str(e)[:60]
            timings.append(t_arviz)

    # -- Phase 11: RheoData creation (Pipeline.load equivalent) ----------------
    with timed_phase("RheoData creation (Pipeline.load)") as t:
        from rheojax.core.data import RheoData

        _rheo_data = RheoData(x=X, y=y, metadata={"test_mode": config["test_mode"]})
    timings.append(t)

    return timings


def run_cprofile(model_key: str) -> str:
    """Run cProfile on a single model's NLSQ fit for detailed function breakdown."""
    model_cls = _load_model_class(model_key)
    config = MODEL_CONFIGS[model_key]
    np.random.seed(42)
    X, y = config["data_fn"](n_points=200)

    model = model_cls()
    # Warm JIT first
    model.fit(X, y, test_mode=config["test_mode"])

    # Profile the warm fit
    model2 = model_cls()
    profiler = cProfile.Profile()
    profiler.enable()
    model2.fit(X, y, test_mode=config["test_mode"])
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(25)
    return stream.getvalue()


# -- Report generation ---------------------------------------------------------


def print_report(all_timings: dict[str, list[TimingResult]]) -> None:
    """Print a formatted bottleneck report."""
    print("\n")
    print("=" * 70)
    print("  PIPELINE BOTTLENECK REPORT")
    print("=" * 70)

    for model_key, timings in all_timings.items():
        total = sum(t.wall_seconds for t in timings)
        print(f"\n  Model: {MODEL_CONFIGS[model_key]['label']}")
        print(f"  Total pipeline time: {total * 1000:.1f} ms")
        print("-" * 70)
        print(f"  {'Phase':<35} {'Time (ms)':>10}  {'%':>6}")
        print("-" * 70)

        for t in sorted(timings, key=lambda x: x.wall_seconds, reverse=True):
            pct = (t.wall_seconds / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            detail_str = ""
            if t.details:
                parts = [f"{k}={v}" for k, v in t.details.items()]
                detail_str = f"  ({', '.join(parts)})"
            print(f"  {t.name:<35} {t.wall_ms:>10.1f}  {pct:>5.1f}%  {bar}{detail_str}")

        print("-" * 70)

    # -- Cross-model comparison ------------------------------------------------
    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON (key phases)")
    print("=" * 70)

    phase_names = [
        "NLSQ fit (cold -- includes JIT)",
        "NLSQ fit (warm -- JIT cached)",
        "predict() (post-fit)",
        "score() (includes predict)",
        "fit_bayesian() (NUTS)",
    ]

    header = f"  {'Phase':<35}"
    for model_key in all_timings:
        header += f" {model_key:>12}"
    print(header)
    print("-" * 70)

    for phase in phase_names:
        row = f"  {phase:<35}"
        for model_key in all_timings:
            timing = next((t for t in all_timings[model_key] if t.name == phase), None)
            if timing:
                row += f" {timing.wall_ms:>10.1f}ms"
            else:
                row += f" {'---':>12}"
        print(row)

    # -- Bottleneck summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print("  IDENTIFIED BOTTLENECKS")
    print("=" * 70)

    for model_key, timings in all_timings.items():
        total = sum(t.wall_seconds for t in timings)
        top3 = sorted(timings, key=lambda x: x.wall_seconds, reverse=True)[:3]
        print(f"\n  {model_key}:")
        for i, t in enumerate(top3, 1):
            pct = (t.wall_seconds / total * 100) if total > 0 else 0
            print(f"    {i}. {t.name} -- {t.wall_ms:.0f}ms ({pct:.0f}%)")

    # -- Redundant computation analysis ----------------------------------------
    print("\n" + "=" * 70)
    print("  REDUNDANT COMPUTATION ANALYSIS")
    print("=" * 70)

    for model_key, timings in all_timings.items():
        predict_t = next((t for t in timings if t.name == "predict() (post-fit)"), None)
        score_t = next((t for t in timings if t.name == "score() (includes predict)"), None)
        nlsq_cold = next((t for t in timings if "cold" in t.name), None)

        if predict_t and score_t and nlsq_cold:
            fit_score_overhead = score_t.wall_ms
            overhead_pct = (fit_score_overhead / nlsq_cold.wall_ms * 100) if nlsq_cold.wall_ms > 0 else 0

            print(f"\n  {model_key}:")
            print(f"    predict() call: {predict_t.wall_ms:.1f}ms")
            print(f"    score() in fit(): {fit_score_overhead:.1f}ms ({overhead_pct:.1f}% of fit time)")
            print("    -> fit() calls score() (line 374), which calls predict() again")
            print(f"    -> This adds ~{fit_score_overhead:.0f}ms overhead to every fit() call")


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile RheoJAX pipeline bottlenecks")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Model to profile (default: all)",
    )
    parser.add_argument("--skip-bayesian", action="store_true", help="Skip Bayesian inference")
    parser.add_argument("--verbose", action="store_true", help="Include cProfile output")
    parser.add_argument("--warmup", type=int, default=100, help="NUTS warmup iterations")
    parser.add_argument("--samples", type=int, default=200, help="NUTS sample iterations")
    parser.add_argument("--chains", type=int, default=1, help="Number of MCMC chains")
    args = parser.parse_args()

    # Suppress JAX/NumPyro logging noise
    os.environ.setdefault("JAX_LOG_LEVEL", "WARNING")

    models_to_profile = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    all_timings: dict[str, list[TimingResult]] = {}
    for model_key in models_to_profile:
        timings = profile_model(
            model_key,
            skip_bayesian=args.skip_bayesian,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            verbose=args.verbose,
        )
        all_timings[model_key] = timings

        if args.verbose:
            print(f"\n--- cProfile for {model_key} (warm fit) ---")
            print(run_cprofile(model_key))

        # Clean up between models
        gc.collect()
        try:
            import jax as _jax

            _jax.clear_caches()
        except Exception:
            pass

    print_report(all_timings)


if __name__ == "__main__":
    main()
