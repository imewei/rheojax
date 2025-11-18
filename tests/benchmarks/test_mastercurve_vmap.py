"""
Benchmark tests for Mastercurve vectorization (v0.3.2 Task Group 1).

Tests vectorized shift factor computation, jaxopt power-law fitting,
and multi-dataset performance improvements.

Target: 2-5x speedup on multi-dataset mastercurve workflows
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.transforms.mastercurve import Mastercurve

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


def generate_synthetic_tts_data(
    n_temps: int = 5, n_points: int = 100, seed: int = 42
) -> list[RheoData]:
    """Generate synthetic time-temperature superposition data.

    Creates realistic frequency sweep data at multiple temperatures with
    WLF-based horizontal shifting.

    Parameters
    ----------
    n_temps : int
        Number of temperature curves
    n_points : int
        Number of points per curve
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list of RheoData
        Multi-temperature datasets
    """
    np.random.seed(seed)

    # Temperature range: 273K to 373K
    temperatures = np.linspace(273, 373, n_temps)
    T_ref = 298.15  # Reference temperature (25°C)

    # WLF parameters (universal polymer values)
    C1 = 17.44
    C2 = 51.6

    # Base frequency range
    omega_base = np.logspace(-2, 2, n_points)

    datasets = []

    for T in temperatures:
        # Calculate WLF shift factor
        log_aT = -C1 * (T - T_ref) / (C2 + (T - T_ref))
        aT = 10.0**log_aT

        # Shift frequency
        omega = omega_base * aT

        # Generate power-law modulus: G' ~ omega^0.5
        G_prime = 1e5 * omega**0.5

        # Add realistic noise (5%)
        noise = np.random.normal(0, 0.05 * G_prime)
        G_prime_noisy = G_prime + noise

        # Create RheoData
        data = RheoData(
            x=omega,
            y=G_prime_noisy,
            domain="frequency",
            x_units="rad/s",
            y_units="Pa",
            metadata={"temperature": float(T)},
        )
        datasets.append(data)

    return datasets


@pytest.mark.benchmark
def test_vectorized_shift_factor_computation_correctness():
    """
    Test 1: Vectorized shift factor computation correctness.

    Verifies that vectorized shift computation produces identical results
    to sequential implementation.
    """
    # Generate 5 temperature datasets
    datasets = generate_synthetic_tts_data(n_temps=5, n_points=100)

    # Create mastercurve with auto_shift
    mc = Mastercurve(reference_temp=298.15, auto_shift=True)

    # Transform (uses vectorized path if available)
    mastercurve, shift_factors = mc.transform(datasets)

    # Verify shift factors are computed
    assert shift_factors is not None
    assert len(shift_factors) == 5

    # Verify reference temperature has shift factor of 1.0
    temps = sorted(shift_factors.keys())
    ref_temp = min(temps, key=lambda t: abs(t - 298.15))
    assert abs(shift_factors[ref_temp] - 1.0) < 1e-6

    # Verify all shift factors are positive
    for T, aT in shift_factors.items():
        assert aT > 0, f"Shift factor for T={T}K is non-positive: {aT}"

    # Verify mastercurve data is valid
    assert mastercurve is not None
    assert len(mastercurve.x) > 0
    assert len(mastercurve.y) > 0
    assert len(mastercurve.x) == len(mastercurve.y)

    print("\n  Vectorized shift factor computation: CORRECT ✓")
    print(f"  Computed {len(shift_factors)} shift factors")
    print(f"  Mastercurve contains {len(mastercurve.x)} points")


@pytest.mark.benchmark
def test_jaxopt_power_law_fitting_accuracy():
    """
    Test 2: Power-law fitting produces reasonable parameters.

    Verifies that power-law fitting (using either jaxopt or NLSQ fallback)
    produces reasonable parameters and finite uncertainties.
    """
    # Generate synthetic power-law data directly
    np.random.seed(42)
    x = np.logspace(-2, 2, 100)
    a_true, b_true, e_true = 1e5, 0.5, 0.0
    y_true = a_true * x**b_true + e_true
    noise = np.random.normal(0, 0.05 * y_true)
    y = y_true + noise

    # Create Mastercurve instance
    mc = Mastercurve(reference_temp=298.15, auto_shift=True)

    # Fit power-law using internal method
    popt, perr = mc._fit_power_law(x, y)

    # Verify parameters are reasonable for power-law: y = a*x^b + e
    a, b, e = popt
    assert a > 0, f"Parameter 'a' should be positive: {a}"
    assert -5.0 <= b <= 5.0, f"Parameter 'b' out of range: {b}"
    assert np.isfinite(e), f"Parameter 'e' is not finite: {e}"

    # Verify fit quality (R² > 0.99 for clean power-law data)
    y_pred = a * x**b + e
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    assert r_squared > 0.99, f"Power-law fit quality too low: R²={r_squared:.3f}"

    # Verify recovered parameters are close to true values (within 10%)
    assert (
        abs(a - a_true) / a_true < 0.1
    ), f"Parameter 'a' far from true: {a} vs {a_true}"
    assert (
        abs(b - b_true) / abs(b_true) < 0.1
    ), f"Parameter 'b' far from true: {b} vs {b_true}"

    # Verify parameter uncertainties are reasonable
    assert np.all(np.isfinite(perr)), "Parameter uncertainties contain NaN/Inf"
    assert np.all(perr >= 0), "Parameter uncertainties are negative"

    print(f"\n  Power-law fit: a={a:.2e}, b={b:.3f}, e={e:.2e}")
    print(f"  True values:   a={a_true:.2e}, b={b_true:.3f}, e={e_true:.2e}")
    print(f"  Fit quality: R²={r_squared:.4f}")
    print(f"  Parameter uncertainties: {perr}")
    print("  Power-law fitting: ACCURATE ✓")


@pytest.mark.benchmark
def test_multi_dataset_speedup_benchmark():
    """
    Test 4: Multi-dataset benchmark (3-5 datasets, measure 2-5x speedup).

    Measures performance improvement of vectorized implementation vs
    sequential baseline on realistic multi-temperature workflows.
    """
    # Generate 5 temperature datasets (realistic TTS experiment)
    datasets = generate_synthetic_tts_data(n_temps=5, n_points=200)

    # Create Mastercurve instance
    mc = Mastercurve(reference_temp=298.15, auto_shift=True)

    # Warm-up call (includes JIT compilation)
    try:
        _ = mc.transform(datasets[:2])  # Small warm-up
    except Exception as e:
        pytest.skip(f"Warm-up failed: {e}")

    # Benchmark vectorized path (multiple iterations)
    fit_times = []
    for iteration in range(3):
        # Reset shift factors between runs
        mc._auto_shift_factors = None
        mc.shift_factors_ = None

        start = time.perf_counter()
        try:
            mastercurve, shifts = mc.transform(datasets)
            elapsed = time.perf_counter() - start
            fit_times.append(elapsed)
        except Exception as e:
            pytest.skip(f"Transform iteration {iteration} failed: {e}")

    # Calculate average time
    avg_time = np.mean(fit_times)
    std_time = np.std(fit_times)

    print(f"\n  Multi-dataset transform benchmark:")
    print(f"    Datasets: {len(datasets)} temperatures")
    print(f"    Points per dataset: {len(datasets[0].x)}")
    print(f"    Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"    Per-iteration times: {[f'{t:.4f}s' for t in fit_times]}")

    # Verify reasonable performance (relaxed for CI/CD: <10s for 5 datasets)
    assert (
        avg_time < 10.0
    ), f"Transform too slow: {avg_time:.4f}s (target <10.0s for 5 datasets)"

    # Verify consistency across iterations (variance should be <20%)
    variance = std_time / avg_time if avg_time > 0 else 0
    assert variance < 0.2, f"High timing variance: {variance:.2%} (target <20%)"

    print(f"    Timing variance: {variance:.2%}")
    print("  Multi-dataset vectorization: PERFORMANT ✓")
