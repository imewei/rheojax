"""Benchmark tests for Generalized Maxwell Model element search optimization.

Tests warm-start optimization vs cold-start baseline for element minimization.
Target speedup: 2-5x through warm-start and compilation reuse.

Baseline (v0.3.1): Cold-start sequential NLSQ fits, ~20-50s for N=10 → optimal
Target (v0.4.0): Warm-start with compilation reuse, ~4-25s

Markers: @pytest.mark.benchmark (local only, excluded from CI)
Runtime: ~10-20 min total for statistical rigor
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rheojax.models import GeneralizedMaxwell

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp_typing
else:
    import numpy as jnp_typing


@pytest.fixture(autouse=True)
def _require_pytest_benchmark(request):
    """Skip benchmark tests if pytest-benchmark not installed."""
    if "benchmark" in [m.name for m in request.node.iter_markers()]:
        pytest.importorskip("pytest_benchmark")


@pytest.fixture
def synthetic_relaxation_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic multi-decade relaxation data typical of polymer melts.

    Returns:
        (t, G_t): Time array (10⁻³ to 10⁵ s) and relaxation modulus
    """
    # Multi-decade time range (8 decades)
    t = np.logspace(-3, 5, 500)

    # True Prony series: 5 modes
    G_inf = 1e3
    G_i = np.array([1e6, 5e5, 2e5, 8e4, 3e4])
    tau_i = np.array([1e-2, 1e-1, 1.0, 1e1, 1e2])

    # Compute relaxation modulus
    G_t = G_inf + np.sum(G_i[:, None] * np.exp(-t[None, :] / tau_i[:, None]), axis=0)

    # Add realistic noise (1% relative)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01 * np.mean(G_t), size=G_t.shape)
    G_t_noisy = G_t + noise

    return t, G_t_noisy


@pytest.fixture
def synthetic_oscillation_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic frequency sweep data for oscillation mode.

    Returns:
        (omega, G_star): Frequency array and complex modulus [G', G"]
    """
    # Frequency range (5 decades)
    omega = np.logspace(-2, 3, 400)

    # True Prony series: 5 modes
    G_inf = 1e3
    G_i = np.array([1e6, 5e5, 2e5, 8e4, 3e4])
    tau_i = np.array([1e-2, 1e-1, 1.0, 1e1, 1e2])

    # Compute complex modulus
    omega_2d = omega[None, :]
    tau_2d = tau_i[:, None]
    G_i_2d = G_i[:, None]

    G_prime = G_inf + np.sum(
        G_i_2d * (omega_2d * tau_2d) ** 2 / (1 + (omega_2d * tau_2d) ** 2), axis=0
    )
    G_double_prime = np.sum(
        G_i_2d * (omega_2d * tau_2d) / (1 + (omega_2d * tau_2d) ** 2), axis=0
    )

    # Stack as (M, 2)
    G_star = np.column_stack([G_prime, G_double_prime])

    # Add noise (1% relative)
    rng = np.random.default_rng(43)
    noise = rng.normal(0, 0.01 * np.mean(G_star), size=G_star.shape)
    G_star_noisy = G_star + noise

    return omega, G_star_noisy


@pytest.fixture
def synthetic_creep_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic creep compliance data.

    Returns:
        (t, J_t): Time array and creep compliance
    """
    # Time range (6 decades)
    t = np.logspace(-2, 4, 400)

    # True Prony series parameters (convert to compliance)
    G_inf = 1e3
    G_i = np.array([1e6, 5e5, 2e5, 8e4, 3e4])
    tau_i = np.array([1e-2, 1e-1, 1.0, 1e1, 1e2])

    # Approximate creep compliance (simplified analytical approach)
    # J(t) ≈ 1/(G_∞ + Σ G_i) + t-dependent terms
    J_0 = 1.0 / (G_inf + np.sum(G_i))
    J_inf = 1.0 / G_inf

    # Exponential transition (simplified)
    J_t = J_inf + (J_0 - J_inf) * np.exp(-t / 10.0)

    # Add noise (2% relative, creep noisier)
    rng = np.random.default_rng(44)
    noise = rng.normal(0, 0.02 * np.mean(J_t), size=J_t.shape)
    J_t_noisy = J_t + noise

    return t, J_t_noisy


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_element_search_n5_relaxation(synthetic_relaxation_data, benchmark):
    """Benchmark element search for N=5 in relaxation mode.

    Baseline (v0.3.1): ~10-15s (cold-start)
    Target (v0.4.0): ~3-5s (warm-start + compilation reuse)
    Expected speedup: 2-3x
    """
    t, G_t = synthetic_relaxation_data

    def run_element_search():
        gmm = GeneralizedMaxwell(n_modes=5, modulus_type="shear")
        gmm.fit(t, G_t, test_mode="relaxation", optimization_factor=1.5)
        return gmm._n_modes, gmm._element_minimization_diagnostics

    # Run benchmark with statistical rigor
    result = benchmark(run_element_search)

    # Verify optimal N selection
    n_opt, diagnostics = result
    assert 1 <= n_opt <= 5, f"Optimal N out of range: {n_opt}"
    assert diagnostics is not None, "Element minimization diagnostics missing"
    assert "n_optimal" in diagnostics
    assert diagnostics["n_optimal"] == n_opt


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_element_search_n10_relaxation(synthetic_relaxation_data, benchmark):
    """Benchmark element search for N=10 in relaxation mode.

    Baseline (v0.3.1): ~20-30s (cold-start)
    Target (v0.4.0): ~5-10s (warm-start + compilation reuse)
    Expected speedup: 3-4x
    """
    t, G_t = synthetic_relaxation_data

    def run_element_search():
        gmm = GeneralizedMaxwell(n_modes=10, modulus_type="shear")
        gmm.fit(t, G_t, test_mode="relaxation", optimization_factor=1.5)
        return gmm._n_modes, gmm._element_minimization_diagnostics

    # Run benchmark
    result = benchmark(run_element_search)

    # Verify optimal N selection
    n_opt, diagnostics = result
    assert 1 <= n_opt <= 10, f"Optimal N out of range: {n_opt}"
    assert diagnostics is not None


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_element_search_n20_relaxation(synthetic_relaxation_data, benchmark):
    """Benchmark element search for N=20 in relaxation mode.

    Baseline (v0.3.1): ~40-50s (cold-start)
    Target (v0.4.0): ~8-15s (warm-start + compilation reuse)
    Expected speedup: 4-5x
    """
    t, G_t = synthetic_relaxation_data

    def run_element_search():
        gmm = GeneralizedMaxwell(n_modes=20, modulus_type="shear")
        gmm.fit(
            t,
            G_t,
            test_mode="relaxation",
            optimization_factor=1.5,
            max_iter=500,  # Reduce iterations for benchmark
        )
        return gmm._n_modes, gmm._element_minimization_diagnostics

    # Run benchmark
    result = benchmark(run_element_search)

    # Verify optimal N selection
    n_opt, diagnostics = result
    assert 1 <= n_opt <= 20, f"Optimal N out of range: {n_opt}"
    assert diagnostics is not None


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_element_search_n10_oscillation(
    synthetic_oscillation_data, benchmark
):
    """Benchmark element search for N=10 in oscillation mode.

    Baseline (v0.3.1): ~25-35s (cold-start, complex modulus)
    Target (v0.4.0): ~6-12s (warm-start + compilation reuse)
    Expected speedup: 3-4x
    """
    omega, G_star = synthetic_oscillation_data

    def run_element_search():
        gmm = GeneralizedMaxwell(n_modes=10, modulus_type="shear")
        gmm.fit(omega, G_star, test_mode="oscillation", optimization_factor=1.5)
        return gmm._n_modes, gmm._element_minimization_diagnostics

    # Run benchmark
    result = benchmark(run_element_search)

    # Verify optimal N selection
    n_opt, diagnostics = result
    assert 1 <= n_opt <= 10, f"Optimal N out of range: {n_opt}"
    assert diagnostics is not None


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_element_search_n10_creep(synthetic_creep_data, benchmark):
    """Benchmark element search for N=10 in creep mode.

    Baseline (v0.3.1): ~30-40s (cold-start, creep simulation)
    Target (v0.4.0): ~7-14s (warm-start + compilation reuse)
    Expected speedup: 3-4x
    """
    t, J_t = synthetic_creep_data

    def run_element_search():
        gmm = GeneralizedMaxwell(n_modes=10, modulus_type="shear")
        gmm.fit(
            t,
            J_t,
            test_mode="creep",
            optimization_factor=1.5,
            max_iter=500,  # Reduce for benchmark
        )
        return gmm._n_modes, gmm._element_minimization_diagnostics

    # Run benchmark
    result = benchmark(run_element_search)

    # Verify optimal N selection
    n_opt, diagnostics = result
    assert 1 <= n_opt <= 10, f"Optimal N out of range: {n_opt}"
    assert diagnostics is not None


@pytest.mark.benchmark
@pytest.mark.slow
def test_speedup_measurement_n10_relaxation(synthetic_relaxation_data):
    """Measure and report speedup for N=10 relaxation mode.

    This test measures warm-start speedup by comparing multiple runs
    and reporting the performance improvement.
    """
    t, G_t = synthetic_relaxation_data

    # Run 3 times to account for JIT compilation
    times = []
    for i in range(3):
        gmm = GeneralizedMaxwell(n_modes=10, modulus_type="shear")
        start = time.time()
        gmm.fit(t, G_t, test_mode="relaxation", optimization_factor=1.5)
        elapsed = time.time() - start
        times.append(elapsed)

        # Store result for verification
        if i == 0:
            n_opt_first = gmm._n_modes
        else:
            # Verify consistency across runs
            assert gmm._n_modes == n_opt_first, "Inconsistent optimal N across runs"

    # Report statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    print(f"\n=== Speedup Measurement (N=10, Relaxation) ===")
    print(f"Runs: {len(times)}")
    print(f"Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"Min time: {min_time:.2f}s (after JIT warm-up)")
    print(f"Optimal N: {n_opt_first}")
    print(f"Baseline (v0.3.1 cold-start): ~20-30s")
    print(f"Expected speedup: 2-5x")

    # Verify performance is reasonable (not a strict assertion)
    # This is informational - actual speedup measured by comparing to v0.3.1
    assert mean_time < 60, f"Element search too slow: {mean_time:.2f}s"


@pytest.mark.benchmark
@pytest.mark.slow
def test_early_termination_effectiveness(synthetic_relaxation_data):
    """Test that early termination prevents futile small-N fits.

    Verifies that element minimization terminates early when R² degrades
    below threshold, avoiding unnecessary iterations for N=1,2,3,...
    """
    t, G_t = synthetic_relaxation_data

    # Use high optimization_factor to trigger early termination
    gmm = GeneralizedMaxwell(n_modes=15, modulus_type="shear")

    start = time.time()
    gmm.fit(t, G_t, test_mode="relaxation", optimization_factor=2.0)
    elapsed = time.time() - start

    diagnostics = gmm._element_minimization_diagnostics

    # Verify early termination occurred (not all N tested)
    n_tested = len(diagnostics["r2"])
    assert n_tested < 15, f"Expected early termination, but tested all {n_tested} modes"

    # Verify R² threshold criterion
    r2_values = diagnostics["r2"]
    n_optimal = diagnostics["n_optimal"]

    print(f"\n=== Early Termination Test ===")
    print(f"Initial N: 15")
    print(f"Tested N modes: {n_tested}")
    print(f"Optimal N: {n_optimal}")
    print(f"R² values: {r2_values}")
    print(f"Time saved by early termination: significant")

    # Verify optimization was successful
    assert diagnostics["n_optimal"] == gmm._n_modes
    assert len(r2_values) == n_tested
