"""Performance tests for ITT-MCT model.

Tests cover:
- Prony decomposition convergence and quality
- JIT compilation time benchmarks
- Precompilation effectiveness
"""

import time

import numpy as np
import pytest

from rheojax.utils.mct_kernels import prony_decompose_memory


class TestPronyDecomposition:
    """Tests for Prony series decomposition robustness."""

    @pytest.mark.smoke
    def test_prony_leastsq_convergence(self):
        """Test that leastsq method converges on typical memory kernel."""
        # Generate synthetic memory kernel (sum of exponentials)
        t = np.logspace(-3, 2, 100)
        # True Prony series: m(t) = 0.5*exp(-t/0.1) + 0.3*exp(-t/1) + 0.2*exp(-t/10)
        m_true = (
            0.5 * np.exp(-t / 0.1) + 0.3 * np.exp(-t / 1.0) + 0.2 * np.exp(-t / 10.0)
        )

        # Fit with leastsq method
        g, tau = prony_decompose_memory(t, m_true, n_modes=5, method="leastsq")

        # Reconstruct and check fit quality
        m_fit = np.sum(g[None, :] * np.exp(-t[:, None] / tau[None, :]), axis=1)

        # R² should be > 0.99 for clean data
        ss_res = np.sum((m_true - m_fit) ** 2)
        ss_tot = np.sum((m_true - m_true.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.99, f"R² = {r_squared:.4f}, expected > 0.99"

    @pytest.mark.smoke
    def test_prony_multistart_convergence(self):
        """Test that multistart method converges on ill-conditioned kernel."""
        # Generate ill-conditioned kernel (closely spaced modes)
        t = np.logspace(-2, 3, 200)
        # Modes at τ = 0.5, 1, 2 (closely spaced in log-space)
        m_true = (
            0.4 * np.exp(-t / 0.5) + 0.35 * np.exp(-t / 1.0) + 0.25 * np.exp(-t / 2.0)
        )

        # Fit with multistart method
        g, tau = prony_decompose_memory(
            t, m_true, n_modes=5, method="multistart", n_starts=5
        )

        # Reconstruct and check fit quality
        m_fit = np.sum(g[None, :] * np.exp(-t[:, None] / tau[None, :]), axis=1)

        ss_res = np.sum((m_true - m_fit) ** 2)
        ss_tot = np.sum((m_true - m_true.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.95, f"R² = {r_squared:.4f}, expected > 0.95"

    def test_prony_log_spacing_fallback(self):
        """Test that log_spacing fallback produces reasonable fit."""
        t = np.logspace(-2, 2, 100)
        m_true = np.exp(-t / 1.0)  # Simple exponential decay

        g, tau = prony_decompose_memory(t, m_true, n_modes=5, method="log_spacing")

        # Reconstruct
        m_fit = np.sum(g[None, :] * np.exp(-t[:, None] / tau[None, :]), axis=1)

        ss_res = np.sum((m_true - m_fit) ** 2)
        ss_tot = np.sum((m_true - m_true.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Log-spacing should still give reasonable fit (>0.9)
        assert r_squared > 0.9, f"R² = {r_squared:.4f}, expected > 0.9"

    def test_prony_respects_positive_constraint(self):
        """Test that Prony amplitudes are non-negative."""
        t = np.logspace(-2, 2, 100)
        m_true = 0.5 * np.exp(-t / 0.1) + 0.5 * np.exp(-t / 10.0)

        g, tau = prony_decompose_memory(t, m_true, n_modes=5, method="leastsq")

        # All amplitudes should be non-negative (enforced by bounds)
        assert np.all(g >= 0), f"Found negative amplitudes: {g}"

    def test_prony_sorted_by_tau(self):
        """Test that output is sorted by relaxation time."""
        t = np.logspace(-2, 2, 100)
        m_true = np.exp(-t / 1.0)

        g, tau = prony_decompose_memory(t, m_true, n_modes=5, method="leastsq")

        # τ should be sorted ascending
        assert np.all(np.diff(tau) >= 0), f"τ not sorted: {tau}"

    def test_prony_handles_noisy_data(self):
        """Test Prony decomposition with noisy data."""
        rng = np.random.default_rng(42)
        t = np.logspace(-2, 2, 100)
        m_clean = 0.5 * np.exp(-t / 0.1) + 0.5 * np.exp(-t / 10.0)
        noise = rng.normal(0, 0.05 * m_clean.mean(), size=m_clean.shape)
        m_noisy = m_clean + noise
        m_noisy = np.maximum(m_noisy, 1e-10)  # Keep positive

        g, tau = prony_decompose_memory(
            t, m_noisy, n_modes=5, method="multistart", n_starts=3
        )

        # Reconstruct
        m_fit = np.sum(g[None, :] * np.exp(-t[:, None] / tau[None, :]), axis=1)

        ss_res = np.sum((m_clean - m_fit) ** 2)
        ss_tot = np.sum((m_clean - m_clean.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Should still capture main trend (>0.8 with noise)
        assert r_squared > 0.8, f"R² = {r_squared:.4f}, expected > 0.8"


class TestDiffraxCompilation:
    """Tests for diffrax JIT compilation performance."""

    @pytest.fixture
    def itt_mct_model(self):
        """Create ITTMCTSchematic model for testing."""
        from rheojax.models.itt_mct import ITTMCTSchematic

        return ITTMCTSchematic(epsilon=-0.1)  # Fluid state for faster convergence

    @pytest.mark.slow
    def test_diffrax_compilation_occurs(self, itt_mct_model):
        """Test that JIT compilation happens on first call."""
        pytest.importorskip("diffrax")

        gamma_dot = np.logspace(-1, 1, 5)

        # First call triggers compilation
        start = time.time()
        _ = itt_mct_model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        first_call_time = time.time() - start

        # Second call should be much faster
        start = time.time()
        _ = itt_mct_model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        second_call_time = time.time() - start

        # First call should be significantly slower (compilation)
        # Note: exact times vary by hardware, so we use a generous threshold
        assert first_call_time > second_call_time * 2, (
            f"First call ({first_call_time:.1f}s) should be >2x slower than "
            f"second call ({second_call_time:.1f}s)"
        )

    @pytest.mark.slow
    def test_precompile_method_works(self, itt_mct_model):
        """Test that precompile() method triggers compilation."""
        pytest.importorskip("diffrax")

        # Precompile
        compile_time = itt_mct_model.precompile()

        # Should return a positive time
        assert compile_time > 0, "precompile() should return positive compilation time"

        # Now first prediction should be fast
        gamma_dot = np.logspace(-1, 1, 5)
        start = time.time()
        _ = itt_mct_model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        predict_time = time.time() - start

        # After precompilation, prediction should be fast (< 5s for simple case)
        assert (
            predict_time < 10
        ), f"After precompile(), prediction took {predict_time:.1f}s (expected <10s)"

    @pytest.mark.slow
    def test_precompile_reduces_first_call_time(self):
        """Test that precompile() reduces the effective first-call time."""
        pytest.importorskip("diffrax")

        from rheojax.models.itt_mct import ITTMCTSchematic

        gamma_dot = np.logspace(-1, 1, 5)

        # Test WITHOUT precompile
        model1 = ITTMCTSchematic(epsilon=-0.1)
        start = time.time()
        _ = model1.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        time_without_precompile = time.time() - start

        # Test WITH precompile (on a fresh model with same n_modes)
        model2 = ITTMCTSchematic(epsilon=-0.1)
        _ = model2.precompile()  # This triggers compilation

        # Now prediction should be cached from model1's compilation
        start = time.time()
        _ = model2.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        time_with_precompile = time.time() - start

        # The predict() call after precompile should be much faster
        # (compilation was already done by precompile or cached from model1)
        assert time_with_precompile < time_without_precompile, (
            f"With precompile: {time_with_precompile:.1f}s, "
            f"without: {time_without_precompile:.1f}s"
        )


class TestPronyPerformance:
    """Performance benchmarks for Prony decomposition."""

    @pytest.mark.slow
    def test_prony_multistart_vs_single_start(self):
        """Compare convergence quality of multistart vs single start."""
        # Generate challenging kernel (multiple time scales)
        t = np.logspace(-3, 3, 200)
        m_true = (
            0.3 * np.exp(-t / 0.01) + 0.3 * np.exp(-t / 1.0) + 0.4 * np.exp(-t / 100.0)
        )

        # Single-start (leastsq)
        g_single, tau_single = prony_decompose_memory(
            t, m_true, n_modes=5, method="leastsq"
        )
        m_fit_single = np.sum(
            g_single[None, :] * np.exp(-t[:, None] / tau_single[None, :]), axis=1
        )
        ss_res_single = np.sum((m_true - m_fit_single) ** 2)
        ss_tot = np.sum((m_true - m_true.mean()) ** 2)
        r2_single = 1 - ss_res_single / ss_tot

        # Multi-start
        g_multi, tau_multi = prony_decompose_memory(
            t, m_true, n_modes=5, method="multistart", n_starts=5
        )
        m_fit_multi = np.sum(
            g_multi[None, :] * np.exp(-t[:, None] / tau_multi[None, :]), axis=1
        )
        ss_res_multi = np.sum((m_true - m_fit_multi) ** 2)
        r2_multi = 1 - ss_res_multi / ss_tot

        # Multi-start should be at least as good as single-start
        assert (
            r2_multi >= r2_single - 0.01
        ), f"Multi-start R²={r2_multi:.4f} should be >= single-start R²={r2_single:.4f}"

        # Both should achieve reasonable fit
        assert r2_multi > 0.95, f"Multi-start R²={r2_multi:.4f}, expected > 0.95"
