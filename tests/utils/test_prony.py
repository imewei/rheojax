"""Tests for Prony series utilities.

Test coverage focuses on critical Prony operations:
- Parameter validation (bounds checking, positivity)
- ParameterSet creation for N modes
- Log-space transforms for wide time-scale ranges
- R² computation
- Element minimization logic
- Softmax penalty differentiability
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.utils.prony import (
    compute_r_squared,
    create_prony_parameter_set,
    iterative_n_reduction,
    log_tau_to_tau,
    select_optimal_n,
    softmax_penalty,
    tau_to_log_tau,
    validate_prony_parameters,
)

# Safe JAX import
jax, jnp = safe_import_jax()


class TestParameterValidation:
    """Test validate_prony_parameters() bounds and positivity checks."""

    def test_valid_parameters(self):
        """Valid parameters should pass validation."""
        E_inf = 1e3
        E_i = np.array([1e5, 1e4, 1e3])
        tau_i = np.array([1e-2, 1e-1, 1.0])

        valid, msg = validate_prony_parameters(E_inf, E_i, tau_i)

        assert valid is True
        assert msg == ""

    def test_negative_E_inf(self):
        """Negative equilibrium modulus should fail."""
        E_inf = -100.0
        E_i = np.array([1e5])
        tau_i = np.array([1.0])

        valid, msg = validate_prony_parameters(E_inf, E_i, tau_i)

        assert valid is False
        assert "E_inf must be non-negative" in msg

    def test_negative_mode_strength(self):
        """Negative mode strength should fail."""
        E_inf = 1e3
        E_i = np.array([1e5, -1e4, 1e3])
        tau_i = np.array([1e-2, 1e-1, 1.0])

        valid, msg = validate_prony_parameters(E_inf, E_i, tau_i)

        assert valid is False
        assert "E_i must be positive" in msg
        assert "[1]" in msg  # Index of negative value

    def test_mismatched_array_lengths(self):
        """Mismatched E_i and tau_i lengths should fail."""
        E_inf = 1e3
        E_i = np.array([1e5, 1e4])
        tau_i = np.array([1.0])

        valid, msg = validate_prony_parameters(E_inf, E_i, tau_i)

        assert valid is False
        assert "same length" in msg


class TestParameterSetCreation:
    """Test create_prony_parameter_set() for N modes."""

    def test_shear_modulus_creation(self):
        """Create shear modulus parameters (G_inf, G_i, tau_i)."""
        params = create_prony_parameter_set(n_modes=3, modulus_type="shear")

        expected_names = ["G_inf", "G_1", "G_2", "G_3", "tau_1", "tau_2", "tau_3"]
        assert list(params.keys()) == expected_names
        assert len(params) == 7  # 2*3 + 1

    def test_tensile_modulus_creation(self):
        """Create tensile modulus parameters (E_inf, E_i, tau_i)."""
        params = create_prony_parameter_set(n_modes=2, modulus_type="tensile")

        expected_names = ["E_inf", "E_1", "E_2", "tau_1", "tau_2"]
        assert list(params.keys()) == expected_names
        assert len(params) == 5  # 2*2 + 1

    def test_invalid_n_modes(self):
        """n_modes < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_modes must be ≥ 1"):
            create_prony_parameter_set(n_modes=0)


class TestLogSpaceTransforms:
    """Test tau_to_log_tau() and inverse for wide time-scale ranges."""

    def test_forward_transform(self):
        """Transform tau to log10(tau)."""
        tau = np.array([1e-3, 1e-1, 1e1, 1e3])
        log_tau = tau_to_log_tau(tau)

        expected = np.array([-3.0, -1.0, 1.0, 3.0])
        np.testing.assert_allclose(log_tau, expected, rtol=1e-10)

    def test_inverse_transform(self):
        """Transform log10(tau) back to tau."""
        log_tau = np.array([-3.0, -1.0, 1.0, 3.0])
        tau = log_tau_to_tau(log_tau)

        expected = np.array([1e-3, 1e-1, 1e1, 1e3])
        np.testing.assert_allclose(tau, expected, rtol=1e-10)

    def test_roundtrip_consistency(self):
        """tau → log → tau should be identity."""
        tau_original = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])

        log_tau = tau_to_log_tau(tau_original)
        tau_recovered = log_tau_to_tau(log_tau)

        np.testing.assert_allclose(tau_recovered, tau_original, rtol=1e-10)


class TestRSquaredComputation:
    """Test compute_r_squared() metric for element minimization."""

    def test_perfect_fit(self):
        """R² = 1.0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        r2 = compute_r_squared(y_true, y_pred)
        assert np.isclose(r2, 1.0, atol=1e-10)

    def test_mean_baseline(self):
        """R² = 0.0 when predictions equal mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))

        r2 = compute_r_squared(y_true, y_pred)
        assert np.isclose(r2, 0.0, atol=1e-10)

    def test_good_fit(self):
        """R² close to 1.0 for good predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

        r2 = compute_r_squared(y_true, y_pred)
        assert r2 > 0.99
        assert r2 <= 1.0


class TestElementMinimization:
    """Test iterative_n_reduction() and select_optimal_n() logic."""

    def test_iterative_n_reduction_diagnostics(self):
        """Track R² vs N for visualization."""
        fit_results = {10: 0.998, 8: 0.995, 6: 0.990, 4: 0.980, 2: 0.950}
        diagnostics = iterative_n_reduction(fit_results)

        assert "n_modes" in diagnostics
        assert "r2" in diagnostics
        assert "r2_min" in diagnostics
        assert "r2_max" in diagnostics
        assert diagnostics["r2_min"] == 0.950
        assert diagnostics["r2_max"] == 0.998
        np.testing.assert_array_equal(diagnostics["n_modes"], [2, 4, 6, 8, 10])

    def test_select_optimal_n_maximum_quality(self):
        """optimization_factor=1.0 requires best R² (no degradation)."""
        r2 = {5: 0.998, 3: 0.995, 2: 0.980, 1: 0.900}
        # R²_max = 0.998
        # factor=1.0: degradation = 0, threshold = 0.998
        # Only N=5 achieves R² = 0.998
        n_opt = select_optimal_n(r2, optimization_factor=1.0)
        assert n_opt == 5

    def test_select_optimal_n_balanced(self):
        """optimization_factor=1.5 balances parsimony and quality."""
        r2 = {5: 0.998, 3: 0.995, 2: 0.980, 1: 0.900}
        # R²_max = 0.998, degradation_room = 1 - 0.998 = 0.002
        # factor=1.5: allowed_degradation = 0.002 × 0.5 = 0.001
        # threshold = 0.998 - 0.001 = 0.997
        # Smallest N with R² ≥ 0.997: N=3 (R²=0.995 < 0.997), so N=5
        n_opt = select_optimal_n(r2, optimization_factor=1.5)
        # Actually N=5 since threshold 0.997 > 0.995
        assert n_opt == 5

    def test_select_optimal_n_maximum_parsimony(self):
        """optimization_factor=2.0 maximizes parsimony (100% degradation allowed)."""
        r2 = {5: 0.998, 3: 0.990, 2: 0.950, 1: 0.800}
        # R²_max = 0.998, degradation_room = 0.002
        # factor=2.0: allowed_degradation = 0.002 × 1.0 = 0.002
        # threshold = 0.998 - 0.002 = 0.996
        # Smallest N with R² ≥ 0.996: still N=5
        n_opt = select_optimal_n(r2, optimization_factor=2.0)
        assert n_opt == 5  # Even with max parsimony, only N=5 meets threshold

    def test_select_optimal_n_wide_gap_scenario(self):
        """Test scenario with large R² gaps between N values."""
        r2 = {10: 0.999, 8: 0.998, 6: 0.990, 4: 0.950, 2: 0.800}
        # R²_max = 0.999, degradation_room = 0.001
        # factor=1.5: allowed_degradation = 0.001 × 0.5 = 0.0005
        # threshold = 0.999 - 0.0005 = 0.9985
        # Smallest N with R² ≥ 0.9985: N=8 (0.998 < 0.9985), so N=10
        n_opt = select_optimal_n(r2, optimization_factor=1.5)
        assert n_opt == 10


class TestSoftmaxPenalty:
    """Test softmax_penalty() for constrained optimization."""

    def test_negative_value_increases_penalty(self):
        """Negative Eᵢ should increase penalty significantly."""
        E_i_positive = np.array([1e5, 1e4, 1e3])
        E_i_negative = np.array([1e5, 1e4, -1e3])

        penalty_pos = float(softmax_penalty(E_i_positive, scale=1e3))
        penalty_neg = float(softmax_penalty(E_i_negative, scale=1e3))

        assert penalty_neg > penalty_pos
        assert penalty_neg > 600.0  # Significant penalty for negative value

    def test_jax_differentiability(self):
        """Penalty should be differentiable with JAX."""
        E_i = jnp.array([1e5, 1e4, 1e3])

        # Compute gradient using JAX
        grad_fn = jax.grad(lambda E: softmax_penalty(E, scale=1e3))
        gradient = grad_fn(E_i)

        # Gradient should be finite
        assert jnp.all(jnp.isfinite(gradient))
        # Gradient should be negative (penalty decreases as E increases)
        assert jnp.all(gradient < 0)
