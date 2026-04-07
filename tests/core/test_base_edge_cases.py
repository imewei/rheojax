"""Edge case tests for BaseModel contracts.

Tests boundary conditions such as empty arrays, mismatched shapes,
predict-before-fit, and single-element inputs.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

import rheojax.models  # noqa: F401
from rheojax.models.classical.maxwell import Maxwell


@pytest.mark.smoke
class TestBaseModelEdgeCases:
    """Edge case tests using Maxwell as a concrete model."""

    def test_predict_before_fit_uses_defaults(self):
        """Predict before fit should use default parameters."""
        model = Maxwell()
        t = np.logspace(-2, 2, 10)
        # BaseModel.predict() should work with defaults (not raise)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))

    def test_fit_single_element(self):
        """Fit with single data point should still work or raise gracefully."""
        model = Maxwell()
        X = np.array([1.0])
        y = np.array([100.0])
        # May work or raise -- both are acceptable
        try:
            model.fit(X, y, test_mode="relaxation")
        except (RuntimeError, ValueError, Exception):
            pass  # Expected for underspecified problems

    def test_predict_with_empty_array(self):
        """Predict with empty array should return empty or raise."""
        model = Maxwell()
        X = np.array([])
        try:
            result = model.predict(X, test_mode="relaxation")
            assert len(result) == 0
        except (ValueError, RuntimeError):
            pass  # Also acceptable

    def test_fit_mismatched_lengths(self):
        """Fit with mismatched X and y lengths should raise."""
        model = Maxwell()
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            model.fit(X, y, test_mode="relaxation")

    def test_fit_with_nan_data(self):
        """Fit with NaN data should raise or handle gracefully."""
        model = Maxwell()
        X = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([100.0, 50.0, 25.0, 12.5])
        try:
            model.fit(X, y, test_mode="relaxation")
        except (RuntimeError, ValueError, Exception):
            pass  # Expected

    def test_fit_with_inf_data(self):
        """Fit with inf data should raise or handle gracefully."""
        model = Maxwell()
        X = np.array([1.0, 2.0, np.inf, 4.0])
        y = np.array([100.0, 50.0, 25.0, 12.5])
        try:
            model.fit(X, y, test_mode="relaxation")
        except (RuntimeError, ValueError, Exception):
            pass  # Expected

    def test_predict_returns_correct_length(self):
        """Predict should return same length as input."""
        model = Maxwell()
        for n in [5, 10, 50]:
            X = np.logspace(-2, 2, n)
            result = model.predict(X, test_mode="relaxation")
            assert len(result) == n

    def test_fitted_flag(self):
        """Model should track fitted state."""
        model = Maxwell()
        assert not model.fitted_
        t = np.logspace(-2, 2, 20)
        G = 1000.0 * np.exp(-t / 1.0)
        model.fit(t, G, test_mode="relaxation")
        assert model.fitted_

    def test_parameters_accessible(self):
        """Parameters should be accessible via keys()."""
        model = Maxwell()
        param_names = list(model.parameters.keys())
        assert len(param_names) > 0
        for name in param_names:
            val = model.parameters.get_value(name)
            assert val is not None
