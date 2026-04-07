"""Tests for rheojax.utils.metrics (fit quality metrics)."""

import numpy as np
import pytest

from rheojax.utils.metrics import compute_fit_quality, r2_complex, r2_complex_components


@pytest.mark.smoke
class TestComputeFitQuality:
    """Tests for compute_fit_quality."""

    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_fit_quality(y, y)
        assert metrics["R2"] == 1.0
        assert metrics["RMSE"] == 0.0

    def test_good_fit(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        metrics = compute_fit_quality(y_true, y_pred)
        assert metrics["R2"] > 0.99
        assert metrics["RMSE"] < 0.15

    def test_bad_fit(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # Reversed
        metrics = compute_fit_quality(y_true, y_pred)
        assert metrics["R2"] < 0.0

    def test_empty_input(self):
        metrics = compute_fit_quality(np.array([]), np.array([]))
        assert np.isnan(metrics["R2"])
        assert np.isnan(metrics["RMSE"])

    def test_complex_input_raises(self):
        y = np.array([1 + 2j, 3 + 4j])
        with pytest.raises(TypeError, match="complex"):
            compute_fit_quality(y, y)

    def test_nrmse_computed(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_fit_quality(y_true, y_pred)
        assert "nrmse" in metrics

    def test_constant_data(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        metrics = compute_fit_quality(y_true, y_pred)
        assert metrics["R2"] == 1.0


@pytest.mark.smoke
class TestR2Complex:
    """Tests for r2_complex (magnitude-based)."""

    def test_perfect_complex(self):
        y = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = r2_complex(y, y)
        assert result == 1.0

    def test_real_input(self):
        y = np.array([1.0, 2.0, 3.0])
        result = r2_complex(y, y)
        assert result == 1.0


@pytest.mark.smoke
class TestR2ComplexComponents:
    """Tests for r2_complex_components (real+imag averaged)."""

    def test_perfect_components(self):
        y = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = r2_complex_components(y, y)
        assert result == 1.0

    def test_real_only(self):
        y = np.array([1.0, 2.0, 3.0])
        result = r2_complex_components(y, y)
        assert result == 1.0
