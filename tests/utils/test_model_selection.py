"""Tests for model selection and comparison utilities."""

import numpy as np
import pytest

from rheojax.core.fit_result import ModelComparison
from rheojax.utils.model_selection import build_fit_result, compare_models


@pytest.mark.smoke
class TestCompareModels:
    """Tests for compare_models function."""

    def test_basic_comparison(self):
        """Compare 2 classical models on relaxation data."""
        t = np.logspace(-2, 2, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)

        comparison = compare_models(
            t, G_t,
            models=["maxwell", "zener"],
            test_mode="relaxation",
            criterion="aic",
        )

        assert isinstance(comparison, ModelComparison)
        assert len(comparison.results) >= 1
        assert comparison.best_model != ""

    def test_akaike_weights(self):
        """Akaike weights should sum to 1."""
        t = np.logspace(-2, 2, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)

        comparison = compare_models(
            t, G_t,
            models=["maxwell", "zener"],
            test_mode="relaxation",
        )

        if comparison.weights:
            total = sum(comparison.weights.values())
            assert abs(total - 1.0) < 1e-10

    def test_invalid_criterion(self):
        """Invalid criterion should raise ValueError."""
        with pytest.raises(ValueError, match="criterion"):
            compare_models(
                np.array([1.0]),
                np.array([1.0]),
                models=["maxwell"],
                criterion="invalid",
            )

    def test_nonexistent_model_skipped(self):
        """Nonexistent models should be skipped, not crash."""
        t = np.logspace(-2, 2, 50)
        G_t = 1000.0 * np.exp(-t)

        comparison = compare_models(
            t, G_t,
            models=["maxwell", "definitely_not_a_real_model"],
            test_mode="relaxation",
        )
        # Should have at least maxwell
        assert len(comparison.results) >= 1


@pytest.mark.smoke
class TestBuildFitResult:
    """Tests for build_fit_result helper."""

    def test_basic(self):
        """build_fit_result should produce a valid FitResult."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t = np.logspace(-2, 2, 50)
        G_t = 1000.0 * np.exp(-t)
        model.fit(t, G_t, test_mode="relaxation")

        result = build_fit_result(model, t, G_t, test_mode="relaxation")
        assert result.model_name == "maxwell"
        assert result.n_params == 2
        assert result.r_squared is not None
        assert result.fitted_curve is not None

    def test_unfitted_model(self):
        """build_fit_result on unfitted model should still work."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        result = build_fit_result(model, np.array([1.0]), np.array([1.0]))
        assert result.model_name == "maxwell"
        assert result.fitted_curve is None
