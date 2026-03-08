"""Tests for FitResult, ModelInfo, and ModelComparison (Phase 1).

Tests cover:
- FitResult construction and property delegation to OptimizationResult
- Serialization round-trip (JSON, NPZ)
- Summary format and to_latex output
- ModelComparison ranking and Akaike weight normalization
- ModelInfo construction from known model
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.fit_result import FitResult, ModelComparison, ModelInfo
from rheojax.utils.optimization import OptimizationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimization_result(
    n_params: int = 2,
    n_data: int = 50,
    r2_target: float = 0.98,
) -> OptimizationResult:
    """Create a realistic OptimizationResult for testing."""
    x = np.array([1.0] * n_params)
    y_data = np.random.default_rng(42).normal(10, 1, n_data)
    # Residuals that give approximately the target R²
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    ss_res_target = (1 - r2_target) * ss_tot
    residuals = np.random.default_rng(42).normal(0, np.sqrt(ss_res_target / n_data), n_data)
    rss = float(np.sum(residuals**2))
    pcov = np.eye(n_params) * 0.01

    return OptimizationResult(
        x=x,
        fun=rss,
        jac=np.zeros((n_data, n_params)),
        pcov=pcov,
        success=True,
        message="converged",
        nit=100,
        nfev=200,
        residuals=residuals,
        y_data=y_data,
        n_data=n_data,
    )


def _make_fit_result(
    model_name: str = "maxwell",
    model_class_name: str = "Maxwell",
    protocol: str = "relaxation",
    n_params: int = 2,
    n_data: int = 50,
    include_opt_result: bool = True,
) -> FitResult:
    """Create a FitResult for testing."""
    opt = _make_optimization_result(n_params, n_data) if include_opt_result else None
    X = np.linspace(0.01, 10, n_data)
    y = np.exp(-X)
    fitted = np.exp(-X) + np.random.default_rng(42).normal(0, 0.01, n_data)
    return FitResult(
        model_name=model_name,
        model_class_name=model_class_name,
        protocol=protocol,
        params={"G0": 1000.0, "tau": 1.0},
        params_units={"G0": "Pa", "tau": "s"},
        n_params=n_params,
        optimization_result=opt,
        fitted_curve=fitted,
        X=X,
        y=y,
    )


# ---------------------------------------------------------------------------
# FitResult tests
# ---------------------------------------------------------------------------


class TestFitResult:
    """Tests for FitResult construction and property delegation."""

    def test_construction(self):
        result = _make_fit_result()
        assert result.model_name == "maxwell"
        assert result.model_class_name == "Maxwell"
        assert result.protocol == "relaxation"
        assert result.n_params == 2
        assert len(result.params) == 2
        assert result.params["G0"] == 1000.0

    def test_delegated_r_squared(self):
        result = _make_fit_result()
        r2 = result.r_squared
        assert r2 is not None
        assert isinstance(r2, float)

    def test_delegated_aic_bic(self):
        result = _make_fit_result()
        assert result.aic is not None
        assert result.bic is not None
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)

    def test_delegated_rmse_mae(self):
        result = _make_fit_result()
        assert result.rmse is not None
        assert result.mae is not None
        assert result.rmse >= 0
        assert result.mae >= 0

    def test_aicc(self):
        result = _make_fit_result(n_data=50, n_params=2)
        aicc = result.aicc
        assert aicc is not None
        # AICc should be close to AIC for large n
        assert abs(aicc - result.aic) < 1.0

    def test_aicc_small_sample(self):
        # n - k - 1 <= 0 should return NaN
        result = _make_fit_result(n_data=3, n_params=3)
        aicc = result.aicc
        assert aicc is None or (aicc is not None and math.isnan(aicc))

    def test_success_property(self):
        result = _make_fit_result()
        assert result.success is True

    def test_n_data_property(self):
        result = _make_fit_result(n_data=50)
        assert result.n_data == 50

    def test_confidence_intervals(self):
        result = _make_fit_result()
        ci = result.confidence_intervals(0.95)
        assert ci is not None
        assert ci.shape == (2, 2)

    def test_no_optimization_result(self):
        result = _make_fit_result(include_opt_result=False)
        assert result.r_squared is None
        assert result.aic is None
        assert result.bic is None
        assert result.rmse is None
        assert result.success is False
        assert result.confidence_intervals() is None

    def test_timestamp_auto(self):
        result = _make_fit_result()
        assert result.timestamp is not None
        assert len(result.timestamp) > 10

    def test_metadata(self):
        result = _make_fit_result()
        result.metadata["deformation_mode"] = "tension"
        assert result.metadata["deformation_mode"] == "tension"


class TestFitResultSummary:
    """Tests for summary and formatting methods."""

    def test_summary_format(self):
        result = _make_fit_result()
        summary = result.summary()
        assert "FitResult: Maxwell" in summary
        assert "relaxation" in summary
        assert "G0" in summary
        assert "tau" in summary
        assert "R²" in summary

    def test_to_latex(self):
        result = _make_fit_result()
        latex = result.to_latex()
        assert "Maxwell" in latex
        assert r"\\" in latex
        assert "&" in latex

    def test_to_dict(self):
        result = _make_fit_result()
        d = result.to_dict()
        assert d["model_name"] == "maxwell"
        assert d["n_params"] == 2
        assert "params" in d
        assert "X" in d
        assert "y" in d


class TestFitResultSerialization:
    """Tests for save/load round-trips."""

    def test_json_round_trip(self):
        result = _make_fit_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        result.save(path)
        loaded = FitResult.load(path)
        assert loaded.model_name == result.model_name
        assert loaded.n_params == result.n_params
        assert loaded.params["G0"] == result.params["G0"]
        np.testing.assert_array_almost_equal(loaded.X, result.X)
        Path(path).unlink()

    def test_npz_round_trip(self):
        result = _make_fit_result()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        result.save(path)
        loaded = FitResult.load(path)
        assert loaded.model_name == result.model_name
        assert loaded.n_params == result.n_params
        np.testing.assert_array_almost_equal(loaded.X, result.X)
        Path(path).unlink()

    def test_unsupported_extension(self):
        result = _make_fit_result()
        with pytest.raises(ValueError, match="Unsupported extension"):
            result.save("/tmp/test.xyz")

    def test_load_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported extension"):
            FitResult.load("/tmp/test.xyz")


class TestFitResultPlot:
    """Smoke tests for plotting."""

    @pytest.mark.smoke
    def test_plot_creates_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        result = _make_fit_result()
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    @pytest.mark.smoke
    def test_plot_no_data_raises(self):
        result = _make_fit_result()
        result.X = None
        with pytest.raises(ValueError, match="Cannot plot"):
            result.plot()


# ---------------------------------------------------------------------------
# ModelComparison tests
# ---------------------------------------------------------------------------


class TestModelComparison:
    """Tests for ModelComparison ranking and weights."""

    def _make_comparison(self) -> ModelComparison:
        results = []
        for name, cls_name, n_params in [
            ("maxwell", "Maxwell", 2),
            ("zener", "Zener", 3),
            ("springpot", "Springpot", 2),
        ]:
            results.append(_make_fit_result(
                model_name=name,
                model_class_name=cls_name,
                n_params=n_params,
            ))
        return ModelComparison(results=results, criterion="aic")

    def test_rankings_computed(self):
        comp = self._make_comparison()
        assert len(comp.rankings) == 3
        assert comp.best_model in comp.rankings
        assert comp.best_model == comp.rankings[0]

    def test_delta_criterion(self):
        comp = self._make_comparison()
        # Best model should have delta = 0
        assert comp.delta_criterion[comp.best_model] == 0.0
        # All deltas should be >= 0
        for d in comp.delta_criterion.values():
            assert d >= 0.0

    def test_akaike_weights_sum_to_one(self):
        comp = self._make_comparison()
        total = sum(comp.weights.values())
        assert abs(total - 1.0) < 1e-10

    def test_akaike_weights_positive(self):
        comp = self._make_comparison()
        for w in comp.weights.values():
            assert w >= 0.0

    def test_summary_format(self):
        comp = self._make_comparison()
        summary = comp.summary()
        assert "Model Comparison" in summary
        assert "AIC" in summary
        assert "maxwell" in summary

    @pytest.mark.smoke
    def test_plot_creates_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        comp = self._make_comparison()
        fig = comp.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_empty_results(self):
        comp = ModelComparison(results=[], criterion="aic")
        assert comp.rankings == []
        assert comp.best_model == ""

    def test_single_result(self):
        results = [_make_fit_result()]
        comp = ModelComparison(results=results, criterion="aic")
        assert len(comp.rankings) == 1
        assert comp.weights[comp.best_model] == 1.0


# ---------------------------------------------------------------------------
# ModelInfo tests
# ---------------------------------------------------------------------------


class TestModelInfo:
    """Tests for ModelInfo construction."""

    @pytest.mark.smoke
    def test_from_registry_maxwell(self):
        """Test ModelInfo construction from a known registered model."""
        try:
            info = ModelInfo.from_registry("maxwell")
            assert info.name == "maxwell"
            assert info.class_name == "Maxwell"
            assert info.n_params > 0
            assert len(info.param_names) == info.n_params
            assert info.supports_bayesian is True
            assert len(info.protocols) > 0
        except KeyError:
            pytest.skip("Maxwell model not registered in test environment")

    def test_from_registry_nonexistent(self):
        with pytest.raises(KeyError, match="not found"):
            ModelInfo.from_registry("nonexistent_model_xyz")
