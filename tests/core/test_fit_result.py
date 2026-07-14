"""Tests for FitResult, ModelInfo, and ModelComparison."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.fit_result import FitResult, ModelComparison, ModelInfo


@pytest.mark.smoke
class TestFitResultConstruction:
    """Test FitResult dataclass construction."""

    def test_minimal_construction(self):
        result = FitResult(
            model_name="maxwell",
            model_class_name="Maxwell",
            protocol="relaxation",
            params={"G": 1000.0, "eta": 100.0},
            params_units={"G": "Pa", "eta": "Pa·s"},
            n_params=2,
            optimization_result=None,
        )
        assert result.model_name == "maxwell"
        assert result.n_params == 2
        assert result.protocol == "relaxation"

    def test_params_access(self):
        result = FitResult(
            model_name="maxwell",
            model_class_name="Maxwell",
            protocol="relaxation",
            params={"G": 1000.0, "eta": 100.0},
            params_units={"G": "Pa", "eta": "Pa·s"},
            n_params=2,
            optimization_result=None,
        )
        assert result.params["G"] == 1000.0
        assert result.params["eta"] == 100.0

    def test_success_false_without_opt_result(self):
        result = FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={},
            params_units={},
            n_params=0,
            optimization_result=None,
        )
        assert result.success is False

    def test_r_squared_none_without_opt_result(self):
        result = FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={},
            params_units={},
            n_params=0,
            optimization_result=None,
        )
        assert result.r_squared is None

    def test_timestamp_set(self):
        result = FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={},
            params_units={},
            n_params=0,
            optimization_result=None,
        )
        assert len(result.timestamp) > 0

    def test_n_data_from_y(self):
        result = FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={},
            params_units={},
            n_params=0,
            optimization_result=None,
            y=np.array([1.0, 2.0, 3.0]),
        )
        assert result.n_data == 3


@pytest.mark.smoke
class TestFitResultSerialization:
    """Test to_dict and save/load."""

    def _make_result(self):
        return FitResult(
            model_name="maxwell",
            model_class_name="Maxwell",
            protocol="relaxation",
            params={"G": 1000.0, "eta": 100.0},
            params_units={"G": "Pa", "eta": "Pa·s"},
            n_params=2,
            optimization_result=None,
            X=np.linspace(0.01, 10, 20),
            y=np.exp(-np.linspace(0.01, 10, 20)),
            metadata={"test_key": "test_value"},
        )

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["model_name"] == "maxwell"
        assert d["n_params"] == 2
        assert "params" in d
        assert d["params"]["G"] == 1000.0

    def test_to_dict_has_x_y(self):
        result = self._make_result()
        d = result.to_dict()
        assert "X" in d
        assert "y" in d
        assert len(d["X"]) == 20

    def test_save_load_json(self):
        result = self._make_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = FitResult.load(path)
            assert loaded.model_name == "maxwell"
            assert loaded.params["G"] == 1000.0
            assert loaded.n_params == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_npz(self):
        result = self._make_result()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = FitResult.load(path)
            assert loaded.model_name == "maxwell"
            assert loaded.params["G"] == 1000.0
            # Regression: params_units and metadata used to be silently
            # dropped by the npz round-trip (unlike the JSON path).
            assert loaded.params_units == {"G": "Pa", "eta": "Pa·s"}
            assert loaded.metadata == {"test_key": "test_value"}
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_npz_missing_params_units_metadata_backward_compat(self):
        """Loading a .npz saved before params_units/metadata were persisted
        must fall back to {} rather than raising KeyError."""
        import json as _json

        def _str_to_bytes(s: str) -> np.ndarray:
            return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            np.savez(
                path,
                model_name=_str_to_bytes("maxwell"),
                model_class_name=_str_to_bytes("Maxwell"),
                protocol=_str_to_bytes("relaxation"),
                n_params=np.array(2),
                timestamp=_str_to_bytes("2024-01-01T00:00:00"),
                param_names=_str_to_bytes(_json.dumps(["G", "eta"])),
                param_values=np.array([1000.0, 100.0], dtype=np.float64),
                success=np.array(True),
                n_data=np.array(20),
            )
            loaded = FitResult.load(path)
            assert loaded.params_units == {}
            assert loaded.metadata == {}
            assert loaded.params["G"] == 1000.0
        finally:
            Path(path).unlink(missing_ok=True)

    def _make_log_residual_result(self):
        """FitResult wrapping an OptimizationResult with _use_log_residuals=True.

        Mirrors the common production case (SGR/STZ/fIKH/fluidity/EPM/Giesekus
        default to use_log_residuals=True) where G'/G'' data spans multiple
        decades, so r_squared/aic/bic must be computed in log10 space to avoid
        a spurious R^2 (see OptimizationResult.r_squared).
        """
        from rheojax.utils.optimization import OptimizationResult

        rng = np.random.default_rng(0)
        t = np.linspace(0, 4, 40)  # log10 decades 0..4
        y = 10.0**t
        log_noise = rng.normal(scale=0.3, size=t.shape)
        fitted = 10.0 ** (t + log_noise)
        log_residuals = np.log10(fitted) - np.log10(y)

        opt_result = OptimizationResult(
            x=np.array([1.0, 2.0]),
            fun=float(np.sum(log_residuals**2)),
            success=True,
            y_data=y,
            residuals=log_residuals,
            n_data=len(y),
            _use_log_residuals=True,
        )
        return FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={"a": 1.0, "b": 2.0},
            params_units={},
            n_params=2,
            optimization_result=opt_result,
            fitted_curve=fitted,
            y=y,
        )

    def test_save_load_json_preserves_log_residual_stats(self):
        """load() must return the persisted stats, not recompute them from
        raw (linear-space) y/fitted_curve arrays, which silently discards
        _use_log_residuals semantics and gives wrong numbers with no error."""
        result = self._make_log_residual_result()
        original_r2 = result.r_squared
        original_aic = result.aic
        original_bic = result.bic
        original_residuals = result.residuals
        assert original_r2 is not None

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = FitResult.load(path)
            assert loaded.r_squared == pytest.approx(original_r2)
            assert loaded.aic == pytest.approx(original_aic)
            assert loaded.bic == pytest.approx(original_bic)
            # Regression: .residuals used to be silently reconstructed as
            # linear-space (y - fitted_curve) on load, even though the
            # original fit used log10 residuals -- no exception, just a
            # completely different array with matching r_squared/aic/bic.
            np.testing.assert_allclose(loaded.residuals, original_residuals)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_npz_preserves_log_residual_stats(self):
        result = self._make_log_residual_result()
        original_r2 = result.r_squared
        original_aic = result.aic
        original_bic = result.bic
        original_residuals = result.residuals
        assert original_r2 is not None

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = FitResult.load(path)
            assert loaded.r_squared == pytest.approx(original_r2)
            assert loaded.aic == pytest.approx(original_aic)
            assert loaded.bic == pytest.approx(original_bic)
            np.testing.assert_allclose(loaded.residuals, original_residuals)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_plot_residual_panel_uses_log_space_for_log_residual_fits(self):
        """plot()'s residual panel must not silently ignore
        _use_log_residuals=True and show raw linear residuals (y - fitted),
        which would contradict the log-space R^2 shown in the title above
        it and look badly heteroscedastic for a genuinely good fit."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        result = self._make_log_residual_result()
        result.X = np.linspace(0, 4, len(result.y))

        fig = result.plot()
        try:
            ax_res = fig.axes[1]
            plotted_y = ax_res.lines[0].get_ydata()
            np.testing.assert_allclose(plotted_y, result.residuals)
            linear_residuals = result.y - np.asarray(result.fitted_curve)
            assert not np.allclose(plotted_y, linear_residuals)
            assert ax_res.get_ylabel() == "Log10 Residual"
        finally:
            plt.close(fig)


@pytest.mark.smoke
class TestFitResultSummary:
    """Test summary and display methods."""

    def test_summary_string(self):
        result = FitResult(
            model_name="maxwell",
            model_class_name="Maxwell",
            protocol="relaxation",
            params={"G": 1000.0},
            params_units={"G": "Pa"},
            n_params=1,
            optimization_result=None,
        )
        summary = result.summary()
        assert "Maxwell" in summary
        assert "relaxation" in summary

    def test_to_latex(self):
        result = FitResult(
            model_name="maxwell",
            model_class_name="Maxwell",
            protocol="relaxation",
            params={"G": 1000.0},
            params_units={},
            n_params=1,
            optimization_result=None,
        )
        latex = result.to_latex()
        assert "Maxwell" in latex
        assert "\\\\" in latex

    def test_aicc_none_when_insufficient_dof(self):
        """aicc must return None (matching every sibling stat property's
        undefined-value convention), not np.nan, when n - k - 1 <= 0.
        A bare np.nan breaks JSON serialization (not RFC-8259-valid) and
        defeats the `if val is not None` guards in summary()/exporters."""
        from rheojax.utils.optimization import OptimizationResult

        y = np.array([1.0, 2.0, 3.0])
        fitted = np.array([1.1, 1.9, 3.2])
        residuals = y - fitted
        opt_result = OptimizationResult(
            x=np.zeros(3),
            fun=float(np.sum(residuals**2)),
            success=True,
            y_data=y,
            residuals=residuals,
            n_data=len(y),
        )
        result = FitResult(
            model_name="test",
            model_class_name="Test",
            protocol=None,
            params={"a": 1.0, "b": 2.0, "c": 3.0},
            params_units={},
            n_params=3,  # n_data(3) - n_params(3) - 1 = -1 <= 0
            optimization_result=opt_result,
            fitted_curve=fitted,
            y=y,
        )
        assert result.aic is not None
        assert result.aicc is None
        assert "AICc" not in result.summary()


@pytest.mark.smoke
class TestModelInfo:
    """Test ModelInfo.from_registry."""

    def test_from_registry(self):
        import rheojax.models  # noqa: F401

        info = ModelInfo.from_registry("maxwell")
        assert info.name == "maxwell"
        assert info.n_params > 0
        assert len(info.param_names) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            ModelInfo.from_registry("nonexistent_model_xyz")

    def test_supports_bayesian_requires_model_function(self):
        """supports_bayesian must reflect BayesianMixin's actual runtime
        contract (model_function), not hasattr(instance, "fit_bayesian")
        which is always True since BaseModel defines fit_bayesian
        unconditionally for every subclass."""
        from rheojax.core.base import BaseModel
        from rheojax.core.registry import ModelRegistry

        class _NoModelFunction(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.fitted_ = True
                return self

            def _predict(self, X, **kwargs):
                return X

        name = "_test_no_model_function_xyz"
        ModelRegistry.register(name)(_NoModelFunction)
        try:
            info = ModelInfo.from_registry(name)
            assert info.supports_bayesian is False
        finally:
            ModelRegistry.unregister(name)


@pytest.mark.smoke
class TestModelComparison:
    """Test ModelComparison rankings."""

    def test_empty_results(self):
        comp = ModelComparison(results=[])
        assert comp.best_model == ""

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion must be"):
            ModelComparison(results=[], criterion="invalid")

    def test_duplicate_model_names_not_silently_dropped(self):
        """Two FitResults sharing model_name (e.g. two pre-instantiated
        instances of the same model class) must both survive ranking.

        Regression test: rankings/delta_criterion/weights used to be dict
        comprehensions keyed purely by model_name, so a duplicate silently
        collided and discarded one result with no warning.
        """

        def _make(aic):
            return FitResult(
                model_name="maxwell",
                model_class_name="Maxwell",
                protocol="relaxation",
                params={"G": 1.0},
                params_units={},
                n_params=1,
                optimization_result=None,
                _persisted_stats={"aic": aic},
            )

        r1 = _make(10.0)
        r2 = _make(20.0)
        comp = ModelComparison(results=[r1, r2], criterion="aic")

        assert len(comp.rankings) == 2
        assert len(comp.delta_criterion) == 2
        assert len(comp.weights) == 2
        assert sorted(comp.rankings.values()) == [1, 2]
        assert sorted(comp.delta_criterion.values()) == [0.0, 10.0]

        # Display should still show the original, un-suffixed model_name
        # for both rows rather than the internal disambiguation key.
        summary = comp.summary()
        assert summary.count("maxwell") == 2
        assert "#" not in summary
