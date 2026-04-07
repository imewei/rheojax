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
        finally:
            Path(path).unlink(missing_ok=True)


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


@pytest.mark.smoke
class TestModelComparison:
    """Test ModelComparison rankings."""

    def test_empty_results(self):
        comp = ModelComparison(results=[])
        assert comp.best_model == ""

    def test_invalid_criterion(self):
        with pytest.raises(ValueError, match="criterion must be"):
            ModelComparison(results=[], criterion="invalid")
