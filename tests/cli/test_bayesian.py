"""Tests for rheojax.cli.bayesian — bayesian subcommand.

Covers the command-execution path in main() (data loading, validation,
model creation, test-mode resolution, warm-start, NUTS invocation, and
output formatting). Real MCMC is stubbed at the model boundary so the
tests exercise CLI wiring without running NumPyro — see StubModel.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from rheojax.cli.bayesian import create_parser, main
from rheojax.core.bayesian_result import BayesianResult
from rheojax.core.data import RheoData


def _make_result() -> BayesianResult:
    """Minimal BayesianResult with two parameters and clean diagnostics."""
    return BayesianResult(
        posterior_samples={
            "a": np.linspace(4.0, 6.0, 20),
            "b": np.linspace(0.4, 0.6, 20),
        },
        summary={},
        diagnostics={
            "divergences": 0,
            "r_hat": {"a": 1.001, "b": 1.002},
            "ess": {"a": 95.0, "b": 88.0},
        },
        num_samples=20,
        num_chains=1,
    )


class StubModel:
    """Lightweight stand-in for a registered model — no real MCMC."""

    def __init__(self, *, fit_raises: bool = False):
        self._fit_raises = fit_raises
        self._last_fit_kwargs: dict = {"stale": True}
        self._test_mode = "stale"
        self.fitted_ = True
        self.fit_called = False
        self.bayesian_kwargs: dict | None = None

    def fit(self, x, y, test_mode=None):
        self.fit_called = True
        if self._fit_raises:
            raise RuntimeError("nlsq boom")
        return self

    def fit_bayesian(self, x, y, *, test_mode, num_warmup, num_samples, num_chains, seed):
        self.bayesian_kwargs = {
            "test_mode": test_mode,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "seed": seed,
        }
        return _make_result()


@pytest.fixture
def data_file(tmp_path):
    """An input path that exists on disk (contents unused — auto_load stubbed)."""
    p = tmp_path / "data.csv"
    p.write_text("time,G_t\n0.1,5.0\n0.2,4.5\n")
    return p


@pytest.fixture
def rheo_data():
    x = np.linspace(0.1, 5.0, 10)
    y = 5.0 * np.exp(-0.5 * x)
    return RheoData(x=x, y=y, domain="time", initial_test_mode="relaxation", validate=False)


@pytest.fixture
def wired(monkeypatch, rheo_data):
    """Patch auto_load + ModelRegistry.create; return the stub model in use.

    Returns a dict so individual tests can swap the returned data/model.
    """
    state: dict = {"data": rheo_data, "model": StubModel()}

    def fake_auto_load(path, **kwargs):
        return state["data"]

    monkeypatch.setattr("rheojax.io.auto_load", fake_auto_load)
    monkeypatch.setattr(
        "rheojax.core.registry.ModelRegistry.create",
        lambda name: state["model"],
    )
    return state


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_parser(self):
        assert isinstance(create_parser(), argparse.ArgumentParser)

    @pytest.mark.unit
    def test_defaults(self):
        ns = create_parser().parse_args(["data.csv", "--model", "maxwell"])
        assert ns.warmup == 1000
        assert ns.samples == 2000
        assert ns.chains == 4
        assert ns.seed == 0
        assert ns.warm_start is False
        assert ns.json_output is False

    @pytest.mark.unit
    def test_model_required(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(["data.csv"])

    @pytest.mark.unit
    def test_flags_parsed(self):
        ns = create_parser().parse_args(
            ["data.csv", "-m", "springpot", "--warm-start", "--json", "--chains", "2"]
        )
        assert ns.warm_start is True
        assert ns.json_output is True
        assert ns.chains == 2


# --------------------------------------------------------------------------- #
# Validation (lines 152-166)
# --------------------------------------------------------------------------- #


class TestInputValidation:
    @pytest.mark.smoke
    def test_missing_file_returns_1(self, tmp_path):
        result = main([str(tmp_path / "nope.csv"), "--model", "maxwell"])
        assert result == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("flag", ["--warmup", "--samples", "--chains"])
    def test_nonpositive_sampling_param_returns_1(self, data_file, flag):
        result = main([str(data_file), "--model", "maxwell", flag, "0"])
        assert result == 1


# --------------------------------------------------------------------------- #
# Data loading + array validation (lines 168-219)
# --------------------------------------------------------------------------- #


class TestDataLoading:
    @pytest.mark.unit
    def test_load_exception_returns_1(self, data_file, monkeypatch, capsys):
        def boom(path, **kwargs):
            raise ValueError("bad file")

        monkeypatch.setattr("rheojax.io.auto_load", boom)
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 1
        assert "Error loading data" in capsys.readouterr().err

    @pytest.mark.unit
    def test_empty_segment_list_returns_1(self, data_file, monkeypatch):
        monkeypatch.setattr("rheojax.io.auto_load", lambda path, **k: [])
        monkeypatch.setattr(
            "rheojax.core.registry.ModelRegistry.create", lambda name: StubModel()
        )
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 1

    @pytest.mark.unit
    def test_multi_segment_uses_first(self, data_file, monkeypatch, rheo_data, capsys):
        segments = [rheo_data, rheo_data]
        monkeypatch.setattr("rheojax.io.auto_load", lambda path, **k: segments)
        monkeypatch.setattr(
            "rheojax.core.registry.ModelRegistry.create", lambda name: StubModel()
        )
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 0
        assert "using first segment" in capsys.readouterr().err

    @pytest.mark.unit
    def test_nan_in_x_returns_1(self, data_file, wired, capsys):
        x = np.array([0.1, np.nan, 0.3])
        wired["data"] = RheoData(
            x=x, y=np.array([1.0, 2.0, 3.0]), domain="time",
            initial_test_mode="relaxation", validate=False,
        )
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 1
        assert "NaN/Inf" in capsys.readouterr().err

    @pytest.mark.unit
    def test_nan_in_y_returns_1(self, data_file, wired, capsys):
        y = np.array([1.0, np.inf, 3.0])
        wired["data"] = RheoData(
            x=np.array([0.1, 0.2, 0.3]), y=y, domain="time",
            initial_test_mode="relaxation", validate=False,
        )
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 1
        assert "NaN/Inf" in capsys.readouterr().err

    @pytest.mark.unit
    def test_nan_in_complex_y_returns_1(self, data_file, wired, capsys):
        y = np.array([1 + 1j, complex(np.nan, 0.0), 3 + 0j])
        wired["data"] = RheoData(
            x=np.array([0.1, 0.2, 0.3]), y=y, domain="frequency",
            initial_test_mode="oscillation", validate=False,
        )
        result = main([str(data_file), "--model", "maxwell", "-t", "oscillation"])
        assert result == 1
        assert "complex y column" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Model creation + test mode (lines 221-247)
# --------------------------------------------------------------------------- #


class TestModelAndTestMode:
    @pytest.mark.unit
    def test_model_create_failure_returns_1(self, data_file, monkeypatch, rheo_data, capsys):
        monkeypatch.setattr("rheojax.io.auto_load", lambda path, **k: rheo_data)

        def raise_key(name):
            raise KeyError(name)

        monkeypatch.setattr("rheojax.core.registry.ModelRegistry.create", raise_key)
        result = main([str(data_file), "--model", "nonexistent", "-t", "relaxation"])
        assert result == 1
        assert "Could not create model" in capsys.readouterr().err

    @pytest.mark.unit
    def test_test_mode_autodetected_from_data(self, data_file, wired):
        # rheo_data was built with initial_test_mode="relaxation"; no -t flag.
        result = main([str(data_file), "--model", "maxwell"])
        assert result == 0
        assert wired["model"].bayesian_kwargs["test_mode"] == "relaxation"

    @pytest.mark.unit
    def test_missing_test_mode_returns_1(self, data_file, wired, capsys):
        # A real RheoData auto-detects a mode, so use a bare stub whose
        # test_mode is genuinely None and which exposes no metadata.
        class _NoModeData:
            x = np.linspace(0.1, 1.0, 5)
            y = np.linspace(1.0, 0.5, 5)
            test_mode = None

        wired["data"] = _NoModeData()
        result = main([str(data_file), "--model", "maxwell"])
        assert result == 1
        assert "auto-detect test mode" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Warm-start (lines 249-266)
# --------------------------------------------------------------------------- #


class TestWarmStart:
    @pytest.mark.unit
    def test_warm_start_calls_fit(self, data_file, wired):
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation", "--warm-start"])
        assert result == 0
        assert wired["model"].fit_called is True

    @pytest.mark.unit
    def test_warm_start_failure_resets_state_and_continues(self, data_file, wired, capsys):
        model = StubModel(fit_raises=True)
        wired["model"] = model
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation", "--warm-start"])
        assert result == 0
        # Partially-mutated state must be reset after a failed fit.
        assert model._last_fit_kwargs == {}
        assert model._test_mode is None
        assert model.fitted_ is False
        assert "warm-start failed" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Inference + output (lines 268-385)
# --------------------------------------------------------------------------- #


class TestInferenceAndOutput:
    @pytest.mark.unit
    def test_forwards_sampling_params(self, data_file, wired):
        result = main([
            str(data_file), "--model", "maxwell", "-t", "relaxation",
            "--warmup", "3", "--samples", "5", "--chains", "2", "--seed", "7",
        ])
        assert result == 0
        kw = wired["model"].bayesian_kwargs
        assert kw == {
            "test_mode": "relaxation",
            "num_warmup": 3,
            "num_samples": 5,
            "num_chains": 2,
            "seed": 7,
        }

    @pytest.mark.unit
    def test_inference_exception_returns_1(self, data_file, wired, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("nuts exploded")

        monkeypatch.setattr(wired["model"], "fit_bayesian", boom)
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 1
        assert "Error during Bayesian inference" in capsys.readouterr().err

    @pytest.mark.smoke
    def test_table_output(self, data_file, wired, capsys):
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Posterior Summary" in out
        assert "Model: maxwell" in out
        # Both parameters appear in the table.
        assert "a" in out and "b" in out

    @pytest.mark.smoke
    def test_json_output(self, data_file, wired, capsys):
        result = main([str(data_file), "--model", "maxwell", "-t", "relaxation", "--json"])
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["model"] == "maxwell"
        assert payload["test_mode"] == "relaxation"
        assert payload["diagnostics"]["divergences"] == 0
        assert set(payload["summary"]) == {"a", "b"}
        assert payload["diagnostics"]["r_hat"]["a"] == pytest.approx(1.001)

    @pytest.mark.unit
    def test_output_written_to_file(self, data_file, wired, tmp_path, capsys):
        out_file = tmp_path / "result.json"
        result = main([
            str(data_file), "--model", "maxwell", "-t", "relaxation",
            "--json", "-o", str(out_file),
        ])
        assert result == 0
        assert "Results written to" in capsys.readouterr().err
        payload = json.loads(out_file.read_text())
        assert payload["model"] == "maxwell"

    @pytest.mark.unit
    def test_output_write_failure_returns_1(self, data_file, wired, tmp_path, capsys):
        bad = tmp_path / "missing_dir" / "result.txt"
        result = main([
            str(data_file), "--model", "maxwell", "-t", "relaxation", "-o", str(bad),
        ])
        assert result == 1
        assert "Error writing to" in capsys.readouterr().err
