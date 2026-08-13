"""Tests for rheojax.cli.fit — fit subcommand.

fit.py had no dedicated test file: its use of the shared
cli._common.load_and_flatten() helper was only exercised indirectly
(single-segment happy path via tests/io/test_io_fixes.py) and by one
manual end-to-end smoke run during development. These three tests mirror
the equivalent, already-proven coverage in test_bayesian.py
(test_load_exception_returns_1 / test_empty_segment_list_returns_1 /
test_multi_segment_uses_first) so fit.py's own wiring of the shared
loader — not just the loader itself — has a regression test.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.cli.fit import main
from rheojax.core.data import RheoData


class StubModel:
    """Lightweight stand-in for a registered model — no real NLSQ solve."""

    class _Param:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.parameters = {"G0": self._Param(5.0), "eta": self._Param(10.0)}
        self.fit_called = False

    def fit(self, x, y, **kwargs):
        self.fit_called = True
        return self


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
    return RheoData(
        x=x, y=y, domain="time", initial_test_mode="relaxation", validate=False
    )


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
        assert "segments, using first segment" in capsys.readouterr().err
