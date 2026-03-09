"""Tests for rheojax.cli._envelope — JSON envelope dataclass."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rheojax.cli._envelope import (
    Envelope,
    create_data_envelope,
    create_fit_envelope,
)


class TestEnvelopeRoundTrip:
    @pytest.mark.smoke
    def test_to_json_from_json_round_trip(self):
        original = Envelope(
            rheojax_version="0.1.0",
            envelope_type="data",
            data={"x": [1.0, 2.0], "y": [3.0, 4.0]},
            metadata={"test_mode": "relaxation"},
        )
        json_str = original.to_json()
        restored = Envelope.from_json(json_str)

        assert restored.rheojax_version == original.rheojax_version
        assert restored.envelope_type == original.envelope_type
        assert restored.data == original.data
        assert restored.metadata == original.metadata

    @pytest.mark.unit
    def test_to_json_returns_string(self):
        env = Envelope(rheojax_version="0.1.0", envelope_type="data")
        result = env.to_json()
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_to_json_contains_required_keys(self):
        env = Envelope(rheojax_version="0.6.0", envelope_type="fit_result")
        parsed = json.loads(env.to_json())
        assert "rheojax_version" in parsed
        assert "envelope_type" in parsed

    @pytest.mark.unit
    def test_from_json_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid envelope JSON"):
            Envelope.from_json("not valid json {{{")

    @pytest.mark.unit
    def test_from_json_raises_on_missing_required_keys(self):
        with pytest.raises(ValueError, match="rheojax_version"):
            Envelope.from_json(json.dumps({"envelope_type": "data"}))

    @pytest.mark.unit
    def test_optional_fields_absent_when_none(self):
        env = Envelope(rheojax_version="0.1.0", envelope_type="data")
        parsed = json.loads(env.to_json())
        assert "fit_result" not in parsed
        assert "bayesian_result" not in parsed

    @pytest.mark.unit
    def test_fit_result_included_when_set(self):
        env = Envelope(
            rheojax_version="0.1.0",
            envelope_type="fit_result",
            fit_result={"model": "maxwell", "parameters": {"G_e": 1000.0}},
        )
        parsed = json.loads(env.to_json())
        assert "fit_result" in parsed
        assert parsed["fit_result"]["model"] == "maxwell"


class TestCreateDataEnvelope:
    @pytest.mark.smoke
    def test_create_data_envelope_basic(self):
        x = np.linspace(0, 1, 5)
        y = np.ones(5)
        env = create_data_envelope(x, y)

        assert env.envelope_type == "data"
        assert env.data is not None
        assert len(env.data["x"]) == 5
        assert len(env.data["y"]) == 5

    @pytest.mark.unit
    def test_create_data_envelope_contains_version(self):
        env = create_data_envelope([1, 2, 3], [4, 5, 6])
        assert env.rheojax_version != ""

    @pytest.mark.unit
    def test_create_data_envelope_json_has_version_key(self):
        env = create_data_envelope([1.0], [2.0])
        parsed = json.loads(env.to_json())
        assert "rheojax_version" in parsed

    @pytest.mark.unit
    def test_create_data_envelope_with_metadata(self):
        env = create_data_envelope([1.0], [2.0], metadata={"test_mode": "creep"})
        assert env.metadata["test_mode"] == "creep"

    @pytest.mark.unit
    def test_create_data_envelope_list_input(self):
        env = create_data_envelope([0.1, 0.2, 0.3], [100.0, 90.0, 80.0])
        assert env.data is not None
        assert env.data["x"] == [0.1, 0.2, 0.3]


class TestCreateFitEnvelope:
    @pytest.mark.unit
    def test_create_fit_envelope_basic(self):
        mock_model = MagicMock()
        mock_model.name = "maxwell"
        params = {"G_e": 1000.0, "tau": 0.1}
        env = create_fit_envelope(mock_model, params, test_mode="relaxation")

        assert env.envelope_type == "fit_result"
        assert env.fit_result is not None
        assert env.fit_result["model"] == "maxwell"
        assert env.fit_result["test_mode"] == "relaxation"

    @pytest.mark.unit
    def test_create_fit_envelope_parameters_serialised(self):
        mock_model = MagicMock()
        mock_model.name = "zener"
        params = {"G_e": 500.0, "G_m": 200.0, "eta": 0.5}
        env = create_fit_envelope(mock_model, params, test_mode="oscillation")
        parsed = json.loads(env.to_json())
        assert parsed["fit_result"]["parameters"]["G_e"] == 500.0

    @pytest.mark.unit
    def test_create_fit_envelope_uses_class_name_fallback(self):
        class FakeModel:
            pass  # No .name attribute

        env = create_fit_envelope(FakeModel(), {}, test_mode="relaxation")
        assert env.fit_result["model"] == "FakeModel"


class TestFromStdin:
    @pytest.mark.unit
    def test_from_stdin_returns_none_when_tty(self, monkeypatch):
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("sys.stdin", mock_stdin)

        result = Envelope.from_stdin()
        assert result is None

    @pytest.mark.unit
    def test_from_stdin_parses_valid_envelope(self, monkeypatch):
        env = Envelope(rheojax_version="0.1.0", envelope_type="data", data={"x": [], "y": []})
        json_str = env.to_json()

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = json_str
        monkeypatch.setattr("sys.stdin", mock_stdin)

        result = Envelope.from_stdin()
        assert result is not None
        assert result.envelope_type == "data"

    @pytest.mark.unit
    def test_from_stdin_returns_none_on_empty_stdin(self, monkeypatch):
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "   "
        monkeypatch.setattr("sys.stdin", mock_stdin)

        result = Envelope.from_stdin()
        assert result is None
