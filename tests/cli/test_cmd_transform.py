"""Tests for rheojax.cli.cmd_transform — transform subcommand."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from rheojax.cli.cmd_transform import _load_from_envelope, create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_transform_name_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["fft_analysis"])
        assert ns.transform_name == "fft_analysis"

    @pytest.mark.unit
    def test_has_input_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["fft_analysis", "--input", "data.csv"])
        assert ns.input == "data.csv"

    @pytest.mark.unit
    def test_has_output_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["fft_analysis", "--output", "result.csv"])
        assert ns.output == "result.csv"

    @pytest.mark.unit
    def test_has_param_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["fft_analysis", "--param", "n_harmonics=10"])
        assert ns.param == ["n_harmonics=10"]

    @pytest.mark.unit
    def test_has_json_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["fft_analysis", "--json"])
        assert ns.json_output is True


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["fft_analysis", "--help"])
        assert exc_info.value.code == 0


class TestMainErrors:
    @pytest.mark.smoke
    def test_main_no_input_returns_1(self, monkeypatch):
        # Patch stdin to appear as a tty so it won't try to read from it
        import sys
        from unittest.mock import MagicMock

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr(sys, "stdin", mock_stdin)

        result = main(["fft_analysis"])
        assert result == 1

    @pytest.mark.unit
    def test_main_unknown_transform_returns_1(self, tmp_path, monkeypatch):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("time,stress\n0.1,100\n0.2,90\n")
        result = main(["totally_unknown_transform_xyz", "--input", str(csv_file)])
        assert result == 1


class TestMainJsonOutputForTupleReturningTransforms:
    """PronyConversion and SpectrumInversion's _transform returns
    (RheoData, extras_dict), not a bare RheoData. Before the fix,
    `hasattr(result, "x")` was False on that tuple, so --json silently
    wrote {"data": {"x": [], "y": []}} and exited 0 instead of surfacing
    real (or a failure) output."""

    @pytest.mark.unit
    def test_prony_conversion_json_output_is_not_empty(self, tmp_path, capsys):
        csv_file = tmp_path / "relax.csv"
        t = [0.01 * (1.3**i) for i in range(30)]
        g = [1e5 * pow(2.718281828, -ti) + 1e3 for ti in t]
        rows = "\n".join(f"{ti},{gi}" for ti, gi in zip(t, g))
        csv_file.write_text(f"time,G_t\n{rows}\n")

        result = main(
            [
                "prony_conversion",
                "--input",
                str(csv_file),
                "--x-col",
                "time",
                "--y-col",
                "G_t",
                "--json",
            ]
        )

        assert result == 0
        envelope = json.loads(capsys.readouterr().out)
        assert len(envelope["data"]["x"]) > 0
        assert len(envelope["data"]["y"]) > 0


class TestLoadFromEnvelope:
    @pytest.mark.unit
    def test_accepts_a_real_create_data_envelope_payload(self):
        # Regression: _load_from_envelope used to read envelope["x"] directly,
        # but rheojax load --json (create_data_envelope) nests the payload under
        # envelope["data"]["x"], so every real pipe from `rheojax load --json`
        # raised "missing required 'x' key" before the fix.
        from rheojax.cli._envelope import create_data_envelope

        envelope_json = create_data_envelope(
            [0.1, 0.2, 0.3], [10.0, 9.0, 8.0], metadata={"test_mode": "relaxation"}
        ).to_json()

        data = _load_from_envelope(envelope_json)

        np.testing.assert_array_equal(data.x, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(data.y, [10.0, 9.0, 8.0])
