"""Tests for rheojax.cli.cmd_export — export subcommand."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from rheojax.cli.cmd_export import _build_pipeline_from_envelope, create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)  # nosec B101

    @pytest.mark.unit
    def test_has_input_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/"])
        assert ns.input == "results/"  # nosec B101

    @pytest.mark.unit
    def test_has_output_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "bundle.h5"])
        assert str(ns.output) == "bundle.h5"  # nosec B101

    @pytest.mark.unit
    def test_has_format_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/", "--format", "excel"])
        assert ns.export_format == "excel"  # nosec B101

    @pytest.mark.unit
    def test_hdf5_format_not_a_valid_choice(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["results/", "--output", "out/", "--format", "hdf5"])

    @pytest.mark.unit
    def test_has_json_output_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/", "--json"])
        assert ns.json_output is True  # nosec B101

    @pytest.mark.unit
    def test_default_format_is_directory(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/"])
        assert ns.export_format == "directory"  # nosec B101


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0  # nosec B101

    @pytest.mark.unit
    def test_main_nonexistent_input_returns_1(self, tmp_path, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr(sys, "stdin", mock_stdin)

        result = main(
            [
                str(tmp_path / "nonexistent_results"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert result == 1  # nosec B101


class TestBuildPipelineFromEnvelope:
    @pytest.mark.unit
    def test_accepts_a_real_create_data_envelope_payload(self):
        # Regression: _build_pipeline_from_envelope used to read envelope["x"]
        # directly, but rheojax load --json (create_data_envelope) nests the
        # payload under envelope["data"]["x"], so every real pipe from
        # `rheojax load --json | rheojax export -` raised on an empty x/y
        # array instead of exporting the piped data.
        from rheojax.cli._envelope import create_data_envelope

        envelope = json.loads(
            create_data_envelope([0.1, 0.2, 0.3], [10.0, 9.0, 8.0]).to_json()
        )

        pipe = _build_pipeline_from_envelope(envelope)

        # Same private-attribute access _build_pipeline_from_envelope itself
        # uses (cmd_export.py: `pipe._data = data  # type: ignore[attr-defined]`).
        np.testing.assert_array_equal(pipe._data.x, [0.1, 0.2, 0.3])  # type: ignore[attr-defined]
        np.testing.assert_array_equal(pipe._data.y, [10.0, 9.0, 8.0])  # type: ignore[attr-defined]
