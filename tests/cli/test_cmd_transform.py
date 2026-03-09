"""Tests for rheojax.cli.cmd_transform — transform subcommand."""

from __future__ import annotations

import argparse

import pytest

from rheojax.cli.cmd_transform import create_parser, main


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
