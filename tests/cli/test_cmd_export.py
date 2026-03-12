"""Tests for rheojax.cli.cmd_export — export subcommand."""

from __future__ import annotations

import argparse

import pytest

from rheojax.cli.cmd_export import create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_input_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/"])
        assert ns.input == "results/"

    @pytest.mark.unit
    def test_has_output_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "bundle.h5"])
        assert str(ns.output) == "bundle.h5"

    @pytest.mark.unit
    def test_has_format_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/", "--format", "hdf5"])
        assert ns.export_format == "hdf5"

    @pytest.mark.unit
    def test_has_json_output_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/", "--json"])
        assert ns.json_output is True

    @pytest.mark.unit
    def test_default_format_is_directory(self):
        parser = create_parser()
        ns = parser.parse_args(["results/", "--output", "out/"])
        assert ns.export_format == "directory"


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

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
        assert result == 1
