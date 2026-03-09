"""Tests for rheojax.cli.cmd_load — load subcommand."""

from __future__ import annotations

import argparse
import json

import pytest

from rheojax.cli.cmd_load import create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_input_file_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv"])
        assert str(ns.input_file) == "data.csv"

    @pytest.mark.unit
    def test_has_format_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv", "--format", "csv"])
        assert ns.file_format == "csv"

    @pytest.mark.unit
    def test_has_x_col_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv", "--x-col", "time"])
        assert ns.x_col == "time"

    @pytest.mark.unit
    def test_has_y_col_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv", "--y-col", "G_t"])
        assert ns.y_col == "G_t"

    @pytest.mark.unit
    def test_has_json_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv", "--json"])
        assert ns.json_output is True

    @pytest.mark.unit
    def test_has_test_mode_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["data.csv", "--test-mode", "relaxation"])
        assert ns.test_mode == "relaxation"


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestMainMissingFile:
    @pytest.mark.smoke
    def test_main_nonexistent_file_returns_1(self, tmp_path):
        result = main([str(tmp_path / "does_not_exist.csv")])
        assert result == 1


class TestMainWithCsv:
    @pytest.fixture
    def csv_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time,stress\n0.1,100\n0.2,90\n0.3,80\n")
        return f

    @pytest.mark.smoke
    def test_main_with_valid_csv_returns_0(self, csv_file, capsys):
        result = main(
            [
                str(csv_file),
                "--x-col",
                "time",
                "--y-col",
                "stress",
                "--test-mode",
                "relaxation",
            ]
        )
        assert result == 0

    @pytest.mark.unit
    def test_main_prints_summary(self, csv_file, capsys):
        main(
            [
                str(csv_file),
                "--x-col",
                "time",
                "--y-col",
                "stress",
                "--test-mode",
                "relaxation",
            ]
        )
        captured = capsys.readouterr()
        assert "3" in captured.out or "Points" in captured.out

    @pytest.mark.unit
    def test_main_json_flag_produces_valid_json(self, csv_file, capsys):
        result = main(
            [
                str(csv_file),
                "--x-col",
                "time",
                "--y-col",
                "stress",
                "--test-mode",
                "relaxation",
                "--json",
            ]
        )
        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert isinstance(parsed, dict)
