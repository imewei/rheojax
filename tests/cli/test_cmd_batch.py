"""Tests for rheojax.cli.cmd_batch — batch subcommand."""

from __future__ import annotations

import argparse

import pytest

from rheojax.cli.cmd_batch import create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_pattern_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["data/*.csv", "--model", "maxwell"])
        assert ns.pattern == "data/*.csv"

    @pytest.mark.unit
    def test_has_model_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "springpot"])
        assert ns.model == "springpot"

    @pytest.mark.unit
    def test_has_test_mode_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--test-mode", "relaxation"])
        assert ns.test_mode == "relaxation"

    @pytest.mark.unit
    def test_has_output_dir_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--output-dir", "results"])
        assert str(ns.output_dir) == "results"

    @pytest.mark.unit
    def test_has_json_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--json"])
        assert ns.json_output is True

    @pytest.mark.unit
    def test_has_max_iter_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--max-iter", "500"])
        assert ns.max_iter == 500


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestMainErrors:
    @pytest.mark.smoke
    def test_main_no_matching_files_returns_1(self, tmp_path):
        # Pattern that matches nothing in tmp_path
        result = main([
            str(tmp_path / "*.totally_nonexistent_extension_xyz"),
            "--model",
            "maxwell",
            "--test-mode",
            "relaxation",
        ])
        assert result == 1

    @pytest.mark.unit
    def test_main_invalid_max_iter_returns_1(self, tmp_path):
        result = main([
            str(tmp_path / "*.csv"),
            "--model",
            "maxwell",
            "--max-iter",
            "0",
        ])
        assert result == 1
