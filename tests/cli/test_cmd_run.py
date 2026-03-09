"""Tests for rheojax.cli.cmd_run — run subcommand."""

from __future__ import annotations

import argparse

import pytest

from rheojax.cli.cmd_run import create_parser, main


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_config_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["pipeline.yaml"])
        assert str(ns.config) == "pipeline.yaml"

    @pytest.mark.unit
    def test_has_override_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["pipeline.yaml", "--override", "steps.1.model=zener"])
        assert ns.override == ["steps.1.model=zener"]

    @pytest.mark.unit
    def test_override_flag_repeatable(self):
        parser = create_parser()
        ns = parser.parse_args(
            ["pipeline.yaml", "--override", "a=1", "--override", "b=2"]
        )
        assert len(ns.override) == 2

    @pytest.mark.unit
    def test_has_dry_run_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["pipeline.yaml", "--dry-run"])
        assert ns.dry_run is True


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestMainErrors:
    @pytest.mark.smoke
    def test_main_nonexistent_config_returns_1(self, tmp_path):
        result = main([str(tmp_path / "nonexistent.yaml")])
        assert result == 1

    @pytest.mark.unit
    def test_main_dry_run_on_nonexistent_file_returns_1(self, tmp_path):
        result = main([str(tmp_path / "nope.yaml"), "--dry-run"])
        assert result == 1
