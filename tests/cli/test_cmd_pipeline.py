"""Tests for rheojax.cli.cmd_pipeline — pipeline management subcommand."""

from __future__ import annotations

import argparse

import pytest
import yaml

from rheojax.cli.cmd_pipeline import create_parser, main

_VALID_PIPELINE_YAML = """\
version: "1"
name: "Test Pipeline"
steps:
  - type: load
    file: data.csv
  - type: fit
    model: maxwell
  - type: export
    output: results/
"""


@pytest.fixture
def valid_yaml_file(tmp_path):
    f = tmp_path / "pipeline.yaml"
    f.write_text(_VALID_PIPELINE_YAML)
    return f


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_init_subcommand(self):
        parser = create_parser()
        ns = parser.parse_args(["init", "--template", "basic"])
        assert ns.command == "init"

    @pytest.mark.unit
    def test_has_validate_subcommand(self):
        parser = create_parser()
        ns = parser.parse_args(["validate", "pipeline.yaml"])
        assert ns.command == "validate"

    @pytest.mark.unit
    def test_has_show_subcommand(self):
        parser = create_parser()
        ns = parser.parse_args(["show", "pipeline.yaml"])
        assert ns.command == "show"


class TestMainNoSubcommand:
    @pytest.mark.smoke
    def test_main_no_subcommand_returns_0(self, capsys):
        result = main([])
        assert result == 0


class TestMainInit:
    @pytest.mark.smoke
    def test_main_init_creates_file(self, tmp_path):
        output = tmp_path / "new_pipeline.yaml"
        result = main(["init", "--template", "basic", "--output", str(output)])
        assert result == 0
        assert output.exists()

    @pytest.mark.unit
    def test_main_init_output_is_valid_yaml(self, tmp_path):
        output = tmp_path / "pipeline.yaml"
        main(["init", "--template", "basic", "--output", str(output)])
        parsed = yaml.safe_load(output.read_text())
        assert isinstance(parsed, dict)
        assert "steps" in parsed

    @pytest.mark.unit
    def test_main_init_existing_file_returns_1_without_force(self, tmp_path):
        output = tmp_path / "pipeline.yaml"
        output.write_text("existing content")
        result = main(["init", "--template", "basic", "--output", str(output)])
        assert result == 1

    @pytest.mark.unit
    def test_main_init_force_overwrites(self, tmp_path):
        output = tmp_path / "pipeline.yaml"
        output.write_text("existing content")
        result = main(
            ["init", "--template", "basic", "--output", str(output), "--force"]
        )
        assert result == 0


class TestMainValidate:
    @pytest.mark.smoke
    def test_main_validate_valid_config_returns_0(self, valid_yaml_file, capsys):
        result = main(["validate", str(valid_yaml_file)])
        assert result == 0

    @pytest.mark.unit
    def test_main_validate_missing_file_returns_1(self, tmp_path):
        result = main(["validate", str(tmp_path / "nope.yaml")])
        assert result == 1

    @pytest.mark.unit
    def test_main_validate_invalid_yaml_returns_1(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: '1'\nname: 'T'\nsteps:\n  - type: bayesian\n")
        result = main(["validate", str(bad)])
        assert result == 1
