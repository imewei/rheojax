"""Tests for rheojax.cli._yaml_schema — YAML config loading and validation."""

from __future__ import annotations

import pytest

from rheojax.cli._yaml_schema import PipelineConfig, load_config, validate_config

_VALID_CONFIG_YAML = """\
version: "1"
name: "Test Pipeline"
defaults:
  test_mode: relaxation
steps:
  - type: load
    file: data.csv
  - type: fit
    model: maxwell
  - type: export
    output: results/
"""


@pytest.fixture
def valid_config_file(tmp_path):
    f = tmp_path / "pipeline.yaml"
    f.write_text(_VALID_CONFIG_YAML)
    return f


class TestPipelineConfigDataclass:
    @pytest.mark.unit
    def test_fields_present(self):
        config = PipelineConfig(version="1", name="Test")
        assert config.version == "1"
        assert config.name == "Test"
        assert config.defaults == {}
        assert config.steps == []

    @pytest.mark.unit
    def test_defaults_and_steps_populated(self):
        steps = [{"type": "load", "file": "data.csv"}]
        config = PipelineConfig(
            version="1", name="Test", defaults={"test_mode": "relaxation"}, steps=steps
        )
        assert config.defaults["test_mode"] == "relaxation"
        assert len(config.steps) == 1


class TestValidateConfig:
    @pytest.mark.smoke
    def test_valid_config_returns_empty_errors(self):
        config = PipelineConfig(
            version="1",
            name="Pipeline",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "fit", "model": "maxwell"},
            ],
        )
        errors = validate_config(config)
        assert errors == []

    @pytest.mark.unit
    def test_missing_load_step_is_error(self):
        config = PipelineConfig(
            version="1",
            name="Pipeline",
            steps=[{"type": "fit", "model": "maxwell"}],
        )
        errors = validate_config(config)
        assert any("load" in e for e in errors)

    @pytest.mark.unit
    def test_bayesian_before_fit_is_error(self):
        config = PipelineConfig(
            version="1",
            name="Pipeline",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "bayesian"},
            ],
        )
        errors = validate_config(config)
        assert any("bayesian" in e.lower() or "fit" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_unknown_step_type_is_error(self):
        config = PipelineConfig(
            version="1",
            name="Pipeline",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "frobnicate"},
            ],
        )
        errors = validate_config(config)
        assert any("frobnicate" in e for e in errors)

    @pytest.mark.unit
    def test_empty_steps_is_error(self):
        config = PipelineConfig(version="1", name="Pipeline", steps=[])
        errors = validate_config(config)
        assert errors  # Non-empty

    @pytest.mark.unit
    def test_missing_name_is_error(self):
        config = PipelineConfig(
            version="1",
            name="",
            steps=[{"type": "load", "file": "data.csv"}],
        )
        errors = validate_config(config)
        assert any("name" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_unsupported_version_is_error(self):
        config = PipelineConfig(
            version="99",
            name="Pipeline",
            steps=[{"type": "load", "file": "data.csv"}],
        )
        errors = validate_config(config)
        assert any("version" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_fit_required_key_missing_is_error(self):
        config = PipelineConfig(
            version="1",
            name="Pipeline",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "fit"},  # Missing model
            ],
        )
        errors = validate_config(config)
        assert any("model" in e for e in errors)


class TestPathSafety:
    @pytest.mark.unit
    def test_absolute_file_path_rejected(self):
        config = PipelineConfig(
            version="1",
            name="P",
            steps=[{"type": "load", "file": "/etc/passwd"}],
        )
        errors = validate_config(config)
        assert any("absolute" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_dotdot_file_path_rejected(self):
        config = PipelineConfig(
            version="1",
            name="P",
            steps=[{"type": "load", "file": "../../../etc/passwd"}],
        )
        errors = validate_config(config)
        assert any(".." in e for e in errors)

    @pytest.mark.unit
    def test_absolute_output_path_rejected(self):
        config = PipelineConfig(
            version="1",
            name="P",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "export", "output": "/tmp/evil"},
            ],
        )
        errors = validate_config(config)
        assert any("absolute" in e.lower() for e in errors)

    @pytest.mark.unit
    def test_dotdot_output_path_rejected(self):
        config = PipelineConfig(
            version="1",
            name="P",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "export", "output": "../../evil"},
            ],
        )
        errors = validate_config(config)
        assert any(".." in e for e in errors)

    @pytest.mark.unit
    def test_relative_paths_accepted(self):
        config = PipelineConfig(
            version="1",
            name="P",
            steps=[
                {"type": "load", "file": "data/sample.csv"},
                {"type": "fit", "model": "maxwell"},
                {"type": "export", "output": "results/out"},
            ],
        )
        errors = validate_config(config)
        # No path-related errors
        assert not any("absolute" in e.lower() or ".." in e for e in errors)


class TestLoadConfig:
    @pytest.mark.smoke
    def test_load_config_valid_file(self, valid_config_file):
        config = load_config(valid_config_file)
        assert isinstance(config, PipelineConfig)
        assert config.name == "Test Pipeline"
        assert config.version == "1"
        assert len(config.steps) == 3

    @pytest.mark.unit
    def test_load_config_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    @pytest.mark.unit
    def test_load_config_raises_on_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: '1'\nname: 'T'\nsteps: [{ type: fit }]\n")
        with pytest.raises(ValueError):
            load_config(bad)

    @pytest.mark.unit
    def test_load_config_raises_on_non_mapping_yaml(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_config(bad)

    @pytest.mark.unit
    def test_load_config_defaults_populated(self, valid_config_file):
        config = load_config(valid_config_file)
        assert config.defaults.get("test_mode") == "relaxation"
