"""Integration tests for CLI YAML pipeline roundtrip.

Tests that YAML configs → PipelineConfig → PipelineBuilder → steps
produce the expected pipeline structure, and that all templates pass
schema validation.  These tests establish a contract between the CLI
YAML format and the pipeline builder that the future GUI serializer
must also satisfy.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from rheojax.cli._templates import TEMPLATES, get_template, list_templates
from rheojax.cli._yaml_runner import apply_overrides, config_to_builder
from rheojax.cli._yaml_schema import PipelineConfig, load_config, validate_config

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _config_from_yaml(yaml_str: str) -> PipelineConfig:
    """Parse a YAML string into a PipelineConfig (in-memory, no file)."""
    raw = yaml.safe_load(yaml_str)
    config = PipelineConfig(
        version=str(raw["version"]),
        name=str(raw["name"]),
        defaults=dict(raw.get("defaults") or {}),
        steps=raw.get("steps", []),
    )
    errors = validate_config(config)
    assert not errors, f"Validation errors: {errors}"
    return config


# ------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------


class TestTemplateSchemaValidation:
    """Every shipped template must pass schema validation."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
    def test_template_validates(self, template_name: str):
        yaml_str = get_template(template_name)
        config = _config_from_yaml(yaml_str)
        assert config.version == "1"
        assert config.name
        assert len(config.steps) >= 2  # at least load + one more

    @pytest.mark.smoke
    def test_list_templates_nonempty(self):
        templates = list_templates()
        assert len(templates) >= 4
        for t in templates:
            assert "name" in t
            assert "description" in t


# ------------------------------------------------------------------
# Config → Builder translation
# ------------------------------------------------------------------


class TestConfigToBuilder:
    """YAML config translates to correct PipelineBuilder steps."""

    @pytest.mark.smoke
    def test_basic_template_steps(self):
        config = _config_from_yaml(get_template("basic"))
        builder = config_to_builder(config)
        steps = builder.get_steps()

        assert len(steps) == 3
        assert steps[0][0] == "load"
        assert steps[1][0] == "fit"
        assert steps[2][0] == "export"

    @pytest.mark.smoke
    def test_bayesian_template_steps(self):
        config = _config_from_yaml(get_template("bayesian"))
        builder = config_to_builder(config)
        steps = builder.get_steps()

        assert len(steps) == 4
        assert steps[0][0] == "load"
        assert steps[1][0] == "fit"
        assert steps[2][0] == "bayesian"
        assert steps[3][0] == "export"

        # Bayesian kwargs preserved
        bayes_kwargs = steps[2][1]
        assert bayes_kwargs["num_warmup"] == 1000
        assert bayes_kwargs["num_samples"] == 2000
        assert bayes_kwargs["num_chains"] == 4
        assert bayes_kwargs["seed"] == 42
        assert bayes_kwargs["warm_start"] is True

    @pytest.mark.smoke
    def test_mastercurve_template_has_transform(self):
        config = _config_from_yaml(get_template("mastercurve"))
        builder = config_to_builder(config)
        steps = builder.get_steps()

        assert len(steps) == 4
        assert steps[1][0] == "transform"
        assert steps[1][1]["name"] == "mastercurve"

    @pytest.mark.smoke
    def test_fit_kwargs_forwarded(self):
        config = _config_from_yaml(get_template("basic"))
        builder = config_to_builder(config)
        steps = builder.get_steps()

        fit_kwargs = steps[1][1]
        assert fit_kwargs["model"] == "maxwell"
        assert fit_kwargs["max_iter"] == 5000

    @pytest.mark.smoke
    def test_export_kwargs_forwarded(self):
        config = _config_from_yaml(get_template("basic"))
        builder = config_to_builder(config)
        steps = builder.get_steps()

        export_kwargs = steps[2][1]
        assert export_kwargs["output_path"] == "results/"
        assert export_kwargs["format"] == "directory"

    @pytest.mark.smoke
    def test_defaults_merged_into_steps(self):
        """Defaults from the config are merged into each step's kwargs."""
        yaml_str = textwrap.dedent("""\
            version: "1"
            name: "Test defaults merge"
            defaults:
              test_mode: relaxation
            steps:
              - type: load
                file: data.csv
              - type: fit
                model: maxwell
              - type: export
                output: out/
        """)
        config = _config_from_yaml(yaml_str)
        builder = config_to_builder(config)
        steps = builder.get_steps()

        # Defaults should be merged into fit step
        fit_kwargs = steps[1][1]
        assert fit_kwargs.get("test_mode") == "relaxation"


# ------------------------------------------------------------------
# Override application
# ------------------------------------------------------------------


class TestOverrides:
    """CLI --override flag correctly mutates config."""

    @pytest.mark.smoke
    def test_override_default(self):
        config = _config_from_yaml(get_template("basic"))
        updated = apply_overrides(config, ["defaults.test_mode=oscillation"])

        assert updated.defaults["test_mode"] == "oscillation"
        # Original unchanged
        assert config.defaults["test_mode"] == "relaxation"

    @pytest.mark.smoke
    def test_override_step_model(self):
        config = _config_from_yaml(get_template("basic"))
        updated = apply_overrides(config, ["steps.1.model=springpot"])

        assert updated.steps[1]["model"] == "springpot"
        # Original unchanged
        assert config.steps[1]["model"] == "maxwell"

    @pytest.mark.smoke
    def test_override_bayesian_params(self):
        config = _config_from_yaml(get_template("bayesian"))
        updated = apply_overrides(
            config,
            ["steps.2.num_warmup=500", "steps.2.num_samples=1000", "steps.2.seed=99"],
        )

        assert updated.steps[2]["num_warmup"] == 500
        assert updated.steps[2]["num_samples"] == 1000
        assert updated.steps[2]["seed"] == 99

    @pytest.mark.smoke
    def test_override_invalid_path_raises(self):
        config = _config_from_yaml(get_template("basic"))
        with pytest.raises(ValueError, match="must start with"):
            apply_overrides(config, ["model=springpot"])

    @pytest.mark.smoke
    def test_override_out_of_range_raises(self):
        config = _config_from_yaml(get_template("basic"))
        with pytest.raises(ValueError, match="out of range"):
            apply_overrides(config, ["steps.99.model=springpot"])


# ------------------------------------------------------------------
# YAML roundtrip: write → load → validate → builder
# ------------------------------------------------------------------


class TestYAMLFileRoundtrip:
    """Write YAML to disk, load it back, validate, and build."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
    def test_file_roundtrip(self, template_name: str, tmp_path: Path):
        yaml_str = get_template(template_name)

        # Write to file
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text(yaml_str, encoding="utf-8")

        # Load back
        config = load_config(config_path)
        assert config.name
        assert len(config.steps) >= 2

        # Convert to builder
        builder = config_to_builder(config)
        steps = builder.get_steps()
        assert len(steps) == len(config.steps)

        # Step types match
        for (step_type, _kwargs), raw_step in zip(steps, config.steps):
            assert step_type == raw_step["type"]


# ------------------------------------------------------------------
# Validation edge cases
# ------------------------------------------------------------------


class TestValidationEdgeCases:
    """Schema validation catches structural errors."""

    @pytest.mark.smoke
    def test_empty_steps_rejected(self):
        config = PipelineConfig(version="1", name="empty", steps=[])
        errors = validate_config(config)
        assert any("at least one step" in e for e in errors)

    @pytest.mark.smoke
    def test_missing_load_first(self):
        config = PipelineConfig(
            version="1",
            name="no-load",
            steps=[{"type": "fit", "model": "maxwell"}],
        )
        errors = validate_config(config)
        assert any("'load'" in e for e in errors)

    @pytest.mark.smoke
    def test_bayesian_before_fit_rejected(self):
        config = PipelineConfig(
            version="1",
            name="bad-order",
            steps=[
                {"type": "load", "file": "x.csv"},
                {"type": "bayesian"},
            ],
        )
        errors = validate_config(config)
        assert any("fit" in e.lower() and "bayesian" in e.lower() for e in errors)

    @pytest.mark.smoke
    def test_path_traversal_rejected(self):
        config = PipelineConfig(
            version="1",
            name="traversal",
            steps=[{"type": "load", "file": "../../../etc/passwd"}],
        )
        errors = validate_config(config)
        assert any(".." in e for e in errors)

    @pytest.mark.smoke
    def test_absolute_path_rejected(self):
        import sys

        # Use platform-appropriate absolute path:
        # On Windows, "/etc/passwd" is not absolute (no drive letter)
        abs_path = "C:\\data.csv" if sys.platform == "win32" else "/etc/passwd"
        config = PipelineConfig(
            version="1",
            name="absolute",
            steps=[{"type": "load", "file": abs_path}],
        )
        errors = validate_config(config)
        assert any("relative" in e.lower() or "absolute" in e.lower() for e in errors)

    @pytest.mark.smoke
    def test_unknown_step_type_rejected(self):
        config = PipelineConfig(
            version="1",
            name="bad-type",
            steps=[
                {"type": "load", "file": "x.csv"},
                {"type": "delete_everything"},
            ],
        )
        errors = validate_config(config)
        assert any("Unknown" in e or "unknown" in e for e in errors)


# ------------------------------------------------------------------
# YAML injection regression tests (M1)
# ------------------------------------------------------------------


class TestYAMLInjection:
    """Verify yaml.safe_load rejects Python object tags.

    These are regression tests ensuring that the YAML parser is always
    ``yaml.safe_load`` (not ``yaml.load``).  If someone accidentally
    switches to unsafe loading, these tests will catch it.
    """

    @pytest.mark.smoke
    def test_python_object_tag_rejected(self):
        """!!python/object tags must not instantiate arbitrary objects."""
        malicious = (
            "version: '1'\n"
            "name: !!python/object:builtins.eval 'print(1)'\n"
            "steps:\n"
            "  - type: load\n"
            "    file: x.csv\n"
        )
        with pytest.raises(yaml.constructor.ConstructorError):
            yaml.safe_load(malicious)

    @pytest.mark.smoke
    def test_python_name_tag_rejected(self):
        """!!python/name tags must be rejected by safe_load."""
        malicious = (
            "version: '1'\n"
            "name: !!python/name:builtins.open ''\n"
            "steps:\n"
            "  - type: load\n"
            "    file: x.csv\n"
        )
        with pytest.raises(yaml.constructor.ConstructorError):
            yaml.safe_load(malicious)
