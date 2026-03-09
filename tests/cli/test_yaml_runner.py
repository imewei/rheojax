"""Tests for rheojax.cli._yaml_runner — config-to-builder translation and overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rheojax.cli._yaml_runner import (
    _coerce_value,
    apply_overrides,
    config_to_builder,
    dry_run_pipeline,
)
from rheojax.cli._yaml_schema import PipelineConfig


def _make_config(**kwargs) -> PipelineConfig:
    defaults = dict(
        version="1",
        name="Test",
        defaults={},
        steps=[
            {"type": "load", "file": "data.csv"},
            {"type": "fit", "model": "maxwell"},
        ],
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


class TestConfigToBuilder:
    @pytest.mark.smoke
    def test_returns_pipeline_builder(self):
        from rheojax.pipeline.builder import PipelineBuilder

        config = _make_config()
        builder = config_to_builder(config)
        assert isinstance(builder, PipelineBuilder)

    @pytest.mark.unit
    def test_load_step_added(self):
        config = _make_config()
        builder = config_to_builder(config)
        step_types = [s[0] for s in builder.steps]
        assert "load" in step_types

    @pytest.mark.unit
    def test_fit_step_added(self):
        config = _make_config()
        builder = config_to_builder(config)
        step_types = [s[0] for s in builder.steps]
        assert "fit" in step_types

    @pytest.mark.unit
    def test_export_step_added(self):
        config = PipelineConfig(
            version="1",
            name="Test",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "fit", "model": "maxwell"},
                {"type": "export", "output": "results/"},
            ],
        )
        builder = config_to_builder(config)
        step_types = [s[0] for s in builder.steps]
        assert "export" in step_types

    @pytest.mark.unit
    def test_defaults_merged_into_steps(self):
        config = PipelineConfig(
            version="1",
            name="Test",
            defaults={"test_mode": "relaxation"},
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "fit", "model": "maxwell"},
            ],
        )
        builder = config_to_builder(config)
        # The fit step kwargs should have test_mode from defaults
        fit_step = next(s for s in builder.steps if s[0] == "fit")
        assert fit_step[1].get("test_mode") == "relaxation"

    @pytest.mark.unit
    def test_unknown_step_type_raises(self):
        config = PipelineConfig(
            version="1",
            name="Test",
            steps=[
                {"type": "load", "file": "data.csv"},
                {"type": "unknown_step"},
            ],
        )
        with pytest.raises(ValueError, match="Unrecognised"):
            config_to_builder(config)


class TestApplyOverrides:
    @pytest.mark.smoke
    def test_override_defaults_test_mode(self):
        config = _make_config(defaults={"test_mode": "relaxation"})
        result = apply_overrides(config, ["defaults.test_mode=oscillation"])
        assert result.defaults["test_mode"] == "oscillation"

    @pytest.mark.unit
    def test_override_step_model(self):
        config = _make_config()
        result = apply_overrides(config, ["steps.1.model=zener"])
        assert result.steps[1]["model"] == "zener"

    @pytest.mark.unit
    def test_override_does_not_mutate_original(self):
        config = _make_config(defaults={"test_mode": "relaxation"})
        apply_overrides(config, ["defaults.test_mode=oscillation"])
        assert config.defaults.get("test_mode") == "relaxation"

    @pytest.mark.unit
    def test_empty_overrides_returns_equivalent_config(self):
        config = _make_config()
        result = apply_overrides(config, [])
        assert result.name == config.name
        assert result.version == config.version

    @pytest.mark.unit
    def test_malformed_override_raises(self):
        config = _make_config()
        with pytest.raises(ValueError):
            apply_overrides(config, ["no_equals_sign"])

    @pytest.mark.unit
    def test_out_of_range_step_index_raises(self):
        config = _make_config()
        with pytest.raises(ValueError):
            apply_overrides(config, ["steps.99.model=maxwell"])


class TestCoerceValue:
    @pytest.mark.unit
    def test_coerce_int(self):
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    @pytest.mark.unit
    def test_coerce_float(self):
        assert _coerce_value("3.14") == pytest.approx(3.14)
        assert isinstance(_coerce_value("3.14"), float)

    @pytest.mark.unit
    def test_coerce_true(self):
        assert _coerce_value("true") is True
        assert _coerce_value("True") is True

    @pytest.mark.unit
    def test_coerce_false(self):
        assert _coerce_value("false") is False

    @pytest.mark.unit
    def test_coerce_string_fallback(self):
        assert _coerce_value("maxwell") == "maxwell"

    @pytest.mark.unit
    def test_coerce_null(self):
        assert _coerce_value("null") is None


class TestDryRunPipeline:
    @pytest.mark.unit
    def test_dry_run_does_not_raise(self):
        config = _make_config()
        dry_run_pipeline(config)  # Must not raise

    @pytest.mark.unit
    def test_dry_run_with_defaults(self):
        config = _make_config(defaults={"test_mode": "relaxation"})
        dry_run_pipeline(config)  # Must not raise
