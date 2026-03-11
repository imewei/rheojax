"""Tests for GUI pipeline serializer YAML round-trip.

Covers to_yaml, from_yaml, to_pipeline_builder, and from_pipeline_builder.
These tests do not require a Qt event loop or StateStore — they operate
entirely on plain Python dataclasses and YAML strings.
"""

import pytest

from rheojax.gui.state.store import StepStatus
from rheojax.gui.utils.pipeline_serializer import (
    PipelineStepConfig,
    VisualPipelineState,
    from_pipeline_builder,
    from_yaml,
    to_pipeline_builder,
    to_yaml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_type: str,
    config: dict,
    step_id: str = "test-id-001",
    name: str | None = None,
    position: int = 0,
) -> PipelineStepConfig:
    """Build a PipelineStepConfig for use in serializer tests."""
    if name is None:
        name = step_type.capitalize()
    return PipelineStepConfig(
        id=step_id,
        step_type=step_type,
        name=name,
        config=config,
        status=StepStatus.PENDING,
        position=position,
    )


def _make_load_step(**kwargs) -> PipelineStepConfig:
    return _make_step("load", {"file": "data/sample.csv", "format": "csv"}, **kwargs)


def _make_fit_step(**kwargs) -> PipelineStepConfig:
    return _make_step("fit", {"model": "Maxwell", "max_iter": 1000}, **kwargs)


def _make_transform_step(**kwargs) -> PipelineStepConfig:
    return _make_step("transform", {"name": "fft"}, **kwargs)


def _make_bayesian_step(**kwargs) -> PipelineStepConfig:
    return _make_step(
        "bayesian",
        {"num_warmup": 500, "num_samples": 1000, "num_chains": 2},
        **kwargs,
    )


def _make_export_step(**kwargs) -> PipelineStepConfig:
    return _make_step("export", {"output": "results/", "format": "csv"}, **kwargs)


def _basic_steps() -> list[PipelineStepConfig]:
    """Return a minimal load + fit pipeline for round-trip tests."""
    return [
        _make_load_step(step_id="id-load", position=0),
        _make_fit_step(step_id="id-fit", name="Fit Maxwell", position=1),
    ]


# ---------------------------------------------------------------------------
# TestToYaml
# ---------------------------------------------------------------------------


class TestToYaml:
    @pytest.mark.smoke
    def test_basic_pipeline_to_yaml(self):
        """to_yaml must produce a non-empty YAML string for a simple pipeline."""
        steps = _basic_steps()
        result = to_yaml(steps, pipeline_name="Test Pipeline")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.smoke
    def test_yaml_has_version_and_name(self):
        """to_yaml output must contain version '1' and the pipeline name."""
        import yaml

        steps = _basic_steps()
        yaml_str = to_yaml(steps, pipeline_name="My Pipeline")
        doc = yaml.safe_load(yaml_str)
        assert doc["version"] == "1"
        assert doc["name"] == "My Pipeline"

    def test_yaml_has_correct_step_types(self):
        """Each step in the YAML steps list must carry the correct 'type' key."""
        import yaml

        steps = [
            _make_load_step(position=0),
            _make_transform_step(position=1),
            _make_fit_step(position=2),
        ]
        yaml_str = to_yaml(steps)
        doc = yaml.safe_load(yaml_str)
        types = [s["type"] for s in doc["steps"]]
        assert types == ["load", "transform", "fit"]

    def test_yaml_includes_config_values(self):
        """Config keys from each step must appear as top-level keys in the YAML step."""
        import yaml

        step = _make_fit_step()
        yaml_str = to_yaml([step])
        doc = yaml.safe_load(yaml_str)
        yaml_step = doc["steps"][0]
        assert yaml_step["model"] == "Maxwell"
        assert yaml_step["max_iter"] == 1000

    def test_yaml_empty_pipeline(self):
        """to_yaml with no steps must produce valid YAML with an empty steps list."""
        import yaml

        yaml_str = to_yaml([], pipeline_name="Empty")
        doc = yaml.safe_load(yaml_str)
        assert doc["steps"] == []
        assert doc["name"] == "Empty"

    def test_yaml_defaults_block_included_when_provided(self):
        """to_yaml must include a 'defaults' block when the defaults arg is given."""
        import yaml

        steps = _basic_steps()
        defaults = {"seed": 42, "log_level": "INFO"}
        yaml_str = to_yaml(steps, defaults=defaults)
        doc = yaml.safe_load(yaml_str)
        assert "defaults" in doc
        assert doc["defaults"]["seed"] == 42

    def test_yaml_no_defaults_block_when_empty(self):
        """to_yaml must omit the 'defaults' block when defaults is None or empty."""
        import yaml

        yaml_str = to_yaml(_basic_steps(), defaults=None)
        doc = yaml.safe_load(yaml_str)
        assert "defaults" not in doc

    def test_yaml_export_step_uses_output_key(self):
        """Export step config key 'output' must pass through to the YAML unchanged."""
        import yaml

        step = _make_export_step()
        yaml_str = to_yaml([step])
        doc = yaml.safe_load(yaml_str)
        assert doc["steps"][0]["output"] == "results/"

    def test_yaml_bayesian_step_serialized(self):
        """to_yaml must correctly serialize a bayesian step with all sampling kwargs."""
        import yaml

        step = _make_bayesian_step()
        yaml_str = to_yaml([step])
        doc = yaml.safe_load(yaml_str)
        bayes = doc["steps"][0]
        assert bayes["type"] == "bayesian"
        assert bayes["num_warmup"] == 500
        assert bayes["num_samples"] == 1000
        assert bayes["num_chains"] == 2


# ---------------------------------------------------------------------------
# TestFromYaml
# ---------------------------------------------------------------------------


class TestFromYaml:
    @pytest.mark.smoke
    def test_basic_yaml_to_steps(self):
        """from_yaml must parse a minimal pipeline YAML into a step list."""
        yaml_str = (
            "version: '1'\n"
            "name: Test\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/test.csv\n"
        )
        steps, name = from_yaml(yaml_str)
        assert len(steps) == 1
        assert name == "Test"

    def test_step_types_preserved(self):
        """from_yaml must preserve step_type for each step in the list."""
        yaml_str = (
            "version: '1'\n"
            "name: Pipeline\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/x.csv\n"
            "  - type: fit\n"
            "    model: Maxwell\n"
            "  - type: export\n"
            "    output: out/\n"
        )
        steps, _ = from_yaml(yaml_str)
        assert [s.step_type for s in steps] == ["load", "fit", "export"]

    def test_config_values_preserved(self):
        """from_yaml must put all non-'type' YAML keys into the config dict."""
        yaml_str = (
            "version: '1'\n"
            "name: Cfg\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/input.csv\n"
            "  - type: fit\n"
            "    model: Giesekus\n"
            "    max_iter: 5000\n"
        )
        steps, _ = from_yaml(yaml_str)
        cfg = steps[1].config
        assert cfg["model"] == "Giesekus"
        assert cfg["max_iter"] == 5000

    def test_display_names_generated(self):
        """from_yaml must generate sensible display names (e.g., 'Fit: Maxwell')."""
        yaml_str = (
            "version: '1'\n"
            "name: N\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/input.csv\n"
            "  - type: fit\n"
            "    model: Maxwell\n"
        )
        steps, _ = from_yaml(yaml_str)
        assert "Maxwell" in steps[1].name

    def test_positions_assigned_in_order(self):
        """from_yaml must assign zero-based positions matching list order."""
        yaml_str = (
            "version: '1'\n"
            "name: P\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/x.csv\n"
            "  - type: fit\n"
            "    model: Maxwell\n"
        )
        steps, _ = from_yaml(yaml_str)
        assert steps[0].position == 0
        assert steps[1].position == 1

    def test_steps_have_pending_status(self):
        """All steps from from_yaml must start with PENDING status."""
        yaml_str = (
            "version: '1'\n"
            "name: S\n"
            "steps:\n"
            "  - type: load\n"
            "    file: data/f.csv\n"
        )
        steps, _ = from_yaml(yaml_str)
        assert steps[0].status == StepStatus.PENDING

    def test_missing_version_raises(self):
        """from_yaml must raise ValueError when 'version' is missing."""
        yaml_str = "name: Bad\nsteps:\n  - type: load\n    file: data/x.csv\n"
        with pytest.raises(ValueError, match="version"):
            from_yaml(yaml_str)

    def test_missing_type_raises(self):
        """from_yaml must raise ValueError when a step is missing 'type'."""
        yaml_str = (
            "version: '1'\n"
            "name: Bad\n"
            "steps:\n"
            "  - file: data/x.csv\n"
        )
        with pytest.raises(ValueError, match="type"):
            from_yaml(yaml_str)

    def test_absolute_path_rejected(self):
        """from_yaml must reject absolute file paths in strict mode."""
        yaml_str = (
            "version: '1'\n"
            "name: Bad\n"
            "steps:\n"
            "  - type: load\n"
            "    file: /etc/passwd\n"
        )
        with pytest.raises(ValueError, match="absolute"):
            from_yaml(yaml_str, strict=True)

    def test_absolute_path_warned_in_non_strict_mode(self):
        """from_yaml warns but accepts absolute paths in non-strict (GUI) mode."""
        yaml_str = (
            "version: '1'\n"
            "name: GUI draft\n"
            "steps:\n"
            "  - type: load\n"
            "    file: /Users/data/sample.csv\n"
        )
        # Should not raise — GUI file picker returns absolute paths.
        steps, name = from_yaml(yaml_str, strict=False)
        assert len(steps) == 1
        assert steps[0].config["file"] == "/Users/data/sample.csv"

    def test_path_traversal_rejected(self):
        """from_yaml must reject path traversal via schema validation."""
        yaml_str = (
            "version: '1'\n"
            "name: Bad\n"
            "steps:\n"
            "  - type: load\n"
            "    file: ../../etc/passwd\n"
        )
        with pytest.raises(ValueError, match="\\.\\."):
            from_yaml(yaml_str)


# ---------------------------------------------------------------------------
# TestYamlRoundtrip
# ---------------------------------------------------------------------------


class TestYamlRoundtrip:
    @pytest.mark.smoke
    def test_roundtrip_preserves_structure(self):
        """Steps serialized via to_yaml then parsed via from_yaml must have the same
        step_types in the same order."""
        original = [
            _make_load_step(position=0),
            _make_transform_step(position=1),
            _make_fit_step(position=2),
            _make_bayesian_step(position=3),
            _make_export_step(position=4),
        ]
        yaml_str = to_yaml(original, pipeline_name="RT")
        recovered, name = from_yaml(yaml_str)
        assert name == "RT"
        assert len(recovered) == len(original)
        for orig, rec in zip(original, recovered):
            assert rec.step_type == orig.step_type

    def test_roundtrip_preserves_config(self):
        """Config values must survive the to_yaml -> from_yaml round-trip intact."""
        original = [
            _make_load_step(position=0),
            _make_fit_step(position=1),
        ]
        yaml_str = to_yaml(original)
        recovered, _ = from_yaml(yaml_str)
        assert recovered[1].config["model"] == "Maxwell"
        assert recovered[1].config["max_iter"] == 1000

    def test_roundtrip_assigns_fresh_uuids(self):
        """from_yaml must generate new UUIDs (not reuse the original step IDs)."""
        original = [_make_load_step(step_id="fixed-id")]
        yaml_str = to_yaml(original)
        recovered, _ = from_yaml(yaml_str)
        assert recovered[0].id != "fixed-id"

    def test_roundtrip_pipeline_name(self):
        """The pipeline name must survive the to_yaml -> from_yaml round-trip."""
        yaml_str = to_yaml(_basic_steps(), pipeline_name="Preserved Name")
        _, name = from_yaml(yaml_str)
        assert name == "Preserved Name"


# ---------------------------------------------------------------------------
# TestToPipelineBuilder
# ---------------------------------------------------------------------------


class TestToPipelineBuilder:
    @pytest.mark.smoke
    def test_builder_created_with_correct_steps(self):
        """to_pipeline_builder must produce a builder whose steps match the input."""
        steps = [
            _make_load_step(position=0),
            _make_fit_step(position=1),
        ]
        builder = to_pipeline_builder(steps)
        # The builder stores (step_type, kwargs) tuples.
        step_types = [t for t, _ in builder.steps]
        assert step_types == ["load", "fit"]

    @pytest.mark.smoke
    def test_builder_step_count_matches(self):
        """to_pipeline_builder must produce a builder with the same step count."""
        steps = [
            _make_load_step(position=0),
            _make_transform_step(position=1),
            _make_fit_step(position=2),
            _make_bayesian_step(position=3),
            _make_export_step(position=4),
        ]
        builder = to_pipeline_builder(steps)
        assert len(builder.steps) == 5

    def test_builder_fit_step_has_model_name(self):
        """The fit step in the builder must carry the correct model name."""
        steps = [_make_load_step(position=0), _make_fit_step(position=1)]
        builder = to_pipeline_builder(steps)
        fit_kwargs = next(kw for t, kw in builder.steps if t == "fit")
        assert fit_kwargs["model"] == "Maxwell"

    def test_unknown_step_type_raises(self):
        """to_pipeline_builder must raise ValueError for an unrecognised step type."""
        bad_step = _make_step("unknown_type", {})
        with pytest.raises(ValueError, match="Unrecognised"):
            to_pipeline_builder([bad_step])

    def test_load_step_file_path_key(self):
        """to_pipeline_builder must pass the file path via file_path (not file)."""
        steps = [_make_load_step(position=0)]
        builder = to_pipeline_builder(steps)
        load_kwargs = next(kw for t, kw in builder.steps if t == "load")
        assert "file_path" in load_kwargs
        assert load_kwargs["file_path"] == "data/sample.csv"

    def test_export_step_output_path_key(self):
        """to_pipeline_builder must remap GUI 'output' key to builder 'output_path'."""
        steps = [_make_load_step(position=0), _make_export_step(position=1)]
        builder = to_pipeline_builder(steps)
        export_kwargs = next(kw for t, kw in builder.steps if t == "export")
        assert "output_path" in export_kwargs
        assert export_kwargs["output_path"] == "results/"


# ---------------------------------------------------------------------------
# TestFromPipelineBuilder
# ---------------------------------------------------------------------------


class TestFromPipelineBuilder:
    @pytest.mark.smoke
    def test_builder_to_visual_state(self):
        """from_pipeline_builder must convert builder steps to a VisualPipelineState."""
        from rheojax.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder()
        builder.add_load_step(file_path="data/t.csv", format="csv")
        builder.add_fit_step("Maxwell")
        state = from_pipeline_builder(builder, pipeline_name="From Builder")
        assert isinstance(state, VisualPipelineState)
        assert len(state.steps) == 2
        assert state.pipeline_name == "From Builder"

    @pytest.mark.smoke
    def test_display_names_from_builder(self):
        """from_pipeline_builder must generate display names for fit steps."""
        from rheojax.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder()
        builder.add_load_step(file_path="data/t.csv")
        builder.add_fit_step("Giesekus")
        state = from_pipeline_builder(builder)
        fit_step = next(s for s in state.steps if s.step_type == "fit")
        assert "Giesekus" in fit_step.name

    def test_load_step_file_key_remapped(self):
        """from_pipeline_builder must translate 'file_path' -> 'file' for load steps."""
        from rheojax.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder()
        builder.add_load_step(file_path="data/input.csv")
        state = from_pipeline_builder(builder)
        load_step = state.steps[0]
        assert "file" in load_step.config
        assert load_step.config["file"] == "data/input.csv"

    def test_export_step_output_key_remapped(self):
        """from_pipeline_builder must translate 'output_path' -> 'output' for export."""
        from rheojax.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder()
        builder.add_load_step(file_path="data/input.csv")
        builder.add_fit_step("Maxwell")
        builder.add_export_step("out/", format="csv")
        state = from_pipeline_builder(builder)
        export_step = next(s for s in state.steps if s.step_type == "export")
        assert "output" in export_step.config
        assert export_step.config["output"] == "out/"

    def test_step_positions_assigned_sequentially(self):
        """from_pipeline_builder must assign zero-based positions."""
        from rheojax.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder()
        builder.add_load_step(file_path="data/t.csv")
        builder.add_fit_step("Maxwell")
        builder.add_export_step("out/")
        state = from_pipeline_builder(builder)
        for idx, step in enumerate(state.steps):
            assert step.position == idx

    def test_roundtrip_builder_preserves_step_types(self):
        """GUI steps -> builder -> GUI state must preserve all step types."""
        original_steps = [
            _make_load_step(position=0),
            _make_fit_step(position=1),
            _make_export_step(position=2),
        ]
        builder = to_pipeline_builder(original_steps)
        recovered_state = from_pipeline_builder(builder, pipeline_name="BRT")
        recovered_types = [s.step_type for s in recovered_state.steps]
        assert recovered_types == ["load", "fit", "export"]
