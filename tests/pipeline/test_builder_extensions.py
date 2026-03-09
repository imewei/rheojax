"""Tests for PipelineBuilder extension steps: bayesian and export."""

from __future__ import annotations

import pytest

from rheojax.pipeline.builder import PipelineBuilder


class TestAddBayesianStep:
    @pytest.mark.smoke
    def test_add_bayesian_step_appends_step(self):
        builder = PipelineBuilder()
        builder.add_bayesian_step()
        step_types = [s[0] for s in builder.steps]
        assert "bayesian" in step_types

    @pytest.mark.unit
    def test_add_bayesian_step_returns_builder_for_chaining(self):
        builder = PipelineBuilder()
        result = builder.add_bayesian_step()
        assert result is builder

    @pytest.mark.unit
    def test_add_bayesian_step_default_values(self):
        builder = PipelineBuilder()
        builder.add_bayesian_step()
        _, kwargs = next(s for s in builder.steps if s[0] == "bayesian")
        assert kwargs["num_warmup"] == 1000
        assert kwargs["num_samples"] == 2000
        assert kwargs["num_chains"] == 4
        assert kwargs["seed"] == 0
        assert kwargs["warm_start"] is True

    @pytest.mark.unit
    def test_add_bayesian_step_custom_values(self):
        builder = PipelineBuilder()
        builder.add_bayesian_step(num_warmup=500, num_samples=1000, num_chains=2, seed=42)
        _, kwargs = next(s for s in builder.steps if s[0] == "bayesian")
        assert kwargs["num_warmup"] == 500
        assert kwargs["num_samples"] == 1000
        assert kwargs["num_chains"] == 2
        assert kwargs["seed"] == 42


class TestAddExportStep:
    @pytest.mark.smoke
    def test_add_export_step_appends_step(self):
        builder = PipelineBuilder()
        builder.add_export_step("results/")
        step_types = [s[0] for s in builder.steps]
        assert "export" in step_types

    @pytest.mark.unit
    def test_add_export_step_returns_builder_for_chaining(self):
        builder = PipelineBuilder()
        result = builder.add_export_step("results/")
        assert result is builder

    @pytest.mark.unit
    def test_add_export_step_stores_output_path(self):
        builder = PipelineBuilder()
        builder.add_export_step("/tmp/output")
        _, kwargs = next(s for s in builder.steps if s[0] == "export")
        assert kwargs["output_path"] == "/tmp/output"

    @pytest.mark.unit
    def test_add_export_step_default_format(self):
        builder = PipelineBuilder()
        builder.add_export_step("results/")
        _, kwargs = next(s for s in builder.steps if s[0] == "export")
        assert kwargs["format"] == "directory"

    @pytest.mark.unit
    def test_add_export_step_custom_format(self):
        builder = PipelineBuilder()
        builder.add_export_step("bundle.h5", format="hdf5")
        _, kwargs = next(s for s in builder.steps if s[0] == "export")
        assert kwargs["format"] == "hdf5"


class TestBuilderValidation:
    @pytest.mark.smoke
    def test_bayesian_before_fit_raises_value_error(self):
        builder = PipelineBuilder()
        builder.add_load_step("data.csv")
        builder.add_bayesian_step()
        with pytest.raises(ValueError, match="fit"):
            builder.build(validate=True)

    @pytest.mark.unit
    def test_export_without_load_raises_value_error(self):
        builder = PipelineBuilder()
        builder.add_export_step("results/")
        with pytest.raises(ValueError):
            builder.build(validate=True)

    @pytest.mark.unit
    def test_empty_pipeline_raises_value_error(self):
        builder = PipelineBuilder()
        with pytest.raises(ValueError):
            builder.build(validate=True)

    @pytest.mark.unit
    def test_first_step_not_load_raises_value_error(self):
        builder = PipelineBuilder()
        builder.steps.append(("fit", {"model": "maxwell"}))
        with pytest.raises(ValueError, match="load"):
            builder.build(validate=True)


class TestBuilderChaining:
    @pytest.mark.smoke
    def test_load_fit_bayesian_export_chain_validates(self):
        builder = PipelineBuilder()
        (
            builder.add_load_step("data.csv")
            .add_fit_step("maxwell")
            .add_bayesian_step()
            .add_export_step("results/")
        )
        step_types = [s[0] for s in builder.steps]
        assert step_types == ["load", "fit", "bayesian", "export"]

    @pytest.mark.unit
    def test_step_count_after_chaining(self):
        builder = PipelineBuilder()
        (
            builder.add_load_step("data.csv")
            .add_fit_step("maxwell")
            .add_bayesian_step()
        )
        assert len(builder) == 3
