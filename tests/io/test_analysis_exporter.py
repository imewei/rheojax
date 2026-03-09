"""Tests for the AnalysisExporter and Pipeline.export()."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.core.base import BaseModel, BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.io.analysis_exporter import AnalysisExporter
from rheojax.pipeline import Pipeline

jax, jnp = safe_import_jax()


# ---------------------------------------------------------------------------
# Mock model & transform
# ---------------------------------------------------------------------------

class _ExportTestModel(BaseModel):
    """Simple model for export testing."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="amplitude", value=100.0, bounds=(0.1, 1e5))
        self.parameters.add(name="decay", value=0.5, bounds=(0.01, 10))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value("amplitude", float(np.mean(np.abs(y))))
        self.parameters.set_value("decay", 0.5)
        return self

    def _predict(self, X):
        a = self.parameters.get_value("amplitude")
        b = self.parameters.get_value("decay")
        return a * np.exp(-b * X)

    def model_function(self, X, params, test_mode=None, **kwargs):
        return params[0] * jnp.exp(-params[1] * X)


class _ExportTestTransform(BaseTransform):
    """Simple transform that doubles data."""

    def _transform(self, data):
        return RheoData(
            x=data.x,
            y=data.y * 2.0,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata={**(data.metadata or {}), "transform": "export_test"},
        )


@pytest.fixture(scope="module", autouse=True)
def register_mocks():
    """Register mock model and transform, clean up after module."""
    try:
        ModelRegistry.create("export_test_model")
    except (KeyError, ValueError):
        ModelRegistry.register("export_test_model")(_ExportTestModel)
    try:
        TransformRegistry.create("export_test_scale")
    except (KeyError, ValueError):
        TransformRegistry.register("export_test_scale")(_ExportTestTransform)
    yield
    # Clean up to avoid polluting other test modules (e.g. protocol validation)
    try:
        ModelRegistry.unregister("export_test_model")
    except (KeyError, ValueError):
        pass
    try:
        TransformRegistry.unregister("export_test_scale")
    except (KeyError, ValueError):
        pass


@pytest.fixture
def sample_data():
    """Sample time-domain data."""
    np.random.seed(42)
    t = np.linspace(0.1, 10, 50)
    y = 100 * np.exp(-0.5 * t) + np.random.normal(0, 2, 50)
    return RheoData(
        x=t, y=y, x_units="s", y_units="Pa", domain="time",
        metadata={"test_mode": "relaxation"},
    )


@pytest.fixture
def fitted_pipeline(sample_data):
    """Pipeline with fitted model."""
    p = Pipeline(data=sample_data)
    p.fit(_ExportTestModel())
    return p


@pytest.fixture
def full_pipeline(sample_data):
    """Pipeline with transform + fit + plot."""
    p = Pipeline(data=sample_data)
    p.transform(_ExportTestTransform())
    p.fit(_ExportTestModel())
    p.plot_fit(show_residuals=False, show_uncertainty=False)
    return p


# ---------------------------------------------------------------------------
# AnalysisExporter tests
# ---------------------------------------------------------------------------

class TestAnalysisExporter:
    """Tests for AnalysisExporter directly."""

    @pytest.mark.smoke
    def test_export_directory_basic(self, fitted_pipeline, tmp_path):
        """Directory export creates expected structure."""
        exporter = AnalysisExporter()
        out = exporter.export_directory(fitted_pipeline, tmp_path / "export")

        assert out.exists()
        assert (out / "summary.json").exists()
        assert (out / "summary.txt").exists()
        assert (out / "data" / "current_data.h5").exists()
        assert (out / "results" / "fit_result.json").exists()
        assert (out / "results" / "fit_result.npz").exists()

    @pytest.mark.smoke
    def test_export_directory_summary_content(self, fitted_pipeline, tmp_path):
        """Summary JSON has expected structure."""
        exporter = AnalysisExporter()
        out = exporter.export_directory(fitted_pipeline, tmp_path / "export")

        with open(out / "summary.json") as f:
            summary = json.load(f)

        assert "timestamp" in summary
        assert "metadata" in summary
        assert "fit" in summary
        assert summary["fit"]["model_class"] == "_ExportTestModel"
        assert "amplitude" in summary["fit"]["params"]
        assert "decay" in summary["fit"]["params"]
        assert "statistics" in summary["fit"]

    def test_export_directory_with_transforms(self, full_pipeline, tmp_path):
        """Directory export includes transform results."""
        exporter = AnalysisExporter()
        out = exporter.export_directory(full_pipeline, tmp_path / "export")

        transform_dir = out / "results" / "transforms"
        assert transform_dir.exists()
        assert (transform_dir / "index.json").exists()

        with open(transform_dir / "index.json") as f:
            index = json.load(f)
        transform_names = [e["transform_name"] for e in index["transforms"]]
        assert "_ExportTestTransform" in transform_names

    def test_export_directory_with_figures(self, full_pipeline, tmp_path):
        """Directory export saves figures."""
        exporter = AnalysisExporter(figure_formats=("png",))
        out = exporter.export_directory(full_pipeline, tmp_path / "export")

        fig_dir = out / "figures"
        assert fig_dir.exists()
        assert (fig_dir / "last_plot.png").exists()
        plt.close("all")

    def test_export_directory_npz_format(self, fitted_pipeline, tmp_path):
        """Directory export works with npz data format."""
        exporter = AnalysisExporter()
        out = exporter.export_directory(
            fitted_pipeline, tmp_path / "export", data_format="npz"
        )
        assert (out / "data" / "current_data.npz").exists()

    def test_export_directory_no_data(self, tmp_path):
        """Export without data doesn't crash."""
        p = Pipeline()
        exporter = AnalysisExporter()
        out = exporter.export_directory(p, tmp_path / "export")
        assert (out / "summary.json").exists()

    def test_export_directory_no_fit(self, sample_data, tmp_path):
        """Export without fit creates summary but no fit results."""
        p = Pipeline(data=sample_data)
        exporter = AnalysisExporter()
        out = exporter.export_directory(p, tmp_path / "export")

        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert "fit" not in summary

    def test_summary_txt_content(self, fitted_pipeline, tmp_path):
        """Human-readable summary contains key info."""
        exporter = AnalysisExporter()
        out = exporter.export_directory(fitted_pipeline, tmp_path / "export")

        txt = (out / "summary.txt").read_text()
        assert "RheoJAX Analysis Report" in txt
        assert "_ExportTestModel" in txt
        assert "amplitude" in txt
        assert "decay" in txt


class TestAnalysisExporterExcel:
    """Tests for Excel export."""

    @pytest.mark.smoke
    def test_export_excel_basic(self, fitted_pipeline, tmp_path):
        """Excel export creates valid xlsx."""
        exporter = AnalysisExporter()
        out = exporter.export_excel(
            fitted_pipeline, tmp_path / "report.xlsx", include_plots=False
        )
        assert out.exists()
        assert out.suffix == ".xlsx"

    def test_export_excel_has_sheets(self, fitted_pipeline, tmp_path):
        """Excel export has expected sheets."""
        import pandas as pd

        exporter = AnalysisExporter()
        out = exporter.export_excel(
            fitted_pipeline, tmp_path / "report.xlsx", include_plots=False
        )

        with pd.ExcelFile(out) as xls:
            sheet_names = xls.sheet_names
            assert "Summary" in sheet_names
            assert "Parameters" in sheet_names
            assert "Fit Quality" in sheet_names
            assert "Data" in sheet_names

    def test_export_excel_parameters_content(self, fitted_pipeline, tmp_path):
        """Parameters sheet has correct values."""
        import pandas as pd

        exporter = AnalysisExporter()
        out = exporter.export_excel(
            fitted_pipeline, tmp_path / "report.xlsx", include_plots=False
        )

        params_df = pd.read_excel(out, sheet_name="Parameters")
        assert "amplitude" in params_df["Parameter"].values
        assert "decay" in params_df["Parameter"].values

    def test_export_excel_with_transforms(self, full_pipeline, tmp_path):
        """Excel export includes transforms sheet."""
        import pandas as pd

        exporter = AnalysisExporter()
        out = exporter.export_excel(
            full_pipeline, tmp_path / "report.xlsx", include_plots=False
        )

        with pd.ExcelFile(out) as xls:
            assert "Transforms" in xls.sheet_names

    def test_export_excel_no_data(self, tmp_path):
        """Excel export without data still creates file."""
        p = Pipeline()
        exporter = AnalysisExporter()
        out = exporter.export_excel(p, tmp_path / "empty.xlsx", include_plots=False)
        assert out.exists()


# ---------------------------------------------------------------------------
# Pipeline.export() integration tests
# ---------------------------------------------------------------------------

class TestPipelineExport:
    """Tests for Pipeline.export() method."""

    @pytest.mark.smoke
    def test_export_directory_via_pipeline(self, full_pipeline, tmp_path):
        """Pipeline.export() creates directory export."""
        result = full_pipeline.export(tmp_path / "analysis")
        assert result is full_pipeline  # Returns self
        assert (tmp_path / "analysis" / "summary.json").exists()
        plt.close("all")

    @pytest.mark.smoke
    def test_export_excel_via_pipeline(self, fitted_pipeline, tmp_path):
        """Pipeline.export() with .xlsx creates Excel."""
        result = fitted_pipeline.export(tmp_path / "report.xlsx")
        assert result is fitted_pipeline
        assert (tmp_path / "report.xlsx").exists()

    def test_export_auto_format_directory(self, fitted_pipeline, tmp_path):
        """Auto format detection for directory path."""
        fitted_pipeline.export(tmp_path / "output_dir")
        assert (tmp_path / "output_dir" / "summary.json").exists()

    def test_export_auto_format_excel(self, fitted_pipeline, tmp_path):
        """Auto format detection for .xlsx path."""
        fitted_pipeline.export(tmp_path / "output.xlsx")
        assert (tmp_path / "output.xlsx").exists()

    def test_export_records_history(self, fitted_pipeline, tmp_path):
        """Export records operation in pipeline history."""
        fitted_pipeline.export(tmp_path / "analysis")
        ops = [h[0] for h in fitted_pipeline.history]
        assert "export" in ops

    def test_export_chaining(self, full_pipeline, tmp_path):
        """Export supports fluent chaining."""
        result = (
            full_pipeline
            .export(tmp_path / "analysis")
        )
        assert result is full_pipeline
        plt.close("all")

    def test_export_invalid_format_raises(self, fitted_pipeline, tmp_path):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown export format"):
            fitted_pipeline.export(tmp_path / "out", format="pdf")


# ---------------------------------------------------------------------------
# Full workflow test
# ---------------------------------------------------------------------------

class TestExportFullWorkflow:
    """End-to-end: load → transform → fit → plot → export."""

    @pytest.mark.smoke
    def test_complete_workflow(self, sample_data, tmp_path):
        """Full pipeline with export at the end."""
        result = (
            Pipeline(data=sample_data)
            .transform(_ExportTestTransform())
            .fit(_ExportTestModel())
            .plot_fit(show_residuals=False, show_uncertainty=False)
            .export(tmp_path / "full_analysis")
        )

        out = tmp_path / "full_analysis"
        assert (out / "summary.json").exists()
        assert (out / "data" / "current_data.h5").exists()
        assert (out / "results" / "fit_result.json").exists()
        assert (out / "results" / "transforms" / "index.json").exists()
        assert (out / "figures" / "last_plot.pdf").exists()

        # Verify summary content
        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert "transforms" in summary
        assert "_ExportTestTransform" in summary["transforms"]
        assert summary["fit"]["model_class"] == "_ExportTestModel"

        # Verify history
        ops = [h[0] for h in result.history]
        assert ops == ["transform", "fit", "plot_fit", "export"]

        plt.close("all")
