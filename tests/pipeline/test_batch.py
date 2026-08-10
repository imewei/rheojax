"""Tests for BatchPipeline.

This module tests batch processing of multiple datasets with the same pipeline.
"""

import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry
from rheojax.pipeline import BatchPipeline, Pipeline


# Mock model for testing
class BatchTestModel(BaseModel):
    """Simple mock model for batch tests."""

    def __init__(self):
        super().__init__()
        self.parameters.add(name="a", value=1.0, bounds=(0, 100))

    def _fit(self, X, y, **kwargs):
        self.parameters.set_value("a", float(np.mean(y)))
        return self

    def _predict(self, X):
        a = self.parameters.get_value("a")
        return a * np.ones_like(X)


@pytest.fixture(scope="module", autouse=True)
def register_batch_model():
    """Register batch test model."""
    ModelRegistry.register("batch_test_model")(BatchTestModel)
    yield
    ModelRegistry.unregister("batch_test_model")


@pytest.fixture
def temp_csv_files():
    """Create multiple temporary CSV files."""
    files = []

    for i in range(3):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y\n")
            for j in range(10):
                f.write(f"{j},{j * (i + 1)}\n")
            files.append(f.name)

    yield files

    for f in files:
        if os.path.exists(f):
            os.unlink(f)


@pytest.fixture
def temp_directory_with_files():
    """Create temporary directory with CSV files."""
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()

    for i in range(3):
        file_path = os.path.join(temp_dir, f"data_{i}.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("x,y\n")
            for j in range(10):
                f.write(f"{j},{j * (i + 1)}\n")

    yield temp_dir

    shutil.rmtree(temp_dir)


class TestBatchPipelineInitialization:
    """Test batch pipeline initialization."""

    def test_init_empty(self):
        """Test initialization without template."""
        batch = BatchPipeline()
        assert batch.template_pipeline is None  # nosec B101
        assert len(batch.results) == 0  # nosec B101
        assert len(batch.errors) == 0  # nosec B101

    def test_init_with_template(self):
        """Test initialization with template pipeline."""
        template = Pipeline()
        batch = BatchPipeline(template)

        assert batch.template_pipeline is template  # nosec B101

    def test_set_template(self):
        """Test setting template after initialization."""
        batch = BatchPipeline()
        template = Pipeline()

        batch.set_template(template)
        assert batch.template_pipeline is template  # nosec B101


class TestBatchProcessing:
    """Test batch file processing."""

    def test_process_files(self, temp_csv_files):
        """Test processing multiple files."""
        # Create template pipeline (no need to fit, BatchPipeline will apply template to each file)
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        assert len(batch.errors) == 0  # nosec B101
        assert len(batch.results) == len(temp_csv_files)  # nosec B101

    def test_process_files_without_template(self, temp_csv_files):
        """Test that processing without template raises error."""
        batch = BatchPipeline()

        with pytest.raises(ValueError, match="No template pipeline"):
            batch.process_files(temp_csv_files)

    def test_process_directory(self, temp_directory_with_files):
        """Test processing directory of files."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_directory(
            temp_directory_with_files,
            pattern="*.csv",
            format="csv",
            x_col="x",
            y_col="y",
        )

        assert len(batch.errors) == 0  # nosec B101
        total_processed = len(batch.results) + len(batch.errors)
        assert total_processed == 3  # nosec B101

    def test_process_directory_not_found(self):
        """Test processing non-existent directory."""
        batch = BatchPipeline(Pipeline())

        with pytest.raises(FileNotFoundError):
            batch.process_directory("/nonexistent/directory")

    def test_process_directory_no_matches(self, temp_directory_with_files):
        """Test processing directory with no matching files."""
        batch = BatchPipeline(Pipeline())

        with pytest.warns(UserWarning, match="No files matching"):
            batch.process_directory(
                temp_directory_with_files,
                pattern="*.xyz",  # No files with this extension
            )


class TestBatchResults:
    """Test batch result handling."""

    def test_get_results(self, temp_csv_files):
        """Test getting results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        results = batch.get_results()
        assert isinstance(results, list)  # nosec B101

    def test_get_results_copy(self, temp_csv_files):
        """Test that get_results returns a copy."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files[:1], format="csv", x_col="x", y_col="y")

        results1 = batch.get_results()
        results2 = batch.get_results()

        assert results1 is not results2  # nosec B101

    def test_get_errors(self, temp_csv_files):
        """Test getting errors."""
        template = Pipeline()
        batch = BatchPipeline(template)

        # Add a non-existent file to trigger error
        batch.process_files(
            temp_csv_files + ["/nonexistent.csv"], format="csv", x_col="x", y_col="y"
        )

        errors = batch.get_errors()
        assert isinstance(errors, list)  # nosec B101

    def test_get_summary_dataframe(self, temp_csv_files):
        """Test getting summary DataFrame."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        df = batch.get_summary_dataframe()
        assert df is not None  # nosec B101
        # DataFrame might be empty if no results processed


class TestBatchStatistics:
    """Test batch statistics."""

    def test_get_statistics_empty(self):
        """Test statistics with no results."""
        batch = BatchPipeline()
        stats = batch.get_statistics()

        assert isinstance(stats, dict)  # nosec B101
        assert len(stats) == 0  # nosec B101

    def test_get_statistics_with_results(self, temp_csv_files):
        """Test statistics with results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        stats = batch.get_statistics()
        assert isinstance(stats, dict)  # nosec B101

    def test_get_statistics_filters_non_finite_values(self):
        """One divergent fit (NaN/inf r_squared or rmse) must not poison
        the aggregate stats for the whole batch."""
        batch = BatchPipeline()
        data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
        batch.results = [
            (Path("a.csv"), data, {"r_squared": 0.9, "rmse": 1.0}),
            (Path("b.csv"), data, {"r_squared": 0.8, "rmse": 2.0}),
            (Path("c.csv"), data, {"r_squared": float("nan"), "rmse": 1.5}),
            (Path("d.csv"), data, {"r_squared": 0.7, "rmse": float("inf")}),
        ]

        stats = batch.get_statistics()

        assert np.isfinite(stats["mean_r_squared"])  # nosec B101
        assert np.isfinite(stats["mean_rmse"])  # nosec B101
        assert stats["mean_r_squared"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)  # nosec B101
        assert stats["mean_rmse"] == pytest.approx((1.0 + 2.0 + 1.5) / 3)  # nosec B101

    def test_export_path_distinct_for_same_stem_different_dirs(
        self, tmp_path, monkeypatch
    ):
        """Two files with the same stem but different parent directories
        must get distinct export subdirectories -- the actual collision
        scenario the hash-suffix naming fix targets, not just its shape."""
        export_calls = []

        def _record_export(self, output_path, format="auto", **kwargs):
            export_calls.append(Path(output_path))
            return self

        monkeypatch.setattr(Pipeline, "export", _record_export)

        data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
        template = Pipeline()
        template.steps = [
            ("export", {"output_path": str(tmp_path / "exports"), "format": "json"}),
        ]
        batch = BatchPipeline(template)

        batch._process_file(Path("dir_a") / "input.csv", preloaded_data=data)
        batch._process_file(Path("dir_b") / "input.csv", preloaded_data=data)

        assert len(export_calls) == 2  # nosec B101
        assert export_calls[0] != export_calls[1], (  # nosec B101
            f"same-stem files from different directories collided: {export_calls}"
        )

    def test_length(self, temp_csv_files):
        """Test batch length."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        assert len(batch) == len(batch.results)  # nosec B101


class TestBatchFiltering:
    """Test batch result filtering."""

    def test_apply_filter(self, temp_csv_files):
        """Test applying filter to results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        initial_count = len(batch.results)

        # Filter to keep only results with certain criteria
        batch.apply_filter(lambda p, d, m: True)  # Keep all

        assert len(batch.results) == initial_count  # nosec B101

    def test_filter_removes_results(self, temp_csv_files):
        """Test that filter can remove results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        # Filter to remove all
        batch.apply_filter(lambda p, d, m: False)

        assert len(batch.results) == 0  # nosec B101


class TestBatchUtilities:
    """Test batch utility methods."""

    def test_clear(self, temp_csv_files):
        """Test clearing results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        batch.clear()

        assert len(batch.results) == 0  # nosec B101
        assert len(batch.errors) == 0  # nosec B101

    def test_repr(self):
        """Test string representation."""
        batch = BatchPipeline()
        repr_str = repr(batch)

        assert "BatchPipeline" in repr_str  # nosec B101
        assert "results=0" in repr_str  # nosec B101
        assert "errors=0" in repr_str  # nosec B101


class TestBatchExport:
    """Test batch export functionality."""

    def test_export_summary_excel(self, temp_csv_files):
        """Test exporting summary to Excel."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            output_path = f.name

        try:
            # This might fail if no pandas-excel support
            # Just test the interface
            with contextlib.suppress(Exception):
                batch.export_summary(output_path, format="excel")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_summary_csv(self, temp_csv_files):
        """Test exporting summary to CSV."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            batch.export_summary(output_path, format="csv")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_empty_warning(self):
        """Test that exporting empty results produces warning."""
        batch = BatchPipeline()

        with pytest.warns(UserWarning, match="No results"):
            batch.export_summary("output.xlsx")

    def test_export_summary_excel_sanitizes_formula_injection(self):
        """CWE-1236: a malicious file-derived path/name in the summary must
        be neutralized before it reaches the workbook (regression test for
        ASSESSMENT.md's Excel/formula-injection security finding)."""
        openpyxl = pytest.importorskip("openpyxl")

        batch = BatchPipeline()
        data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
        batch.results = [
            (Path("=HYPERLINK(evil.com,click).csv"), data, {"r_squared": 0.9}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            output_path = f.name

        try:
            batch.export_summary(output_path, format="excel")
            wb = openpyxl.load_workbook(output_path)
            ws = wb.active
            header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            name_col = header.index("file_name") + 1
            file_name_value = ws.cell(row=2, column=name_col).value
            assert file_name_value.startswith("'"), (  # nosec B101
                f"file_name cell must be neutralized, got: {file_name_value!r}"
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_summary_csv_sanitizes_formula_injection(self):
        """CWE-1236 applies to CSV too: a malicious file-derived path/name
        must be neutralized before it reaches the file, since spreadsheet
        apps treat leading '=' etc. as a formula trigger on CSV import."""
        batch = BatchPipeline()
        data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
        batch.results = [
            (Path("=HYPERLINK(evil.com,click).csv"), data, {"r_squared": 0.9}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            batch.export_summary(output_path, format="csv")
            content = Path(output_path).read_text()
            assert "'=HYPERLINK" in content, (  # nosec B101
                f"file_name must be neutralized in CSV output, got: {content!r}"
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class _FailingTransform:
    """Transform whose replay always fails, to exercise the
    transform_replay_failed skip-guard on downstream steps."""

    stateless = True

    def transform(self, data):
        raise RuntimeError("intentional transform failure")


class TestBatchTransformReplayFailureSkipsDownstream:
    """A failed transform replay must not let fit/fit_bayesian/export run
    against silently-unprocessed data (rheojax/pipeline/batch.py's
    transform_replay_failed guard, extended to all three step types)."""

    def test_transform_failure_skips_fit_bayesian_and_export(self, tmp_path):
        data = RheoData(
            x=np.linspace(0.1, 10, 20), y=np.linspace(1, 20, 20), validate=False
        )
        template = Pipeline()
        template.steps = [
            ("transform", _FailingTransform()),
            ("fit", BatchTestModel()),
            ("fit_bayesian", BatchTestModel()),
            (
                "export",
                {"output_path": str(tmp_path / "exports"), "format": "json"},
            ),
        ]

        batch = BatchPipeline(template)
        _result, metrics = batch._process_file(
            tmp_path / "input.csv", preloaded_data=data
        )

        assert metrics["transform_replay_failed"] is True  # nosec B101
        assert "r_squared" not in metrics  # nosec B101
        assert "rmse" not in metrics  # nosec B101
        assert "bayesian_completed" not in metrics  # nosec B101
        assert "export_path" not in metrics  # nosec B101
        # Nothing should have been written to disk for the skipped export.
        assert not (tmp_path / "exports").exists()  # nosec B101
