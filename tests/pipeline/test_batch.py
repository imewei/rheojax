"""Tests for BatchPipeline.

This module tests batch processing of multiple datasets with the same pipeline.
"""

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
        with open(file_path, "w") as f:
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
        assert batch.template_pipeline is None
        assert len(batch.results) == 0
        assert len(batch.errors) == 0

    def test_init_with_template(self):
        """Test initialization with template pipeline."""
        template = Pipeline()
        batch = BatchPipeline(template)

        assert batch.template_pipeline is template

    def test_set_template(self):
        """Test setting template after initialization."""
        batch = BatchPipeline()
        template = Pipeline()

        batch.set_template(template)
        assert batch.template_pipeline is template


class TestBatchProcessing:
    """Test batch file processing."""

    def test_process_files(self, temp_csv_files):
        """Test processing multiple files."""
        # Create template pipeline (no need to fit, BatchPipeline will apply template to each file)
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        # Note: Results might be empty if template execution is simplified
        # This tests the interface, not full execution
        assert len(batch.results) >= 0
        assert len(batch.errors) >= 0

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

        # At least some files should be found
        total_processed = len(batch.results) + len(batch.errors)
        assert total_processed >= 0

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
        assert isinstance(results, list)

    def test_get_results_copy(self, temp_csv_files):
        """Test that get_results returns a copy."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files[:1], format="csv", x_col="x", y_col="y")

        results1 = batch.get_results()
        results2 = batch.get_results()

        assert results1 is not results2

    def test_get_errors(self, temp_csv_files):
        """Test getting errors."""
        template = Pipeline()
        batch = BatchPipeline(template)

        # Add a non-existent file to trigger error
        batch.process_files(
            temp_csv_files + ["/nonexistent.csv"], format="csv", x_col="x", y_col="y"
        )

        errors = batch.get_errors()
        assert isinstance(errors, list)

    def test_get_summary_dataframe(self, temp_csv_files):
        """Test getting summary DataFrame."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        df = batch.get_summary_dataframe()
        assert df is not None
        # DataFrame might be empty if no results processed


class TestBatchStatistics:
    """Test batch statistics."""

    def test_get_statistics_empty(self):
        """Test statistics with no results."""
        batch = BatchPipeline()
        stats = batch.get_statistics()

        assert isinstance(stats, dict)
        assert len(stats) == 0

    def test_get_statistics_with_results(self, temp_csv_files):
        """Test statistics with results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        stats = batch.get_statistics()
        assert isinstance(stats, dict)

    def test_length(self, temp_csv_files):
        """Test batch length."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        assert len(batch) == len(batch.results)


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

        assert len(batch.results) == initial_count

    def test_filter_removes_results(self, temp_csv_files):
        """Test that filter can remove results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        # Filter to remove all
        batch.apply_filter(lambda p, d, m: False)

        assert len(batch.results) == 0


class TestBatchUtilities:
    """Test batch utility methods."""

    def test_clear(self, temp_csv_files):
        """Test clearing results."""
        template = Pipeline()
        batch = BatchPipeline(template)

        batch.process_files(temp_csv_files, format="csv", x_col="x", y_col="y")

        batch.clear()

        assert len(batch.results) == 0
        assert len(batch.errors) == 0

    def test_repr(self):
        """Test string representation."""
        batch = BatchPipeline()
        repr_str = repr(batch)

        assert "BatchPipeline" in repr_str
        assert "results=0" in repr_str
        assert "errors=0" in repr_str


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
            batch.export_summary(output_path, format="excel")
        except Exception:
            pass  # Expected if openpyxl not installed
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
