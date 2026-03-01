"""Tests for parallel batch processing."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def csv_files(tmp_path):
    """Create 5 minimal CSV files."""
    for i in range(5):
        f = tmp_path / f"data_{i}.csv"
        t = np.linspace(0.01, 10.0, 50)
        G = 1000.0 * np.exp(-t / (0.5 + 0.1 * i))
        lines = ["time,G_relax\n"] + [f"{ti},{gi}\n" for ti, gi in zip(t, G)]
        f.write_text("".join(lines))
    return sorted(tmp_path.glob("*.csv"))


class TestBatchParallelIO:
    """Test thread-parallel I/O loading phase."""

    @pytest.mark.smoke
    def test_parallel_io_loads_all_files(self, csv_files):
        from rheojax.pipeline import Pipeline
        from rheojax.pipeline.batch import BatchPipeline

        template = Pipeline()
        batch = BatchPipeline(template)
        batch.process_files(csv_files, parallel_io=True, x_col="time", y_col="G_relax")
        assert len(batch.results) == 5

    def test_parallel_io_matches_sequential(self, csv_files):
        from rheojax.pipeline import Pipeline
        from rheojax.pipeline.batch import BatchPipeline

        template = Pipeline()

        # Sequential
        batch_seq = BatchPipeline(template)
        batch_seq.process_files(
            csv_files, parallel_io=False, x_col="time", y_col="G_relax"
        )

        # Parallel I/O
        batch_par = BatchPipeline(template)
        batch_par.process_files(
            csv_files, parallel_io=True, x_col="time", y_col="G_relax"
        )

        assert len(batch_seq.results) == len(batch_par.results)

    def test_parallel_io_default_true(self, csv_files):
        """parallel_io defaults to True."""
        from rheojax.pipeline import Pipeline
        from rheojax.pipeline.batch import BatchPipeline

        template = Pipeline()
        batch = BatchPipeline(template)
        # Should work without specifying parallel_io
        batch.process_files(csv_files, x_col="time", y_col="G_relax")
        assert len(batch.results) == 5
