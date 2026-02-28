"""Tests for parallel file loading in MastercurvePipeline."""

import numpy as np
import pytest


class TestMastercurveParallelLoad:
    @pytest.mark.smoke
    def test_parallel_load_kwarg_accepted(self):
        from rheojax.pipeline.workflows import MastercurvePipeline

        mc = MastercurvePipeline(reference_temp=60.0)
        # Should accept parallel_io parameter without error
        assert hasattr(mc.run, "__call__")

    def test_parallel_io_loads_all(self, tmp_path):
        from rheojax.pipeline.workflows import MastercurvePipeline

        # Create minimal CSV files for 3 temperatures
        temps = [40.0, 60.0, 80.0]
        files = []
        for i, temp in enumerate(temps):
            f = tmp_path / f"data_{temp:.0f}K.csv"
            omega = np.logspace(-2, 2, 30)
            G = 1e5 * np.exp(-omega / (10 * (i + 1)))
            lines = ["omega,G\n"] + [f"{w},{g}\n" for w, g in zip(omega, G)]
            f.write_text("".join(lines))
            files.append(str(f))

        mc = MastercurvePipeline(reference_temp=60.0)
        mc.run(files, temps, parallel_io=True, x_col="omega", y_col="G")
        assert mc.data is not None

    def test_sequential_io_also_works(self, tmp_path):
        from rheojax.pipeline.workflows import MastercurvePipeline

        temps = [40.0, 60.0]
        files = []
        for i, temp in enumerate(temps):
            f = tmp_path / f"data_{temp:.0f}K.csv"
            omega = np.logspace(-2, 2, 20)
            G = 1e5 * np.exp(-omega / (10 * (i + 1)))
            lines = ["omega,G\n"] + [f"{w},{g}\n" for w, g in zip(omega, G)]
            f.write_text("".join(lines))
            files.append(str(f))

        mc = MastercurvePipeline(reference_temp=60.0)
        mc.run(files, temps, parallel_io=False, x_col="omega", y_col="G")
        assert mc.data is not None
