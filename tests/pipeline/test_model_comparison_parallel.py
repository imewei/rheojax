"""Tests for parallel model comparison."""

import numpy as np
import pytest


class TestModelComparisonParallel:
    @pytest.fixture
    def relaxation_data(self):
        from rheojax.core.data import RheoData

        t = np.linspace(0.01, 10.0, 100)
        G = 1000.0 * np.exp(-t / 1.0) + 50.0
        return RheoData(x=t, y=G, initial_test_mode="relaxation")

    @pytest.mark.smoke
    def test_parallel_kwarg_accepted(self, relaxation_data):
        from rheojax.pipeline.workflows import ModelComparisonPipeline

        mc = ModelComparisonPipeline(models=["maxwell", "zener"])
        # Should accept parallel parameter
        mc.run(relaxation_data, parallel=False, test_mode="relaxation")
        assert len(mc.results) == 2

    def test_parallel_matches_sequential(self, relaxation_data):
        from rheojax.pipeline.workflows import ModelComparisonPipeline

        mc_seq = ModelComparisonPipeline(models=["maxwell", "zener"])
        mc_seq.run(relaxation_data, parallel=False, test_mode="relaxation")

        mc_par = ModelComparisonPipeline(models=["maxwell", "zener"])
        mc_par.run(
            relaxation_data, parallel=True, n_workers=2, test_mode="relaxation"
        )

        # Both should find results for both models
        assert set(mc_seq.results.keys()) == set(mc_par.results.keys())

        # Metrics should be numerically close
        for model in mc_seq.results:
            seq_r2 = mc_seq.results[model].get("r_squared", 0)
            par_r2 = mc_par.results[model].get("r_squared", 0)
            assert abs(seq_r2 - par_r2) < 0.05, (
                f"Model {model}: R-squared mismatch seq={seq_r2:.4f} vs par={par_r2:.4f}"
            )

    def test_parallel_single_model_fallback(self, relaxation_data):
        """With only 1 model, parallel=True should not use pool."""
        from rheojax.pipeline.workflows import ModelComparisonPipeline

        mc = ModelComparisonPipeline(models=["maxwell"])
        mc.run(relaxation_data, parallel=True, test_mode="relaxation")
        assert "maxwell" in mc.results
