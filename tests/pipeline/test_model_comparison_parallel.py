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
        mc_par.run(relaxation_data, parallel=True, n_workers=2, test_mode="relaxation")

        # Both should find results for both models
        assert set(mc_seq.results.keys()) == set(mc_par.results.keys())

        # Metrics should be numerically close
        for model in mc_seq.results:
            seq_r2 = mc_seq.results[model].get("r_squared", 0)
            par_r2 = mc_par.results[model].get("r_squared", 0)
            assert abs(seq_r2 - par_r2) < 0.05, (
                f"Model {model}: R-squared mismatch seq={seq_r2:.4f} vs par={par_r2:.4f}"
            )

            # Tight parity on rmse/aic/bic: the parallel path
            # (_fit_model_in_subprocess) must prefer nlsq_result.rmse/.aic/.bic
            # exactly like the sequential path, not fall back to a manually
            # recomputed value. A loose r_squared check alone doesn't catch a
            # regression here because r_squared computation is unaffected.
            assert mc_seq.results[model]["rmse"] == pytest.approx(
                mc_par.results[model]["rmse"], rel=1e-6
            ), f"Model {model}: rmse mismatch between sequential and parallel paths"
            assert mc_seq.results[model]["aic"] == pytest.approx(
                mc_par.results[model]["aic"], rel=1e-6
            ), f"Model {model}: aic mismatch between sequential and parallel paths"
            assert mc_seq.results[model]["bic"] == pytest.approx(
                mc_par.results[model]["bic"], rel=1e-6
            ), f"Model {model}: bic mismatch between sequential and parallel paths"

        # The actual regression this fix prevents: ranking by a metric must
        # not depend on the parallel flag. r_squared alone wouldn't have
        # caught the original bug since aic/bic were the ones diverging.
        assert mc_seq.get_best_model(metric="aic") == mc_par.get_best_model(
            metric="aic"
        ), "get_best_model(metric='aic') disagrees between sequential and parallel runs"

    def test_parallel_single_model_fallback(self, relaxation_data):
        """With only 1 model, parallel=True should not use pool."""
        from rheojax.pipeline.workflows import ModelComparisonPipeline

        mc = ModelComparisonPipeline(models=["maxwell"])
        mc.run(relaxation_data, parallel=True, test_mode="relaxation")
        assert "maxwell" in mc.results
