"""Integration tests for parallel execution.

End-to-end tests verifying parallel pipeline produces numerically
equivalent results to sequential execution.
"""

import numpy as np
import pytest

# ── Module-level functions (must be picklable on macOS spawn context) ──


def _fit_model(args):
    """Fit a model in subprocess — picklable entry point."""
    model_name, t, G = args
    from rheojax.core.jax_config import safe_import_jax

    safe_import_jax()
    from rheojax.models import _ensure_all_registered

    _ensure_all_registered()
    from rheojax.core.registry import ModelRegistry

    model = ModelRegistry.create(model_name)
    model.fit(t, G, test_mode="relaxation", max_iter=200)
    pred = np.asarray(model.predict(t, test_mode="relaxation"))
    ss_res = np.sum((G - pred) ** 2)
    ss_tot = np.sum((G - np.mean(G)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    params = {
        name: float(model.parameters.get_value(name))
        for name in model.parameters.keys()
    }
    return {
        "model": model_name,
        "r_squared": r_squared,
        "params": params,
    }


def _fit_single_maxwell(t_G_pair):
    """Fit Maxwell model from (t, G) tuple."""
    t, G = t_G_pair
    from rheojax.core.jax_config import safe_import_jax

    safe_import_jax()
    from rheojax.models import _ensure_all_registered

    _ensure_all_registered()
    from rheojax.core.registry import ModelRegistry

    model = ModelRegistry.create("maxwell")
    model.fit(t, G, test_mode="relaxation", max_iter=200)
    return float(model.parameters.get_value("G0"))


# ── Test classes ──


class TestParallelFitIntegration:
    """Test parallel model fitting produces correct results."""

    @pytest.fixture
    def relaxation_data(self):
        """Single-exponential decay — fits Maxwell and Zener well."""
        t = np.linspace(0.01, 10.0, 100)
        G = 1000.0 * np.exp(-t / 1.0)
        return t, G

    @pytest.mark.smoke
    def test_parallel_map_with_fit(self, relaxation_data):
        """Parallel_map submits fits to subprocesses, collects results."""
        from rheojax.parallel import parallel_map

        t, G = relaxation_data
        results = list(
            parallel_map(
                _fit_model,
                [("maxwell", t, G), ("maxwell", t, G)],
                n_workers=2,
            )
        )
        assert len(results) == 2
        for r in results:
            # Verify fit ran and returned valid params (not testing model accuracy)
            assert "params" in r
            assert r["params"].get("G0", 0) > 0, "G0 should be positive"

    def test_parallel_matches_sequential(self, relaxation_data):
        """Parallel and sequential produce numerically equivalent fits."""
        from rheojax.parallel import parallel_map

        t, G = relaxation_data
        items = [("maxwell", t, G), ("maxwell", t, G)]

        # Sequential
        seq_results = [_fit_model(item) for item in items]

        # Parallel
        par_results = list(parallel_map(_fit_model, items, n_workers=2))

        for seq, par in zip(seq_results, par_results):
            assert seq["model"] == par["model"]
            # R² should be close (independent JIT caches → minor float diffs)
            assert abs(seq["r_squared"] - par["r_squared"]) < 0.05, (
                f"{seq['model']}: seq R²={seq['r_squared']:.4f} vs "
                f"par R²={par['r_squared']:.4f}"
            )


class TestParallelLoadIntegration:
    """Test parallel file loading end-to-end."""

    @pytest.mark.smoke
    def test_parallel_load_csv_files(self, tmp_path):
        """Load multiple CSV files in parallel threads."""
        from rheojax.parallel import parallel_load

        t = np.linspace(0.01, 10.0, 50)
        for i in range(4):
            G = (500.0 + 100 * i) * np.exp(-t / (0.5 + 0.5 * i))
            lines = ["time,stress"] + [f"{ti:.4f},{gi:.4f}" for ti, gi in zip(t, G)]
            (tmp_path / f"data_{i}.csv").write_text("\n".join(lines))

        files = sorted(tmp_path.glob("*.csv"))
        results = parallel_load(files, x_col="time", y_col="stress")
        assert len(results) == 4
        for r in results:
            assert len(r.x) == 50
            assert len(r.y) == 50

    def test_parallel_load_matches_sequential(self, tmp_path):
        """Parallel and sequential loading produce identical data."""
        from unittest.mock import patch

        from rheojax.parallel import parallel_load

        t = np.linspace(0.1, 5.0, 20)
        G = 1000.0 * np.exp(-t / 1.0)
        for i in range(3):
            lines = ["time,stress"] + [f"{ti:.6f},{gi:.6f}" for ti, gi in zip(t, G)]
            (tmp_path / f"data_{i}.csv").write_text("\n".join(lines))

        files = sorted(tmp_path.glob("*.csv"))

        # Sequential
        with patch.dict("os.environ", {"RHEOJAX_SEQUENTIAL": "1"}):
            seq = parallel_load(files, x_col="time", y_col="stress")

        # Parallel
        par = parallel_load(files, x_col="time", y_col="stress")

        for s, p in zip(seq, par):
            np.testing.assert_allclose(np.asarray(s.x), np.asarray(p.x), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(s.y), np.asarray(p.y), rtol=1e-12)


class TestParallelMapFanOut:
    """Test fan-out pattern: same model, different datasets."""

    def test_fan_out_same_model_multiple_datasets(self):
        """Fit Maxwell to several datasets in parallel."""
        from rheojax.parallel import parallel_map

        rng = np.random.default_rng(42)
        datasets = []
        for _ in range(4):
            t = np.linspace(0.01, 10.0, 50)
            G0 = rng.uniform(500, 2000)
            tau = rng.uniform(0.5, 2.0)
            G = G0 * np.exp(-t / tau) + rng.normal(0, G0 * 0.01, size=len(t))
            datasets.append((t, G))

        results = list(parallel_map(_fit_single_maxwell, datasets, n_workers=2))
        assert len(results) == 4
        for g0 in results:
            assert g0 > 0, f"G0 should be positive, got {g0}"
