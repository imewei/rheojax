from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("arviz")

from rheojax.gui.services.export_service import ExportService


def test_export_posterior_netcdf_roundtrip(tmp_path):
    import arviz as az

    result = {
        "posterior_samples": {"a": np.random.default_rng(0).normal(size=400)},
        "sample_stats": {"diverging": np.zeros(400, dtype=bool)},
        "num_chains": 4,
    }
    path = tmp_path / "posterior.nc"
    ExportService().export_posterior_netcdf(result, path)
    assert path.exists()
    idata = az.from_netcdf(path)
    assert "a" in idata.posterior.data_vars
