"""Regression tests for GMM alias handling and compatibility."""

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService
from rheojax.gui.services.model_service import ModelService, normalize_model_name


def test_gmm_alias_normalization():
    service = ModelService()
    assert service._normalize_model_name("GMM") == "generalized_maxwell"
    assert normalize_model_name("gmm") == "generalized_maxwell"


def test_gmm_compatibility_alias():
    data = RheoData(x=np.array([0.0, 1.0]), y=np.array([1.0, 0.5]), metadata={"test_mode": "relaxation"})
    report = ModelService().check_compatibility("GMM", data, "relaxation")
    # Should return a dict with compatibility keys even when using alias
    assert "compatible" in report


def test_bayesian_service_accepts_gmm_alias():
    priors = BayesianService().get_default_priors("GMM")
    assert isinstance(priors, dict)
    # The generalized_maxwell model has parameters; ensure some are returned
    assert len(priors) > 0

