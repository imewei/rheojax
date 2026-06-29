# tests/refactor/test_dmta_removal.py
import inspect
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.models import Maxwell

RJ = Path(__file__).resolve().parents[2] / "rheojax"
IDENTS = (
    "DeformationMode",
    "deformation_mode",
    "poisson_ratio",
    "modulus_conversion",
    "convert_modulus",
    "convert_rheodata",
    "POISSON_PRESETS",
    "is_tensile",
    "modulus_type",
)


def _hits(subdir, idents):
    return sorted(
        str(p)
        for p in (RJ / subdir).rglob("*.py")
        if any(i in p.read_text() for i in idents)
    )


def test_no_model_imports_deformationmode():
    assert _hits("models", ("DeformationMode",)) == []


def test_base_signatures_clean():
    for n in ("fit", "predict", "fit_bayesian"):
        p = inspect.signature(getattr(BaseModel, n)).parameters
        assert "deformation_mode" not in p and "poisson_ratio" not in p, n


def test_reject_dmta_kwargs():
    model = Maxwell()
    t = np.logspace(-2, 2, 10)
    G = np.ones_like(t)

    # Test fit
    with pytest.raises(TypeError, match="shear-only"):
        model.fit(t, G, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(TypeError, match="shear-only"):
        model.fit(t, G, test_mode="relaxation", poisson_ratio=0.5)

    # Test predict
    with pytest.raises(TypeError, match="shear-only"):
        model.predict(t, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(TypeError, match="shear-only"):
        model.predict(t, test_mode="relaxation", poisson_ratio=0.5)

    # Test fit_bayesian
    with pytest.raises(TypeError, match="shear-only"):
        model.fit_bayesian(t, G, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(TypeError, match="shear-only"):
        model.fit_bayesian(t, G, test_mode="relaxation", poisson_ratio=0.5)


def test_no_tensile_modulus_type():
    from rheojax.models.multimode.generalized_maxwell import GeneralizedMaxwell
    from rheojax.utils.prony import create_prony_parameter_set

    assert "modulus_type" not in inspect.signature(create_prony_parameter_set).parameters
    assert "modulus_type" not in inspect.signature(GeneralizedMaxwell.__init__).parameters
    with pytest.raises(TypeError):
        GeneralizedMaxwell(modulus_type="tensile")


def test_registry_clean():
    from rheojax.core.registry import ModelRegistry, Registry
    assert "deformation_mode" not in inspect.signature(ModelRegistry.find).parameters
    assert "deformation_mode" not in inspect.signature(Registry.find_compatible).parameters
    assert not hasattr(ModelRegistry.get_info("maxwell"), "deformation_modes")


def test_rheodata_clean():
    from rheojax.core.data import RheoData
    d = RheoData(x=np.array([1.0, 2, 3]), y=np.array([1.0, 2, 3]))
    assert not hasattr(d, "deformation_mode")


def test_pipeline_clean():
    assert _hits("pipeline", ("deformation_mode", "poisson_ratio")) == []


def test_legacy_kwargs_raise():
    from rheojax.core.registry import ModelRegistry
    m = ModelRegistry.create("maxwell")
    x = np.logspace(-1, 1, 10)
    y = np.exp(-x)
    for call in (
        lambda: m.fit(x, y, deformation_mode="tension"),
        lambda: m.predict(x, poisson_ratio=0.5),
        lambda: m.fit_bayesian(x, y, deformation_mode="tension"),
        lambda: ModelRegistry.find(protocol="oscillation", deformation_mode="shear"),
    ):
        with pytest.raises(TypeError, match="shear-only"):
            call()

