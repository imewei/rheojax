# tests/refactor/test_dmta_removal.py
import inspect
import pathlib

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.models import Maxwell

RJ = pathlib.Path("rheojax")
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
    with pytest.raises(ValueError, match="deformation_mode"):
        model.fit(t, G, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(ValueError, match="poisson_ratio"):
        model.fit(t, G, test_mode="relaxation", poisson_ratio=0.5)

    # Test predict
    with pytest.raises(ValueError, match="deformation_mode"):
        model.predict(t, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(ValueError, match="poisson_ratio"):
        model.predict(t, test_mode="relaxation", poisson_ratio=0.5)

    # Test fit_bayesian
    with pytest.raises(ValueError, match="deformation_mode"):
        model.fit_bayesian(t, G, test_mode="relaxation", deformation_mode="tensile")
    with pytest.raises(ValueError, match="poisson_ratio"):
        model.fit_bayesian(t, G, test_mode="relaxation", poisson_ratio=0.5)
