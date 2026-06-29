# tests/refactor/test_dmta_removal.py
import importlib
import importlib.util
import inspect
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rheojax.core.base import BaseModel
from rheojax.models import Maxwell

RJ = Path(__file__).resolve().parents[2] / "rheojax"
RETIRED_TOKENS = (
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

REJECTION_BOUNDARY = RJ / "core" / "_validation.py"
OBFUSCATED_REMOVED_KEYS = ('deforma" + "tion_mode', 'poisson" + "_ratio')


def _hits(subdir, idents):
    return sorted(
        str(p)
        for p in (RJ / subdir).rglob("*.py")
        if any(i in p.read_text() for i in idents)
    )


def test_retired_symbols_are_absent_from_production_tree():
    """Only the central removed-option rejection boundary may name old keys."""
    violations = []
    for path in RJ.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in RETIRED_TOKENS:
            if token in text:
                violations.append((str(path.relative_to(RJ)), token))
        if path != REJECTION_BOUNDARY:
            for fragment in OBFUSCATED_REMOVED_KEYS:
                if fragment in text:
                    violations.append((str(path.relative_to(RJ)), fragment))

    assert violations == []


@pytest.mark.parametrize(
    "module_name",
    (
        "rheojax.core.deformation_converter",
        "rheojax.utils.modulus_conversion",
    ),
)
def test_deleted_modules_have_no_import_spec(module_name):
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    ("module_name", "public_name"),
    (
        ("rheojax.core", "DeformationMode"),
        ("rheojax.core", "DeformationConverter"),
        ("rheojax.utils", "convert_modulus"),
        ("rheojax.utils", "convert_rheodata"),
        ("rheojax.utils", "POISSON_PRESETS"),
        ("rheojax.utils", "is_tensile"),
    ),
)
def test_former_public_names_cannot_be_imported(module_name, public_name):
    module = import_module(module_name)
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {public_name}", {})
    assert not hasattr(module, public_name)


def test_no_model_imports_deformationmode():
    assert _hits("models", ("DeformationMode",)) == []


def test_base_signatures_clean():
    for n in ("fit", "predict", "fit_bayesian"):
        p = inspect.signature(getattr(BaseModel, n)).parameters
        assert "deformation_mode" not in p and "poisson_ratio" not in p, n


def test_removed_options_are_rejected():
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


def test_csv_reader_does_not_create_retired_metadata(tmp_path):
    from rheojax.io.readers.csv_reader import load_csv

    path = tmp_path / "shear.csv"
    path.write_text("time,modulus\n1,10\n2,20\n", encoding="utf-8")
    data = load_csv(path, x_col="time", y_col="modulus")

    assert "deformation_mode" not in data.metadata


def test_excel_reader_does_not_create_retired_metadata(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    from rheojax.io.readers.excel_reader import load_excel

    path = tmp_path / "shear.xlsx"
    pd.DataFrame({"time": [1.0, 2.0], "modulus": [10.0, 20.0]}).to_excel(
        path, index=False
    )
    data = load_excel(path, x_col="time", y_col="modulus")

    assert "deformation_mode" not in data.metadata


def test_anton_paar_oscillation_does_not_create_retired_metadata():
    import pandas as pd

    from rheojax.io.readers.anton_paar import (
        IntervalBlock,
        _interval_to_rheodata_oscillation,
    )

    frame = pd.DataFrame(
        {
            "angular_frequency": [1.0, 2.0],
            "storage_modulus": [10.0, 20.0],
            "loss_modulus": [2.0, 4.0],
        }
    )
    block = IntervalBlock(interval_index=1, n_points=2, units={}, df=frame)
    data = _interval_to_rheodata_oscillation(block, {}, frame, {})

    assert "deformation_mode" not in data.metadata


def test_analysis_exporter_does_not_copy_retired_metadata():
    from rheojax.core.data import RheoData
    from rheojax.io.analysis_exporter import AnalysisExporter

    data = RheoData(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        metadata={"deformation_mode": "shear", "test_mode": "relaxation"},
    )
    pipeline = SimpleNamespace(data=data, history=[])

    state = AnalysisExporter()._collect_state(pipeline)

    assert "deformation_mode" not in state["metadata"]


def test_legacy_hdf5_deformation_metadata_is_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    from rheojax.io._exceptions import UnsupportedDataError
    from rheojax.io.writers.hdf5_writer import load_hdf5

    path = tmp_path / "legacy-tensile.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.array([1.0, 2.0]))
        handle.create_dataset("y", data=np.array([3.0, 4.0]))
        handle.attrs["deformation_mode"] = "tensile"

    with pytest.raises(UnsupportedDataError, match="tensile"):
        load_hdf5(path)


@pytest.mark.parametrize("removed_key", ("deformation_mode", "poisson_ratio"))
def test_plotting_guard_reuses_removed_option_policy(removed_key):
    from rheojax.visualization.fit_plotter import _reject_removed_plot_options

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        _reject_removed_plot_options({removed_key: "legacy"})


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


@pytest.mark.parametrize("removed_key", ("deformation_mode", "poisson_ratio"))
def test_registry_find_compatible_rejects_removed_option(removed_key):
    from rheojax.core.registry import Registry

    registry = Registry()
    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        registry.find_compatible(protocol="relaxation", **{removed_key: "legacy"})


@pytest.mark.parametrize("removed_key", ("deformation_mode", "poisson_ratio"))
def test_model_registry_create_rejects_removed_option(removed_key):
    from rheojax.core.registry import ModelRegistry

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        ModelRegistry.create("maxwell", **{removed_key: "legacy"})


def test_model_registry_discovers_exact_inventory_in_fresh_process():
    code = """
import rheojax.models
from rheojax.core.registry import ModelRegistry
models = ModelRegistry.list_models()
assert len(models) == 59, models
assert len(models) == len(set(models)), models
"""
    subprocess.run([sys.executable, "-c", code], check=True)
