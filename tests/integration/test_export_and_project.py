"""Integration smokes for export and project save/load flows."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.services.export_service import ExportService
from rheojax.gui.services.model_service import ModelService


def _fit_simple_relaxation() -> tuple[RheoData, Any]:
    t = np.linspace(0, 5, 25)
    y = np.exp(-t / 2.0)
    data = RheoData(x=t, y=y, domain="time", metadata={"test_mode": "relaxation"})

    svc = ModelService()
    result = svc.fit("maxwell", data, params={}, test_mode="relaxation")
    return data, result


def test_export_parameters_and_data(tmp_path: Path):
    data, result = _fit_simple_relaxation()
    exporter = ExportService()

    csv_path = tmp_path / "params.csv"
    json_path = tmp_path / "params.json"
    h5_path = tmp_path / "data.hdf5"
    pdf_report = tmp_path / "report.pdf"

    exporter.export_parameters(result, csv_path, "csv")
    exporter.export_parameters(result, json_path, "json")
    exporter.export_data(data, h5_path, "hdf5")
    exporter.generate_report(
        {
            "model_name": "maxwell",
            "parameters": result.parameters,
            "test_mode": "relaxation",
        },
        template="summary",
        path=pdf_report,
    )

    assert csv_path.exists()
    assert json.loads(json_path.read_text())
    assert h5_path.exists()
    assert pdf_report.exists()


def test_project_save_load_round_trip(tmp_path: Path):
    data, result = _fit_simple_relaxation()
    exporter = ExportService()

    state = {
        "model_name": "maxwell",
        "test_mode": "relaxation",
        "data": data,
        "parameters": result.parameters,
    }

    proj_path = tmp_path / "sample.rheo"
    exporter.save_project(state, proj_path)

    loaded = exporter.load_project(proj_path)

    assert proj_path.exists()
    assert "data" in loaded and isinstance(loaded["data"], RheoData)
    assert "parameters" in loaded and loaded["parameters"]
