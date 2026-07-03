import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.foundation import import_service
from rheojax.gui.foundation.import_service import import_dataset


def test_import_dataset_sets_protocol_and_metadata(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "omega,G',G''\n0.1,100.0,10.0\n1.0,150.0,20.0\n10.0,200.0,50.0\n"
    )

    ref, data = import_dataset(csv_path, "oscillation")

    assert ref.protocol_type == "oscillation"
    assert ref.origin == "imported"
    assert ref.id  # non-empty uuid4 hex
    assert data.test_mode == "oscillation"


def test_import_dataset_converts_hz_to_rad_per_s(tmp_path, monkeypatch):
    # rheojax.io's generic-CSV auto-detect only recognizes bare x-column
    # headers ("omega", "frequency", ...) -- a real Hz-labeled instrument
    # export signals units via a separate metadata row/field, not a header
    # a hand-rolled CSV can trivially reproduce. Stub auto_load to return
    # what the reader hands back in that case (x_units="Hz", x unconverted)
    # so this test isolates import_dataset's own conversion/unit-recording
    # logic -- the thing this fix actually changes -- from that reader detail.
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("placeholder\n")
    raw = RheoData(
        x=np.array([0.1, 1.0, 10.0]),
        y=np.array([100.0 + 10.0j, 150.0 + 20.0j, 200.0 + 50.0j]),
        x_units="Hz",
        y_units="Pa",
        domain="frequency",
    )
    monkeypatch.setattr(import_service, "auto_load", lambda path: raw)

    ref, data = import_dataset(csv_path, "oscillation")

    np.testing.assert_allclose(
        np.asarray(data.x), np.array([0.1, 1.0, 10.0]) * (2 * np.pi)
    )
    assert data.x_units == "rad/s"
    assert ref.units["x"] == "rad/s"


def test_import_dataset_does_not_double_convert_rad_per_s(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("placeholder\n")
    raw = RheoData(
        x=np.array([0.1, 1.0, 10.0]),
        y=np.array([100.0 + 10.0j, 150.0 + 20.0j, 200.0 + 50.0j]),
        x_units="rad/s",
        y_units="Pa",
        domain="frequency",
    )
    monkeypatch.setattr(import_service, "auto_load", lambda path: raw)

    ref, data = import_dataset(csv_path, "oscillation")

    np.testing.assert_allclose(np.asarray(data.x), np.array([0.1, 1.0, 10.0]))
    assert data.x_units == "rad/s"
    assert ref.units["x"] == "rad/s"


def test_import_dataset_rejects_unknown_protocol(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n0.1,10\n")
    with pytest.raises(ValueError):
        import_dataset(csv_path, "not_a_real_protocol")


def test_import_dataset_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_dataset(tmp_path / "does_not_exist.csv", "oscillation")


def test_import_dataset_rejects_wrong_column_count(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("time,stress\n0.1,10\n1.0,20\n10.0,30\n")
    with pytest.raises(ValueError):
        import_dataset(csv_path, "oscillation")
