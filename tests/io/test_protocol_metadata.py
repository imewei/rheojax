"""Tests for protocol metadata kwargs in CSV and Excel readers (Phase 2)."""
import numpy as np
import pandas as pd
import pytest

from rheojax.io.readers.csv_reader import load_csv
from rheojax.io.readers.excel_reader import load_excel


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame({"time": [0.1, 0.2, 0.3], "stress": [100.0, 90.0, 80.0]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def excel_file(tmp_path):
    df = pd.DataFrame({"time": [0.1, 0.2, 0.3], "stress": [100.0, 90.0, 80.0]})
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_csv_strain_amplitude(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress", strain_amplitude=0.01)
    assert data.metadata["gamma_0"] == pytest.approx(0.01)


@pytest.mark.smoke
def test_csv_angular_frequency(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress", angular_frequency=6.28)
    assert data.metadata["omega"] == pytest.approx(6.28)


@pytest.mark.smoke
def test_csv_applied_stress(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress", applied_stress=100.0)
    assert data.metadata["sigma_applied"] == pytest.approx(100.0)


@pytest.mark.smoke
def test_csv_shear_rate(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress", shear_rate=1.0)
    assert data.metadata["gamma_dot"] == pytest.approx(1.0)


@pytest.mark.smoke
def test_csv_reference_gamma_dot(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress", reference_gamma_dot=0.1)
    assert data.metadata["reference_gamma_dot"] == pytest.approx(0.1)


@pytest.mark.smoke
def test_csv_column_mapping(tmp_path):
    df = pd.DataFrame({"t": [0.1, 0.2, 0.3], "sigma": [100.0, 90.0, 80.0]})
    path = tmp_path / "remapped.csv"
    df.to_csv(path, index=False)
    data = load_csv(
        path,
        x_col="time",
        y_col="stress",
        column_mapping={"t": "time", "sigma": "stress"},
    )
    assert len(data.x) == 3
    np.testing.assert_allclose(data.x, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.y, [100.0, 90.0, 80.0])


@pytest.mark.smoke
def test_csv_multiple_metadata(csv_file):
    data = load_csv(
        csv_file,
        x_col="time",
        y_col="stress",
        strain_amplitude=0.05,
        angular_frequency=1.0,
        shear_rate=2.0,
        applied_stress=50.0,
        reference_gamma_dot=0.5,
    )
    assert data.metadata["gamma_0"] == pytest.approx(0.05)
    assert data.metadata["omega"] == pytest.approx(1.0)
    assert data.metadata["gamma_dot"] == pytest.approx(2.0)
    assert data.metadata["sigma_applied"] == pytest.approx(50.0)
    assert data.metadata["reference_gamma_dot"] == pytest.approx(0.5)


@pytest.mark.smoke
def test_csv_no_metadata_default(csv_file):
    data = load_csv(csv_file, x_col="time", y_col="stress")
    for key in ("gamma_0", "omega", "sigma_applied", "gamma_dot", "reference_gamma_dot"):
        assert key not in data.metadata


# ---------------------------------------------------------------------------
# Excel tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_excel_strain_amplitude(excel_file):
    data = load_excel(excel_file, x_col="time", y_col="stress", strain_amplitude=0.01)
    assert data.metadata["gamma_0"] == pytest.approx(0.01)


@pytest.mark.smoke
def test_excel_column_mapping(tmp_path):
    df = pd.DataFrame({"t": [0.1, 0.2, 0.3], "sigma": [100.0, 90.0, 80.0]})
    path = tmp_path / "remapped.xlsx"
    df.to_excel(path, index=False)
    data = load_excel(
        path,
        x_col="time",
        y_col="stress",
        column_mapping={"t": "time", "sigma": "stress"},
    )
    assert len(data.x) == 3
    np.testing.assert_allclose(data.x, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.y, [100.0, 90.0, 80.0])


@pytest.mark.smoke
def test_excel_multiple_metadata(excel_file):
    data = load_excel(
        excel_file,
        x_col="time",
        y_col="stress",
        strain_amplitude=0.05,
        angular_frequency=1.0,
        shear_rate=2.0,
        applied_stress=50.0,
        reference_gamma_dot=0.5,
    )
    assert data.metadata["gamma_0"] == pytest.approx(0.05)
    assert data.metadata["omega"] == pytest.approx(1.0)
    assert data.metadata["gamma_dot"] == pytest.approx(2.0)
    assert data.metadata["sigma_applied"] == pytest.approx(50.0)
    assert data.metadata["reference_gamma_dot"] == pytest.approx(0.5)


@pytest.mark.smoke
def test_excel_no_metadata_default(excel_file):
    data = load_excel(excel_file, x_col="time", y_col="stress")
    for key in ("gamma_0", "omega", "sigma_applied", "gamma_dot", "reference_gamma_dot"):
        assert key not in data.metadata
