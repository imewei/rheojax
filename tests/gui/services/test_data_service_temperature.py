"""Regression check: DataService._extract_scalar_temperature must detect
the file's real delimiter, not assume comma; and must normalize to Kelvin
like its filename-parsing sibling (get_shift_factor's WLF/Arrhenius shift
factors expect Kelvin -- returning raw °C here evaluated ~273 K low)."""

from pathlib import Path

import pytest

from rheojax.gui.services.data_service import DataService


def test_extract_scalar_temperature_tab_delimited(tmp_path: Path):
    """A tab-delimited file collapses to one column under pandas' default
    comma separator, so the temperature column lookup silently fails and
    the metadata is dropped with no user-visible error.
    """
    tsv_path = tmp_path / "sweep.tsv"
    tsv_path.write_text("omega\tGp\tGpp\tTemp\n1\t101\t11\t25.0\n2\t102\t12\t25.0\n")

    value = DataService._extract_scalar_temperature(tsv_path, "Temp")

    # No unit marker on the "Temp" header -> defaults to Celsius -> Kelvin.
    assert value == 298.15


def test_extract_scalar_temperature_column_k_unit_passthrough(tmp_path: Path):
    """A column header explicitly marked K is not re-converted."""
    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text("omega,Gp,Gpp,Temp [K]\n1,101,11,298.15\n2,102,12,298.15\n")

    value = DataService._extract_scalar_temperature(csv_path, "Temp [K]")

    assert value == 298.15


def test_extract_scalar_temperature_column_f_unit_converted(tmp_path: Path):
    """A column header marked °F is converted through Kelvin."""
    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text("omega,Gp,Gpp,Temp (F)\n1,101,11,77.0\n2,102,12,77.0\n")

    value = DataService._extract_scalar_temperature(csv_path, "Temp (F)")

    assert value == pytest.approx(298.15, abs=1e-2)


class TestExtractTemperatureFromFilename:
    """Regression for the filename-parsing sibling: it must return Kelvin,
    not Celsius, or get_shift_factor's WLF/Arrhenius shift evaluates ~273 K
    below the intended point.
    """

    def test_celsius_marker_converted_to_kelvin(self, tmp_path: Path):
        f = tmp_path / "foam_dma_-5C.csv"
        f.write_text("x,y\n1,2\n")
        assert DataService._extract_temperature_from_filename(f) == pytest.approx(
            268.15
        )

    def test_positive_celsius_marker_converted_to_kelvin(self, tmp_path: Path):
        f = tmp_path / "sweep_60C.csv"
        f.write_text("x,y\n1,2\n")
        assert DataService._extract_temperature_from_filename(f) == pytest.approx(
            333.15
        )

    def test_kelvin_marker_passed_through(self, tmp_path: Path):
        f = tmp_path / "sweep_298K.csv"
        f.write_text("x,y\n1,2\n")
        assert DataService._extract_temperature_from_filename(f) == pytest.approx(298.0)

    def test_bare_trailing_number_not_treated_as_temperature(self, tmp_path: Path):
        """A unit marker is required -- run indices/dates must not be
        fabricated into temperature metadata."""
        f = tmp_path / "run_3.csv"
        f.write_text("x,y\n1,2\n")
        assert DataService._extract_temperature_from_filename(f) is None

    def test_no_match_returns_none(self, tmp_path: Path):
        f = tmp_path / "sample_20240115.csv"
        f.write_text("x,y\n1,2\n")
        assert DataService._extract_temperature_from_filename(f) is None
