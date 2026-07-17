"""Regression check: DataService._extract_scalar_temperature must detect
the file's real delimiter, not assume comma."""

from pathlib import Path

from rheojax.gui.services.data_service import DataService


def test_extract_scalar_temperature_tab_delimited(tmp_path: Path):
    """A tab-delimited file collapses to one column under pandas' default
    comma separator, so the temperature column lookup silently fails and
    the metadata is dropped with no user-visible error.
    """
    tsv_path = tmp_path / "sweep.tsv"
    tsv_path.write_text("omega\tGp\tGpp\tTemp\n1\t101\t11\t25.0\n2\t102\t12\t25.0\n")

    value = DataService._extract_scalar_temperature(tsv_path, "Temp")

    assert value == 25.0
