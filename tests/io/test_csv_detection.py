"""Robustness tests for CSV/auto reader delimiter and encoding inference."""

from pathlib import Path

import numpy as np
import pytest

from rheojax.io.readers import auto_load, load_csv


def _write(path: Path, content: str, encoding="utf-8"):
    path.write_bytes(content.encode(encoding))


def test_load_csv_semicolon_windows1252(tmp_path: Path):
    csv_path = tmp_path / "semicol.csv"
    _write(csv_path, "Time (s);Stress [Pa]\n1,0;10,5\n2,0;20,5\n", encoding="cp1252")

    data = load_csv(csv_path, x_col=0, y_col=1, delimiter=None)
    assert np.isclose(data.x[0], 1.0)
    assert np.isclose(data.y[1], 20.5)


def test_load_csv_tab_bom(tmp_path: Path):
    csv_path = tmp_path / "tab.tsv"
    # UTF-8 BOM + tab
    content = "\ufefftime\tstress\n1\t3.0\n"
    _write(csv_path, content, encoding="utf-8")

    data = load_csv(csv_path, x_col="time", y_col="stress", delimiter=None)
    assert len(data.x) == 1
    assert np.isclose(data.y[0], 3.0)


def test_auto_load_excel_fallback_on_dat(tmp_path: Path):
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        pytest.skip("pandas not installed")

    excel_path = tmp_path / "sample.dat"
    # Simple two-column Excel file
    import pandas as pd

    df = pd.DataFrame({"time": [1, 2], "stress": [3.0, 4.0]})
    df.to_excel(excel_path, index=False)

    data = auto_load(excel_path, x_col="time", y_col="stress")
    assert len(data.x) == 2
    assert np.isclose(data.y[1], 4.0)
    # Verify actual column values are correct (guards against pandas index offset)
    np.testing.assert_allclose(data.x, [1.0, 2.0], rtol=1e-10)
    np.testing.assert_allclose(data.y, [3.0, 4.0], rtol=1e-10)


def test_load_csv_quoted_semicolon_with_preamble(tmp_path: Path):
    csv_path = tmp_path / "quoted.csv"
    content = '# header line\n# second\n"Time (s)";"Stress [Pa]"\n"1,000";"10,5"\n'
    _write(csv_path, content, encoding="cp1252")

    data = load_csv(csv_path, x_col=0, y_col=1, delimiter=None, header=2)
    assert np.isclose(data.x[0], 1.0)
    assert np.isclose(data.y[0], 10.5)


def test_load_csv_utf16_tab(tmp_path: Path):
    csv_path = tmp_path / "utf16.tsv"
    content = "time\tstress\n1\t5.5\n"
    _write(csv_path, content, encoding="utf-16le")

    data = load_csv(csv_path, x_col=0, y_col=1, delimiter="\t", header=0)
    assert np.isclose(data.y[0], 5.5)


def test_load_csv_comment_preamble_autodetect(tmp_path: Path):
    """Test automatic detection of # comment preamble lines."""
    csv_path = tmp_path / "preamble.csv"
    csv_path.write_text("# comment1\n# comment2\ntime,stress\n1.0,100\n2.0,200\n")
    data = load_csv(str(csv_path), x_col="time", y_col="stress")
    assert len(data.x) == 2
    np.testing.assert_allclose(data.x, [1.0, 2.0], rtol=1e-10)
    np.testing.assert_allclose(data.y, [100.0, 200.0], rtol=1e-10)


def test_load_csv_semicolon_values(tmp_path: Path):
    """Test that column values are correctly extracted after delimiter detection."""
    csv_path = tmp_path / "simple.csv"
    _write(csv_path, "time,stress\n1.0,3.0\n2.0,4.0\n")

    data = load_csv(csv_path, x_col="time", y_col="stress")
    np.testing.assert_allclose(data.x, [1.0, 2.0], rtol=1e-10)
    np.testing.assert_allclose(data.y, [3.0, 4.0], rtol=1e-10)


def test_to_float_eu_locale_mixed_with_plain_and_thousands():
    """Regression test for two _to_float bugs found in PR #67 review:

    1. A plain US-style decimal ("5.5") in an EU-detected column must not
       have its dot blanket-stripped into 55.0.
    2. A dot-only EU-thousands-grouped value ("1.234" meaning 1234) must not
       be left as the literal decimal 1.234 just because it lacks a comma.
    """
    from rheojax.io.readers.csv_reader import _to_float

    result = _to_float(np.array(["1.234.567,89", "1.234", "42", "5,5"], dtype=object))
    np.testing.assert_allclose(result, [1234567.89, 1234.0, 42.0, 5.5])


def test_load_csv_auto_detected_units_normalized_to_si(tmp_path: Path):
    """load_csv must normalize auto-detected header units to SI, matching
    the anton_paar/TRIOS readers (PR #67), not just extract the raw unit
    string and leave the numeric values unconverted."""
    csv_path = tmp_path / "units.csv"
    _write(csv_path, "Frequency (Hz),Storage Modulus (kPa)\n1.0,10.0\n2.0,20.0\n")

    data = load_csv(csv_path, x_col="Frequency (Hz)", y_col="Storage Modulus (kPa)")

    assert data.x_units == "rad/s"
    np.testing.assert_allclose(data.x, [2 * np.pi, 4 * np.pi])
    assert data.y_units == "Pa"
    np.testing.assert_allclose(data.y, [10000.0, 20000.0])


def test_load_csv_percent_strain_normalized_to_fraction(tmp_path: Path):
    """A '%' unit header must be converted to a dimensionless fraction, not
    left as a raw percentage (a 100x silent error if downstream code assumes
    fractional strain)."""
    csv_path = tmp_path / "strain.csv"
    _write(csv_path, "Time (s),Strain (%)\n1.0,5.0\n2.0,10.0\n")

    data = load_csv(csv_path, x_col="Time (s)", y_col="Strain (%)")

    assert data.y_units == "dimensionless"
    np.testing.assert_allclose(data.y, [0.05, 0.10])
