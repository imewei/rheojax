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


def test_load_csv_quoted_semicolon_with_preamble(tmp_path: Path):
    csv_path = tmp_path / "quoted.csv"
    content = "# header line\n# second\n\"Time (s)\";\"Stress [Pa]\"\n\"1,000\";\"10,5\"\n"
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
