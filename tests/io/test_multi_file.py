"""Tests for multi_file loaders: load_tts, load_srfs, load_series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rheojax.core.data import RheoData
from rheojax.io.readers.multi_file import load_series, load_srfs, load_tts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, name: str, temperature: float | None = None) -> Path:
    """Write a minimal time/stress CSV and return its path."""
    p = tmp_path / name
    df = pd.DataFrame({"time": [0.1, 0.2, 0.3], "stress": [100.0, 90.0, 80.0]})
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_tts tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_load_tts_basic(tmp_path):
    """Three files with explicit temperatures in Kelvin."""
    paths = [
        _make_csv(tmp_path, "T1.csv"),
        _make_csv(tmp_path, "T2.csv"),
        _make_csv(tmp_path, "T3.csv"),
    ]
    temps = [300.0, 320.0, 340.0]
    results = load_tts(
        paths, T_ref=300.0, temperatures=temps, x_col="time", y_col="stress"
    )
    assert len(results) == 3
    for rd in results:
        assert isinstance(rd, RheoData)
        assert rd.metadata["T_ref"] == 300.0
    # Sorted ascending
    assert results[0].metadata["temperature"] == pytest.approx(300.0)
    assert results[1].metadata["temperature"] == pytest.approx(320.0)
    assert results[2].metadata["temperature"] == pytest.approx(340.0)


@pytest.mark.smoke
def test_load_tts_celsius(tmp_path):
    """Temperatures provided in Celsius are converted to Kelvin."""
    paths = [_make_csv(tmp_path, "A.csv"), _make_csv(tmp_path, "B.csv")]
    results = load_tts(
        paths,
        T_ref=298.15,
        temperatures=[25.0, 50.0],
        temperature_unit="C",
        x_col="time",
        y_col="stress",
    )
    assert results[0].metadata["temperature"] == pytest.approx(298.15)
    assert results[1].metadata["temperature"] == pytest.approx(323.15)


@pytest.mark.smoke
def test_load_tts_auto_extract(tmp_path):
    """temperatures=None extracts temperature from file metadata."""
    # Manually craft files and patch metadata after load by pre-seeding metadata.
    # We do this by having the file read succeed and then relying on the fact that
    # auto_load injects a metadata dict — we write temperature into the CSV
    # itself via a custom approach: instead, we monkeypatch auto_load.
    import unittest.mock as mock

    def _make_rd(temp: float) -> RheoData:
        return RheoData(
            x=np.array([0.1, 0.2]),
            y=np.array([100.0, 90.0]),
            metadata={"temperature": temp, "temperature_unit": "K"},
        )

    paths = [tmp_path / "f1.csv", tmp_path / "f2.csv"]
    for p in paths:
        p.write_text("time,stress\n0.1,100\n0.2,90\n")

    with mock.patch(
        "rheojax.io.readers.multi_file.auto_load",
        side_effect=[_make_rd(310.0), _make_rd(290.0)],
    ):
        results = load_tts(paths, T_ref=300.0)

    assert len(results) == 2
    # Sorted: 290 first, then 310
    assert results[0].metadata["temperature"] == pytest.approx(290.0)
    assert results[1].metadata["temperature"] == pytest.approx(310.0)


@pytest.mark.smoke
def test_load_tts_mismatch(tmp_path):
    """Mismatched temperatures length raises ValueError."""
    paths = [_make_csv(tmp_path, "X.csv"), _make_csv(tmp_path, "Y.csv")]
    with pytest.raises(ValueError, match="does not match"):
        load_tts(paths, T_ref=300.0, temperatures=[300.0], x_col="time", y_col="stress")


@pytest.mark.smoke
def test_load_tts_sorted(tmp_path):
    """Output is sorted by temperature ascending regardless of input order."""
    paths = [
        _make_csv(tmp_path, "hot.csv"),
        _make_csv(tmp_path, "cold.csv"),
        _make_csv(tmp_path, "mid.csv"),
    ]
    results = load_tts(
        paths,
        T_ref=300.0,
        temperatures=[400.0, 250.0, 325.0],
        x_col="time",
        y_col="stress",
    )
    temps = [r.metadata["temperature"] for r in results]
    assert temps == sorted(temps)


@pytest.mark.smoke
def test_load_tts_glob(tmp_path):
    """Glob pattern expands and loads all matching files."""
    for name in ["t_300.csv", "t_320.csv", "t_340.csv"]:
        _make_csv(tmp_path, name)

    pattern = str(tmp_path / "t_*.csv")
    results = load_tts(
        pattern,
        T_ref=300.0,
        temperatures=[300.0, 320.0, 340.0],
        x_col="time",
        y_col="stress",
    )
    assert len(results) == 3
    # Verify temperature ordering is ascending (not just alphabetical file order)
    temps = [r.metadata["temperature"] for r in results]
    assert temps == sorted(temps)


# ---------------------------------------------------------------------------
# load_srfs tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_load_srfs_basic(tmp_path):
    """Three files with explicit gamma_dots."""
    paths = [
        _make_csv(tmp_path, "g1.csv"),
        _make_csv(tmp_path, "g2.csv"),
        _make_csv(tmp_path, "g3.csv"),
    ]
    gamma_dots = [0.1, 1.0, 10.0]
    results = load_srfs(paths, gamma_dots, x_col="time", y_col="stress")
    assert len(results) == 3
    for rd in results:
        assert isinstance(rd, RheoData)
        assert "reference_gamma_dot" in rd.metadata
    gamma_values = [r.metadata["reference_gamma_dot"] for r in results]
    assert gamma_values == sorted(gamma_values)


@pytest.mark.smoke
def test_load_srfs_sorted(tmp_path):
    """Output is sorted by gamma_dot ascending regardless of input order."""
    paths = [_make_csv(tmp_path, "a.csv"), _make_csv(tmp_path, "b.csv")]
    results = load_srfs(paths, [10.0, 0.5], x_col="time", y_col="stress")
    assert results[0].metadata["reference_gamma_dot"] == pytest.approx(0.5)
    assert results[1].metadata["reference_gamma_dot"] == pytest.approx(10.0)


@pytest.mark.smoke
def test_load_srfs_mismatch(tmp_path):
    """Mismatched gamma_dot count raises ValueError."""
    paths = [_make_csv(tmp_path, "m1.csv"), _make_csv(tmp_path, "m2.csv")]
    with pytest.raises(ValueError, match="does not match"):
        load_srfs(paths, [0.1], x_col="time", y_col="stress")


# ---------------------------------------------------------------------------
# load_series tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_load_series_basic(tmp_path):
    """Generic loader tags all files with protocol."""
    paths = [_make_csv(tmp_path, "s1.csv"), _make_csv(tmp_path, "s2.csv")]
    results = load_series(paths, "relaxation", x_col="time", y_col="stress")
    assert len(results) == 2
    for rd in results:
        assert isinstance(rd, RheoData)
        assert rd.metadata["protocol"] == "relaxation"


@pytest.mark.smoke
def test_load_series_with_metadata(tmp_path):
    """metadata_key and metadata_values tag each file."""
    paths = [
        _make_csv(tmp_path, "c1.csv"),
        _make_csv(tmp_path, "c2.csv"),
        _make_csv(tmp_path, "c3.csv"),
    ]
    results = load_series(
        paths,
        "creep",
        metadata_key="strain_level",
        metadata_values=[0.01, 0.05, 0.10],
        x_col="time",
        y_col="stress",
    )
    assert results[0].metadata["strain_level"] == pytest.approx(0.01)
    assert results[1].metadata["strain_level"] == pytest.approx(0.05)
    assert results[2].metadata["strain_level"] == pytest.approx(0.10)


@pytest.mark.smoke
def test_load_series_sort_by(tmp_path):
    """sort_by sorts output by the given metadata key."""
    paths = [_make_csv(tmp_path, "r1.csv"), _make_csv(tmp_path, "r2.csv")]
    results = load_series(
        paths,
        "oscillation",
        metadata_key="amplitude",
        metadata_values=[5.0, 1.0],
        sort_by="amplitude",
        x_col="time",
        y_col="stress",
    )
    assert results[0].metadata["amplitude"] == pytest.approx(1.0)
    assert results[1].metadata["amplitude"] == pytest.approx(5.0)


@pytest.mark.smoke
def test_load_series_metadata_key_without_values_raises(tmp_path):
    """metadata_key without metadata_values raises ValueError."""
    paths = [_make_csv(tmp_path, "e1.csv")]
    with pytest.raises(ValueError, match="metadata_values"):
        load_series(paths, "flow", metadata_key="rate", x_col="time", y_col="stress")


@pytest.mark.smoke
def test_load_series_metadata_values_mismatch(tmp_path):
    """metadata_values wrong length raises ValueError."""
    paths = [_make_csv(tmp_path, "v1.csv"), _make_csv(tmp_path, "v2.csv")]
    with pytest.raises(ValueError, match="does not match"):
        load_series(
            paths,
            "flow",
            metadata_key="rate",
            metadata_values=[1.0],
            x_col="time",
            y_col="stress",
        )
