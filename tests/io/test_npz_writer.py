"""Tests for NumPy .npz writer/reader (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io.writers.npz_writer import load_npz, save_npz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    *,
    n: int = 5,
    complex_y: bool = False,
    x_units: str | None = "s",
    y_units: str | None = "Pa",
    domain: str | None = "time",
    initial_test_mode: str | None = "relaxation",
    metadata: dict | None = None,
) -> RheoData:
    x = np.linspace(0.1, 1.0, n)
    if complex_y:
        y = np.array(
            [100.0 + 50.0j, 90.0 + 45.0j, 80.0 + 40.0j, 70.0 + 35.0j, 60.0 + 30.0j]
        )
    else:
        y = np.linspace(100.0, 60.0, n)
    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        initial_test_mode=initial_test_mode,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_roundtrip_real(tmp_path):
    """Save and load real y data; verify all fields round-trip correctly."""
    data = _make_data()
    path = tmp_path / "real.npz"
    save_npz(data, path)
    loaded = load_npz(path)

    np.testing.assert_allclose(loaded.x, data.x)
    np.testing.assert_allclose(loaded.y, data.y)
    assert loaded.x_units == "s"
    assert loaded.y_units == "Pa"
    assert loaded.domain == "time"
    assert loaded._explicit_test_mode == "relaxation"


@pytest.mark.smoke
def test_roundtrip_complex(tmp_path):
    """Save and load complex y data (G* = G' + iG'')."""
    data = _make_data(
        complex_y=True,
        x_units="rad/s",
        domain="frequency",
        initial_test_mode="oscillation",
    )
    path = tmp_path / "complex.npz"
    save_npz(data, path)
    loaded = load_npz(path)

    np.testing.assert_allclose(loaded.y.real, data.y.real)
    np.testing.assert_allclose(loaded.y.imag, data.y.imag)
    assert np.iscomplexobj(loaded.y)


@pytest.mark.smoke
def test_roundtrip_metadata(tmp_path):
    """Rich metadata dict survives save/load round-trip."""
    meta = {
        "sample_name": "PDMS-50k",
        "temperature": 298.15,
        "n_modes": 3,
        "is_calibrated": True,
        "geometry": {"gap": 1.0, "diameter": 25.0},
        "tags": ["oscillation", "SAOS"],
    }
    data = _make_data(metadata=meta)
    path = tmp_path / "meta.npz"
    save_npz(data, path)
    loaded = load_npz(path)

    assert loaded.metadata["sample_name"] == "PDMS-50k"
    assert loaded.metadata["temperature"] == pytest.approx(298.15)
    assert loaded.metadata["n_modes"] == 3
    assert loaded.metadata["is_calibrated"] is True
    assert loaded.metadata["geometry"]["gap"] == pytest.approx(1.0)
    assert loaded.metadata["tags"] == ["oscillation", "SAOS"]


@pytest.mark.smoke
def test_compressed_default(tmp_path):
    """compressed=True is the default; resulting file must be loadable."""
    data = _make_data()
    path = tmp_path / "default.npz"
    save_npz(data, path)  # no explicit compressed kwarg
    loaded = load_npz(path)
    np.testing.assert_allclose(loaded.x, data.x)


@pytest.mark.smoke
def test_uncompressed(tmp_path):
    """compressed=False produces a valid loadable archive."""
    data = _make_data()
    path = tmp_path / "uncompressed.npz"
    save_npz(data, path, compressed=False)
    loaded = load_npz(path)
    np.testing.assert_allclose(loaded.x, data.x)
    np.testing.assert_allclose(loaded.y, data.y)


@pytest.mark.smoke
def test_missing_file(tmp_path):
    """load_npz raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "nonexistent.npz")


@pytest.mark.smoke
def test_roundtrip_none_units(tmp_path):
    """x_units=None and y_units=None are preserved as None after round-trip."""
    data = _make_data(x_units=None, y_units=None)
    path = tmp_path / "no_units.npz"
    save_npz(data, path)
    loaded = load_npz(path)
    assert loaded.x_units is None
    assert loaded.y_units is None


@pytest.mark.smoke
def test_roundtrip_temperature(tmp_path):
    """Metadata with a temperature key survives round-trip."""
    data = _make_data(metadata={"temperature": 303.15})
    path = tmp_path / "temp.npz"
    save_npz(data, path)
    loaded = load_npz(path)
    assert loaded.metadata["temperature"] == pytest.approx(303.15)


@pytest.mark.smoke
def test_roundtrip_large_metadata(tmp_path):
    """Metadata with nested dicts and lists survives round-trip."""
    meta = {
        "level1": {
            "level2": {
                "value": 42,
                "array": [1, 2, 3, 4, 5],
            }
        },
        "tags": ["a", "b", "c"],
        "floats": [1.1, 2.2, 3.3],
    }
    data = _make_data(metadata=meta)
    path = tmp_path / "nested.npz"
    save_npz(data, path)
    loaded = load_npz(path)
    assert loaded.metadata["level1"]["level2"]["value"] == 42
    assert loaded.metadata["level1"]["level2"]["array"] == [1, 2, 3, 4, 5]
    assert loaded.metadata["tags"] == ["a", "b", "c"]
    assert loaded.metadata["floats"] == pytest.approx([1.1, 2.2, 3.3])


@pytest.mark.smoke
def test_save_creates_parent_dirs(tmp_path):
    """save_npz creates missing parent directories automatically."""
    data = _make_data()
    nested_path = tmp_path / "level1" / "level2" / "output.npz"
    assert not nested_path.parent.exists()
    save_npz(data, nested_path)
    assert nested_path.exists()
    loaded = load_npz(nested_path)
    np.testing.assert_allclose(loaded.x, data.x)
