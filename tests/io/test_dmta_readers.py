"""Tests for DMTA/DMA deformation mode detection in IO readers.

Tests column pattern detection, deformation_mode propagation to RheoData,
and pyvisco-style CSV format support.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.io.readers._utils import (
    detect_deformation_mode_from_columns,
    detect_test_mode_from_columns,
)

# ── Column-level deformation mode detection ──────────────────────────────


@pytest.mark.smoke
def test_detect_tension_from_e_prime():
    """E' / E'' columns should detect tension mode."""
    result = detect_deformation_mode_from_columns(["E' (Pa)", 'E" (Pa)'])
    assert result == "tension"


@pytest.mark.smoke
def test_detect_tension_from_e_stor_loss():
    """E_stor / E_loss columns (pyvisco style) should detect tension mode."""
    result = detect_deformation_mode_from_columns(["E_stor", "E_loss"])
    assert result == "tension"


@pytest.mark.smoke
def test_detect_shear_from_g_prime():
    """G' / G'' columns should detect shear mode."""
    result = detect_deformation_mode_from_columns(["G' (Pa)", 'G" (Pa)'])
    assert result == "shear"


@pytest.mark.smoke
def test_detect_none_for_generic_columns():
    """Generic columns (stress, strain) should return None."""
    result = detect_deformation_mode_from_columns(["stress (Pa)", "strain"])
    assert result is None


@pytest.mark.smoke
def test_detect_tension_from_units():
    """E* in units string should help detect tension."""
    result = detect_deformation_mode_from_columns(
        ["Storage Modulus", "Loss Modulus"], y_units="E* (Pa)"
    )
    assert result == "tension"


# ── Test mode detection with E' columns ──────────────────────────────────


@pytest.mark.smoke
def test_oscillation_detected_from_e_prime():
    """E' columns should still detect oscillation test mode."""
    result = detect_test_mode_from_columns("frequency (Hz)", ["E' (Pa)", 'E" (Pa)'])
    assert result == "oscillation"


@pytest.mark.smoke
def test_oscillation_detected_from_e_stor():
    """E_stor columns should detect oscillation test mode."""
    result = detect_test_mode_from_columns("f", ["E_stor", "E_loss"])
    assert result == "oscillation"


# ── CSV reader with deformation_mode ─────────────────────────────────────


@pytest.mark.smoke
def test_csv_explicit_deformation_mode():
    """load_csv with explicit deformation_mode='tension' sets metadata."""
    from rheojax.io.readers.csv_reader import load_csv

    # Create synthetic DMTA CSV data
    omega = np.logspace(-1, 2, 20)
    E_prime = 3e9 * omega**2 / (1 + omega**2)
    E_double_prime = 3e9 * omega / (1 + omega**2)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("omega (rad/s),E' (Pa),E'' (Pa)\n")
        for w, ep, epp in zip(omega, E_prime, E_double_prime):
            f.write(f"{w},{ep},{epp}\n")
        fpath = f.name

    data = load_csv(
        fpath,
        x_col="omega (rad/s)",
        y_cols=["E' (Pa)", "E'' (Pa)"],
        deformation_mode="tension",
    )

    assert data.metadata["deformation_mode"] == "tension"
    assert np.iscomplexobj(data.y)
    assert len(data.x) == 20
    Path(fpath).unlink()


@pytest.mark.smoke
def test_csv_auto_detect_deformation_mode():
    """load_csv auto-detects tension from E' column names."""
    from rheojax.io.readers.csv_reader import load_csv

    omega = np.logspace(-1, 2, 15)
    E_prime = 1e6 * np.ones_like(omega)
    E_double_prime = 1e5 * np.ones_like(omega)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("omega (rad/s),E' (Pa),E'' (Pa)\n")
        for w, ep, epp in zip(omega, E_prime, E_double_prime):
            f.write(f"{w},{ep},{epp}\n")
        fpath = f.name

    data = load_csv(
        fpath,
        x_col="omega (rad/s)",
        y_cols=["E' (Pa)", "E'' (Pa)"],
    )

    # Auto-detected from column names
    assert data.metadata.get("deformation_mode") == "tension"
    Path(fpath).unlink()


@pytest.mark.smoke
def test_csv_pyvisco_format():
    """pyvisco-style CSV (f, E_stor, E_loss, T, Set) should load correctly."""
    from rheojax.io.readers.csv_reader import load_csv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("f,E_stor,E_loss,T,Set\n")
        # 5 frequency points
        for freq in [0.1, 1.0, 10.0, 50.0, 100.0]:
            e_stor = 1e3 * (1 + freq**0.5)
            e_loss = 1e2 * freq**0.3
            f.write(f"{freq},{e_stor},{e_loss},25.0,1\n")
        fpath = f.name

    data = load_csv(
        fpath,
        x_col="f",
        y_cols=["E_stor", "E_loss"],
        deformation_mode="tension",
    )

    assert data.metadata["deformation_mode"] == "tension"
    assert np.iscomplexobj(data.y)
    assert len(data.x) == 5
    Path(fpath).unlink()


@pytest.mark.smoke
def test_csv_shear_columns_no_deformation_mode():
    """G'/G'' columns should auto-detect shear (or no deformation_mode set)."""
    from rheojax.io.readers.csv_reader import load_csv

    omega = np.logspace(-1, 2, 10)
    G_prime = 1e6 * np.ones_like(omega)
    G_double_prime = 1e5 * np.ones_like(omega)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("omega (rad/s),G' (Pa),G'' (Pa)\n")
        for w, gp, gpp in zip(omega, G_prime, G_double_prime):
            f.write(f"{w},{gp},{gpp}\n")
        fpath = f.name

    data = load_csv(
        fpath,
        x_col="omega (rad/s)",
        y_cols=["G' (Pa)", "G'' (Pa)"],
    )

    # Should detect shear
    assert data.metadata.get("deformation_mode") == "shear"
    Path(fpath).unlink()


# ── TRIOS reader tensile modulus detection ────────────────────────────────


@pytest.mark.smoke
def test_trios_tensile_column_mapping():
    """TRIOS column mappings include tensile modulus entries."""
    from rheojax.io.readers.trios.common import TRIOS_COLUMN_MAPPINGS

    assert "tensile_storage_modulus" in TRIOS_COLUMN_MAPPINGS
    assert "tensile_loss_modulus" in TRIOS_COLUMN_MAPPINGS
    tsm = TRIOS_COLUMN_MAPPINGS["tensile_storage_modulus"]
    assert tsm.is_y_candidate is True
    assert "oscillation" in tsm.applicable_modes


@pytest.mark.smoke
def test_trios_detect_oscillation_from_tensile():
    """TRIOS detect_test_type recognises E'/E'' as oscillation."""
    import pandas as pd

    from rheojax.io.readers.trios.common import detect_test_type

    df = pd.DataFrame({"angular_frequency": [1.0], "e'": [1e6], "e''": [1e5]})
    assert detect_test_type(df) == "oscillation"


@pytest.mark.smoke
def test_trios_select_xy_tensile_columns():
    """TRIOS select_xy_columns picks E'/E'' for complex modulus."""
    import pandas as pd

    from rheojax.io.readers.trios.common import select_xy_columns

    df = pd.DataFrame(
        {
            "angular_frequency": [1.0, 10.0],
            "e'": [1e6, 2e6],
            "e''": [1e5, 2e5],
        }
    )
    x_col, y_col, y2_col = select_xy_columns(df, "oscillation")
    assert x_col == "angular_frequency"
    # y_col and y2_col should be the E' and E'' columns
    assert y2_col is not None  # Complex modulus detected


# ── Anton Paar reader tensile modulus detection ───────────────────────────


@pytest.mark.smoke
def test_anton_paar_tensile_column_mapping():
    """Anton Paar COLUMN_MAPPINGS include tensile modulus entries."""
    from rheojax.io.readers.anton_paar import COLUMN_MAPPINGS

    assert "tensile_storage_modulus" in COLUMN_MAPPINGS
    assert "tensile_loss_modulus" in COLUMN_MAPPINGS
    patterns, unit, modes = COLUMN_MAPPINGS["tensile_storage_modulus"]
    assert unit == "Pa"
    assert "oscillation" in modes
