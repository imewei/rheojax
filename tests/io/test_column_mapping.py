"""Tests for the consolidated canonical column field registry."""

from __future__ import annotations

import pytest

from rheojax.io.readers._column_mapping import (
    CANONICAL_FIELDS,
    CanonicalField,
    match_column,
    match_columns,
)

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_time_exact():
    cf = match_column("time")
    assert cf is not None
    assert cf.canonical_name == "time"


@pytest.mark.smoke
def test_match_time_t():
    cf = match_column("t")
    assert cf is not None
    assert cf.canonical_name == "time"


@pytest.mark.smoke
def test_match_time_zeit():
    cf = match_column("Zeit")
    assert cf is not None
    assert cf.canonical_name == "time"


@pytest.mark.smoke
def test_match_time_step_time():
    cf = match_column("Step time")
    assert cf is not None
    assert cf.canonical_name == "time"


# ---------------------------------------------------------------------------
# Angular frequency
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_frequency_angular_frequency():
    cf = match_column("Angular Frequency")
    assert cf is not None
    assert cf.canonical_name == "angular_frequency"


@pytest.mark.smoke
def test_match_frequency_omega():
    cf = match_column("omega")
    assert cf is not None
    assert cf.canonical_name == "angular_frequency"


@pytest.mark.smoke
def test_match_frequency_unicode_omega():
    cf = match_column("ω")
    assert cf is not None
    assert cf.canonical_name == "angular_frequency"


@pytest.mark.smoke
def test_match_frequency_frequency():
    cf = match_column("Frequency")
    assert cf is not None
    assert cf.canonical_name == "angular_frequency"


# ---------------------------------------------------------------------------
# Shear moduli G'/G''
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_storage_modulus_gprime_symbol():
    cf = match_column("G'")
    assert cf is not None
    assert cf.canonical_name == "storage_modulus"


@pytest.mark.smoke
def test_match_storage_modulus_text():
    cf = match_column("Storage Modulus")
    assert cf is not None
    assert cf.canonical_name == "storage_modulus"


@pytest.mark.smoke
def test_match_loss_modulus_gdoubleprime():
    cf = match_column("G''")
    assert cf is not None
    assert cf.canonical_name == "loss_modulus"


@pytest.mark.smoke
def test_match_loss_modulus_text():
    cf = match_column("Loss Modulus")
    assert cf is not None
    assert cf.canonical_name == "loss_modulus"


# ---------------------------------------------------------------------------
# Tensile moduli E'/E'' (DMTA)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_tensile_storage_eprime():
    cf = match_column("E'")
    assert cf is not None
    assert cf.canonical_name == "tensile_storage_modulus"


@pytest.mark.smoke
def test_match_tensile_storage_estor():
    cf = match_column("E_stor")
    assert cf is not None
    assert cf.canonical_name == "tensile_storage_modulus"


@pytest.mark.smoke
def test_match_tensile_loss_edoubleprime():
    cf = match_column("E''")
    assert cf is not None
    assert cf.canonical_name == "tensile_loss_modulus"


@pytest.mark.smoke
def test_match_tensile_loss_eloss():
    cf = match_column("E_loss")
    assert cf is not None
    assert cf.canonical_name == "tensile_loss_modulus"


# ---------------------------------------------------------------------------
# Viscosity — word-boundary safety
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_viscosity_text():
    cf = match_column("Viscosity")
    assert cf is not None
    assert cf.canonical_name == "viscosity"


@pytest.mark.smoke
def test_match_viscosity_eta_unicode():
    cf = match_column("η")
    assert cf is not None
    assert cf.canonical_name == "viscosity"


@pytest.mark.smoke
def test_match_viscosity_eta_ascii():
    cf = match_column("eta")
    assert cf is not None
    assert cf.canonical_name == "viscosity"


@pytest.mark.smoke
def test_no_match_theta():
    # "theta" must NOT match viscosity
    cf = match_column("theta")
    assert cf is None or cf.canonical_name != "viscosity"


@pytest.mark.smoke
def test_no_match_beta():
    cf = match_column("beta")
    assert cf is None or cf.canonical_name != "viscosity"


@pytest.mark.smoke
def test_no_match_zeta():
    cf = match_column("zeta")
    assert cf is None or cf.canonical_name != "viscosity"


# ---------------------------------------------------------------------------
# Shear rate
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_shear_rate_text():
    cf = match_column("Shear Rate")
    assert cf is not None
    assert cf.canonical_name == "shear_rate"


@pytest.mark.smoke
def test_match_shear_rate_unicode():
    cf = match_column("γ̇")
    assert cf is not None
    assert cf.canonical_name == "shear_rate"


@pytest.mark.smoke
def test_match_shear_rate_gamma_dot():
    cf = match_column("gamma dot")
    assert cf is not None
    assert cf.canonical_name == "shear_rate"


# ---------------------------------------------------------------------------
# Header with unit suffix
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_with_unit_suffix_parens():
    # "omega (rad/s)" should still match angular_frequency
    cf = match_column("omega (rad/s)")
    assert cf is not None
    assert cf.canonical_name == "angular_frequency"


@pytest.mark.smoke
def test_match_time_with_unit_suffix():
    cf = match_column("time (s)")
    assert cf is not None
    assert cf.canonical_name == "time"


# ---------------------------------------------------------------------------
# No match
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_no_match_random_column():
    cf = match_column("random_column")
    assert cf is None


@pytest.mark.smoke
def test_no_match_empty_ish():
    cf = match_column("   ")
    assert cf is None


# ---------------------------------------------------------------------------
# match_columns (batch)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_match_columns_basic():
    headers = ["time", "G'", "G''", "Temperature", "unknown_col"]
    result = match_columns(headers)
    assert "time" in result
    assert result["time"].canonical_name == "time"
    assert "G'" in result
    assert result["G'"].canonical_name == "storage_modulus"
    assert "G''" in result
    assert result["G''"].canonical_name == "loss_modulus"
    assert "Temperature" in result
    assert result["Temperature"].canonical_name == "temperature"
    # Unknown column should be absent from result
    assert "unknown_col" not in result


@pytest.mark.smoke
def test_match_columns_empty():
    assert match_columns([]) == {}


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_angular_frequency_higher_priority_than_time():
    # angular_frequency priority=5, time priority=10 → angular_frequency wins first
    omega_field = CANONICAL_FIELDS["angular_frequency"]
    time_field = CANONICAL_FIELDS["time"]
    assert omega_field.priority < time_field.priority


# ---------------------------------------------------------------------------
# is_x_candidate / is_y_candidate flags
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_time_is_x_candidate():
    assert CANONICAL_FIELDS["time"].is_x_candidate is True
    assert CANONICAL_FIELDS["time"].is_y_candidate is False


@pytest.mark.smoke
def test_angular_frequency_is_x_candidate():
    assert CANONICAL_FIELDS["angular_frequency"].is_x_candidate is True


@pytest.mark.smoke
def test_storage_modulus_is_y_candidate():
    assert CANONICAL_FIELDS["storage_modulus"].is_y_candidate is True
    assert CANONICAL_FIELDS["storage_modulus"].is_x_candidate is False


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_case_insensitive_uppercase():
    cf = match_column("TIME")
    assert cf is not None
    assert cf.canonical_name == "time"


@pytest.mark.smoke
def test_case_insensitive_mixed():
    cf = match_column("Loss MODULUS")
    assert cf is not None
    assert cf.canonical_name == "loss_modulus"


# ---------------------------------------------------------------------------
# CanonicalField dataclass
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_canonical_field_is_dataclass():
    cf = CANONICAL_FIELDS["time"]
    assert isinstance(cf, CanonicalField)
    assert isinstance(cf.patterns, list)
    assert isinstance(cf.si_unit, str)
    assert isinstance(cf.applicable_modes, list)
