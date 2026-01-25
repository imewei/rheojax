"""Tests for FIKH thermal coupling utilities.

Tests cover:
- Arrhenius viscosity
- Thermal yield stress
- Temperature evolution
- Steady-state temperature
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fikh._thermal import (
    arrhenius_viscosity,
    arrhenius_modulus,
    thermal_yield_stress,
    temperature_evolution_rate,
    steady_state_temperature,
    update_thermal_parameters,
    compute_adiabatic_temperature_rise,
    R_GAS,
)

jax, jnp = safe_import_jax()


class TestArrheniusViscosity:
    """Test Arrhenius viscosity calculation."""

    @pytest.mark.smoke
    def test_reference_temperature_gives_reference_viscosity(self):
        """Test η(T_ref) = η_ref."""
        eta_ref = 1000.0
        T_ref = 300.0
        E_a = 50000.0

        eta = arrhenius_viscosity(eta_ref, jnp.array(T_ref), T_ref, E_a)
        assert jnp.isclose(eta, eta_ref)

    def test_higher_temperature_lower_viscosity(self):
        """Test viscosity decreases with temperature."""
        eta_ref = 1000.0
        T_ref = 300.0
        E_a = 50000.0

        T_high = jnp.array(350.0)
        T_low = jnp.array(250.0)

        eta_high = arrhenius_viscosity(eta_ref, T_high, T_ref, E_a)
        eta_low = arrhenius_viscosity(eta_ref, T_low, T_ref, E_a)

        assert float(eta_high) < float(eta_ref)
        assert float(eta_low) > float(eta_ref)

    def test_zero_activation_energy(self):
        """Test E_a = 0 gives constant viscosity."""
        eta_ref = 1000.0
        T_ref = 300.0
        E_a = 0.0

        for T in [250.0, 300.0, 350.0]:
            eta = arrhenius_viscosity(eta_ref, jnp.array(T), T_ref, E_a)
            assert jnp.isclose(eta, eta_ref)

    def test_typical_polymer_values(self):
        """Test with typical polymer melt parameters."""
        eta_ref = 1e6  # Pa·s
        T_ref = 373.15  # 100°C in K
        E_a = 50e3  # J/mol (typical for polymers)

        # At T_ref + 30K, viscosity should drop significantly
        T_high = jnp.array(403.15)
        eta_high = arrhenius_viscosity(eta_ref, T_high, T_ref, E_a)

        # Viscosity ratio: exp(E_a/R * (1/T_high - 1/T_ref))
        expected_ratio = jnp.exp(E_a / R_GAS * (1 / 403.15 - 1 / 373.15))
        actual_ratio = float(eta_high) / eta_ref

        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-5)


class TestThermalYieldStress:
    """Test thermal yield stress calculation."""

    @pytest.mark.smoke
    def test_reference_conditions(self):
        """Test σ_y at T_ref with λ=1, m_y=1."""
        sigma_y0 = 100.0
        lam = jnp.array(1.0)
        m_y = 1.0
        T = jnp.array(300.0)
        T_ref = 300.0
        E_y = 30000.0

        sigma_y = thermal_yield_stress(sigma_y0, lam, m_y, T, T_ref, E_y)
        assert jnp.isclose(sigma_y, sigma_y0)

    def test_structure_dependence(self):
        """Test yield stress increases with structure parameter."""
        sigma_y0 = 100.0
        m_y = 1.0
        T = jnp.array(300.0)
        T_ref = 300.0
        E_y = 0.0  # No temperature dependence

        lam_high = jnp.array(1.0)
        lam_low = jnp.array(0.5)

        sigma_y_high = thermal_yield_stress(sigma_y0, lam_high, m_y, T, T_ref, E_y)
        sigma_y_low = thermal_yield_stress(sigma_y0, lam_low, m_y, T, T_ref, E_y)

        assert float(sigma_y_high) > float(sigma_y_low)

    def test_temperature_dependence(self):
        """Test yield stress depends on temperature."""
        sigma_y0 = 100.0
        lam = jnp.array(1.0)
        m_y = 1.0
        T_ref = 300.0
        E_y = 30000.0  # Positive E_y means lower σ_y at higher T

        T_high = jnp.array(350.0)
        T_low = jnp.array(250.0)

        sigma_y_high_T = thermal_yield_stress(sigma_y0, lam, m_y, T_high, T_ref, E_y)
        sigma_y_low_T = thermal_yield_stress(sigma_y0, lam, m_y, T_low, T_ref, E_y)

        # Higher T gives lower yield stress for E_y > 0
        assert float(sigma_y_high_T) < float(sigma_y_low_T)


class TestTemperatureEvolution:
    """Test temperature evolution rate calculation."""

    @pytest.mark.smoke
    def test_no_shear_no_heating(self):
        """Test dT/dt = 0 when γ̇ᵖ = 0 and T = T_env."""
        T = jnp.array(300.0)
        sigma = jnp.array(100.0)
        gamma_dot_p = jnp.array(0.0)
        T_env = 300.0
        rho_cp = 4e6
        chi = 0.9
        h = 100.0

        dT_dt = temperature_evolution_rate(T, sigma, gamma_dot_p, T_env, rho_cp, chi, h)
        assert jnp.isclose(dT_dt, 0.0)

    def test_viscous_heating(self):
        """Test viscous heating increases temperature."""
        T = jnp.array(300.0)
        sigma = jnp.array(1000.0)  # 1 kPa
        gamma_dot_p = jnp.array(10.0)  # 10 /s
        T_env = 300.0
        rho_cp = 4e6
        chi = 0.9
        h = 0.0  # No cooling

        dT_dt = temperature_evolution_rate(T, sigma, gamma_dot_p, T_env, rho_cp, chi, h)

        # dT/dt = χ·σ·γ̇ᵖ / (ρ·c_p)
        expected = chi * 1000.0 * 10.0 / rho_cp
        assert jnp.isclose(dT_dt, expected)

    def test_cooling(self):
        """Test convective cooling decreases temperature."""
        T = jnp.array(350.0)  # Above environment
        sigma = jnp.array(0.0)  # No stress
        gamma_dot_p = jnp.array(0.0)
        T_env = 300.0
        rho_cp = 4e6
        chi = 0.9
        h = 1000.0  # Strong cooling

        dT_dt = temperature_evolution_rate(T, sigma, gamma_dot_p, T_env, rho_cp, chi, h)

        # dT/dt = -h·(T-T_env) / (ρ·c_p)
        expected = -h * (350.0 - 300.0) / rho_cp
        assert jnp.isclose(dT_dt, expected)


class TestSteadyStateTemperature:
    """Test steady-state temperature calculation."""

    @pytest.mark.smoke
    def test_no_shear_ambient(self):
        """Test T_ss = T_env when no shear."""
        sigma = jnp.array(0.0)
        gamma_dot_p = jnp.array(0.0)
        T_env = 300.0
        chi = 0.9
        h = 100.0

        T_ss = steady_state_temperature(sigma, gamma_dot_p, T_env, chi, h)
        assert jnp.isclose(T_ss, T_env)

    def test_heating_above_ambient(self):
        """Test T_ss > T_env with viscous heating."""
        sigma = jnp.array(1000.0)
        gamma_dot_p = jnp.array(10.0)
        T_env = 300.0
        chi = 0.9
        h = 100.0

        T_ss = steady_state_temperature(sigma, gamma_dot_p, T_env, chi, h)

        # T_ss = T_env + χ·σ·γ̇ᵖ / h
        expected = T_env + chi * 1000.0 * 10.0 / h
        assert jnp.isclose(T_ss, expected)


class TestAdiabaticTemperatureRise:
    """Test adiabatic temperature rise calculation."""

    @pytest.mark.smoke
    def test_zero_strain_no_rise(self):
        """Test ΔT = 0 when γ = 0."""
        gamma_total = jnp.array(0.0)
        sigma_avg = jnp.array(1000.0)
        rho_cp = 4e6
        chi = 0.9

        delta_T = compute_adiabatic_temperature_rise(gamma_total, sigma_avg, rho_cp, chi)
        assert jnp.isclose(delta_T, 0.0)

    def test_adiabatic_rise_formula(self):
        """Test adiabatic formula: ΔT = χ·σ·γ / (ρ·c_p)."""
        gamma_total = jnp.array(10.0)
        sigma_avg = jnp.array(1000.0)
        rho_cp = 4e6
        chi = 0.9

        delta_T = compute_adiabatic_temperature_rise(gamma_total, sigma_avg, rho_cp, chi)

        expected = chi * 1000.0 * 10.0 / rho_cp
        assert jnp.isclose(delta_T, expected)


class TestUpdateThermalParameters:
    """Test thermal parameter update utility."""

    @pytest.mark.smoke
    def test_updates_viscosity(self):
        """Test update_thermal_parameters adds eta_T."""
        params = {
            "eta": 1e6,
            "G": 1e3,
            "sigma_y0": 100.0,
            "T_ref": 300.0,
            "E_a": 50000.0,
            "E_y": 30000.0,
            "m_y": 1.0,
        }
        T = jnp.array(350.0)

        updated = update_thermal_parameters(params, T)

        assert "eta_T" in updated
        # eta_T should be different from eta (T ≠ T_ref)
        assert not jnp.isclose(updated["eta_T"], params["eta"])
