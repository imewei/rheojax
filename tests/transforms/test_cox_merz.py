"""Tests for CoxMerz transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.cox_merz import CoxMerz


@pytest.mark.smoke
class TestCoxMerz:
    """Tests for the Cox-Merz rule validation transform."""

    def _make_matching_data(self):
        """Create oscillation + flow data that satisfy Cox-Merz exactly."""
        # Maxwell model: η_0 = 1000 Pa·s, τ = 1 s
        eta_0 = 1000.0
        tau = 1.0

        omega = np.logspace(-2, 2, 50)
        gamma_dot = np.logspace(-2, 2, 50)

        # |η*(ω)| = η_0 / sqrt(1 + ω²τ²)
        eta_star = eta_0 / np.sqrt(1.0 + (omega * tau) ** 2)
        # G* = η* · iω
        G_star = 1j * omega * eta_star  # simplified: G' + iG''
        # More precisely for Maxwell:
        G_prime = eta_0 * omega**2 * tau / (1.0 + (omega * tau) ** 2)
        G_double_prime = eta_0 * omega / (1.0 + (omega * tau) ** 2)
        G_star = G_prime + 1j * G_double_prime

        # Steady shear: η(γ̇) = η_0 (for Newtonian case)
        # For Cox-Merz to hold exactly, η(γ̇) = |η*(ω)| at ω = γ̇
        eta_steady = eta_0 / np.sqrt(1.0 + (gamma_dot * tau) ** 2)

        osc_data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})
        flow_data = RheoData(
            x=gamma_dot,
            y=eta_steady,
            metadata={"test_mode": "flow_curve", "is_viscosity": True},
        )
        return osc_data, flow_data

    def test_perfect_match_passes(self):
        """Cox-Merz should pass for data that satisfies the rule."""
        osc, flow = self._make_matching_data()
        transform = CoxMerz(tolerance=0.1)
        result, meta = transform.transform([osc, flow])

        assert "cox_merz_result" in meta
        cm = meta["cox_merz_result"]
        assert cm.passes is True
        assert cm.mean_deviation < 0.1

    def test_large_deviation_fails(self):
        """Cox-Merz should fail when data diverges significantly."""
        osc, flow = self._make_matching_data()
        # Corrupt flow data by a factor of 10
        flow_bad = RheoData(
            x=flow.x,
            y=np.asarray(flow.y) * 10.0,
            metadata={"test_mode": "flow_curve", "is_viscosity": True},
        )
        transform = CoxMerz(tolerance=0.1)
        result, meta = transform.transform([osc, flow_bad])
        assert meta["cox_merz_result"].passes is False

    def test_wrong_input_count(self):
        """Should raise ValueError for wrong number of inputs."""
        data = RheoData(x=np.array([1.0]), y=np.array([1.0]))
        transform = CoxMerz()
        with pytest.raises(ValueError, match="exactly 2"):
            transform.transform([data])

    def test_no_overlap_raises(self):
        """Should raise ValueError when frequency ranges don't overlap."""
        osc = RheoData(
            x=np.logspace(0, 2, 20),
            y=np.ones(20) + 1j * np.ones(20),
        )
        flow = RheoData(
            x=np.logspace(-4, -2, 20),
            y=np.ones(20),
            metadata={"is_viscosity": True},
        )
        transform = CoxMerz()
        with pytest.raises(ValueError, match="No overlapping"):
            transform.transform([osc, flow])

    def test_stress_to_viscosity_conversion(self):
        """When flow data is stress, it should be auto-converted to viscosity."""
        osc, flow = self._make_matching_data()
        gamma_dot = np.asarray(flow.x)
        eta = np.asarray(flow.y)
        sigma = eta * gamma_dot  # Convert back to stress

        flow_stress = RheoData(
            x=gamma_dot, y=sigma, metadata={"test_mode": "flow_curve"}
        )
        transform = CoxMerz(tolerance=0.1)
        result, meta = transform.transform([osc, flow_stress])
        assert meta["cox_merz_result"].mean_deviation < 0.15

    def test_mpa_s_y_units_converted_to_si(self):
        """Flow-curve viscosity labeled 'mPa.s' must be scaled by 1e-3, not
        used at face value, so physically-matching data still passes."""
        osc, flow = self._make_matching_data()
        # Same physical viscosity as `flow`, but expressed in mPa.s (1000x
        # the Pa.s numeric value) and labeled accordingly.
        flow_mpas = RheoData(
            x=flow.x,
            y=np.asarray(flow.y) * 1000.0,
            y_units="mPa.s",
            metadata={"test_mode": "flow_curve", "is_viscosity": True},
        )
        transform = CoxMerz(tolerance=0.1)
        result, meta = transform.transform([osc, flow_mpas])
        assert meta["cox_merz_result"].passes is True
        assert meta["cox_merz_result"].mean_deviation < 0.1

    def test_kpa_stress_y_units_converted_to_si(self):
        """Flow-curve stress labeled 'kPa' must be scaled by 1e3, not used
        at face value, so physically-matching data still passes."""
        osc, flow = self._make_matching_data()
        gamma_dot = np.asarray(flow.x)
        eta = np.asarray(flow.y)
        sigma_kpa = (eta * gamma_dot) / 1000.0  # stress in kPa

        flow_stress_kpa = RheoData(
            x=gamma_dot,
            y=sigma_kpa,
            y_units="kPa",
            metadata={"test_mode": "flow_curve"},
        )
        transform = CoxMerz(tolerance=0.1)
        result, meta = transform.transform([osc, flow_stress_kpa])
        assert meta["cox_merz_result"].mean_deviation < 0.15

    def test_zero_viscosity_no_nan(self):
        """Zero viscosity values should not produce NaN/inf in log interpolation."""
        omega = np.logspace(-1, 1, 20)
        G_star = np.zeros(20, dtype=complex)  # G*=0 → η*=0
        osc = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})

        gamma_dot = np.logspace(-1, 1, 20)
        eta_steady = np.zeros(20)  # Zero viscosity
        flow = RheoData(
            x=gamma_dot,
            y=eta_steady,
            metadata={"test_mode": "flow_curve", "is_viscosity": True},
        )
        transform = CoxMerz(tolerance=0.5)
        result, meta = transform.transform([osc, flow])
        # Must not crash or produce NaN — deviation is finite
        assert np.isfinite(meta["cox_merz_result"].mean_deviation)
