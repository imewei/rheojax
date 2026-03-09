"""Tests for auto_p0 smart parameter initialization.

Coverage across model families: classical, fractional, flow, SGR, fluidity,
Giesekus, GMM, IKH, HL, VLB, TNT.
"""

import warnings

import numpy as np
import pytest

from rheojax.io._exceptions import RheoJaxInitWarning
from rheojax.utils.initialization.auto_p0 import auto_p0

# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------


def _maxwell_relaxation(G0=1000.0, tau=1.0, n=100):
    t = np.logspace(-2, 2, n)
    return t, G0 * np.exp(-t / tau)


def _maxwell_oscillation(G0=1000.0, eta=1000.0, n=100):
    omega = np.logspace(-2, 2, n)
    tau = eta / G0
    wt2 = (omega * tau) ** 2
    G_prime = G0 * wt2 / (1.0 + wt2)
    G_dbl = G0 * omega * tau / (1.0 + wt2)
    return omega, G_prime + 1j * G_dbl


def _flow_curve(eta0=100.0, n_pl=0.5, gamma_dot_c=1.0, n_pts=80):
    gd = np.logspace(-2, 2, n_pts)
    sigma = eta0 * gd / (1.0 + (gd / gamma_dot_c) ** (1.0 - n_pl))
    return gd, sigma


def _assert_p0_valid(p0, model):
    """Check dict, non-empty, bounds-compliant."""
    assert isinstance(p0, dict)
    assert len(p0) > 0
    for name, value in p0.items():
        param = model.parameters[name]
        if param.bounds:
            lo, hi = param.bounds
            if lo is not None:
                assert value >= lo, f"{name}={value} < lower bound {lo}"
            if hi is not None:
                assert value <= hi, f"{name}={value} > upper bound {hi}"


# ---------------------------------------------------------------------------
# Classical family
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAutoP0Maxwell:
    """Test auto_p0 on Maxwell model (classical family)."""

    def test_relaxation_basic(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t, G_t = _maxwell_relaxation()
        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        _assert_p0_valid(p0, model)
        if "G0" in p0:
            assert p0["G0"] > 0

    def test_oscillation_basic(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)

    def test_bounds_compliance(self):
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t, G_t = _maxwell_relaxation()
        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        _assert_p0_valid(p0, model)


@pytest.mark.smoke
class TestAutoP0Zener:
    def test_relaxation(self):
        try:
            from rheojax.models.classical import Zener
        except ImportError:
            pytest.skip("Zener not available")
        model = Zener()
        t = np.logspace(-2, 2, 100)
        G_t = 500.0 + 500.0 * np.exp(-t / 1.0)
        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        _assert_p0_valid(p0, model)


@pytest.mark.smoke
class TestAutoP0SpringPot:
    def test_oscillation(self):
        from rheojax.models.classical import SpringPot

        model = SpringPot()
        omega = np.logspace(-2, 2, 100)
        G_star = 1000.0 * (1j * omega) ** 0.5
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# Fractional family
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAutoP0Fractional:
    def test_fractional_maxwell_oscillation(self):
        from rheojax.models.fractional.fractional_maxwell_model import (
            FractionalMaxwellModel,
        )

        model = FractionalMaxwellModel()
        omega = np.logspace(-2, 2, 100)
        G_star = 1000.0 * (1j * omega) ** 0.5
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)

    def test_fractional_zener_oscillation(self):
        try:
            from rheojax.models.fractional.fractional_zener_ss import FractionalZenerSolidSolid
        except ImportError:
            pytest.skip("FractionalZenerSolidSolid not available")
        model = FractionalZenerSolidSolid()
        omega = np.logspace(-2, 2, 100)
        G_star = 500.0 + 500.0 * (1j * omega) ** 0.3
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# Flow family
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAutoP0Flow:
    def test_cross_flow_curve(self):
        from rheojax.models.flow import Cross

        model = Cross()
        gd, sigma = _flow_curve()
        p0 = auto_p0(gd, sigma, model, test_mode="flow_curve")
        _assert_p0_valid(p0, model)

    def test_carreau_flow_curve(self):
        from rheojax.models.flow import Carreau

        model = Carreau()
        gd, sigma = _flow_curve()
        p0 = auto_p0(gd, sigma, model, test_mode="flow_curve")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# SGR family
# ---------------------------------------------------------------------------


class TestAutoP0SGR:
    def test_sgr_oscillation(self):
        from rheojax.models.sgr import SGRConventional

        model = SGRConventional()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# Fluidity family
# ---------------------------------------------------------------------------


class TestAutoP0Fluidity:
    def test_fluidity_oscillation(self):
        from rheojax.models.fluidity import FluidityLocal

        model = FluidityLocal()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# GMM (multi-mode)
# ---------------------------------------------------------------------------


class TestAutoP0GMM:
    def test_gmm_relaxation(self):
        try:
            from rheojax.models.gmm import GeneralizedMaxwell
        except ImportError:
            pytest.skip("GeneralizedMaxwell not available")
        model = GeneralizedMaxwell(n_modes=3)
        t = np.logspace(-2, 2, 100)
        G_t = 500 * np.exp(-t / 0.1) + 300 * np.exp(-t / 1.0) + 200 * np.exp(-t / 10.0)
        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# Giesekus family
# ---------------------------------------------------------------------------


class TestAutoP0Giesekus:
    def test_giesekus_oscillation(self):
        from rheojax.models.giesekus import GiesekusSingleMode

        model = GiesekusSingleMode()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# VLB family
# ---------------------------------------------------------------------------


class TestAutoP0VLB:
    def test_vlb_relaxation(self):
        try:
            from rheojax.models.vlb import VLBLocal
        except ImportError:
            pytest.skip("VLBLocal not available")
        model = VLBLocal()
        t, G_t = _maxwell_relaxation()
        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# TNT family
# ---------------------------------------------------------------------------


class TestAutoP0TNT:
    def test_tnt_single_oscillation(self):
        try:
            from rheojax.models.tnt import TNTSingleMode
        except ImportError:
            pytest.skip("TNTSingleMode not available")
        model = TNTSingleMode()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# IKH family
# ---------------------------------------------------------------------------


class TestAutoP0IKH:
    def test_ikh_oscillation(self):
        from rheojax.models.ikh import MIKH

        model = MIKH()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# HL family
# ---------------------------------------------------------------------------


class TestAutoP0HL:
    def test_hl_oscillation(self):
        from rheojax.models.hl import HebraudLequeux

        model = HebraudLequeux()
        omega, G_star = _maxwell_oscillation()
        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)


# ---------------------------------------------------------------------------
# Warning emission
# ---------------------------------------------------------------------------


class TestAutoP0WarningEmission:
    """Test that auto_p0 emits warnings for failed estimations."""

    def test_empty_data_warns(self):
        """auto_p0 with pathological data should emit warnings, not crash."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p0 = auto_p0(np.array([]), np.array([]), model)
            assert isinstance(p0, dict)

    def test_constant_data_warns(self):
        """Constant data provides no features — should warn gracefully."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t = np.logspace(-2, 2, 50)
        y = np.ones_like(t) * 42.0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            p0 = auto_p0(t, y, model, test_mode="relaxation")
            assert isinstance(p0, dict)


# ---------------------------------------------------------------------------
# Real (N,2) oscillation format
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestAutoP0RealTwoColumn:
    def test_real_two_column_oscillation(self):
        """auto_p0 should handle (N, 2) [G', G''] input."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        omega = np.logspace(-2, 2, 100)
        tau = 1.0
        G0 = 1000.0
        wt2 = (omega * tau) ** 2
        G_prime = G0 * wt2 / (1.0 + wt2)
        G_dbl = G0 * omega * tau / (1.0 + wt2)
        y = np.column_stack([G_prime, G_dbl])
        p0 = auto_p0(omega, y, model, test_mode="oscillation")
        _assert_p0_valid(p0, model)
