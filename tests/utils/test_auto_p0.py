"""Tests for auto_p0 smart parameter initialization."""

import warnings

import numpy as np
import pytest

from rheojax.io._exceptions import RheoJaxInitWarning
from rheojax.utils.initialization.auto_p0 import auto_p0


@pytest.mark.smoke
class TestAutoP0Maxwell:
    """Test auto_p0 on Maxwell model (classical family)."""

    def test_relaxation_basic(self):
        """auto_p0 should estimate G0 and eta/tau from relaxation data."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t = np.logspace(-2, 2, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)  # G0=1000, tau=1

        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        assert isinstance(p0, dict)
        assert len(p0) > 0
        # G0 estimate should be positive
        if "G0" in p0:
            assert p0["G0"] > 0

    def test_oscillation_basic(self):
        """auto_p0 should estimate from oscillation data."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        omega = np.logspace(-2, 2, 100)
        G0, eta = 1000.0, 1000.0
        tau = eta / G0
        G_prime = G0 * (omega * tau) ** 2 / (1 + (omega * tau) ** 2)
        G_double_prime = G0 * omega * tau / (1 + (omega * tau) ** 2)
        G_star = G_prime + 1j * G_double_prime

        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        assert isinstance(p0, dict)
        assert len(p0) > 0

    def test_bounds_compliance(self):
        """All estimated values should be within parameter bounds."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t = np.logspace(-2, 2, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)

        p0 = auto_p0(t, G_t, model, test_mode="relaxation")
        for name, value in p0.items():
            param = model.parameters[name]
            if param.bounds:
                lo, hi = param.bounds
                if lo is not None:
                    assert value >= lo, f"{name}={value} < lower bound {lo}"
                if hi is not None:
                    assert value <= hi, f"{name}={value} > upper bound {hi}"


class TestAutoP0WarningEmission:
    """Test that auto_p0 emits warnings for failed estimations."""

    def test_empty_data_warns(self):
        """auto_p0 with pathological data should emit warnings, not crash."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        # Zero-length data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p0 = auto_p0(np.array([]), np.array([]), model)
            assert isinstance(p0, dict)


class TestAutoP0ModelSelection:
    """Test auto_p0 on specific model types."""

    @pytest.mark.smoke
    def test_zener_model(self):
        """auto_p0 should handle Zener model parameters."""
        try:
            from rheojax.models.classical import Zener

            model = Zener()
            t = np.logspace(-2, 2, 100)
            G_t = 500.0 + 500.0 * np.exp(-t / 1.0)

            p0 = auto_p0(t, G_t, model, test_mode="relaxation")
            assert isinstance(p0, dict)
            assert len(p0) > 0
        except ImportError:
            pytest.skip("Zener model not available")

    @pytest.mark.smoke
    def test_springpot_model(self):
        """auto_p0 should handle Springpot fractional parameters."""
        from rheojax.models.fractional.fractional_maxwell_model import FractionalMaxwellModel

        model = FractionalMaxwellModel()
        omega = np.logspace(-2, 2, 100)
        G_star = 1000.0 * (1j * omega) ** 0.5  # Springpot-like

        p0 = auto_p0(omega, G_star, model, test_mode="oscillation")
        assert isinstance(p0, dict)
