"""Tests for automatic shift factor calculation via power-law intersection method.

This module tests the power-law intersection algorithm for automatic shift factor
calculation in time-temperature superposition, following the pyvisco algorithm.
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.mastercurve import Mastercurve


class TestPowerLawFitting:
    """Test power-law fitting with NLSQ (no scipy)."""

    def test_power_law_fit_basic(self):
        """Test basic power-law fitting: y = a*x^b + e."""
        # Create synthetic power-law data
        x = np.logspace(-2, 2, 50)
        a_true, b_true, e_true = 1.5, -0.7, 0.1
        y_true = a_true * x**b_true + e_true

        # Add small noise
        np.random.seed(42)
        y_data = y_true + 0.01 * np.random.randn(len(x))

        # Fit power-law
        mc = Mastercurve(reference_temp=298.15, method="wlf")
        popt, perr = mc._fit_power_law(x, y_data)

        # Check parameters recovered within tolerance
        assert len(popt) == 3
        assert np.abs(popt[0] - a_true) < 0.1  # a
        assert np.abs(popt[1] - b_true) < 0.1  # b
        assert np.abs(popt[2] - e_true) < 0.1  # e

        # Check uncertainties are positive
        assert len(perr) == 3
        assert all(perr > 0)

    def test_power_law_outlier_detection_logic(self):
        """Test that outlier detection logic works correctly."""
        # Create clean data (no outliers)
        x = np.logspace(-2, 2, 50)
        a_true, b_true = 2.0, -0.5
        y_data = a_true * x**b_true

        mc = Mastercurve(reference_temp=298.15, method="wlf")

        # Fit full data
        popt_full, perr_full = mc._fit_power_law(x, y_data)

        # Detect outliers (should keep all points since data is clean)
        x_clean, y_clean, popt_clean, perr_clean = mc._detect_outliers(
            x, y_data, popt_full, perr_full
        )

        # For clean data, should keep all points or remove if it helps
        # Just verify the method returns valid results
        assert len(x_clean) > 0
        assert len(y_clean) == len(x_clean)
        assert len(popt_clean) == 3
        assert len(perr_clean) == 3


class TestShiftCalculation:
    """Test intersection-based shift factor calculation."""

    def test_overlap_case(self):
        """Test shift calculation for overlapping curves."""
        # Create two overlapping power-law curves
        x1 = np.logspace(-1, 1, 30)
        x2 = np.logspace(0, 2, 30)  # Overlaps in [1, 10]

        a1, b1 = 1.0, -0.6
        a2, b2 = 1.5, -0.6  # Same exponent, different amplitude

        y1 = a1 * x1**b1
        y2 = a2 * x2**b2

        curve_top = np.column_stack([x1, y1])
        curve_bot = np.column_stack([x2, y2])

        mc = Mastercurve(reference_temp=298.15, method="wlf")

        # Power-law parameters for each curve
        popt_top = np.array([a1, b1, 0.0])
        popt_bot = np.array([a2, b2, 0.0])

        # Compute shift
        log_aT = mc._compute_pairwise_shift(curve_top, curve_bot, popt_top, popt_bot)

        # Check shift is finite and reasonable
        assert np.isfinite(log_aT)
        assert -2.0 < log_aT < 2.0

    def test_gap_case(self):
        """Test shift calculation for non-overlapping curves (gap)."""
        # Create two curves with gap between them
        x1 = np.logspace(-2, 0, 30)  # [0.01, 1]
        x2 = np.logspace(1, 3, 30)  # [10, 1000] - gap from 1 to 10

        a1, b1 = 1.0, -0.5
        a2, b2 = 1.0, -0.5  # Same power-law

        y1 = a1 * x1**b1
        y2 = a2 * x2**b2

        curve_top = np.column_stack([x1, y1])
        curve_bot = np.column_stack([x2, y2])

        mc = Mastercurve(reference_temp=298.15, method="wlf")

        popt_top = np.array([a1, b1, 0.0])
        popt_bot = np.array([a2, b2, 0.0])

        # Compute shift
        log_aT = mc._compute_pairwise_shift(curve_top, curve_bot, popt_top, popt_bot)

        # Check shift is finite
        assert np.isfinite(log_aT)
        assert -3.0 < log_aT < 3.0


class TestSequentialShifting:
    """Test sequential cumulative shift factor calculation."""

    def test_sequential_cumulative_shifts(self):
        """Test sequential pairwise cumulative shifting."""
        # Create synthetic multi-temperature data with known shifts
        temps = [273.15, 298.15, 323.15]  # 3 temperatures
        ref_idx = 1  # Reference at 298.15 K

        # Create power-law curves with known amplitude shifts
        datasets = []
        for i, T in enumerate(temps):
            # Use amplitude shift instead of x-shift for easier verification
            # Higher temp -> lower amplitude (simulating mastercurve behavior)
            amplitude = 1e5 * (1.5 ** -(i - ref_idx))  # Known amplitude pattern

            x = np.logspace(-1, 1, 30)
            y = amplitude * x ** (-0.6)

            data = RheoData(x=x, y=y, domain="frequency", metadata={"temperature": T})
            datasets.append(data)

        mc = Mastercurve(reference_temp=298.15, method="wlf")

        # Compute auto shifts
        log_aT_array = mc._compute_auto_shift_factors(datasets, ref_idx)

        # Check output shape
        assert len(log_aT_array) == len(temps)

        # Check reference temperature has near-zero shift
        assert np.abs(log_aT_array[ref_idx]) < 1e-10


class TestAutoShiftIntegration:
    """Test integration of auto_shift with Mastercurve API."""

    def test_auto_shift_parameter(self):
        """Test auto_shift parameter initialization."""
        # Default should be False
        mc_default = Mastercurve(reference_temp=298.15)
        assert hasattr(mc_default, "_auto_shift")
        assert mc_default._auto_shift is False

        # Explicit True
        mc_auto = Mastercurve(reference_temp=298.15, auto_shift=True)
        assert mc_auto._auto_shift is True

    def test_backward_compatibility(self):
        """Test auto_shift=False preserves existing behavior."""
        # Create multi-temperature datasets
        temps = [273.15, 298.15, 323.15]
        datasets = []

        for T in temps:
            x = np.logspace(-1, 1, 30)
            y = 1e5 * x ** (-0.5)
            data = RheoData(x=x, y=y, domain="frequency", metadata={"temperature": T})
            datasets.append(data)

        # Test with auto_shift=False (default, WLF)
        mc_wlf = Mastercurve(reference_temp=298.15, method="wlf", auto_shift=False)
        mastercurve_wlf, shifts_wlf = mc_wlf.transform(datasets)

        # Should use WLF calculation
        assert len(shifts_wlf) == len(temps)
        assert all(isinstance(v, float) for v in shifts_wlf.values())

        # Check reference temperature has shift factor = 1.0
        assert np.abs(shifts_wlf[298.15] - 1.0) < 1e-10
