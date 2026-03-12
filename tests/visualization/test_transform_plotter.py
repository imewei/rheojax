"""Tests for the TransformPlotter class."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.visualization.transform_plotter import TransformPlotter


@pytest.fixture
def plotter():
    return TransformPlotter()


@pytest.fixture
def time_data():
    """Simple time-domain RheoData."""
    t = np.linspace(0, 10, 100)
    y = np.sin(2 * np.pi * t)
    return RheoData(x=t, y=y, x_units="s", y_units="Pa", domain="time")


@pytest.fixture
def freq_data():
    """Simple frequency-domain RheoData."""
    f = np.logspace(-2, 2, 50)
    return RheoData(x=f, y=np.abs(1 / (1 + 1j * f)), x_units="Hz", domain="frequency")


@pytest.fixture
def complex_data():
    """Complex modulus RheoData."""
    freq = np.logspace(-2, 2, 50)
    G0 = 1e5
    omega_tau = freq * 1.0
    Gp = G0 * omega_tau**2 / (1 + omega_tau**2)
    Gpp = G0 * omega_tau / (1 + omega_tau**2)
    G_star = Gp + 1j * Gpp
    return RheoData(x=freq, y=G_star, x_units="rad/s", y_units="Pa", domain="frequency")


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestTransformPlotterDispatch:
    """Tests for auto-dispatch based on transform name."""

    @pytest.mark.smoke
    def test_unknown_transform_uses_generic(self, plotter, freq_data):
        """Unknown transform falls back to generic plot."""
        fig, axes = plotter.plot("unknown_transform", freq_data)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_known_transform_dispatches(self, plotter, freq_data):
        """Known transform name dispatches to correct method."""
        fig, axes = plotter.plot("fft", freq_data)
        assert fig is not None
        plt.close(fig)

    def test_name_normalization(self, plotter, freq_data):
        """Transform names are normalized (case, hyphens, spaces)."""
        for name in ["FFT", "fft_analysis", "FFT-Analysis", "fft analysis"]:
            fig, _ = plotter.plot(name, freq_data)
            assert fig is not None
            plt.close(fig)


# ---------------------------------------------------------------------------
# FFT tests
# ---------------------------------------------------------------------------


class TestFFTPlot:
    """Tests for FFT analysis visualization."""

    @pytest.mark.smoke
    def test_fft_with_input(self, plotter, time_data, freq_data):
        """FFT plot with input data shows 2 panels."""
        fig, axes = plotter.plot("fft", freq_data, input_data=time_data)
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_fft_without_input(self, plotter, freq_data):
        """FFT plot without input shows single panel."""
        fig, axes = plotter.plot("fft", freq_data)
        assert fig is not None
        assert len(axes) == 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Mastercurve tests
# ---------------------------------------------------------------------------


class TestMastercurvePlot:
    """Tests for mastercurve visualization."""

    @pytest.mark.smoke
    def test_mastercurve_with_shifts(self, plotter, complex_data):
        """Mastercurve plot with shift factors dict."""
        shifts = {300.0: 1.0, 310.0: 0.5, 320.0: 0.25}
        result = (complex_data, shifts)

        datasets = [complex_data] * 3
        fig, axes = plotter.plot(
            "mastercurve",
            result,
            input_data=datasets,
        )
        assert fig is not None
        assert len(axes) == 2  # unshifted + shifted
        plt.close(fig)

    def test_mastercurve_without_input(self, plotter, complex_data):
        """Mastercurve plot without input datasets."""
        shifts = {300.0: 1.0}
        result = (complex_data, shifts)

        fig, axes = plotter.plot("mastercurve", result)
        assert fig is not None
        assert len(axes) == 1  # shifted only
        plt.close(fig)


# ---------------------------------------------------------------------------
# Mutation number tests
# ---------------------------------------------------------------------------


class TestMutationNumberPlot:

    @pytest.mark.smoke
    def test_mutation_number(self, plotter):
        """Mutation number plot shows bar with value."""
        data = RheoData(
            x=np.array([0.0]),
            y=np.array([0.42]),
            domain="scalar",
            y_units="dimensionless",
            metadata={"mutation_number": 0.42},
        )
        fig, ax = plotter.plot("mutation_number", data)
        assert fig is not None
        assert "0.42" in ax.get_title()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Derivative tests
# ---------------------------------------------------------------------------


class TestDerivativePlot:

    @pytest.mark.smoke
    def test_derivative_with_input(self, plotter, time_data):
        """Derivative plot shows original + derivative."""
        deriv = RheoData(
            x=time_data.x,
            y=np.cos(2 * np.pi * time_data.x),
            x_units="s",
            y_units="Pa/s",
            domain="time",
        )
        fig, axes = plotter.plot("derivative", deriv, input_data=time_data)
        assert len(axes) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# SPP tests
# ---------------------------------------------------------------------------


class TestSPPPlot:

    @pytest.mark.smoke
    def test_spp_basic(self, plotter):
        """SPP plot with minimal metadata produces 3-panel layout."""
        t = np.linspace(0, 2 * np.pi, 100)
        stress = np.sin(t)
        data = RheoData(
            x=t,
            y=stress,
            domain="time",
            metadata={
                "spp_results": {
                    "strain": np.sin(t),
                    "stress_reconstructed": stress,
                    "Gp_t": 100 * np.ones_like(t),
                    "Gpp_t": 50 * np.ones_like(t),
                    "time_new": t,
                }
            },
        )
        fig, axes = plotter.plot("spp", data)
        assert len(axes) == 3
        plt.close(fig)


# ---------------------------------------------------------------------------
# Prony conversion tests
# ---------------------------------------------------------------------------


class TestPronyPlot:

    @pytest.mark.smoke
    def test_prony_with_input(self, plotter, time_data, complex_data):
        """Prony conversion shows input + converted domains."""
        from collections import namedtuple

        PronyResult = namedtuple("PronyResult", ["G_i", "tau_i", "G_e", "n_modes"])
        meta = {
            "prony_result": PronyResult(
                G_i=np.array([1e4, 5e3]),
                tau_i=np.array([0.1, 1.0]),
                G_e=100.0,
                n_modes=2,
            )
        }
        result = (complex_data, meta)
        fig, axes = plotter.plot("prony", result, input_data=time_data)
        assert len(axes) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spectrum inversion tests
# ---------------------------------------------------------------------------


class TestSpectrumPlot:

    @pytest.mark.smoke
    def test_spectrum_with_input(self, plotter, complex_data):
        """Spectrum inversion shows G',G'' + H(τ)."""
        tau = np.logspace(-3, 2, 50)
        H = 1e4 * np.exp(-tau)
        spectrum_data = RheoData(x=tau, y=H, domain="time")
        result = (spectrum_data, {"method": "tikhonov"})
        fig, axes = plotter.plot("spectrum_inversion", result, input_data=complex_data)
        assert len(axes) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cox-Merz tests
# ---------------------------------------------------------------------------


class TestCoxMerzPlot:

    @pytest.mark.smoke
    def test_cox_merz(self, plotter):
        """Cox-Merz plot shows both viscosities."""
        from dataclasses import dataclass

        @dataclass
        class CoxMerzResult:
            common_rates: np.ndarray
            eta_complex: np.ndarray
            eta_steady: np.ndarray
            deviation: np.ndarray
            mean_deviation: float
            max_deviation: float
            passes: bool

        rates = np.logspace(-1, 2, 20)
        cm = CoxMerzResult(
            common_rates=rates,
            eta_complex=100 / rates,
            eta_steady=95 / rates,
            deviation=np.ones(20) * 0.05,
            mean_deviation=0.05,
            max_deviation=0.08,
            passes=True,
        )
        dev_data = RheoData(x=rates, y=cm.deviation, domain="frequency")
        result = (dev_data, {"cox_merz_result": cm})
        fig, ax = plotter.plot("cox_merz", result)
        assert "PASS" in ax.get_title()
        plt.close(fig)


# ---------------------------------------------------------------------------
# LVE envelope tests
# ---------------------------------------------------------------------------


class TestLVEEnvelopePlot:

    @pytest.mark.smoke
    def test_envelope_with_input(self, plotter, time_data):
        """LVE envelope shows envelope + startup overlay."""
        t = np.linspace(0, 5, 100)
        sigma = 500 * (1 - np.exp(-t))
        env_data = RheoData(
            x=t,
            y=sigma,
            domain="time",
            metadata={"shear_rate": 1.0},
        )
        result = (env_data, {"lve_result": None})
        fig, ax = plotter.plot("lve_envelope", result, input_data=time_data)
        assert "γ̇" in ax.get_title()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Generic fallback tests
# ---------------------------------------------------------------------------


class TestGenericPlot:

    def test_generic_with_input(self, plotter, time_data, freq_data):
        """Generic plot shows before/after."""
        fig, axes = plotter.plot(
            "some_new_transform",
            freq_data,
            input_data=time_data,
        )
        assert len(axes) == 2
        plt.close(fig)

    def test_generic_without_input(self, plotter, freq_data):
        """Generic plot shows single output panel."""
        fig, axes = plotter.plot("some_new_transform", freq_data)
        assert len(axes) == 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Unpack result tests
# ---------------------------------------------------------------------------


class TestUnpackResult:

    def test_unpack_rheodata(self, plotter, freq_data):
        """Bare RheoData unpacks to (data, None)."""
        data, meta = plotter._unpack_result(freq_data)
        assert isinstance(data, RheoData)
        assert meta is None

    def test_unpack_tuple(self, plotter, freq_data):
        """Tuple[RheoData, dict] unpacks correctly."""
        result = (freq_data, {"key": "value"})
        data, meta = plotter._unpack_result(result)
        assert isinstance(data, RheoData)
        assert meta == {"key": "value"}

    def test_unpack_invalid_raises(self, plotter):
        """Invalid result type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot unpack"):
            plotter._unpack_result("not a result")


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------


class TestTransformPlotterStyles:

    @pytest.mark.parametrize("style", ["default", "publication", "presentation"])
    def test_styles_work(self, plotter, freq_data, style):
        """All styles work without error."""
        fig, _ = plotter.plot("fft", freq_data, style=style)
        assert fig is not None
        plt.close(fig)
