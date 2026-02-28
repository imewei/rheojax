"""Tests for visualization plotting utilities."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from rheojax.core.data import RheoData
from rheojax.visualization.plotter import (
    plot_flow_curve,
    plot_frequency_domain,
    plot_residuals,
    plot_rheo_data,
    plot_time_domain,
)


class TestPlotRheoData:
    """Test plot_rheo_data automatic plot type selection."""

    @pytest.mark.smoke
    def test_plot_time_domain_data(self):
        """Test plotting time-domain data."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        data = RheoData(
            x=time,
            y=stress,
            x_units="s",
            y_units="Pa",
            domain="time",
            metadata={"test_mode": "relaxation"},
        )

        fig, ax = plot_rheo_data(data)

        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() != ""  # Should have a label set
        assert ax.get_ylabel() != ""
        plt.close(fig)

    @pytest.mark.smoke
    def test_plot_frequency_domain_data(self):
        """Test plotting frequency-domain data."""
        frequency = np.logspace(-2, 2, 50)
        # Use proper Maxwell G* = G0*w^2/(1+w^2) + i*G0*w/(1+w^2) so both
        # G' (real) and G'' (imag) are positive — verifies both subplots render.
        G0 = 1e5
        modulus = G0 * frequency**2 / (1 + frequency**2) + 1j * G0 * frequency / (
            1 + frequency**2
        )

        data = RheoData(
            x=frequency,
            y=modulus,
            x_units="rad/s",
            y_units="Pa",
            domain="frequency",
            metadata={"test_mode": "oscillation"},
        )

        fig, ax = plot_rheo_data(data)

        assert fig is not None
        assert isinstance(ax, (list, np.ndarray))  # Should have 2 axes for G', G''
        assert len(ax) >= 2
        # Verify the plot was actually created (check lines or collections exist)
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
        for a in axes:
            assert (
                len(a.get_lines()) > 0 or len(a.collections) > 0
            ), "Plot should have data"
        plt.close(fig)

    @pytest.mark.smoke
    def test_plot_with_custom_kwargs(self):
        """Test plotting with custom matplotlib kwargs."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

        fig, ax = plot_rheo_data(data, color="red", linewidth=2, label="Test Data")

        assert fig is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_plot_rotation_data(self):
        """Test plotting rotation (flow curve) data."""
        shear_rate = np.logspace(-2, 2, 50)
        viscosity = 1.0 + 10 / (1 + shear_rate)

        data = RheoData(
            x=shear_rate,
            y=viscosity,
            x_units="1/s",
            y_units="Pa.s",
            domain="time",
            metadata={"test_mode": "rotation"},
        )

        fig, ax = plot_rheo_data(data)

        assert fig is not None
        assert ax is not None
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)


class TestPlotTimeDomain:
    """Test time-domain plotting functions."""

    @pytest.mark.smoke
    def test_plot_relaxation_data(self):
        """Test plotting relaxation data."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress, x_units="s", y_units="Pa")

        assert fig is not None
        assert ax is not None
        assert "Time" in ax.get_xlabel()
        assert "Stress" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_with_log_scale(self):
        """Test plotting with logarithmic scale."""
        time = np.logspace(-2, 2, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress, log_x=True, log_y=True)

        assert fig is not None
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)


class TestPlotFrequencyDomain:
    """Test frequency-domain plotting functions."""

    def test_plot_complex_modulus(self):
        """Test plotting complex modulus (G' and G'')."""
        frequency = np.logspace(-2, 2, 50)
        # Use proper Maxwell G* so both G' and G'' are positive
        G0 = 1e5
        G_complex = G0 * frequency**2 / (1 + frequency**2) + 1j * G0 * frequency / (
            1 + frequency**2
        )

        fig, axes = plot_frequency_domain(frequency, G_complex)

        assert fig is not None
        assert len(axes) == 2
        # xlabel is on the bottom (second) axis
        assert "Frequency" in axes[1].get_xlabel()
        assert "G'" in axes[0].get_ylabel()
        assert 'G"' in axes[1].get_ylabel()
        plt.close(fig)

    def test_plot_real_modulus_only(self):
        """Test plotting real modulus only."""
        frequency = np.logspace(-2, 2, 50)
        G_prime = 1e5 / (1 + frequency**2)

        fig, ax = plot_frequency_domain(frequency, G_prime)

        assert fig is not None
        # Should return single axis wrapped in an array for real data
        assert isinstance(ax, np.ndarray), f"Expected np.ndarray, got {type(ax)}"
        assert len(ax) == 1
        assert ax[0] is not None
        plt.close(fig)


class TestPlotFlowCurve:
    """Test flow curve plotting functions."""

    def test_plot_viscosity_vs_shear_rate(self):
        """Test plotting viscosity vs shear rate."""
        shear_rate = np.logspace(-2, 2, 50)
        viscosity = 1.0 + 10 / (1 + shear_rate)

        fig, ax = plot_flow_curve(shear_rate, viscosity)

        assert fig is not None
        assert ax is not None
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_plot_shear_stress(self):
        """Test plotting shear stress vs shear rate."""
        shear_rate = np.logspace(-2, 2, 50)
        shear_stress = 10 + 0.5 * shear_rate**0.8

        fig, ax = plot_flow_curve(shear_rate, shear_stress, y_label="Shear Stress (Pa)")

        assert fig is not None
        assert "Shear Stress" in ax.get_ylabel()
        plt.close(fig)


class TestPlotResiduals:
    """Test residual plotting functions."""

    def test_plot_simple_residuals(self):
        """Test plotting residuals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = np.sin(x)
        y_pred = np.sin(x) + 0.1 * np.random.randn(50)
        residuals = y_true - y_pred

        fig, ax = plot_residuals(x, residuals)

        assert fig is not None
        assert ax is not None
        assert "Residuals" in ax.get_ylabel()
        # Check horizontal zero-line exists
        hlines = [
            l
            for l in ax.get_lines()
            if len(l.get_ydata()) > 0 and np.allclose(l.get_ydata(), 0.0)
        ]
        assert len(hlines) >= 1, "Residual plot should have a horizontal zero-line"
        plt.close(fig)

    def test_plot_with_predictions(self):
        """Test plotting data with predictions and residuals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = np.sin(x)
        y_pred = np.sin(x) + 0.05 * np.random.randn(50)

        fig, axes = plot_residuals(x, y_true - y_pred, y_true=y_true, y_pred=y_pred)

        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)


class TestExportFormats:
    """Test export to different file formats."""

    def test_export_to_png(self, tmp_path):
        """Test export to PNG format."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress)

        output_path = tmp_path / "test_plot.png"
        fig.savefig(output_path, dpi=150)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_export_to_pdf(self, tmp_path):
        """Test export to PDF format."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress)

        output_path = tmp_path / "test_plot.pdf"
        fig.savefig(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_export_to_svg(self, tmp_path):
        """Test export to SVG format."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress)

        output_path = tmp_path / "test_plot.svg"
        fig.savefig(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)


class TestPublicationQuality:
    """Test publication-quality output."""

    def test_publication_style_settings(self):
        """Test that publication-quality settings are applied."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress, style="publication")

        assert fig is not None
        # Check figure size matches the PUBLICATION_FIGSIZE constant (6 x 4.5)
        assert fig.get_figwidth() == pytest.approx(6.0, abs=0.5)
        assert fig.get_figheight() == pytest.approx(4.5, abs=0.5)
        plt.close(fig)

    def test_default_style_settings(self):
        """Test default style settings."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress)

        assert fig is not None
        plt.close(fig)


class TestUncertaintyBand:
    """Tests for compute_uncertainty_band and plot_fit_with_uncertainty."""

    def test_basic_linear_model(self):
        """Test uncertainty band for a simple linear model."""
        from rheojax.visualization.plotter import compute_uncertainty_band

        def model_fn(x, params):
            return params[0] * x + params[1]

        x = np.linspace(1, 10, 20)
        popt = np.array([2.0, 1.0])
        pcov = np.array([[0.01, 0.0], [0.0, 0.01]])

        y_fit, y_lower, y_upper = compute_uncertainty_band(model_fn, x, popt, pcov)
        assert y_fit is not None
        assert y_lower is not None
        assert y_upper is not None
        assert np.all(y_lower <= y_fit)
        assert np.all(y_fit <= y_upper)

    def test_complex_output_returns_none_bands(self):
        """Test that complex model output returns None for bands."""
        from rheojax.visualization.plotter import compute_uncertainty_band

        def model_fn(x, params):
            return params[0] / (1 + 1j * x)

        x = np.logspace(-1, 1, 20)
        popt = np.array([1000.0])
        pcov = np.array([[1.0]])

        y_fit, y_lower, y_upper = compute_uncertainty_band(model_fn, x, popt, pcov)
        assert y_fit is not None
        assert y_lower is None
        assert y_upper is None


class TestSaveFigure:
    """Tests for save_figure function."""

    def test_save_pdf(self, tmp_path):
        """Test saving a figure to PDF format."""
        from rheojax.visualization.plotter import save_figure

        fig, _ = plot_time_domain(np.linspace(0, 1, 10), np.ones(10))
        path = save_figure(fig, tmp_path / "out.pdf")
        assert path.exists()
        plt.close(fig)

    def test_save_png(self, tmp_path):
        """Test saving a figure to PNG format."""
        from rheojax.visualization.plotter import save_figure

        fig, _ = plot_time_domain(np.linspace(0, 1, 10), np.ones(10))
        path = save_figure(fig, tmp_path / "out.png")
        assert path.exists()
        plt.close(fig)


def test_plot_frequency_domain_tension_labels():
    """Test E'/E'' labels for tensile deformation mode."""
    freq = np.logspace(-1, 1, 20)
    G = 1e9 / (1 + 1j * freq)
    fig, axes = plot_frequency_domain(freq, G, deformation_mode="tension")
    assert "E" in axes[0].get_ylabel()  # Should show E' not G'
    plt.close(fig)


def test_filter_positive_partial():
    """Test _filter_positive with some negative values."""
    from rheojax.visualization.plotter import _filter_positive

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([-1.0, 2.0, 3.0, 4.0])
    with pytest.warns(UserWarning, match="[Rr]emoved"):
        x_f, y_f = _filter_positive(x, y, warn=True)
    assert len(x_f) == 3
    assert np.all(y_f > 0)


def test_plot_time_domain_jax_input():
    """Test that JAX arrays are accepted."""
    try:
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
    except (ImportError, Exception):
        pytest.skip("JAX not available")
    time = jnp.linspace(0.0, 10.0, 50)
    stress = jnp.exp(-time / 2.0) * 1000.0
    fig, ax = plot_time_domain(time, stress)
    assert fig is not None
    plt.close(fig)


# NOTE: test_plot_with_log_scale uses exponentially decaying data
# that approaches but never reaches zero. Near-zero log-scale
# behavior is handled by _filter_positive in frequency/flow plots.


def test_plot_fit_with_uncertainty_existing_ax():
    """Test plot_fit_with_uncertainty with a pre-existing axis."""
    from rheojax.visualization.plotter import plot_fit_with_uncertainty

    x = np.linspace(1, 10, 20)
    y = x**0.5
    fig_outer, ax = plt.subplots()
    returned_fig, returned_ax = plot_fit_with_uncertainty(x, y, x, y, ax=ax)
    # When ax is provided, fig may be None or the parent figure
    assert returned_fig is None or returned_fig is fig_outer
    assert returned_ax is ax
    plt.close(fig_outer)


# ---------------------------------------------------------------------------
# Coverage Gap-6: _modulus_labels for bending/compression/metadata
# ---------------------------------------------------------------------------


class TestModulusLabels:
    """Gap-6: _modulus_labels returns E'/E'' for tension/bending/compression."""

    @pytest.mark.smoke
    def test_default_shear_labels(self):
        """No deformation mode → G'/G'' labels."""
        from rheojax.visualization.plotter import _modulus_labels

        s, l, g = _modulus_labels()
        assert "G'" in s
        assert 'G"' in l

    def test_tension_labels(self):
        """deformation_mode='tension' → E'/E'' labels."""
        from rheojax.visualization.plotter import _modulus_labels

        data = RheoData(
            x=np.array([1.0]),
            y=np.array([1.0]),
            metadata={"deformation_mode": "tension"},
        )
        s, l, g = _modulus_labels(data=data)
        assert "E'" in s
        assert 'E"' in l

    def test_bending_labels(self):
        """deformation_mode='bending' → E'/E'' labels."""
        from rheojax.visualization.plotter import _modulus_labels

        data = RheoData(
            x=np.array([1.0]),
            y=np.array([1.0]),
            metadata={"deformation_mode": "bending"},
        )
        s, l, g = _modulus_labels(data=data)
        assert "E'" in s
        assert 'E"' in l

    def test_compression_labels(self):
        """deformation_mode='compression' → E'/E'' labels."""
        from rheojax.visualization.plotter import _modulus_labels

        data = RheoData(
            x=np.array([1.0]),
            y=np.array([1.0]),
            metadata={"deformation_mode": "compression"},
        )
        s, l, g = _modulus_labels(data=data)
        assert "E'" in s
        assert 'E"' in l

    def test_custom_units(self):
        """y_units kwarg reflected in label."""
        from rheojax.visualization.plotter import _modulus_labels

        s, l, g = _modulus_labels(y_units="MPa")
        assert "MPa" in s
        assert "MPa" in l

    def test_metadata_deformation_mode(self):
        """deformation_mode from metadata dict is picked up."""
        from rheojax.visualization.plotter import _modulus_labels

        data = RheoData(
            x=np.array([1.0]),
            y=np.array([1.0]),
            metadata={"deformation_mode": "compression"},
        )
        s, _, _ = _modulus_labels(data=data)
        assert "E'" in s
