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
        assert ax.get_xlabel() is not None
        assert ax.get_ylabel() is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_plot_frequency_domain_data(self):
        """Test plotting frequency-domain data."""
        frequency = np.logspace(-2, 2, 50)
        modulus = 1e5 / (1 + 1j * frequency)

        data = RheoData(
            x=frequency,
            y=modulus,
            x_units="rad/s",
            y_units="Pa",
            domain="frequency",
            metadata={"test_mode": "oscillation"},
        )

        # Expect warning about non-positive values in log-scale plot
        with pytest.warns(UserWarning, match="Removed .* non-positive values"):
            fig, ax = plot_rheo_data(data)

        assert fig is not None
        assert isinstance(ax, (list, np.ndarray))  # Should have 2 axes for G', G''
        assert len(ax) >= 2
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
        G_complex = 1e5 / (1 + 1j * frequency)

        # Expect warning about non-positive values in log-scale plot
        with pytest.warns(UserWarning, match="Removed .* non-positive values"):
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
        # Should return single axis for real data
        if isinstance(ax, (list, np.ndarray)):
            assert len(ax) == 1
        else:
            assert ax is not None
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
        x = np.linspace(0, 10, 50)
        y_true = np.sin(x)
        y_pred = np.sin(x) + 0.1 * np.random.randn(50)
        residuals = y_true - y_pred

        fig, ax = plot_residuals(x, residuals)

        assert fig is not None
        assert ax is not None
        assert "Residuals" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_with_predictions(self):
        """Test plotting data with predictions and residuals."""
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
        # Check figure size is appropriate for publication
        assert fig.get_figwidth() >= 6
        assert fig.get_figheight() >= 4
        plt.close(fig)

    def test_default_style_settings(self):
        """Test default style settings."""
        time = np.linspace(0, 10, 100)
        stress = 1000 * np.exp(-time / 2)

        fig, ax = plot_time_domain(time, stress)

        assert fig is not None
        plt.close(fig)
