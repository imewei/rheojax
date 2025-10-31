"""Tests for visualization templates."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from rheojax.core.data import RheoData
from rheojax.visualization.templates import (
    apply_template_style,
    plot_mastercurve,
    plot_model_fit,
    plot_modulus_frequency,
    plot_stress_strain,
)


class TestStressStrainTemplate:
    """Test stress-strain plotting template."""

    def test_plot_relaxation(self):
        """Test stress relaxation plot."""
        time = np.linspace(0, 100, 200)
        stress = 1000 * np.exp(-time / 20)

        data = RheoData(
            x=time,
            y=stress,
            x_units="s",
            y_units="Pa",
            domain="time",
            metadata={"test_mode": "relaxation"},
        )

        fig, ax = plot_stress_strain(data)

        assert fig is not None
        assert ax is not None
        assert "Time" in ax.get_xlabel()
        assert "Stress" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_creep(self):
        """Test creep compliance plot."""
        time = np.linspace(0, 100, 200)
        strain = 0.01 * (1 - np.exp(-time / 20))

        data = RheoData(
            x=time,
            y=strain,
            x_units="s",
            y_units="unitless",
            domain="time",
            metadata={"test_mode": "creep"},
        )

        fig, ax = plot_stress_strain(data)

        assert fig is not None
        assert "Time" in ax.get_xlabel()
        plt.close(fig)


class TestModulusFrequencyTemplate:
    """Test modulus-frequency plotting template."""

    def test_plot_storage_loss_modulus(self):
        """Test G' and G'' vs frequency plot."""
        frequency = np.logspace(-2, 2, 50)
        G_storage = 1e5 * frequency**2 / (1 + frequency**2)
        G_loss = 1e5 * frequency / (1 + frequency**2)
        G_complex = G_storage + 1j * G_loss

        data = RheoData(
            x=frequency,
            y=G_complex,
            x_units="rad/s",
            y_units="Pa",
            domain="frequency",
            metadata={"test_mode": "oscillation"},
        )

        fig, axes = plot_modulus_frequency(data)

        assert fig is not None
        assert len(axes) == 2
        assert "G'" in axes[0].get_ylabel() or "Storage" in axes[0].get_ylabel()
        assert 'G"' in axes[1].get_ylabel() or "Loss" in axes[1].get_ylabel()
        plt.close(fig)

    def test_plot_single_axis_modulus(self):
        """Test plotting on single axis."""
        frequency = np.logspace(-2, 2, 50)
        G_storage = 1e5 * frequency**2 / (1 + frequency**2)
        G_loss = 1e5 * frequency / (1 + frequency**2)
        G_complex = G_storage + 1j * G_loss

        data = RheoData(
            x=frequency, y=G_complex, x_units="rad/s", y_units="Pa", domain="frequency"
        )

        fig, ax = plot_modulus_frequency(data, separate_axes=False)

        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestMastercurveTemplate:
    """Test mastercurve plotting template."""

    def test_plot_single_mastercurve(self):
        """Test plotting a single mastercurve."""
        frequency = np.logspace(-2, 2, 50)
        G_storage = 1e5 * frequency**2 / (1 + frequency**2)

        data = RheoData(
            x=frequency,
            y=G_storage,
            x_units="rad/s",
            y_units="Pa",
            domain="frequency",
            metadata={"temperature": 25},
        )

        fig, ax = plot_mastercurve([data])

        assert fig is not None
        assert ax is not None
        assert "Frequency" in ax.get_xlabel()
        plt.close(fig)

    def test_plot_multi_temperature_mastercurve(self):
        """Test plotting mastercurve with multiple temperatures."""
        datasets = []
        for temp in [20, 25, 30, 35, 40]:
            frequency = np.logspace(-2, 2, 50)
            shift = 10 ** ((temp - 25) / 10)
            G_storage = 1e5 * (frequency * shift) ** 2 / (1 + (frequency * shift) ** 2)

            data = RheoData(
                x=frequency,
                y=G_storage,
                x_units="rad/s",
                y_units="Pa",
                domain="frequency",
                metadata={"temperature": temp},
            )
            datasets.append(data)

        fig, ax = plot_mastercurve(datasets, show_shifts=True)

        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestModelFitTemplate:
    """Test model fit plotting template."""

    def test_plot_fit_with_data(self):
        """Test plotting model fit with experimental data."""
        frequency = np.logspace(-2, 2, 50)
        G_data = 1e5 * frequency**2 / (1 + frequency**2)
        G_fit = 1e5 * frequency**2 / (1 + frequency**2) * 1.05  # Slight variation

        data = RheoData(
            x=frequency, y=G_data, x_units="rad/s", y_units="Pa", domain="frequency"
        )

        fig, axes = plot_model_fit(data, G_fit, show_residuals=True)

        assert fig is not None
        assert len(axes) == 2  # Main plot + residuals
        plt.close(fig)

    def test_plot_fit_without_residuals(self):
        """Test plotting model fit without residuals."""
        frequency = np.logspace(-2, 2, 50)
        G_data = 1e5 / (1 + 1j * frequency)
        G_fit = G_data * 1.02

        data = RheoData(
            x=frequency, y=G_data, x_units="rad/s", y_units="Pa", domain="frequency"
        )

        fig, ax = plot_model_fit(data, G_fit, show_residuals=False)

        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestTemplateStyles:
    """Test template style application."""

    def test_apply_default_style(self):
        """Test applying default style."""
        fig, ax = plt.subplots()
        apply_template_style(ax, style="default")

        assert fig is not None
        plt.close(fig)

    def test_apply_publication_style(self):
        """Test applying publication style."""
        fig, ax = plt.subplots()
        apply_template_style(ax, style="publication")

        assert fig is not None
        # Check that font sizes are appropriate for publication
        assert ax.xaxis.label.get_fontsize() >= 10
        plt.close(fig)

    def test_apply_presentation_style(self):
        """Test applying presentation style."""
        fig, ax = plt.subplots()
        apply_template_style(ax, style="presentation")

        assert fig is not None
        # Check that font sizes are larger for presentation
        assert ax.xaxis.label.get_fontsize() >= 12
        plt.close(fig)

    def test_invalid_style_fallback(self):
        """Test that invalid style falls back to default."""
        fig, ax = plt.subplots()
        apply_template_style(ax, style="invalid_style")

        assert fig is not None
        plt.close(fig)
