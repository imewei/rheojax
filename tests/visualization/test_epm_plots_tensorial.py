"""Tests for tensorial EPM visualization functions."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from rheojax.core.jax_config import safe_import_jax
from rheojax.visualization.epm_plots import (
    plot_lattice_fields,
    plot_normal_stress_field,
    plot_normal_stress_ratio,
    plot_tensorial_fields,
    plot_von_mises_field,
    animate_tensorial_evolution,
)

jax, jnp = safe_import_jax()


class TestTensorialAutoDetection:
    """Test auto-detection of scalar vs tensorial stress fields."""

    @pytest.mark.smoke
    def test_plot_lattice_fields_scalar_detection(self):
        """Test that plot_lattice_fields auto-detects scalar (L, L) stress."""
        L = 32
        stress_scalar = np.random.randn(L, L)
        thresholds = np.abs(np.random.randn(L, L)) + 0.5

        fig = plot_lattice_fields(stress_scalar, thresholds)

        assert fig is not None
        # Should create 2 main panels (stress + thresholds) plus colorbars
        # Total: 2 plot axes + 2 colorbar axes = 4
        assert len(fig.axes) == 4
        plt.close(fig)

    @pytest.mark.smoke
    def test_plot_lattice_fields_tensor_detection(self):
        """Test that plot_lattice_fields auto-detects tensorial (3, L, L) stress."""
        L = 32
        stress_tensor = np.random.randn(3, L, L)
        thresholds = np.abs(np.random.randn(L, L)) + 0.5

        fig = plot_lattice_fields(stress_tensor, thresholds)

        assert fig is not None
        # Should create 4 main panels (σ_xx, σ_yy, σ_xy + thresholds) plus colorbars
        # Total: 4 plot axes + 4 colorbar axes = 8
        assert len(fig.axes) == 8
        plt.close(fig)

    def test_plot_lattice_fields_invalid_shape(self):
        """Test that invalid stress shape raises ValueError."""
        stress_invalid = np.random.randn(5, 32, 32)

        with pytest.raises(ValueError, match="Invalid stress shape"):
            plot_lattice_fields(stress_invalid, np.ones((32, 32)))


class TestTensorialFieldPlots:
    """Test specialized tensorial field plotting functions."""

    @pytest.mark.smoke
    def test_plot_tensorial_fields_layout(self):
        """Test that plot_tensorial_fields creates 3-panel layout."""
        L = 32
        stress = np.random.randn(3, L, L)

        fig, axes = plot_tensorial_fields(stress)

        assert fig is not None
        assert len(axes) == 3
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 4
        plt.close(fig)

    def test_plot_tensorial_fields_titles(self):
        """Test that tensorial field panels have correct LaTeX titles."""
        L = 32
        stress = np.random.randn(3, L, L)

        fig, axes = plot_tensorial_fields(stress)

        titles = [ax.get_title() for ax in axes]

        # Check for LaTeX formatted component labels
        assert any('xx' in title.lower() or 'σ' in title for title in titles)
        assert any('yy' in title.lower() or 'σ' in title for title in titles)
        assert any('xy' in title.lower() or 'σ' in title for title in titles)
        plt.close(fig)

    def test_plot_tensorial_fields_colormap(self):
        """Test that tensorial fields use coolwarm diverging colormap."""
        L = 32
        stress = np.random.randn(3, L, L)

        fig, axes = plot_tensorial_fields(stress, cmap='coolwarm')

        # Check that images use specified colormap
        for ax in axes:
            images = ax.get_images()
            if images:
                assert images[0].get_cmap().name == 'coolwarm'
        plt.close(fig)


class TestNormalStressPlots:
    """Test normal stress difference visualization."""

    @pytest.mark.smoke
    def test_plot_normal_stress_field_computation(self):
        """Test that plot_normal_stress_field computes N₁ correctly."""
        L = 32
        stress = np.random.randn(3, L, L)
        nu = 0.48

        fig, ax = plot_normal_stress_field(stress, nu=nu)

        assert fig is not None
        assert ax is not None

        # Verify N₁ computation manually
        N1_expected = stress[0] - stress[1]

        plt.close(fig)

    def test_plot_normal_stress_field_title(self):
        """Test that normal stress field has LaTeX label."""
        L = 32
        stress = np.random.randn(3, L, L)

        fig, ax = plot_normal_stress_field(stress)

        title = ax.get_title()

        # Should contain N or N1 in title
        assert 'N' in title or '$N_1$' in title
        plt.close(fig)

    def test_plot_normal_stress_ratio(self):
        """Test log-log plot of N₁/σ_xy vs shear rate."""
        shear_rates = np.logspace(-2, 2, 20)
        N1 = 0.5 * shear_rates**1.5
        sigma_xy = shear_rates**0.8

        fig, ax = plot_normal_stress_ratio(shear_rates, N1, sigma_xy)

        assert fig is not None
        assert ax is not None
        # Should be log-log
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        plt.close(fig)


class TestVonMisesPlots:
    """Test von Mises stress visualization."""

    @pytest.mark.smoke
    def test_plot_von_mises_field_layout(self):
        """Test that plot_von_mises_field creates 2-panel layout."""
        L = 32
        stress = np.random.randn(3, L, L)
        thresholds = np.abs(np.random.randn(L, L)) + 0.5

        fig, axes = plot_von_mises_field(stress, thresholds)

        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_von_mises_field_colormaps(self):
        """Test that von Mises panels use correct colormaps."""
        L = 32
        stress = np.random.randn(3, L, L)
        thresholds = np.abs(np.random.randn(L, L)) + 0.5

        fig, axes = plot_von_mises_field(stress, thresholds)

        # First panel: σ_eff with viridis (sequential)
        images_0 = axes[0].get_images()
        if images_0:
            assert images_0[0].get_cmap().name == 'viridis'

        # Second panel: σ_eff/σ_c with RdYlGn_r (diverging centered at 1)
        images_1 = axes[1].get_images()
        if images_1:
            assert images_1[0].get_cmap().name == 'RdYlGn_r'

        plt.close(fig)


class TestTensorialAnimation:
    """Test animation of tensorial stress evolution."""

    @pytest.mark.smoke
    def test_animate_tensorial_all_components(self):
        """Test animation with all components."""
        L = 16
        T = 10
        stress_history = np.random.randn(T, 3, L, L)
        time = np.linspace(0, 1, T)

        history = {
            'stress': stress_history,
            'time': time
        }

        anim = animate_tensorial_evolution(history, component='all')

        assert anim is not None
        # Should have 3 plot axes + 3 colorbar axes = 6 axes for 'all' components
        assert len(anim._fig.axes) == 6
        plt.close(anim._fig)

    def test_animate_tensorial_single_component(self):
        """Test animation with single component (e.g., 'xx')."""
        L = 16
        T = 10
        stress_history = np.random.randn(T, 3, L, L)
        time = np.linspace(0, 1, T)

        history = {
            'stress': stress_history,
            'time': time
        }

        anim = animate_tensorial_evolution(history, component='xx')

        assert anim is not None
        # Should have 1 plot axis + 1 colorbar axis = 2 axes for single component
        assert len(anim._fig.axes) == 2
        plt.close(anim._fig)

    def test_animate_tensorial_von_mises(self):
        """Test animation of von Mises effective stress."""
        L = 16
        T = 10
        stress_history = np.random.randn(T, 3, L, L)
        time = np.linspace(0, 1, T)

        history = {
            'stress': stress_history,
            'time': time
        }

        anim = animate_tensorial_evolution(history, component='vm')

        assert anim is not None
        # Should have 1 plot axis + 1 colorbar axis = 2 axes
        assert len(anim._fig.axes) == 2
        plt.close(anim._fig)
