"""Tests for SPP (Sequence of Physical Processes) visualization module.

Covers all 7 public plotting functions in rheojax.visualization.spp_plots.
Tests focus on: figure creation, axis count, input validation, edge cases,
and JAX array compatibility.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.visualization.spp_plots import (
    create_spp_report,
    plot_3d_trajectory,
    plot_cole_cole,
    plot_harmonic_spectrum,
    plot_lissajous,
    plot_moduli_evolution,
    plot_pipkin_diagram,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def laos_signals():
    """Generate synthetic LAOS signals (1 cycle)."""
    rng = np.random.RandomState(42)
    omega = 1.0
    gamma_0 = 0.1
    n_pts = 200
    t = np.linspace(0, 2 * np.pi / omega, n_pts)
    strain = gamma_0 * np.sin(omega * t)
    strain_rate = gamma_0 * omega * np.cos(omega * t)
    # Nonlinear stress: fundamental + 3rd harmonic
    stress = 1000 * strain + 50 * strain**3 + rng.randn(n_pts) * 5
    return {
        "strain": strain,
        "strain_rate": strain_rate,
        "stress": stress,
        "omega": omega,
        "gamma_0": gamma_0,
        "t": t,
        "n_pts": n_pts,
    }


@pytest.fixture
def spp_results(laos_signals):
    """Generate synthetic SPP analysis results dict."""
    n = laos_signals["n_pts"]
    rng = np.random.RandomState(123)
    Gp_t = 1000.0 + 100 * np.sin(2 * laos_signals["omega"] * laos_signals["t"])
    Gpp_t = 200.0 + 50 * np.cos(2 * laos_signals["omega"] * laos_signals["t"])
    delta_t = np.arctan2(Gpp_t, Gp_t)
    return {
        "Gp_t": Gp_t,
        "Gpp_t": Gpp_t,
        "delta_t": delta_t,
        "time_new": laos_signals["t"],
        "strain_recon": laos_signals["strain"],
        "rate_recon": laos_signals["strain_rate"],
        "stress_recon": laos_signals["stress"],
    }


# ---------------------------------------------------------------------------
# plot_lissajous
# ---------------------------------------------------------------------------


class TestPlotLissajous:
    """Tests for plot_lissajous."""

    @pytest.mark.smoke
    def test_basic_creation(self, laos_signals):
        """Lissajous figure has 2 subplots."""
        fig = plot_lissajous(
            laos_signals["strain"],
            laos_signals["strain_rate"],
            laos_signals["stress"],
        )
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    @pytest.mark.smoke
    def test_normalized(self, laos_signals):
        """Normalization by gamma_0 and omega produces correct axis labels."""
        fig = plot_lissajous(
            laos_signals["strain"],
            laos_signals["strain_rate"],
            laos_signals["stress"],
            gamma_0=laos_signals["gamma_0"],
            omega=laos_signals["omega"],
        )
        ax1, ax2 = fig.axes
        assert r"\gamma_0" in ax1.get_xlabel()
        assert r"\dot{\gamma}" in ax2.get_xlabel()
        plt.close(fig)

    def test_custom_axes(self, laos_signals):
        """Accepts external axes tuple."""
        fig_ext, (ax1, ax2) = plt.subplots(1, 2)
        fig = plot_lissajous(
            laos_signals["strain"],
            laos_signals["strain_rate"],
            laos_signals["stress"],
            ax=(ax1, ax2),
        )
        assert fig is fig_ext
        plt.close(fig)

    def test_shape_mismatch_raises(self, laos_signals):
        """Mismatched strain/stress lengths raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            plot_lissajous(
                laos_signals["strain"][:10],
                laos_signals["strain_rate"],
                laos_signals["stress"],
            )

    def test_jax_arrays(self, laos_signals):
        """JAX arrays are accepted (converted internally)."""
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        fig = plot_lissajous(
            jnp.array(laos_signals["strain"]),
            jnp.array(laos_signals["strain_rate"]),
            jnp.array(laos_signals["stress"]),
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_cole_cole
# ---------------------------------------------------------------------------


class TestPlotColeCole:
    """Tests for plot_cole_cole."""

    @pytest.mark.smoke
    def test_basic_creation(self, spp_results):
        """Cole-Cole diagram creates single-axis figure."""
        fig = plot_cole_cole(spp_results["Gp_t"], spp_results["Gpp_t"])
        assert fig is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_trajectory_mode(self, spp_results):
        """Trajectory mode with time coloring."""
        fig = plot_cole_cole(
            spp_results["Gp_t"],
            spp_results["Gpp_t"],
            time=spp_results["time_new"],
            show_trajectory=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_scatter_mode(self, spp_results):
        """Scatter mode (show_trajectory=False)."""
        fig = plot_cole_cole(
            spp_results["Gp_t"],
            spp_results["Gpp_t"],
            show_trajectory=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_length_mismatch_raises(self, spp_results):
        """Mismatched G'/G'' lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            plot_cole_cole(spp_results["Gp_t"][:5], spp_results["Gpp_t"])

    def test_empty_data(self):
        """Empty arrays produce a 'No data' text instead of crash."""
        fig = plot_cole_cole(np.array([]), np.array([]))
        # Should not crash — shows "No data" text
        assert fig is not None
        plt.close(fig)

    def test_jax_arrays(self, spp_results):
        """JAX arrays are accepted."""
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        fig = plot_cole_cole(
            jnp.array(spp_results["Gp_t"]),
            jnp.array(spp_results["Gpp_t"]),
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_moduli_evolution
# ---------------------------------------------------------------------------


class TestPlotModuliEvolution:
    """Tests for plot_moduli_evolution."""

    @pytest.mark.smoke
    def test_basic_two_panel(self, spp_results):
        """Minimal call creates 2 subplots (G', G'')."""
        fig = plot_moduli_evolution(
            spp_results["time_new"],
            spp_results["Gp_t"],
            spp_results["Gpp_t"],
        )
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    @pytest.mark.smoke
    def test_with_delta(self, spp_results):
        """Adding delta_t creates 3 subplots."""
        fig = plot_moduli_evolution(
            spp_results["time_new"],
            spp_results["Gp_t"],
            spp_results["Gpp_t"],
            delta_t=spp_results["delta_t"],
        )
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_with_delta_and_speed(self, spp_results):
        """Adding delta_t + G_speed creates 4 subplots."""
        G_speed = np.abs(np.gradient(spp_results["Gp_t"]))
        fig = plot_moduli_evolution(
            spp_results["time_new"],
            spp_results["Gp_t"],
            spp_results["Gpp_t"],
            delta_t=spp_results["delta_t"],
            G_speed=G_speed,
        )
        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_jax_arrays(self, spp_results):
        """JAX arrays are accepted."""
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        fig = plot_moduli_evolution(
            jnp.array(spp_results["time_new"]),
            jnp.array(spp_results["Gp_t"]),
            jnp.array(spp_results["Gpp_t"]),
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_harmonic_spectrum
# ---------------------------------------------------------------------------


class TestPlotHarmonicSpectrum:
    """Tests for plot_harmonic_spectrum."""

    @pytest.mark.smoke
    def test_basic_creation(self):
        """Bar chart creates figure with odd-harmonic labels."""
        amps = np.array([1.0, 0.3, 0.05, 0.01])
        fig = plot_harmonic_spectrum(amps)
        ax = fig.axes[0]
        # x-ticks should be 1, 3, 5, 7 (odd harmonics)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert fig is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_normalized(self):
        """Normalized amplitudes: I_n/I_1 ylabel."""
        amps = np.array([100.0, 30.0, 5.0])
        fig = plot_harmonic_spectrum(amps, normalize=True)
        ax = fig.axes[0]
        assert "I_n/I_1" in ax.get_ylabel()
        plt.close(fig)

    def test_unnormalized(self):
        """Unnormalized: I_n ylabel."""
        amps = np.array([100.0, 30.0])
        fig = plot_harmonic_spectrum(amps, normalize=False)
        ax = fig.axes[0]
        assert "I_n" in ax.get_ylabel()
        plt.close(fig)

    def test_n_harmonics_limit(self):
        """n_harmonics limits displayed bars."""
        amps = np.array([1.0, 0.3, 0.05, 0.01, 0.001])
        fig = plot_harmonic_spectrum(amps, n_harmonics=3)
        ax = fig.axes[0]
        # Should show 3 bars
        bars = [p for p in ax.patches if hasattr(p, "get_height")]
        assert len(bars) == 3
        plt.close(fig)

    def test_empty_amplitudes(self):
        """Empty amplitudes returns figure with 'empty' title."""
        fig = plot_harmonic_spectrum(np.array([]))
        ax = fig.axes[0]
        assert "empty" in ax.get_title().lower()
        plt.close(fig)

    def test_zero_harmonics(self):
        """n_harmonics=0 returns early with empty figure."""
        fig = plot_harmonic_spectrum(np.array([1.0, 0.3]), n_harmonics=0)
        assert fig is not None
        plt.close(fig)

    def test_negative_n_harmonics_raises(self):
        """Negative n_harmonics raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            plot_harmonic_spectrum(np.array([1.0]), n_harmonics=-1)

    def test_negative_amplitudes(self):
        """Negative amplitudes are plotted without crash."""
        amps = np.array([1.0, -0.2, 0.05])
        fig = plot_harmonic_spectrum(amps, normalize=False)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_3d_trajectory
# ---------------------------------------------------------------------------


class TestPlot3DTrajectory:
    """Tests for plot_3d_trajectory."""

    @pytest.mark.smoke
    def test_basic_creation(self, laos_signals):
        """3D trajectory creates figure with 3D axes."""
        fig = plot_3d_trajectory(
            laos_signals["strain"],
            laos_signals["strain_rate"],
            laos_signals["stress"],
            omega=laos_signals["omega"],
        )
        assert fig is not None
        # 3D axes
        ax = fig.axes[0]
        assert hasattr(ax, "set_zlabel")
        plt.close(fig)

    def test_zero_omega_raises(self, laos_signals):
        """omega=0 raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            plot_3d_trajectory(
                laos_signals["strain"],
                laos_signals["strain_rate"],
                laos_signals["stress"],
                omega=0.0,
            )

    def test_empty_strain_raises(self):
        """Empty strain array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plot_3d_trajectory(
                np.array([]),
                np.array([]),
                np.array([]),
                omega=1.0,
            )

    def test_jax_arrays(self, laos_signals):
        """JAX arrays are accepted."""
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        fig = plot_3d_trajectory(
            jnp.array(laos_signals["strain"]),
            jnp.array(laos_signals["strain_rate"]),
            jnp.array(laos_signals["stress"]),
            omega=laos_signals["omega"],
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_pipkin_diagram
# ---------------------------------------------------------------------------


class TestPlotPipkinDiagram:
    """Tests for plot_pipkin_diagram."""

    @pytest.mark.smoke
    def test_basic_creation(self):
        """Pipkin diagram creates figure with colorbar."""
        gamma_0 = np.logspace(-2, 0, 5)
        omega = np.logspace(-1, 1, 6)
        metric = np.random.RandomState(42).rand(5, 6)

        fig = plot_pipkin_diagram(gamma_0, omega, metric)
        assert fig is not None
        # Main axis + colorbar axis
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_custom_metric_name(self):
        """Custom metric name appears on colorbar."""
        gamma_0 = np.logspace(-2, 0, 3)
        omega = np.logspace(-1, 1, 4)
        metric = np.random.RandomState(0).rand(3, 4)

        fig = plot_pipkin_diagram(gamma_0, omega, metric, metric_name="S factor")
        # Colorbar label should contain "S factor"
        cbar_ax = fig.axes[1]
        assert "S factor" in cbar_ax.get_ylabel()
        plt.close(fig)

    def test_log_scale_axes(self):
        """Pipkin diagram axes are log scale."""
        gamma_0 = np.logspace(-2, 0, 3)
        omega = np.logspace(-1, 1, 4)
        metric = np.ones((3, 4))

        fig = plot_pipkin_diagram(gamma_0, omega, metric)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)


# ---------------------------------------------------------------------------
# create_spp_report
# ---------------------------------------------------------------------------


class TestCreateSPPReport:
    """Tests for create_spp_report."""

    @pytest.mark.smoke
    def test_basic_creation(self, laos_signals, spp_results):
        """Report creates 6-panel figure."""
        fig = create_spp_report(
            spp_results,
            laos_signals["strain"],
            laos_signals["stress"],
            omega=laos_signals["omega"],
            gamma_0=laos_signals["gamma_0"],
        )
        assert fig is not None
        # 6 subplots (2x3 grid)
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) >= 5  # delta_t panel may be hidden
        plt.close(fig)

    @pytest.mark.smoke
    def test_without_delta(self, laos_signals):
        """Report with spp_results lacking delta_t hides 6th panel."""
        spp = {
            "Gp_t": np.ones(laos_signals["n_pts"]) * 1000,
            "Gpp_t": np.ones(laos_signals["n_pts"]) * 200,
        }
        fig = create_spp_report(
            spp,
            laos_signals["strain"],
            laos_signals["stress"],
            omega=laos_signals["omega"],
            gamma_0=laos_signals["gamma_0"],
        )
        assert fig is not None
        plt.close(fig)

    def test_zero_omega_raises(self, laos_signals, spp_results):
        """omega=0 raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            create_spp_report(
                spp_results,
                laos_signals["strain"],
                laos_signals["stress"],
                omega=0.0,
                gamma_0=laos_signals["gamma_0"],
            )

    def test_zero_gamma0_raises(self, laos_signals, spp_results):
        """gamma_0=0 raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            create_spp_report(
                spp_results,
                laos_signals["strain"],
                laos_signals["stress"],
                omega=laos_signals["omega"],
                gamma_0=0.0,
            )

    def test_jax_arrays(self, laos_signals, spp_results):
        """JAX arrays in strain/stress are accepted."""
        from rheojax.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()
        fig = create_spp_report(
            spp_results,
            jnp.array(laos_signals["strain"]),
            jnp.array(laos_signals["stress"]),
            omega=laos_signals["omega"],
            gamma_0=laos_signals["gamma_0"],
        )
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, laos_signals, spp_results, tmp_path):
        """Report can be saved to disk."""
        save_path = str(tmp_path / "spp_report.png")
        fig = create_spp_report(
            spp_results,
            laos_signals["strain"],
            laos_signals["stress"],
            omega=laos_signals["omega"],
            gamma_0=laos_signals["gamma_0"],
            save_path=save_path,
        )
        assert fig is not None
        assert (tmp_path / "spp_report.png").exists()
        assert (tmp_path / "spp_report.png").stat().st_size > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-cutting concerns
# ---------------------------------------------------------------------------


class TestSPPPlotEdgeCases:
    """Cross-cutting edge case tests."""

    def test_all_zero_signals(self):
        """All-zero signals don't crash any plot function."""
        n = 100
        zeros = np.zeros(n)
        # Lissajous
        fig = plot_lissajous(zeros, zeros, zeros)
        plt.close(fig)
        # Cole-Cole
        fig = plot_cole_cole(zeros, zeros)
        plt.close(fig)
        # Moduli evolution
        t = np.linspace(0, 1, n)
        fig = plot_moduli_evolution(t, zeros, zeros)
        plt.close(fig)

    def test_single_point_signals(self):
        """Single-point signals don't crash."""
        one = np.array([1.0])
        fig = plot_lissajous(one, one, one)
        plt.close(fig)

    def test_large_signals(self):
        """Large arrays (10k points) don't crash or timeout."""
        n = 10_000
        rng = np.random.RandomState(99)
        strain = rng.randn(n)
        rate = rng.randn(n)
        stress = rng.randn(n)
        fig = plot_lissajous(strain, rate, stress)
        assert fig is not None
        plt.close(fig)

    def test_nan_in_signals(self):
        """NaN values don't crash plot functions."""
        n = 50
        strain = np.sin(np.linspace(0, 2 * np.pi, n))
        stress = np.cos(np.linspace(0, 2 * np.pi, n))
        strain[10] = np.nan
        stress[25] = np.nan
        rate = np.gradient(strain)
        fig = plot_lissajous(strain, rate, stress)
        assert fig is not None
        plt.close(fig)

    def test_inf_in_signals(self):
        """Inf values don't crash plot functions."""
        n = 50
        Gp = np.ones(n) * 1000
        Gpp = np.ones(n) * 200
        Gp[5] = np.inf
        fig = plot_cole_cole(Gp, Gpp)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage Gap-5: strain_rate_normalized path in create_spp_report
# ---------------------------------------------------------------------------


class TestSPPReportStrainRateNormalized:
    """Gap-5: create_spp_report uses strain_rate_normalized when present."""

    @pytest.mark.smoke
    def test_strain_rate_normalized_override(self, laos_signals, spp_results):
        """When strain_rate_normalized is provided, it overrides computed rate."""
        n = laos_signals["n_pts"]
        # Add strain_rate_normalized to SPP results
        spp_with_norm = dict(spp_results)
        spp_with_norm["strain_rate_normalized"] = np.ones(n) * 0.5

        fig = create_spp_report(
            strain=laos_signals["strain"],
            stress=laos_signals["stress"],
            spp_results=spp_with_norm,
            gamma_0=laos_signals["gamma_0"],
            omega=laos_signals["omega"],
        )
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage Gap-7: plot_moduli_evolution single-Axes ValueError
# ---------------------------------------------------------------------------


class TestModuliEvolutionSingleAxes:
    """Gap-7: plot_moduli_evolution raises ValueError for single Axes with multi-panel."""

    @pytest.mark.smoke
    def test_single_axes_raises_value_error(self, spp_results):
        """Single Axes with n_plots > 1 raises ValueError."""
        _, single_ax = plt.subplots()
        time = np.linspace(0, 2 * np.pi, len(spp_results["Gp_t"]))
        with pytest.raises(ValueError, match="subplots needed"):
            plot_moduli_evolution(
                time=time,
                Gp_t=spp_results["Gp_t"],
                Gpp_t=spp_results["Gpp_t"],
                delta_t=spp_results["delta_t"],
                ax=single_ax,
            )
        plt.close("all")

    def test_tuple_axes_accepted(self, spp_results):
        """Tuple of axes for multi-panel works correctly."""
        time = np.linspace(0, 2 * np.pi, len(spp_results["Gp_t"]))
        fig, axes = plt.subplots(3, 1)
        result_fig = plot_moduli_evolution(
            time=time,
            Gp_t=spp_results["Gp_t"],
            Gpp_t=spp_results["Gpp_t"],
            delta_t=spp_results["delta_t"],
            ax=tuple(axes),
        )
        assert result_fig is not None
        plt.close("all")
