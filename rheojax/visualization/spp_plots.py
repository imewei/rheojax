"""SPP (Sequence of Physical Processes) visualization module.

This module provides specialized plotting functions for SPP analysis of LAOS data,
matching the visualization capabilities of MATLAB SPPplus and R oreo packages.

Key Plots
---------
- Lissajous-Bowditch plots (sigma vs gamma, sigma vs gamma_dot)
- Cole-Cole diagram (G'' vs G')
- Time-resolved moduli evolution (G'(t), G''(t), delta(t) vs t)
- Pipkin diagram (amplitude-frequency map)
- Harmonic spectrum bar charts
- Frenet-Serret frame 3D trajectory

References
----------
- S.A. Rogers, "In search of physical meaning: defining transient parameters
  for nonlinear viscoelasticity", Rheol. Acta 56, 2017
- MATLAB SPPplus: SPPplus_figures_v2.m
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Module logger
logger = get_logger(__name__)


def _ensure_matplotlib():
    """Ensure matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "Matplotlib is required for SPP visualization. "
            "Install with: pip install matplotlib"
        ) from e


def plot_lissajous(
    strain: np.ndarray,
    strain_rate: np.ndarray,
    stress: np.ndarray,
    gamma_0: float | None = None,
    omega: float | None = None,
    ax: Axes | tuple[Axes, Axes] | None = None,
    style: str = "publication",
    **kwargs,
) -> Figure:
    """
    Create Lissajous-Bowditch plots (stress vs strain and stress vs strain rate).

    Parameters
    ----------
    strain : np.ndarray
        Strain signal gamma(t) (dimensionless)
    strain_rate : np.ndarray
        Strain rate signal gamma_dot(t) (1/s)
    stress : np.ndarray
        Stress signal sigma(t) (Pa)
    gamma_0 : float, optional
        Strain amplitude for normalization
    omega : float, optional
        Angular frequency for normalization
    ax : Axes or tuple of Axes, optional
        Matplotlib axes to plot on. If None, creates new figure with 2 subplots.
    style : str
        Plot style: 'publication', 'presentation', or 'minimal'
    **kwargs
        Additional keyword arguments passed to plot()

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug("Generating plot", plot_type="lissajous", style=style)

    try:
        plt = _ensure_matplotlib()

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        elif isinstance(ax, tuple):
            fig = ax[0].get_figure()
            ax1, ax2 = ax
        else:
            fig = ax.get_figure()
            ax1 = ax2 = ax

        # Normalize if amplitudes provided
        strain_plot = strain / gamma_0 if gamma_0 else strain
        rate_plot = (
            strain_rate / (gamma_0 * omega) if (gamma_0 and omega) else strain_rate
        )
        stress_plot = stress

        # Style settings
        if style == "publication":
            linewidth = 1.5
            fontsize = 12
        elif style == "presentation":
            linewidth = 2.5
            fontsize = 14
        else:
            linewidth = 1.0
            fontsize = 10

        # Plot sigma vs gamma (elastic Lissajous)
        ax1.plot(strain_plot, stress_plot, linewidth=linewidth, **kwargs)
        ax1.set_xlabel(
            r"$\gamma/\gamma_0$" if gamma_0 else r"$\gamma$", fontsize=fontsize
        )
        ax1.set_ylabel(r"$\sigma$ (Pa)", fontsize=fontsize)
        ax1.set_title("Elastic Lissajous", fontsize=fontsize + 2)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax1.axvline(0, color="gray", linestyle="--", linewidth=0.5)

        # Plot sigma vs gamma_dot (viscous Lissajous)
        ax2.plot(rate_plot, stress_plot, linewidth=linewidth, **kwargs)
        ax2.set_xlabel(
            (
                r"$\dot{\gamma}/\dot{\gamma}_0$"
                if (gamma_0 and omega)
                else r"$\dot{\gamma}$ (1/s)"
            ),
            fontsize=fontsize,
        )
        ax2.set_ylabel(r"$\sigma$ (Pa)", fontsize=fontsize)
        ax2.set_title("Viscous Lissajous", fontsize=fontsize + 2)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax2.axvline(0, color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()

        logger.debug("Figure created", plot_type="lissajous")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate lissajous plot",
            plot_type="lissajous",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_cole_cole(
    Gp_t: np.ndarray,
    Gpp_t: np.ndarray,
    time: np.ndarray | None = None,
    ax: Axes | None = None,
    colormap: str = "viridis",
    show_trajectory: bool = True,
    **kwargs,
) -> Figure:
    """
    Create Cole-Cole diagram (G'' vs G') with optional time coloring.

    Parameters
    ----------
    Gp_t : np.ndarray
        Instantaneous storage modulus G'(t) (Pa)
    Gpp_t : np.ndarray
        Instantaneous loss modulus G''(t) (Pa)
    time : np.ndarray, optional
        Time array for colormap. If None, uses index.
    ax : Axes, optional
        Matplotlib axes to plot on
    colormap : str
        Colormap name for trajectory coloring
    show_trajectory : bool
        If True, show as colored trajectory. If False, show as scatter.
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug("Generating plot", plot_type="cole_cole", colormap=colormap)

    try:
        plt = _ensure_matplotlib()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        if time is None:
            time = np.arange(len(Gp_t))

        if show_trajectory:
            # Colored line showing trajectory evolution
            from matplotlib.collections import LineCollection

            points = np.array([Gp_t, Gpp_t]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=colormap, linewidth=2)
            lc.set_array(time[:-1])
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label="Time (s)")
            ax.autoscale()
        else:
            scatter = ax.scatter(Gp_t, Gpp_t, c=time, cmap=colormap, s=10, **kwargs)
            fig.colorbar(scatter, ax=ax, label="Time (s)")

        # Add start/end markers
        ax.plot(Gp_t[0], Gpp_t[0], "go", markersize=10, label="Start", zorder=5)
        ax.plot(Gp_t[-1], Gpp_t[-1], "rs", markersize=10, label="End", zorder=5)

        ax.set_xlabel(r"$G'(t)$ (Pa)", fontsize=12)
        ax.set_ylabel(r"$G''(t)$ (Pa)", fontsize=12)
        ax.set_title("Cole-Cole Diagram", fontsize=14)
        ax.legend(loc="upper right")
        ax.set_aspect("equal", adjustable="box")

        # Add reference lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)

        logger.debug("Figure created", plot_type="cole_cole")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate cole_cole plot",
            plot_type="cole_cole",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_moduli_evolution(
    time: np.ndarray,
    Gp_t: np.ndarray,
    Gpp_t: np.ndarray,
    delta_t: np.ndarray | None = None,
    G_speed: np.ndarray | None = None,
    ax: Axes | tuple | None = None,
    **kwargs,
) -> Figure:
    """
    Plot time-resolved moduli evolution (G'(t), G''(t), delta(t) vs t).

    Parameters
    ----------
    time : np.ndarray
        Time array (s)
    Gp_t : np.ndarray
        Instantaneous storage modulus G'(t) (Pa)
    Gpp_t : np.ndarray
        Instantaneous loss modulus G''(t) (Pa)
    delta_t : np.ndarray, optional
        Instantaneous phase angle delta(t) (radians)
    G_speed : np.ndarray, optional
        Moduli rate magnitude |dG*/dt| (Pa/s)
    ax : Axes or tuple, optional
        Matplotlib axes
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug("Generating plot", plot_type="moduli_evolution")

    try:
        plt = _ensure_matplotlib()

        n_plots = (
            2 + (1 if delta_t is not None else 0) + (1 if G_speed is not None else 0)
        )

        if ax is None:
            fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
            if n_plots == 1:
                axes = [axes]
        else:
            if isinstance(ax, tuple):
                axes = list(ax)
            else:
                axes = [ax]
            fig = axes[0].get_figure()

        idx = 0

        # G'(t) plot
        axes[idx].plot(time, Gp_t, "b-", linewidth=1.5, label=r"$G'(t)$")
        axes[idx].set_ylabel(r"$G'(t)$ (Pa)", color="b")
        axes[idx].tick_params(axis="y", labelcolor="b")
        axes[idx].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        idx += 1

        # G''(t) plot
        axes[idx].plot(time, Gpp_t, "r-", linewidth=1.5, label=r"$G''(t)$")
        axes[idx].set_ylabel(r"$G''(t)$ (Pa)", color="r")
        axes[idx].tick_params(axis="y", labelcolor="r")
        axes[idx].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        idx += 1

        # delta(t) plot
        if delta_t is not None and idx < len(axes):
            axes[idx].plot(
                time, np.degrees(delta_t), "g-", linewidth=1.5, label=r"$\delta(t)$"
            )
            axes[idx].set_ylabel(r"$\delta(t)$ (deg)", color="g")
            axes[idx].tick_params(axis="y", labelcolor="g")
            axes[idx].axhline(45, color="gray", linestyle=":", linewidth=0.5)
            axes[idx].axhline(90, color="gray", linestyle="--", linewidth=0.5)
            idx += 1

        # G_speed plot
        if G_speed is not None and idx < len(axes):
            axes[idx].plot(time, G_speed, "m-", linewidth=1.5, label=r"$|\dot{G}^*|$")
            axes[idx].set_ylabel(r"$|\dot{G}^*|$ (Pa/s)", color="m")
            axes[idx].tick_params(axis="y", labelcolor="m")
            idx += 1

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Time-Resolved Moduli Evolution", fontsize=14)
        plt.tight_layout()

        logger.debug("Figure created", plot_type="moduli_evolution")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate moduli_evolution plot",
            plot_type="moduli_evolution",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_harmonic_spectrum(
    amplitudes: np.ndarray,
    n_harmonics: int | None = None,
    normalize: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> Figure:
    """
    Plot harmonic spectrum bar chart (I_n/I_1 vs harmonic number).

    Parameters
    ----------
    amplitudes : np.ndarray
        Harmonic amplitudes [I_1, I_3, I_5, ...] (Pa)
    n_harmonics : int, optional
        Number of harmonics to show. If None, shows all.
    normalize : bool
        If True, normalize by fundamental (I_1)
    ax : Axes, optional
        Matplotlib axes
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug(
        "Generating plot",
        plot_type="harmonic_spectrum",
        n_harmonics=n_harmonics,
        normalize=normalize,
    )

    try:
        plt = _ensure_matplotlib()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        if n_harmonics is None:
            n_harmonics = len(amplitudes)
        else:
            n_harmonics = min(n_harmonics, len(amplitudes))

        amps = amplitudes[:n_harmonics]
        harmonics = [2 * i + 1 for i in range(n_harmonics)]  # 1, 3, 5, ...

        if normalize and amps[0] > 0:
            amps = amps / amps[0]
            ylabel = r"$I_n/I_1$"
        else:
            ylabel = r"$I_n$ (Pa)"

        bars = ax.bar(harmonics, amps, color="steelblue", edgecolor="black", **kwargs)
        ax.set_xlabel("Harmonic Number n", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title("Fourier Harmonic Spectrum", fontsize=14)
        ax.set_xticks(harmonics)

        # Add value labels on bars
        for bar, amp in zip(bars, amps, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{amp:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        logger.debug("Figure created", plot_type="harmonic_spectrum")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate harmonic_spectrum plot",
            plot_type="harmonic_spectrum",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_3d_trajectory(
    strain: np.ndarray,
    strain_rate: np.ndarray,
    stress: np.ndarray,
    omega: float = 1.0,
    T_vec: np.ndarray | None = None,
    N_vec: np.ndarray | None = None,
    B_vec: np.ndarray | None = None,
    ax: Axes | None = None,
    show_frame: bool = False,
    frame_scale: float = 0.1,
    **kwargs,
) -> Figure:
    """
    Plot 3D (gamma, gamma_dot/omega, sigma) trajectory with optional Frenet-Serret frame.

    Parameters
    ----------
    strain : np.ndarray
        Strain signal gamma(t)
    strain_rate : np.ndarray
        Strain rate signal gamma_dot(t) (1/s)
    stress : np.ndarray
        Stress signal sigma(t) (Pa)
    omega : float
        Angular frequency for rate normalization (rad/s)
    T_vec, N_vec, B_vec : np.ndarray, optional
        Frenet-Serret frame vectors (n_points, 3)
    ax : Axes, optional
        3D matplotlib axes
    show_frame : bool
        If True and frame vectors provided, show Frenet-Serret vectors
    frame_scale : float
        Scale factor for frame vectors
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug("Generating plot", plot_type="3d_trajectory", show_frame=show_frame)

    try:
        plt = _ensure_matplotlib()

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        # Normalize rate by omega
        rate_normalized = strain_rate / omega

        # Plot trajectory
        ax.plot(strain, rate_normalized, stress, linewidth=1.5, **kwargs)

        # Add start/end markers
        ax.scatter(
            [strain[0]],
            [rate_normalized[0]],
            [stress[0]],
            color="g",
            s=100,
            label="Start",
        )
        ax.scatter(
            [strain[-1]],
            [rate_normalized[-1]],
            [stress[-1]],
            color="r",
            s=100,
            label="End",
        )

        # Show Frenet-Serret frame at selected points
        if show_frame and T_vec is not None:
            n_arrows = 10  # Number of frame vectors to show
            indices = np.linspace(0, len(strain) - 1, n_arrows, dtype=int)

            for i in indices:
                pos = np.array([strain[i], rate_normalized[i], stress[i]])

                # Scale vectors for visibility
                scale = frame_scale * np.max(np.abs(stress))

                if T_vec is not None:
                    ax.quiver(
                        *pos, *T_vec[i] * scale, color="red", arrow_length_ratio=0.3
                    )
                if N_vec is not None:
                    ax.quiver(
                        *pos, *N_vec[i] * scale, color="green", arrow_length_ratio=0.3
                    )
                if B_vec is not None:
                    ax.quiver(
                        *pos, *B_vec[i] * scale, color="blue", arrow_length_ratio=0.3
                    )

        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\dot{\gamma}/\omega$")
        ax.set_zlabel(r"$\sigma$ (Pa)")
        ax.set_title("3D Response Trajectory", fontsize=14)
        ax.legend()

        logger.debug("Figure created", plot_type="3d_trajectory")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate 3d_trajectory plot",
            plot_type="3d_trajectory",
            error=str(e),
            exc_info=True,
        )
        raise


def plot_pipkin_diagram(
    gamma_0_values: np.ndarray,
    omega_values: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str = "I_3/I_1",
    ax: Axes | None = None,
    cmap: str = "viridis",
    **kwargs,
) -> Figure:
    """
    Create Pipkin diagram (gamma_0 vs omega map) colored by a nonlinearity metric.

    Parameters
    ----------
    gamma_0_values : np.ndarray
        Strain amplitude values (1D array)
    omega_values : np.ndarray
        Frequency values (1D array, rad/s)
    metric_values : np.ndarray
        Metric values on (gamma_0, omega) grid (2D array)
    metric_name : str
        Name of the metric for colorbar label
    ax : Axes, optional
        Matplotlib axes
    cmap : str
        Colormap name
    **kwargs
        Additional keyword arguments for pcolormesh

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug("Generating plot", plot_type="pipkin_diagram", metric_name=metric_name)

    try:
        plt = _ensure_matplotlib()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        # Create meshgrid
        Omega, Gamma = np.meshgrid(omega_values, gamma_0_values)

        # Plot as heatmap
        mesh = ax.pcolormesh(
            Omega, Gamma, metric_values, cmap=cmap, shading="auto", **kwargs
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(metric_name, fontsize=12)

        ax.set_xlabel(r"$\omega$ (rad/s)", fontsize=12)
        ax.set_ylabel(r"$\gamma_0$", fontsize=12)
        ax.set_title("Pipkin Diagram", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")

        logger.debug("Figure created", plot_type="pipkin_diagram")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate pipkin_diagram plot",
            plot_type="pipkin_diagram",
            error=str(e),
            exc_info=True,
        )
        raise


def create_spp_report(
    spp_results: dict,
    strain: np.ndarray,
    stress: np.ndarray,
    omega: float,
    gamma_0: float,
    save_path: str | None = None,
    **kwargs,
) -> Figure:
    """
    Create comprehensive SPP analysis report figure.

    Generates a multi-panel figure with:
    - Lissajous plots
    - Cole-Cole diagram
    - Time-resolved moduli
    - Harmonic spectrum (if available)

    Parameters
    ----------
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    strain : np.ndarray
        Strain signal
    stress : np.ndarray
        Stress signal
    omega : float
        Angular frequency (rad/s)
    gamma_0 : float
        Strain amplitude
    save_path : str, optional
        Path to save figure
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    logger.debug(
        "Generating plot",
        plot_type="spp_report",
        omega=omega,
        gamma_0=gamma_0,
        save_path=save_path,
    )

    try:
        plt = _ensure_matplotlib()

        fig = plt.figure(figsize=(14, 10))

        # Create grid of subplots
        ax1 = fig.add_subplot(2, 3, 1)  # Elastic Lissajous
        ax2 = fig.add_subplot(2, 3, 2)  # Viscous Lissajous
        ax3 = fig.add_subplot(2, 3, 3)  # Cole-Cole
        ax4 = fig.add_subplot(2, 3, 4)  # G'(t)
        ax5 = fig.add_subplot(2, 3, 5)  # G''(t)
        ax6 = fig.add_subplot(2, 3, 6)  # delta(t)

        # Extract results
        Gp_t = np.asarray(spp_results["Gp_t"])
        Gpp_t = np.asarray(spp_results["Gpp_t"])
        delta_t = np.asarray(spp_results["delta_t"])

        # Compute strain rate
        strain_rate = (
            gamma_0
            * omega
            * np.cos(omega * np.linspace(0, 2 * np.pi / omega, len(strain)))
        )
        if "strain_rate_normalized" in spp_results:
            strain_rate = np.asarray(spp_results["strain_rate_normalized"]) * omega

        time = np.linspace(0, 2 * np.pi / omega, len(strain))

        # Lissajous plots
        ax1.plot(strain / gamma_0, stress, "b-", linewidth=1.5)
        ax1.set_xlabel(r"$\gamma/\gamma_0$")
        ax1.set_ylabel(r"$\sigma$ (Pa)")
        ax1.set_title("Elastic Lissajous")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax1.axvline(0, color="gray", linestyle="--", linewidth=0.5)

        ax2.plot(strain_rate / (gamma_0 * omega), stress, "r-", linewidth=1.5)
        ax2.set_xlabel(r"$\dot{\gamma}/\dot{\gamma}_0$")
        ax2.set_ylabel(r"$\sigma$ (Pa)")
        ax2.set_title("Viscous Lissajous")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax2.axvline(0, color="gray", linestyle="--", linewidth=0.5)

        # Cole-Cole
        ax3.plot(Gp_t, Gpp_t, "k-", linewidth=1.5)
        ax3.plot(Gp_t[0], Gpp_t[0], "go", markersize=8, label="Start")
        ax3.plot(Gp_t[-1], Gpp_t[-1], "rs", markersize=8, label="End")
        ax3.set_xlabel(r"$G'(t)$ (Pa)")
        ax3.set_ylabel(r"$G''(t)$ (Pa)")
        ax3.set_title("Cole-Cole Diagram")
        ax3.legend(loc="upper right")

        # Time-resolved moduli
        ax4.plot(time, Gp_t, "b-", linewidth=1.5)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel(r"$G'(t)$ (Pa)")
        ax4.set_title(r"Storage Modulus $G'(t)$")

        ax5.plot(time, Gpp_t, "r-", linewidth=1.5)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel(r"$G''(t)$ (Pa)")
        ax5.set_title(r"Loss Modulus $G''(t)$")

        ax6.plot(time, np.degrees(delta_t), "g-", linewidth=1.5)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel(r"$\delta(t)$ (deg)")
        ax6.set_title(r"Phase Angle $\delta(t)$")
        ax6.axhline(45, color="gray", linestyle=":", linewidth=0.5)
        ax6.axhline(90, color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.debug("Figure saved", plot_type="spp_report", save_path=save_path)

        logger.debug("Figure created", plot_type="spp_report")
        return fig

    except Exception as e:
        logger.error(
            "Failed to generate spp_report plot",
            plot_type="spp_report",
            error=str(e),
            exc_info=True,
        )
        raise


__all__ = [
    "plot_lissajous",
    "plot_cole_cole",
    "plot_moduli_evolution",
    "plot_harmonic_spectrum",
    "plot_3d_trajectory",
    "plot_pipkin_diagram",
    "create_spp_report",
]
