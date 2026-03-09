"""Standardized transform result visualization.

This module provides the TransformPlotter class for creating publication-quality
visualizations of transform results, with per-transform layout dispatch and
optional before/after comparisons.

Supported transforms:
- FFT analysis (time signal → magnitude spectrum)
- Mastercurve / TTS (multi-T overlay → shifted mastercurve)
- SRFS (strain-rate frequency superposition)
- Mutation number (scalar diagnostic)
- OWChirp (wavelet time-frequency analysis)
- Smooth derivative (original → derivative)
- SPP decomposer (Lissajous, G't/G''t, Cole-Cole)
- Prony conversion (time ↔ frequency domain)
- Spectrum inversion (relaxation spectrum H(τ))
- Cox-Merz (|η*| vs η overlay)
- LVE envelope (startup stress envelope)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rheojax.core.data import RheoData
from rheojax.logging import get_logger
from rheojax.visualization.plotter import (
    _apply_style,
    _ensure_numpy,
    _filter_positive,
    _modulus_labels,
)

logger = get_logger(__name__)

_GRID_ALPHA = 0.3
_GRID_LINESTYLE = "--"


class TransformPlotter:
    """Protocol-aware transform result plotting.

    Dispatches to the correct layout based on transform name. Each transform
    gets a tailored multi-panel figure showing the most useful views of
    its output.

    All methods are stateless — data and results are passed explicitly.

    Examples
    --------
    Plot FFT analysis result:

    >>> plotter = TransformPlotter()
    >>> fig, axes = plotter.plot("fft", fft_result, input_data=raw_data)

    Plot mastercurve with shift factors:

    >>> fig, axes = plotter.plot(
    ...     "mastercurve", (mastercurve_data, shift_factors),
    ...     input_data=datasets
    ... )
    """

    # Registry mapping transform names to plot methods
    _plot_methods: dict[str, str] = {
        "fft": "_plot_fft",
        "fft_analysis": "_plot_fft",
        "mastercurve": "_plot_mastercurve",
        "tts": "_plot_mastercurve",
        "srfs": "_plot_srfs",
        "mutation_number": "_plot_mutation",
        "owchirp": "_plot_owchirp",
        "smooth_derivative": "_plot_derivative",
        "derivative": "_plot_derivative",
        "spp": "_plot_spp",
        "spp_decomposer": "_plot_spp",
        "prony_conversion": "_plot_prony",
        "prony": "_plot_prony",
        "spectrum_inversion": "_plot_spectrum",
        "cox_merz": "_plot_cox_merz",
        "lve_envelope": "_plot_envelope",
    }

    def plot(
        self,
        transform_name: str,
        result: Any,
        input_data: RheoData | list[RheoData] | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray]:
        """Auto-dispatch to the correct plot method for a transform.

        Parameters
        ----------
        transform_name : str
            Registry name of the transform (e.g., 'fft', 'mastercurve').
        result : RheoData or tuple or dict
            Output from ``transform.transform()``.
        input_data : RheoData or list[RheoData], optional
            Original data before the transform. If provided, enables
            before/after comparison panels.
        show_intermediate : bool
            Whether to show intermediate processing steps (default: True).
        style : str
            Plot style ('default', 'publication', 'presentation').
        **kwargs
            Additional arguments forwarded to the specific plot method.

        Returns
        -------
        tuple[Figure, Axes or ndarray]

        Raises
        ------
        ValueError
            If ``transform_name`` is not recognized.
        """
        name_lower = transform_name.lower().replace("-", "_").replace(" ", "_")
        method_name = self._plot_methods.get(name_lower)

        if method_name is None:
            # Fallback: generic before/after plot
            logger.debug(
                "No specific plot method for transform, using generic",
                transform=transform_name,
            )
            return self._plot_generic(
                result, input_data=input_data, style=style,
                transform_name=transform_name, **kwargs,
            )

        method = getattr(self, method_name)
        return method(
            result, input_data=input_data, show_intermediate=show_intermediate,
            style=style, **kwargs,
        )

    # ------------------------------------------------------------------
    # Per-transform plot methods
    # ------------------------------------------------------------------

    def _plot_fft(
        self,
        result: RheoData,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """FFT analysis: time signal + magnitude spectrum.

        Layout:
        ┌──────────────┬──────────────┐
        │ Time signal  │  Spectrum    │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)

        if input_data is not None and show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))

            # Left: time-domain input
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            axes[0].plot(
                x_in, y_in,
                linewidth=style_params["lines.linewidth"],
            )
            axes[0].set_xlabel(
                f"Time ({input_data.x_units})" if input_data.x_units else "Time (s)"
            )
            axes[0].set_ylabel(
                f"Signal ({input_data.y_units})" if input_data.y_units else "Signal"
            )
            axes[0].set_title("Input Signal")
            axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Right: frequency spectrum
            self._plot_spectrum_panel(axes[1], result, style_params)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            self._plot_spectrum_panel(ax, result, style_params)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_spectrum_panel(
        self, ax: Axes, result: RheoData, style_params: dict
    ) -> None:
        """Plot a single FFT spectrum panel."""
        x_spec = _ensure_numpy(result.x)
        y_spec = _ensure_numpy(result.y)

        # Filter positive for log-log
        mask = (x_spec > 0) & (y_spec > 0) & np.isfinite(x_spec) & np.isfinite(y_spec)
        if np.any(mask):
            ax.loglog(
                x_spec[mask], y_spec[mask],
                linewidth=style_params["lines.linewidth"],
            )
        else:
            ax.plot(x_spec, y_spec, linewidth=style_params["lines.linewidth"])

        meta = result.metadata or {}
        is_psd = meta.get("psd", False)
        y_label = "PSD" if is_psd else "Magnitude"
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(y_label)
        ax.set_title("FFT Spectrum")
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

    def _plot_mastercurve(
        self,
        result: Any,
        input_data: list[RheoData] | RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Mastercurve (TTS): unshifted overlay + shifted mastercurve.

        Layout:
        ┌──────────────┬──────────────┐
        │  Unshifted   │   Shifted    │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)

        # Unpack result
        mastercurve_data, shift_factors = self._unpack_result(result)

        has_input = (
            input_data is not None
            and show_intermediate
            and isinstance(input_data, list)
            and len(input_data) > 1
        )

        if has_input:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.6,
                style_params["figure.figsize"][1],
            ))

            # Left: unshifted multi-temperature overlay
            colors = plt.cm.viridis(np.linspace(0, 1, len(input_data)))
            for i, data in enumerate(input_data):
                temp = (data.metadata or {}).get("temperature", f"#{i+1}")
                x_d = _ensure_numpy(data.x)
                y_d = _ensure_numpy(data.y)
                if np.iscomplexobj(y_d):
                    x_f, y_f = _filter_positive(x_d, np.real(y_d), warn=False)
                    axes[0].loglog(
                        x_f, y_f, "o", color=colors[i],
                        markersize=style_params["lines.markersize"],
                        markerfacecolor="none", markeredgewidth=1.0,
                        label=f"{temp}",
                    )
                else:
                    x_f, y_f = _filter_positive(x_d, y_d, warn=False)
                    axes[0].loglog(
                        x_f, y_f, "o", color=colors[i],
                        markersize=style_params["lines.markersize"],
                        markerfacecolor="none", markeredgewidth=1.0,
                        label=f"{temp}",
                    )
            axes[0].set_title("Unshifted Data")
            axes[0].set_xlabel("Frequency")
            axes[0].set_ylabel("Modulus")
            axes[0].legend(fontsize=style_params["legend.fontsize"], loc="best")
            axes[0].grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Right: shifted mastercurve
            self._plot_mastercurve_panel(
                axes[1], mastercurve_data, shift_factors, style_params,
            )

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            self._plot_mastercurve_panel(ax, mastercurve_data, shift_factors, style_params)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_mastercurve_panel(
        self,
        ax: Axes,
        data: RheoData,
        shift_factors: dict | None,
        style_params: dict,
    ) -> None:
        """Single panel for the shifted mastercurve."""
        x_mc = _ensure_numpy(data.x)
        y_mc = _ensure_numpy(data.y)
        storage_lbl, loss_lbl, generic_lbl = _modulus_labels(data)

        if np.iscomplexobj(y_mc):
            x_gp, gp = _filter_positive(x_mc, np.real(y_mc), warn=False)
            x_gpp, gpp = _filter_positive(x_mc, np.imag(y_mc), warn=False)
            ax.loglog(
                x_gp, gp, "o", color="C0",
                markersize=style_params["lines.markersize"],
                markerfacecolor="none", markeredgewidth=1.0,
                label=storage_lbl,
            )
            if len(x_gpp) > 0:
                ax.loglog(
                    x_gpp, gpp, "s", color="C1", alpha=0.7,
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none", markeredgewidth=1.0,
                    label=loss_lbl,
                )
        else:
            x_f, y_f = _filter_positive(x_mc, y_mc, warn=False)
            ax.loglog(
                x_f, y_f, "o", color="C0",
                markersize=style_params["lines.markersize"],
                markerfacecolor="none", markeredgewidth=1.0,
            )

        ref_temp = (data.metadata or {}).get("reference_temperature", "?")
        ax.set_title(f"Mastercurve (T_ref = {ref_temp})")
        ax.set_xlabel("Shifted Frequency")
        ax.set_ylabel(generic_lbl)
        ax.legend(fontsize=style_params["legend.fontsize"], loc="best")
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

    def _plot_srfs(
        self,
        result: Any,
        input_data: list[RheoData] | RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """SRFS: same layout as mastercurve but with strain-rate axis."""
        # Reuse mastercurve layout — SRFS has identical structure
        return self._plot_mastercurve(
            result, input_data=input_data,
            show_intermediate=show_intermediate, style=style, **kwargs,
        )

    def _plot_mutation(
        self,
        result: RheoData,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Mutation number: scalar diagnostic with guidelines.

        Layout: Single panel with Nm value, 0/1 reference lines.
        """
        style_params = _apply_style(style)
        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        meta = result.metadata or {}
        nm = meta.get("mutation_number", float(_ensure_numpy(result.y)[0]))

        # Bar chart showing mutation number
        bar_color = "C0" if nm < 0.5 else "C1"
        ax.barh(["Nm"], [nm], color=bar_color, height=0.4, alpha=0.8)

        # Reference lines
        ax.axvline(x=0, color="C2", linestyle="--", linewidth=1, alpha=0.7, label="Elastic (Nm=0)")
        ax.axvline(x=1, color="C3", linestyle="--", linewidth=1, alpha=0.7, label="Viscous (Nm=1)")
        ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlim(-0.1, 1.1)
        ax.set_xlabel("Mutation Number")
        ax.set_title(f"Mutation Number: Nm = {nm:.4f}")
        ax.legend(fontsize=style_params["legend.fontsize"])
        ax.grid(True, axis="x", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()
        return fig, ax

    def _plot_owchirp(
        self,
        result: RheoData,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """OWChirp: input chirp + wavelet spectrum.

        Layout:
        ┌──────────────┬──────────────┐
        │ Input chirp  │  Spectrum    │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)

        if input_data is not None and show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))

            # Left: input chirp signal
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            axes[0].plot(x_in, y_in, linewidth=style_params["lines.linewidth"])
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Signal")
            axes[0].set_title("Input Chirp")
            axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Right: wavelet spectrum
            x_out = _ensure_numpy(result.x)
            y_out = _ensure_numpy(result.y)
            mask = (x_out > 0) & (y_out > 0) & np.isfinite(x_out) & np.isfinite(y_out)
            if np.any(mask):
                axes[1].loglog(
                    x_out[mask], y_out[mask],
                    linewidth=style_params["lines.linewidth"],
                )
            else:
                axes[1].plot(x_out, y_out, linewidth=style_params["lines.linewidth"])
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Coefficient Magnitude")
            axes[1].set_title("OWChirp Spectrum")
            axes[1].grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            x_out = _ensure_numpy(result.x)
            y_out = _ensure_numpy(result.y)
            mask = (x_out > 0) & (y_out > 0) & np.isfinite(x_out) & np.isfinite(y_out)
            if np.any(mask):
                ax.loglog(x_out[mask], y_out[mask], linewidth=style_params["lines.linewidth"])
            else:
                ax.plot(x_out, y_out, linewidth=style_params["lines.linewidth"])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Coefficient Magnitude")
            ax.set_title("OWChirp Spectrum")
            ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_derivative(
        self,
        result: RheoData,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Smooth derivative: original data + derivative.

        Layout:
        ┌──────────────┬──────────────┐
        │ Original     │  Derivative  │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)

        if input_data is not None and show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))

            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            axes[0].plot(x_in, y_in, linewidth=style_params["lines.linewidth"])
            axes[0].set_xlabel(f"x ({input_data.x_units})" if input_data.x_units else "x")
            axes[0].set_ylabel(f"y ({input_data.y_units})" if input_data.y_units else "y")
            axes[0].set_title("Original Data")
            axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            x_d = _ensure_numpy(result.x)
            y_d = _ensure_numpy(result.y)
            axes[1].plot(x_d, y_d, linewidth=style_params["lines.linewidth"], color="C1")
            axes[1].set_xlabel(f"x ({result.x_units})" if result.x_units else "x")
            axes[1].set_ylabel(f"dy/dx ({result.y_units})" if result.y_units else "dy/dx")
            axes[1].set_title("Smoothed Derivative")
            axes[1].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            x_d = _ensure_numpy(result.x)
            y_d = _ensure_numpy(result.y)
            ax.plot(x_d, y_d, linewidth=style_params["lines.linewidth"], color="C1")
            ax.set_xlabel(f"x ({result.x_units})" if result.x_units else "x")
            ax.set_ylabel(f"dy/dx ({result.y_units})" if result.y_units else "dy/dx")
            ax.set_title("Smoothed Derivative")
            ax.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_spp(
        self,
        result: RheoData,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """SPP decomposer: Lissajous + G't/G''t + Cole-Cole.

        Layout:
        ┌──────────┬──────────┬──────────┐
        │ Lissajous│  G't,G''t│ Cole-Cole│
        │  σ vs γ  │  vs time │  G'' v G'│
        └──────────┴──────────┴──────────┘
        """
        style_params = _apply_style(style)
        meta = result.metadata or {}
        spp = meta.get("spp_results", meta.get("core", {}))

        fig, axes = plt.subplots(1, 3, figsize=(
            style_params["figure.figsize"][0] * 2.0,
            style_params["figure.figsize"][1],
        ))

        # Panel 1: Lissajous (stress vs strain)
        strain = spp.get("strain")
        if strain is None:
            strain = spp.get("gamma")
        stress_recon = spp.get("stress_reconstructed")
        if stress_recon is None:
            stress_recon = spp.get("stress")
        if strain is not None and stress_recon is not None:
            strain = _ensure_numpy(np.asarray(strain))
            stress_r = _ensure_numpy(np.asarray(stress_recon))
            n = min(len(strain), len(stress_r))
            axes[0].plot(
                strain[:n], stress_r[:n],
                linewidth=style_params["lines.linewidth"],
            )
        axes[0].set_xlabel("Strain γ")
        axes[0].set_ylabel("Stress σ (Pa)")
        axes[0].set_title("Lissajous (σ vs γ)")
        axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        # Panel 2: Time-dependent moduli
        Gp_t = spp.get("Gp_t")
        if Gp_t is None:
            Gp_t = spp.get("G_cage")
        Gpp_t = spp.get("Gpp_t")
        time_spp = spp.get("time_new")
        if time_spp is None:
            time_spp = spp.get("time")
        if Gp_t is not None and time_spp is not None:
            Gp_t = _ensure_numpy(np.asarray(Gp_t))
            time_arr = _ensure_numpy(np.asarray(time_spp))
            n = min(len(time_arr), len(Gp_t))
            # Determine label prefix from deformation mode
            _s, _l, _ = _modulus_labels(y_units="Pa")
            _sp = _s.split(" ")[0]  # "G'" or "E'"
            _lp = _l.split(" ")[0]  # 'G"' or 'E"'
            axes[1].plot(
                time_arr[:n], Gp_t[:n],
                label=f"{_sp}(t)", linewidth=style_params["lines.linewidth"],
            )
            if Gpp_t is not None:
                Gpp_t = _ensure_numpy(np.asarray(Gpp_t))
                n2 = min(len(time_arr), len(Gpp_t))
                axes[1].plot(
                    time_arr[:n2], Gpp_t[:n2], "--",
                    label=f'{_lp}(t)', linewidth=style_params["lines.linewidth"],
                )
            axes[1].legend(fontsize=style_params["legend.fontsize"])
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Modulus (Pa)")
        axes[1].set_title("Time-Dependent Moduli")
        axes[1].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        # Panel 3: Cole-Cole
        _s_lbl, _l_lbl, _ = _modulus_labels(y_units="Pa")
        if Gp_t is not None and Gpp_t is not None:
            n3 = min(len(Gp_t), len(Gpp_t))
            axes[2].plot(
                Gp_t[:n3], Gpp_t[:n3],
                linewidth=style_params["lines.linewidth"],
            )
        axes[2].set_xlabel(_s_lbl)
        axes[2].set_ylabel(_l_lbl)
        axes[2].set_title("Cole-Cole Diagram")
        axes[2].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)
        axes[2].set_aspect("equal", adjustable="datalim")

        fig.tight_layout()
        return fig, axes

    def _plot_prony(
        self,
        result: Any,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Prony conversion: input domain + converted domain + modes.

        Layout:
        ┌──────────────┬──────────────┐
        │ Input domain │  Converted   │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)
        output_data, meta_dict = self._unpack_result(result)

        if input_data is not None and show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))

            # Left: input
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            s_lbl, l_lbl, _ = _modulus_labels(input_data)
            if np.iscomplexobj(y_in):
                x_f, gp = _filter_positive(x_in, np.real(y_in), warn=False)
                x_f2, gpp = _filter_positive(x_in, np.imag(y_in), warn=False)
                axes[0].loglog(x_f, gp, "o", label=s_lbl, markersize=4, markerfacecolor="none")
                if len(x_f2) > 0:
                    axes[0].loglog(x_f2, gpp, "s", label=l_lbl, markersize=4, markerfacecolor="none")
                axes[0].legend()
            else:
                axes[0].plot(x_in, y_in, linewidth=style_params["lines.linewidth"])
            axes[0].set_title("Input")
            axes[0].grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Right: converted
            self._plot_prony_output(axes[1], output_data, meta_dict, style_params)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            self._plot_prony_output(ax, output_data, meta_dict, style_params)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_prony_output(
        self, ax: Axes, data: RheoData, meta: dict | None, style_params: dict
    ) -> None:
        """Plot Prony conversion output with mode stems."""
        x_out = _ensure_numpy(data.x)
        y_out = _ensure_numpy(data.y)

        s_lbl, l_lbl, _ = _modulus_labels(data)
        if np.iscomplexobj(y_out):
            x_f, gp = _filter_positive(x_out, np.real(y_out), warn=False)
            x_f2, gpp = _filter_positive(x_out, np.imag(y_out), warn=False)
            ax.loglog(x_f, gp, "-", label=s_lbl, linewidth=style_params["lines.linewidth"])
            if len(x_f2) > 0:
                ax.loglog(x_f2, gpp, "--", label=l_lbl, linewidth=style_params["lines.linewidth"])
            ax.legend()
        else:
            ax.plot(x_out, y_out, linewidth=style_params["lines.linewidth"])

        # Overlay Prony modes as stems if available
        if meta is not None:
            prony_result = meta.get("prony_result")
            if prony_result is not None:
                tau_i = getattr(prony_result, "tau_i", None)
                G_i = getattr(prony_result, "G_i", None)
                if tau_i is not None and G_i is not None:
                    tau_i = _ensure_numpy(np.asarray(tau_i))
                    G_i = _ensure_numpy(np.asarray(G_i))
                    ax.stem(
                        tau_i, G_i, linefmt="C3-", markerfmt="C3o",
                        basefmt="k-", label="Prony modes",
                    )
                    ax.legend()

        ax.set_title("Prony Conversion")
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

    def _plot_spectrum(
        self,
        result: Any,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Spectrum inversion: G',G'' data + relaxation spectrum H(τ).

        Layout:
        ┌──────────────┬──────────────┐
        │   G', G''    │    H(τ)      │
        └──────────────┴──────────────┘
        """
        style_params = _apply_style(style)
        output_data, meta_dict = self._unpack_result(result)

        if input_data is not None and show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))

            # Left: input oscillation data
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            sp_lbl, lp_lbl, gn_lbl = _modulus_labels(input_data)
            if np.iscomplexobj(y_in):
                x_f, gp = _filter_positive(x_in, np.real(y_in), warn=False)
                x_f2, gpp = _filter_positive(x_in, np.imag(y_in), warn=False)
                axes[0].loglog(x_f, gp, "o", label=sp_lbl, markersize=4, markerfacecolor="none")
                if len(x_f2) > 0:
                    axes[0].loglog(x_f2, gpp, "s", label=lp_lbl, markersize=4, markerfacecolor="none")
                axes[0].legend()
            axes[0].set_xlabel("ω (rad/s)")
            axes[0].set_ylabel(gn_lbl)
            axes[0].set_title("Dynamic Moduli")
            axes[0].grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            # Right: H(τ) spectrum
            self._plot_spectrum_inv_panel(axes[1], output_data, style_params)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            self._plot_spectrum_inv_panel(ax, output_data, style_params)
            fig.tight_layout()
            return fig, np.array([ax])

    def _plot_spectrum_inv_panel(
        self, ax: Axes, data: RheoData, style_params: dict
    ) -> None:
        """Plot H(τ) relaxation spectrum."""
        tau = _ensure_numpy(data.x)
        H = _ensure_numpy(data.y)

        mask = (tau > 0) & (H > 0) & np.isfinite(tau) & np.isfinite(H)
        if np.any(mask):
            ax.loglog(
                tau[mask], H[mask],
                linewidth=style_params["lines.linewidth"],
            )
        else:
            ax.plot(tau, H, linewidth=style_params["lines.linewidth"])

        ax.set_xlabel("τ (s)")
        ax.set_ylabel("H(τ) (Pa)")
        ax.set_title("Relaxation Spectrum")
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

    def _plot_cox_merz(
        self,
        result: Any,
        input_data: list[RheoData] | RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Cox-Merz: |η*(ω)| vs η(γ̇) overlay.

        Layout: Single panel with both viscosities overlaid.
        """
        style_params = _apply_style(style)
        output_data, meta_dict = self._unpack_result(result)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        if meta_dict is not None:
            cm_result = meta_dict.get("cox_merz_result")
            if cm_result is not None:
                rates = _ensure_numpy(np.asarray(cm_result.common_rates))
                eta_complex = _ensure_numpy(np.asarray(cm_result.eta_complex))
                eta_steady = _ensure_numpy(np.asarray(cm_result.eta_steady))

                ax.loglog(
                    rates, eta_complex, "o-",
                    label="|η*(ω)|",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    linewidth=style_params["lines.linewidth"],
                )
                ax.loglog(
                    rates, eta_steady, "s--",
                    label="η(γ̇)",
                    markersize=style_params["lines.markersize"],
                    markerfacecolor="none",
                    linewidth=style_params["lines.linewidth"],
                    color="C1",
                )

                passes = getattr(cm_result, "passes", None)
                mean_dev = getattr(cm_result, "mean_deviation", None)
                title = "Cox-Merz Rule"
                if passes is not None:
                    title += f" ({'PASS' if passes else 'FAIL'})"
                if mean_dev is not None:
                    title += f" — mean dev: {mean_dev:.1%}"
                ax.set_title(title)
        else:
            # Fallback: plot output_data
            x_out = _ensure_numpy(output_data.x)
            y_out = _ensure_numpy(output_data.y)
            ax.plot(x_out, y_out, linewidth=style_params["lines.linewidth"])
            ax.set_title("Cox-Merz Deviation")

        ax.set_xlabel("Rate (1/s)")
        ax.set_ylabel("Viscosity (Pa·s)")
        ax.legend(fontsize=style_params["legend.fontsize"])
        ax.grid(True, which="both", alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()
        return fig, ax

    def _plot_envelope(
        self,
        result: Any,
        input_data: RheoData | None = None,
        show_intermediate: bool = True,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """LVE envelope: startup stress envelope σ_LVE(t).

        Layout: Single panel with envelope curve (and optional startup data overlay).
        """
        style_params = _apply_style(style)
        output_data, meta_dict = self._unpack_result(result)

        fig, ax = plt.subplots(figsize=style_params["figure.figsize"])

        # Envelope curve
        x_env = _ensure_numpy(output_data.x)
        y_env = _ensure_numpy(output_data.y)
        ax.plot(
            x_env, y_env, "-",
            linewidth=style_params["lines.linewidth"] * 1.5,
            color="C0", label="LVE Envelope", zorder=3,
        )

        # Overlay startup data if provided
        if input_data is not None and show_intermediate:
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            ax.plot(
                x_in, y_in, "--",
                linewidth=style_params["lines.linewidth"] * 0.7,
                color="C1", alpha=0.6, label="Startup Data", zorder=2,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stress σ (Pa)")

        shear_rate = (output_data.metadata or {}).get("shear_rate")
        title = "LVE Envelope"
        if shear_rate is not None:
            title += f" (γ̇ = {shear_rate} 1/s)"
        ax.set_title(title)
        ax.legend(fontsize=style_params["legend.fontsize"])
        ax.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

        fig.tight_layout()
        return fig, ax

    def _plot_generic(
        self,
        result: Any,
        input_data: RheoData | None = None,
        style: str = "default",
        transform_name: str = "Transform",
        **kwargs: Any,
    ) -> tuple[Figure, np.ndarray]:
        """Generic before/after plot for unrecognized transforms."""
        style_params = _apply_style(style)
        output_data, _ = self._unpack_result(result)

        if input_data is not None and isinstance(input_data, RheoData):
            fig, axes = plt.subplots(1, 2, figsize=(
                style_params["figure.figsize"][0] * 1.5,
                style_params["figure.figsize"][1],
            ))
            x_in = _ensure_numpy(input_data.x)
            y_in = _ensure_numpy(input_data.y)
            if np.iscomplexobj(y_in):
                y_in = np.abs(y_in)
            axes[0].plot(x_in, y_in, linewidth=style_params["lines.linewidth"])
            axes[0].set_title("Before")
            axes[0].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            x_out = _ensure_numpy(output_data.x)
            y_out = _ensure_numpy(output_data.y)
            if np.iscomplexobj(y_out):
                y_out = np.abs(y_out)
            axes[1].plot(x_out, y_out, linewidth=style_params["lines.linewidth"], color="C1")
            axes[1].set_title(f"After ({transform_name})")
            axes[1].grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)

            fig.tight_layout()
            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=style_params["figure.figsize"])
            x_out = _ensure_numpy(output_data.x)
            y_out = _ensure_numpy(output_data.y)
            if np.iscomplexobj(y_out):
                y_out = np.abs(y_out)
            ax.plot(x_out, y_out, linewidth=style_params["lines.linewidth"])
            ax.set_title(f"Transform: {transform_name}")
            ax.grid(True, alpha=_GRID_ALPHA, linestyle=_GRID_LINESTYLE)
            fig.tight_layout()
            return fig, np.array([ax])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_result(result: Any) -> tuple[RheoData, dict | None]:
        """Unpack transform result into (RheoData, metadata_dict)."""
        if isinstance(result, tuple) and len(result) == 2:
            data, meta = result
            if isinstance(data, RheoData):
                if isinstance(meta, dict):
                    return data, meta
                logger.warning(
                    "Transform returned non-dict metadata of type %s; "
                    "treating as None.",
                    type(meta).__name__,
                )
                return data, None
        if isinstance(result, RheoData):
            return result, None
        # Fallback for unexpected types
        raise TypeError(
            f"Cannot unpack transform result of type {type(result).__name__}. "
            "Expected RheoData or tuple[RheoData, dict]."
        )
