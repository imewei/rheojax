"""
Plot Service
===========

Service for generating rheological plots with matplotlib.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc_params_from_file
from matplotlib.figure import Figure

from rheojax.core.data import RheoData
from rheojax.gui.resources import available_plot_styles, load_plot_style
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PlotService:
    """Service for rheological visualization.

    Features:
        - Auto plot selection based on data type
        - Multiple plot styles (default, publication, presentation)
        - ArviZ diagnostics integration
        - Publication-ready export
        - Colorblind-safe palettes

    Example
    -------
    >>> service = PlotService()
    >>> fig = service.create_fit_plot(data, fit_result, style='publication')
    >>> service.apply_style(fig, 'publication')
    """

    def __init__(self) -> None:
        """Initialize plot service."""
        logger.debug("Initializing PlotService")
        self._styles = {
            "default": self._apply_default_style,
            "publication": self._apply_publication_style,
            "presentation": self._apply_presentation_style,
            "dark": self._apply_dark_style,
        }
        self._style_cache = {
            name: load_plot_style(name) for name in available_plot_styles()
        }

        # Wong colorblind-safe palette
        self._colorblind_palette = [
            "#E69F00",  # Orange
            "#56B4E9",  # Sky Blue
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#0072B2",  # Blue
            "#D55E00",  # Vermillion
            "#CC79A7",  # Reddish Purple
        ]
        self._dark_palette = [
            "#8ab4f8",  # light blue
            "#f28b82",  # salmon
            "#81c995",  # green
            "#fdd663",  # yellow
            "#a78bfa",  # purple
            "#ffb2f2",  # pink
            "#5ad1fa",  # cyan
        ]
        logger.debug(
            "PlotService initialized", available_styles=list(self._styles.keys())
        )

    def get_available_styles(self) -> list[str]:
        """Get available plot styles.

        Returns
        -------
        list[str]
            Style names
        """
        logger.debug("Getting available styles")
        styles = sorted(set(self._styles.keys()) | set(self._style_cache.keys()))
        logger.debug("Available styles retrieved", count=len(styles))
        return styles

    def get_colorblind_palette(self) -> list[str]:
        """Get Wong colorblind-safe palette.

        Returns
        -------
        list[str]
            Hex color codes
        """
        logger.debug("Getting colorblind palette")
        return self._colorblind_palette.copy()

    def get_dark_palette(self) -> list[str]:
        """Get dark-theme-friendly palette."""
        logger.debug("Getting dark palette")
        return self._dark_palette.copy()

    def apply_style(self, fig: Figure, style: str) -> None:
        """Apply matplotlib style to figure.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure
        style : str
            Style name ('default', 'publication', 'presentation')
        """
        logger.debug("Applying style to figure", style=style)
        if style in self._styles:
            self._styles[style](fig)
            logger.debug("Style applied successfully", style=style)
        else:
            logger.warning(f"Unknown style: {style}, using default")
            self._apply_default_style(fig)

    def _apply_default_style(self, fig: Figure) -> None:
        """Apply default style."""
        logger.debug("Applying default style")
        for ax in fig.get_axes():
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

    def _apply_publication_style(self, fig: Figure) -> None:
        """Apply publication style (IEEE, Nature, etc.)."""
        logger.debug("Applying publication style")
        # Set DPI for publication quality
        fig.set_dpi(300)

        for ax in fig.get_axes():
            # Font sizes
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            ax.title.set_fontsize(14)

            # Grid and spines
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)

            # Line widths
            for line in ax.get_lines():
                line.set_linewidth(1.5)

    def _apply_presentation_style(self, fig: Figure) -> None:
        """Apply presentation style (larger fonts, bold)."""
        logger.debug("Applying presentation style")
        fig.set_dpi(150)

        for ax in fig.get_axes():
            # Larger fonts for visibility
            ax.tick_params(labelsize=14)
            ax.xaxis.label.set_fontsize(16)
            ax.yaxis.label.set_fontsize(16)
            ax.title.set_fontsize(18, weight="bold")

            # Thicker lines
            for line in ax.get_lines():
                line.set_linewidth(2.5)

            # Grid
            ax.grid(True, alpha=0.4)

    def _apply_dark_style(self, fig: Figure) -> None:
        """Apply dark-theme adjustments after RC params."""
        logger.debug("Applying dark style")
        for ax in fig.get_axes():
            ax.tick_params(colors="#e8eaed")
            ax.xaxis.label.set_color("#e8eaed")
            ax.yaxis.label.set_color("#e8eaed")
            ax.title.set_color("#e8eaed")
            ax.grid(True, color="#3c4043", alpha=0.6, linestyle="--", linewidth=0.8)
            for line, color in zip(ax.get_lines(), self._dark_palette, strict=False):
                line.set_color(color)
            for spine in ax.spines.values():
                spine.set_color("#e8eaed")

    def create_fit_plot(
        self,
        data: RheoData,
        fit_result: Any,
        style: str = "default",
        test_mode: str | None = None,
    ) -> Figure:
        """Create publication-quality fit plot with optional uncertainty bands.

        Parameters
        ----------
        data : RheoData
            Experimental data
        fit_result : FitResult
            Fitting result with predictions (and optional pcov for uncertainty)
        style : str, default='default'
            Plot style
        test_mode : str, optional
            Test mode for appropriate plot type

        Returns
        -------
        Figure
            Matplotlib figure
        """
        logger.debug("create_fit_plot called", style=style, test_mode=test_mode)
        logger.info("Generating plot", plot_type="fit")

        try:
            self._apply_style_context(style)
            fig, ax = plt.subplots(figsize=(8, 6))
            palette = (
                self._get_palette(style) if (style or "").lower() == "dark" else None
            )

            x = np.asarray(data.x)
            y = np.asarray(data.y)
            y_fit = fit_result.y_fit
            if y_fit is None:
                raise ValueError("FitResult does not contain y_fit data")
            y_fit = np.asarray(y_fit)

            # Check for covariance matrix (for uncertainty bands)
            pcov = getattr(fit_result, "pcov", None)
            has_uncertainty = pcov is not None and np.all(np.isfinite(pcov))

            # Determine test mode
            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "oscillation")

            logger.debug(
                "Fit plot configuration",
                data_shape=x.shape,
                has_uncertainty=has_uncertainty,
                resolved_test_mode=test_mode,
            )

            # Helper: compute uncertainty band for real-valued fits
            def _compute_band(x_vals, y_vals):
                """Compute +/-2sigma band if pcov available."""
                if not has_uncertainty:
                    return None, None
                try:
                    # Use finite difference on y_fit to approximate band
                    # This is simpler than full error propagation
                    # Uncertainty ~ sqrt(diag(pcov)) projected to y space
                    param_std = np.sqrt(np.diag(pcov))
                    # Rough approximation: scale y_fit by relative param uncertainty
                    rel_uncertainty = np.mean(
                        param_std
                        / (np.abs(list(fit_result.parameters.values())) + 1e-10)
                    )
                    sigma_y = np.abs(y_vals) * rel_uncertainty
                    z = 1.96  # 95% CI
                    return y_vals - z * sigma_y, y_vals + z * sigma_y
                except Exception:
                    return None, None

            # Plot based on test mode
            if test_mode == "oscillation":
                # Plot G' and G" for oscillation
                if np.iscomplexobj(y):
                    G_prime = np.real(y)
                    # Some datasets/models may use sign conventions; ensure log-safe positivity.
                    G_double_prime = np.abs(np.imag(y))
                    G_prime_fit = np.real(y_fit)
                    G_double_prime_fit = np.abs(np.imag(y_fit))

                    # Avoid log-scale issues with zeros.
                    positive = np.concatenate(
                        [
                            np.ravel(G_prime[np.isfinite(G_prime) & (G_prime > 0)]),
                            np.ravel(
                                G_double_prime[
                                    np.isfinite(G_double_prime) & (G_double_prime > 0)
                                ]
                            ),
                        ]
                    )
                    eps = float(np.nanmin(positive)) * 1e-12 if positive.size else 1e-30
                    G_prime = np.maximum(G_prime, eps)
                    G_double_prime = np.maximum(G_double_prime, eps)
                    G_prime_fit = np.maximum(G_prime_fit, eps)
                    G_double_prime_fit = np.maximum(G_double_prime_fit, eps)

                    ax.loglog(
                        x,
                        G_prime,
                        "o",
                        label="G' (data)",
                        color=palette[0] if palette else None,
                    )
                    ax.loglog(
                        x,
                        G_double_prime,
                        "s",
                        label='G" (data)',
                        color=palette[1] if palette else None,
                    )
                    ax.loglog(
                        x,
                        G_prime_fit,
                        "-",
                        label="G' (fit)",
                        color=palette[0] if palette else None,
                    )
                    ax.loglog(
                        x,
                        G_double_prime_fit,
                        "-",
                        label='G" (fit)',
                        color=palette[1] if palette else None,
                    )

                    ax.set_xlabel("Frequency (rad/s)")
                    ax.set_ylabel("Modulus (Pa)")
                else:
                    # Complex modulus magnitude
                    ax.loglog(
                        x, y, "o", label="Data", color=palette[0] if palette else None
                    )
                    ax.loglog(
                        x,
                        y_fit,
                        "-",
                        label="Fit",
                        color=palette[1] if palette else None,
                    )

                    # Add uncertainty band
                    y_lo, y_hi = _compute_band(x, y_fit)
                    if y_lo is not None and y_hi is not None:
                        y_lo = np.maximum(y_lo, 1e-30)  # Ensure positive for log scale
                        ax.fill_between(
                            x, y_lo, y_hi, alpha=0.3, color="C0", label="95% CI"
                        )

                    ax.set_xlabel("Frequency (rad/s)")
                    ax.set_ylabel("|G*| (Pa)")

            elif test_mode == "relaxation":
                ax.loglog(
                    x, y, "o", label="Data", color=palette[0] if palette else None
                )
                ax.loglog(
                    x, y_fit, "-", label="Fit", color=palette[1] if palette else None
                )

                # Add uncertainty band
                y_lo, y_hi = _compute_band(x, y_fit)
                if y_lo is not None and y_hi is not None:
                    y_lo = np.maximum(y_lo, 1e-30)  # Ensure positive for log scale
                    ax.fill_between(
                        x, y_lo, y_hi, alpha=0.3, color="C0", label="95% CI"
                    )

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Relaxation Modulus G(t) (Pa)")

            elif test_mode == "creep":
                ax.loglog(
                    x, y, "o", label="Data", color=palette[0] if palette else None
                )
                ax.loglog(
                    x, y_fit, "-", label="Fit", color=palette[1] if palette else None
                )

                # Add uncertainty band
                y_lo, y_hi = _compute_band(x, y_fit)
                if y_lo is not None and y_hi is not None:
                    y_lo = np.maximum(y_lo, 1e-30)  # Ensure positive for log scale
                    ax.fill_between(
                        x, y_lo, y_hi, alpha=0.3, color="C0", label="95% CI"
                    )

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Creep Compliance J(t) (1/Pa)")

            elif test_mode == "flow":
                ax.loglog(
                    x, y, "o", label="Data", color=palette[0] if palette else None
                )
                ax.loglog(
                    x, y_fit, "-", label="Fit", color=palette[1] if palette else None
                )

                # Add uncertainty band
                y_lo, y_hi = _compute_band(x, y_fit)
                if y_lo is not None and y_hi is not None:
                    y_lo = np.maximum(y_lo, 1e-30)  # Ensure positive for log scale
                    ax.fill_between(
                        x, y_lo, y_hi, alpha=0.3, color="C0", label="95% CI"
                    )

                ax.set_xlabel("Shear Rate (1/s)")
                ax.set_ylabel("Viscosity (Pa.s)")

            ax.legend()
            ax.set_title(f"{fit_result.model_name} Model Fit")

            # Apply style
            self.apply_style(fig, style)

            logger.info("Plot generated", plot_type="fit")
            logger.debug("create_fit_plot completed successfully")
            return fig

        except Exception as e:
            logger.error(f"Failed to create fit plot: {e}", exc_info=True)
            plt.close("all")
            raise

    def create_residual_plot(
        self,
        data: RheoData,
        fit_result: Any,
        style: str = "default",
    ) -> Figure:
        """Create residual plot.

        Parameters
        ----------
        data : RheoData
            Experimental data
        fit_result : FitResult
            Fitting result with residuals

        Returns
        -------
        Figure
            Matplotlib figure
        """
        logger.debug("create_residual_plot called", style=style)
        logger.info("Generating plot", plot_type="residual")

        try:
            self._apply_style_context(style)
            palette = self._get_palette(style)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            x = np.asarray(data.x)
            residuals = fit_result.residuals
            if residuals is None:
                raise ValueError("FitResult does not contain residuals")
            residuals = np.asarray(residuals)

            logger.debug(
                "Residual plot data", data_points=len(x), residual_shape=residuals.shape
            )

            # Residuals vs x
            ax1.semilogx(x, residuals, "o", color=palette[0])
            ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residual Analysis")

            # Residual histogram
            ax2.hist(residuals, bins=30, color=palette[1], alpha=0.7, edgecolor="black")
            ax2.set_xlabel("Residual Value")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Residual Distribution")

            fig.tight_layout()

            logger.info("Plot generated", plot_type="residual")
            logger.debug("create_residual_plot completed successfully")
            return fig

        except Exception as e:
            logger.error(f"Failed to create residual plot: {e}", exc_info=True)
            plt.close("all")
            raise

    def create_arviz_plot(
        self,
        result: Any,
        plot_type: str,
        style: str = "default",
        **kwargs: Any,
    ) -> Figure:
        """Create ArviZ diagnostic plot.

        Parameters
        ----------
        result : BayesianResult
            Bayesian inference result
        plot_type : str
            ArviZ plot type ('trace', 'pair', 'forest', 'energy', 'autocorr', 'rank', 'ess')
        **kwargs
            Additional ArviZ plot options

        Returns
        -------
        Figure
            Matplotlib figure
        """
        logger.debug("create_arviz_plot called", plot_type=plot_type, style=style)
        logger.info("Generating plot", plot_type=f"arviz_{plot_type}")

        self._apply_style_context(style)
        try:
            import arviz as az
        except ImportError:
            from rheojax.gui.utils._dependency_guard import require_dependency

            require_dependency("arviz", "Bayesian diagnostic plots", "pip install arviz")
        try:

            # Convert to InferenceData
            posterior_samples = result.posterior_samples
            idata_dict = {
                k: v.reshape(1, -1) if v.ndim == 1 else v
                for k, v in posterior_samples.items()
            }
            idata = az.from_dict(idata_dict)

            logger.debug(
                "ArviZ InferenceData created", parameters=list(posterior_samples.keys())
            )

            # Create plot based on type
            if plot_type == "trace":
                axes = az.plot_trace(idata, **kwargs)
                fig = axes.ravel()[0].figure

            elif plot_type == "pair":
                ax = az.plot_pair(idata, **kwargs)
                fig = ax.figure if hasattr(ax, "figure") else ax.ravel()[0].figure

            elif plot_type == "forest":
                ax = az.plot_forest(idata, **kwargs)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

            elif plot_type == "energy":
                ax = az.plot_energy(idata, **kwargs)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

            elif plot_type == "autocorr":
                axes = az.plot_autocorr(idata, **kwargs)
                fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else plt.gcf()

            elif plot_type == "rank":
                axes = az.plot_rank(idata, **kwargs)
                fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else plt.gcf()

            elif plot_type == "ess":
                ax = az.plot_ess(idata, **kwargs)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

            else:
                logger.error(f"Unknown ArviZ plot type: {plot_type}")
                raise ValueError(f"Unknown plot type: {plot_type}")

            logger.info("Plot generated", plot_type=f"arviz_{plot_type}")
            logger.debug(
                "create_arviz_plot completed successfully", plot_type=plot_type
            )
            return fig

        except Exception as e:
            logger.error(f"ArviZ plot failed: {e}", exc_info=True)
            plt.close("all")  # Prevent figure leak on error
            raise RuntimeError(f"Plot creation failed: {e}") from e

    def create_data_plot(
        self,
        data: RheoData,
        style: str = "default",
        test_mode: str | None = None,
    ) -> Figure:
        """Create data visualization plot.

        Parameters
        ----------
        data : RheoData
            Rheological data
        style : str, default='default'
            Plot style
        test_mode : str, optional
            Test mode

        Returns
        -------
        Figure
            Matplotlib figure
        """
        logger.debug("create_data_plot called", style=style, test_mode=test_mode)
        logger.info("Generating plot", plot_type="data")

        try:
            self._apply_style_context(style)
            palette = self._get_palette(style)
            fig, ax = plt.subplots(figsize=(8, 6))

            x = np.asarray(data.x)
            y = np.asarray(data.y)

            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "unknown")

            logger.debug(
                "Data plot configuration",
                data_shape=x.shape,
                is_complex=np.iscomplexobj(y),
                resolved_test_mode=test_mode,
            )

            # Plot based on type
            if np.iscomplexobj(y):
                G_prime = np.real(y)
                G_double_prime = np.imag(y)
                ax.loglog(x, G_prime, "o", label="G'", color=palette[0])
                ax.loglog(x, G_double_prime, "s", label='G"', color=palette[1])
                ax.set_xlabel(data.x_units or "Frequency (rad/s)")
                ax.set_ylabel(data.y_units or "Modulus (Pa)")
            else:
                ax.loglog(x, y, "o", color=palette[0])
                ax.set_xlabel(data.x_units or "X")
                ax.set_ylabel(data.y_units or "Y")

            ax.set_title(f"Rheological Data ({test_mode})")
            if np.iscomplexobj(y):
                ax.legend()

            self.apply_style(fig, style)

            logger.info("Plot generated", plot_type="data")
            logger.debug("create_data_plot completed successfully")
            return fig

        except Exception as e:
            logger.error(f"Failed to create data plot: {e}", exc_info=True)
            plt.close("all")
            raise

    def _apply_style_context(self, style: str) -> None:
        """Apply RC params from bundled styles if available."""
        logger.debug("Applying style context", style=style)
        style_name = style or "default"
        if style_name == "default":
            plt.rcParams.update(plt.rcParamsDefault)
            return
        style_text = self._style_cache.get(style_name, "")
        if style_text:
            plt.rcParams.update(plt.rcParamsDefault)
            # rc_params_from_file expects a path; write to temp file
            with tempfile.NamedTemporaryFile(
                "w", suffix=".mplstyle", delete=False
            ) as tmp:
                tmp.write(style_text)
                tmp_path = tmp.name
            try:
                rc = rc_params_from_file(tmp_path)
                plt.rcParams.update(rc)
                logger.debug("Style context applied from cache", style=style_name)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    logger.debug("Failed to remove temp style file: %s", exc)
        else:
            plt.style.use("default")
            logger.debug("Using default matplotlib style")

    def _get_palette(self, style: str) -> list[str]:
        """Return palette respecting style."""
        logger.debug("Getting palette for style", style=style)
        if (style or "").lower() == "dark":
            return self._dark_palette
        return self._colorblind_palette
