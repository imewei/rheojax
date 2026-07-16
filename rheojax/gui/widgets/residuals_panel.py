"""
Residuals Panel Widget
=====================

Residual analysis visualization for model fitting diagnostics.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from rheojax.gui.compat import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles.tokens import Spacing, themed
from rheojax.gui.utils.layout_helpers import set_toolbar_margins, set_zero_margins
from rheojax.gui.widgets.dropdown import RheoComboBox
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Available plot types for residual analysis
PLOT_TYPES = [
    ("residuals", "Residuals vs Fitted", "Residuals plotted against fitted values"),
    ("qq", "Q-Q Plot", "Normal quantile-quantile plot"),
    ("histogram", "Histogram", "Distribution of residuals"),
    ("scale_location", "Scale-Location", "√|Residuals| vs Fitted"),
    ("residuals_x", "Residuals vs X", "Residuals plotted against X values"),
    ("autocorr", "Autocorrelation", "Residual autocorrelation"),
]


class ResidualsPanel(QWidget):
    """Residuals visualization panel for model fitting diagnostics.

    Features:
        - Multiple residual plot types (vs fitted, Q-Q, histogram, etc.)
        - Statistics display (mean, std, normality tests)
        - Interactive matplotlib canvas
        - Export support

    Signals
    -------
    plot_changed : Signal(str)
        Emitted when plot type changes

    Example
    -------
    >>> panel = ResidualsPanel()  # doctest: +SKIP
    >>> panel.plot_residuals(y_true, y_pred, x)  # doctest: +SKIP
    >>> panel.set_plot_type("qq")  # doctest: +SKIP
    """

    plot_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize residuals panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self._x_values: np.ndarray | None = None
        self._y_true: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._current_plot_type = "residuals"

        self._setup_ui()
        self._connect_signals()
        # closeEvent() only fires for top-level widgets; when embedded as a
        # child (e.g. inside VisualizeStep) it's destroyed via Qt's parent
        # cascade instead, which closeEvent never sees. destroyed fires on
        # every teardown path, so wiring cleanup() to it covers both.
        self.destroyed.connect(lambda: self.cleanup())
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def cleanup(self) -> None:
        """Explicitly release matplotlib resources before Qt widget deletion.

        Must be called before the widget is deleted to prevent segfaults from
        deferred ``draw_idle()`` callbacks firing on a freed C++ object.
        """
        import matplotlib.pyplot as plt

        try:
            if self._canvas is not None:
                self._canvas.callbacks.callbacks.clear()
                self._canvas.draw_idle = lambda: None
            if self._figure is not None:
                plt.close(self._figure)
        except RuntimeError:
            pass  # C++ object already deleted

    def closeEvent(self, event) -> None:  # noqa: N802
        """Cancel pending matplotlib draws before the widget is closed.

        Only fires on an explicit .close() (including qtbot.addWidget()'s
        teardown, which is what this actually needs to fix). When this panel
        is embedded as a child of fit_page.py/step5_visualize.py and the
        parent is torn down via deleteLater() cascade instead, closeEvent
        never reaches this widget -- same limitation base_arviz_widget.py/
        plot_canvas.py already have; call cleanup() directly at any such
        parent-cascade teardown site if that path needs covering too.
        """
        self.cleanup()
        super().closeEvent(event)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        set_zero_margins(layout)
        layout.setSpacing(Spacing.XS)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        set_toolbar_margins(toolbar_layout)

        # Plot type selector
        type_label = QLabel("Plot Type:")
        toolbar_layout.addWidget(type_label)

        self._type_combo = RheoComboBox()
        self._type_combo.setMinimumWidth(140)
        self._type_combo.set_items_safely(
            [(display_name, plot_id) for plot_id, display_name, _tooltip in PLOT_TYPES]
        )
        for idx, (_plot_id, _display_name, tooltip) in enumerate(PLOT_TYPES):
            self._type_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        toolbar_layout.addWidget(self._type_combo)

        toolbar_layout.addStretch()

        # Statistics label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(f"color: {themed('TEXT_SECONDARY')};")
        toolbar_layout.addWidget(self._stats_label)

        # Export button
        self._export_btn = QPushButton("Export")
        self._export_btn.setToolTip("Export plot to file")
        toolbar_layout.addWidget(self._export_btn)

        layout.addLayout(toolbar_layout)

        # Matplotlib figure and canvas
        self._figure = Figure(figsize=(8, 5), dpi=100)
        self._figure.set_layout_engine("tight")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Navigation toolbar
        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)

        layout.addWidget(self._nav_toolbar)
        layout.addWidget(self._canvas, 1)

        self._empty_label = QLabel("No residuals yet. Run a fit to view diagnostics.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {themed('TEXT_SECONDARY')}; padding: 6px;"
        )
        layout.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._export_btn.clicked.connect(self._on_export_clicked)

    def _on_export_clicked(self) -> None:
        """Prompt for a file path and export the current figure (Export button)."""
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not filepath:
            return
        self.export_figure(filepath)

    def _on_type_changed(self, index: int) -> None:
        """Handle plot type change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        plot_type = self._type_combo.current_data()
        self._current_plot_type = plot_type
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="plot_type_changed",
            plot_type=plot_type,
        )
        self.plot_changed.emit(plot_type)

        if self._residuals is not None:
            self._refresh_plot()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x: np.ndarray | None = None,
    ) -> None:
        """Set data and plot residuals.

        Parameters
        ----------
        y_true : np.ndarray
            True/observed values
        y_pred : np.ndarray
            Predicted/fitted values
        x : np.ndarray, optional
            X values (independent variable)
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="plot_residuals",
            y_true_shape=y_true.shape if hasattr(y_true, "shape") else len(y_true),
            y_pred_shape=y_pred.shape if hasattr(y_pred, "shape") else len(y_pred),
        )
        self._y_true = np.asarray(y_true).flatten()
        self._y_pred = np.asarray(y_pred).flatten()
        self._x_values = np.asarray(x).flatten() if x is not None else None

        # Keep the signed (complex) residual. Oscillation-mode fits carry
        # y = G' + i*G'' as a genuine complex array; collapsing to
        # np.abs() here would lose directionality and make the mean/std,
        # Q-Q, and histogram diagnostics meaningless. _residual_parts()
        # splits complex residuals into real/imag components for display.
        self._residuals = self._y_true - self._y_pred

        self._update_statistics()
        self._refresh_plot()

    def set_residuals(self, residuals: np.ndarray) -> None:
        """Set residuals directly.

        Parameters
        ----------
        residuals : np.ndarray
            Residual values
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_residuals",
            residuals_shape=(
                residuals.shape if hasattr(residuals, "shape") else len(residuals)
            ),
        )
        self._residuals = np.asarray(residuals).flatten()
        self._y_true = None
        self._y_pred = None
        self._x_values = None

        self._update_statistics()
        self._refresh_plot()
        self._empty_label.hide()

    def _residual_parts(self) -> list[tuple[str, np.ndarray]]:
        """Split residuals into labeled, real-valued parts.

        Returns ``[("", residuals)]`` for real-valued residuals, or
        ``[("Re", real_part), ("Im", imag_part)]`` for complex (e.g.
        oscillation-mode G'+iG'') residuals, so every diagnostic operates
        on signed values instead of a collapsed magnitude.
        """
        if self._residuals is None:
            return []
        if np.iscomplexobj(self._residuals):
            return [("Re", self._residuals.real), ("Im", self._residuals.imag)]
        return [("", self._residuals)]

    def _fitted_part(self, label: str) -> np.ndarray | None:
        """Return the fitted-value array matching a residual part's label."""
        if self._y_pred is None:
            return None
        if np.iscomplexobj(self._y_pred):
            return self._y_pred.imag if label == "Im" else self._y_pred.real
        return self._y_pred

    def _update_statistics(self) -> None:
        """Update statistics display."""
        if self._residuals is None or len(self._residuals) == 0:
            self._stats_label.setText("")
            return

        n = len(self._residuals)
        stats = " | ".join(
            f"mean{f'({label})' if label else ''}={np.mean(r):.3g} "
            f"std{f'({label})' if label else ''}={np.std(r):.3g}"
            for label, r in self._residual_parts()
        )
        self._stats_label.setText(f"n={n} | {stats}")

    def _refresh_plot(self) -> None:
        """Refresh the current plot."""
        if self._residuals is None:
            self._empty_label.show()
            return

        self._empty_label.hide()

        self._figure.clear()

        plot_funcs = {
            "residuals": self._plot_residuals_vs_fitted,
            "qq": self._plot_qq,
            "histogram": self._plot_histogram,
            "scale_location": self._plot_scale_location,
            "residuals_x": self._plot_residuals_vs_x,
            "autocorr": self._plot_autocorr,
        }

        func = plot_funcs.get(self._current_plot_type)
        if func:
            try:
                func()
            except Exception as e:
                logger.error(
                    "Error generating plot",
                    widget=self.__class__.__name__,
                    plot_type=self._current_plot_type,
                    error=str(e),
                    exc_info=True,
                )
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"Error generating plot:\n{e}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self._empty_label.setText(f"Plot error: {e}")
                self._empty_label.show()
        else:
            self._empty_label.setText("Unsupported residual plot")
            self._empty_label.show()

        try:
            self._canvas.draw()
        except Exception as e:
            # Agg/FreeType rendering can fail for reasons outside this
            # widget's control (host DPI/font-cache quirks triggering a
            # "raster overflow", degenerate tight-layout margins, etc.) --
            # same rationale as the plot-generation try/except above, just
            # covering the draw() call it currently doesn't reach.
            logger.error(
                "Error rendering canvas",
                widget=self.__class__.__name__,
                plot_type=self._current_plot_type,
                error=str(e),
                exc_info=True,
            )
            self._empty_label.setText(f"Render error: {e}")
            self._empty_label.show()

    def _plot_residuals_vs_fitted(self) -> None:
        """Plot residuals vs fitted values."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        for label, residuals in parts:
            fitted = self._fitted_part(label)
            if fitted is None:
                fitted = np.arange(len(residuals))

            sc = ax.scatter(
                fitted, residuals, alpha=0.6, edgecolors="none",
                label=label if multi else None,
            )

            # Add smoothed trend line
            if len(residuals) > 10:
                try:
                    from scipy.ndimage import uniform_filter1d

                    sorted_idx = np.argsort(fitted)
                    smoothed = uniform_filter1d(
                        residuals[sorted_idx], size=max(5, len(residuals) // 20)
                    )
                    ax.plot(
                        fitted[sorted_idx],
                        smoothed,
                        color=sc.get_facecolor()[0],
                        alpha=0.5,
                        linewidth=2,
                    )
                except ImportError as exc:
                    logger.debug(
                        "uniform_filter1d unavailable",
                        widget=self.__class__.__name__,
                        error=str(exc),
                    )

        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
        if multi:
            ax.legend()
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")

    def _plot_qq(self) -> None:
        """Generate Q-Q plot."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        try:
            from scipy import stats
        except ImportError as exc:
            stats = None
            logger.debug(
                "scipy.stats unavailable for QQ plot",
                widget=self.__class__.__name__,
                error=str(exc),
            )

        # VIS-017: Guard against zero std (perfect fit) across all parts
        if all(np.std(r) < 1e-15 for _, r in parts):
            ax.text(
                0.5,
                0.5,
                "Perfect fit\n(zero residuals)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Normal Q-Q Plot")
            return

        for label, residuals in parts:
            std = np.std(residuals)
            if std < 1e-15:
                continue
            if stats is not None:
                standardized = (residuals - np.mean(residuals)) / std
                stats.probplot(standardized, dist="norm", plot=ax)
                if multi:
                    ax.lines[-2].set_label(label)
            else:
                # Fallback without scipy
                sorted_res = np.sort(residuals)
                n = len(sorted_res)
                theoretical = np.linspace(-2, 2, n)

                ax.scatter(
                    theoretical, sorted_res, alpha=0.6, label=label if multi else None
                )
                ax.plot([-3, 3], [-3, 3], "r--", linewidth=1)
                ax.set_xlabel("Theoretical Quantiles")
                ax.set_ylabel("Sample Quantiles")

        if multi:
            ax.legend()
        ax.set_title("Normal Q-Q Plot" if stats is not None else "Q-Q Plot (simplified)")

    def _plot_histogram(self) -> None:
        """Plot histogram of residuals."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        try:
            from scipy import stats
        except ImportError as exc:
            stats = None
            logger.debug(
                "scipy.stats unavailable for histogram fit",
                widget=self.__class__.__name__,
                error=str(exc),
            )

        for label, residuals in parts:
            _n, bins, _patches = ax.hist(
                residuals,
                bins="auto",
                density=True,
                alpha=0.5 if multi else 0.7,
                edgecolor="black",
                label=f"Residuals ({label})" if multi else None,
            )

            # Fit normal distribution
            if stats is not None:
                mu, std = np.mean(residuals), np.std(residuals)
                x = np.linspace(bins[0], bins[-1], 100)
                ax.plot(
                    x,
                    stats.norm.pdf(x, mu, std),
                    linewidth=2,
                    label=f"Normal fit ({label})" if multi else "Normal fit",
                )

        if multi or stats is not None:
            ax.legend()
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")

    def _plot_scale_location(self) -> None:
        """Plot scale-location (sqrt |residuals| vs fitted)."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        for label, residuals in parts:
            fitted = self._fitted_part(label)
            if fitted is None:
                fitted = np.arange(len(residuals))

            sqrt_abs_res = np.sqrt(np.abs(residuals))

            sc = ax.scatter(
                fitted, sqrt_abs_res, alpha=0.6, edgecolors="none",
                label=label if multi else None,
            )

            # Add smoothed trend line
            if len(residuals) > 10:
                try:
                    from scipy.ndimage import uniform_filter1d

                    sorted_idx = np.argsort(fitted)
                    smoothed = uniform_filter1d(
                        sqrt_abs_res[sorted_idx], size=max(5, len(residuals) // 20)
                    )
                    ax.plot(
                        fitted[sorted_idx],
                        smoothed,
                        color=sc.get_facecolor()[0],
                        alpha=0.5,
                        linewidth=2,
                    )
                except ImportError as exc:
                    logger.debug(
                        "uniform_filter1d unavailable",
                        widget=self.__class__.__name__,
                        error=str(exc),
                    )

        if multi:
            ax.legend()
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("√|Residuals|")
        ax.set_title("Scale-Location Plot")

    def _plot_residuals_vs_x(self) -> None:
        """Plot residuals vs X values."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        for label, residuals in parts:
            x = (
                self._x_values
                if self._x_values is not None
                else np.arange(len(residuals))
            )
            ax.scatter(
                x, residuals, alpha=0.6, edgecolors="none",
                label=label if multi else None,
            )

        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
        if multi:
            ax.legend()
        ax.set_xlabel("X Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs X")

    def _plot_autocorr(self) -> None:
        """Plot residual autocorrelation."""
        ax = self._figure.add_subplot(111)
        parts = self._residual_parts()
        multi = len(parts) > 1

        n = len(self._residuals)
        if n < 10:
            ax.text(
                0.5,
                0.5,
                "Not enough data points for autocorrelation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Compute autocorrelation
        max_lag = min(40, n // 4)
        width = 0.8 / len(parts)
        for i, (label, residuals) in enumerate(parts):
            centered = residuals - np.mean(residuals)
            autocorr = np.correlate(centered, centered, mode="full")
            autocorr = autocorr[n - 1 : n - 1 + max_lag]
            # VIS-018: Guard against zero normalization (perfect fit / constant residuals)
            if autocorr[0] < 1e-15:
                autocorr = np.zeros_like(autocorr)
            else:
                autocorr = autocorr / autocorr[0]  # Normalize

            lags = np.arange(max_lag) + i * width
            ax.bar(
                lags, autocorr, width=width, alpha=0.7, label=label if multi else None
            )

        # Confidence bounds (approximate)
        conf_bound = 1.96 / np.sqrt(n)
        ax.axhline(y=conf_bound, color="r", linestyle="--", linewidth=1)
        ax.axhline(y=-conf_bound, color="r", linestyle="--", linewidth=1)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        if multi:
            ax.legend()
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Residual Autocorrelation")

    def set_plot_type(self, plot_type: str) -> None:
        """Set current plot type.

        Parameters
        ----------
        plot_type : str
            Plot type identifier
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="set_plot_type",
            plot_type=plot_type,
        )
        self._type_combo.set_current_data(plot_type)

    def get_plot_type(self) -> str:
        """Get current plot type.

        Returns
        -------
        str
            Plot type identifier
        """
        return self._current_plot_type

    def get_residuals(self) -> np.ndarray | None:
        """Get current residuals.

        Returns
        -------
        np.ndarray or None
            Current residuals or None
        """
        return self._residuals

    def get_statistics(self) -> dict[str, float]:
        """Get residual statistics.

        Returns
        -------
        dict
            Statistics dictionary with keys: mean, std, n, min, max. For
            complex (e.g. oscillation-mode G'+iG'') residuals, real/imaginary
            components are reported separately as ``mean_re``/``mean_im``, etc.
        """
        if self._residuals is None:
            return {}

        n = len(self._residuals)
        parts = self._residual_parts()
        if len(parts) == 1:
            _, r = parts[0]
            return {
                "n": n,
                "mean": float(np.mean(r)),
                "std": float(np.std(r)),
                "min": float(np.min(r)),
                "max": float(np.max(r)),
            }

        stats: dict[str, float] = {"n": n}
        for label, r in parts:
            suffix = label.lower()
            stats[f"mean_{suffix}"] = float(np.mean(r))
            stats[f"std_{suffix}"] = float(np.std(r))
            stats[f"min_{suffix}"] = float(np.min(r))
            stats[f"max_{suffix}"] = float(np.max(r))
        return stats

    def export_figure(self, filepath: str, dpi: int = 150) -> None:
        """Export figure to file.

        Parameters
        ----------
        filepath : str
            Output file path
        dpi : int, optional
            Resolution for raster formats
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="export_figure",
            filepath=filepath,
            dpi=dpi,
        )
        try:
            self._figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.debug(
                "Figure exported successfully",
                widget=self.__class__.__name__,
                filepath=filepath,
            )
        except Exception as e:
            logger.error(
                "Failed to export figure",
                widget=self.__class__.__name__,
                filepath=filepath,
                error=str(e),
                exc_info=True,
            )
            raise

    def clear(self) -> None:
        """Clear the panel."""
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="clear",
        )
        self._x_values = None
        self._y_true = None
        self._y_pred = None
        self._residuals = None
        self._stats_label.setText("")
        self._figure.clear()
        self._canvas.draw()
        self._empty_label.show()

    def get_figure(self) -> Figure:
        """Get matplotlib figure.

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        return self._figure
