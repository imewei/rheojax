"""
Residuals Panel Widget
=====================

Residual analysis visualization for model fitting diagnostics.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from rheojax.gui.compat import (
    Qt,
    Signal,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
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
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(4, 4, 4, 4)

        # Plot type selector
        type_label = QLabel("Plot Type:")
        toolbar_layout.addWidget(type_label)

        self._type_combo = QComboBox()
        self._type_combo.setMinimumWidth(140)
        for plot_id, display_name, tooltip in PLOT_TYPES:
            self._type_combo.addItem(display_name, plot_id)
            idx = self._type_combo.count() - 1
            self._type_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        toolbar_layout.addWidget(self._type_combo)

        toolbar_layout.addStretch()

        # Statistics label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: gray;")
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
        self._empty_label.setStyleSheet("color: #94A3B8; padding: 6px;")
        layout.addWidget(self._empty_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

    def _on_type_changed(self, index: int) -> None:
        """Handle plot type change.

        Parameters
        ----------
        index : int
            New combo box index
        """
        plot_type = self._type_combo.currentData()
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

        # Handle complex data
        if np.iscomplexobj(self._y_true) or np.iscomplexobj(self._y_pred):
            self._residuals = np.abs(self._y_true - self._y_pred)
        else:
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

    def _update_statistics(self) -> None:
        """Update statistics display."""
        if self._residuals is None or len(self._residuals) == 0:
            self._stats_label.setText("")
            return

        mean = np.mean(self._residuals)
        std = np.std(self._residuals)
        n = len(self._residuals)

        self._stats_label.setText(f"n={n} | mean={mean:.3g} | std={std:.3g}")

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

        self._canvas.draw()

    def _plot_residuals_vs_fitted(self) -> None:
        """Plot residuals vs fitted values."""
        ax = self._figure.add_subplot(111)

        if self._y_pred is not None:
            fitted = self._y_pred
        else:
            fitted = np.arange(len(self._residuals))

        ax.scatter(fitted, self._residuals, alpha=0.6, edgecolors="none")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)

        # Add smoothed trend line
        if len(self._residuals) > 10:
            try:
                from scipy.ndimage import uniform_filter1d

                sorted_idx = np.argsort(fitted)
                smoothed = uniform_filter1d(
                    self._residuals[sorted_idx], size=max(5, len(self._residuals) // 20)
                )
                ax.plot(fitted[sorted_idx], smoothed, "r-", alpha=0.5, linewidth=2)
            except ImportError as exc:
                logger.debug(
                    "uniform_filter1d unavailable",
                    widget=self.__class__.__name__,
                    error=str(exc),
                )

        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")

    def _plot_qq(self) -> None:
        """Generate Q-Q plot."""
        ax = self._figure.add_subplot(111)

        try:
            from scipy import stats

            # Standardize residuals
            standardized = (self._residuals - np.mean(self._residuals)) / np.std(
                self._residuals
            )

            # Q-Q plot
            stats.probplot(standardized, dist="norm", plot=ax)
            ax.set_title("Normal Q-Q Plot")
        except ImportError as exc:
            logger.debug(
                "scipy.stats unavailable for QQ plot",
                widget=self.__class__.__name__,
                error=str(exc),
            )
            # Fallback without scipy
            sorted_res = np.sort(self._residuals)
            n = len(sorted_res)
            theoretical = np.linspace(-2, 2, n)

            ax.scatter(theoretical, sorted_res, alpha=0.6)
            ax.plot([-3, 3], [-3, 3], "r--", linewidth=1)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("Q-Q Plot (simplified)")

    def _plot_histogram(self) -> None:
        """Plot histogram of residuals."""
        ax = self._figure.add_subplot(111)

        # Histogram
        n, bins, patches = ax.hist(
            self._residuals, bins="auto", density=True, alpha=0.7, edgecolor="black"
        )

        # Fit normal distribution
        try:
            from scipy import stats

            mu, std = np.mean(self._residuals), np.std(self._residuals)
            x = np.linspace(bins[0], bins[-1], 100)
            ax.plot(
                x, stats.norm.pdf(x, mu, std), "r-", linewidth=2, label="Normal fit"
            )
            ax.legend()
        except ImportError as exc:
            logger.debug(
                "scipy.stats unavailable for histogram fit",
                widget=self.__class__.__name__,
                error=str(exc),
            )

        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")

    def _plot_scale_location(self) -> None:
        """Plot scale-location (sqrt |residuals| vs fitted)."""
        ax = self._figure.add_subplot(111)

        if self._y_pred is not None:
            fitted = self._y_pred
        else:
            fitted = np.arange(len(self._residuals))

        sqrt_abs_res = np.sqrt(np.abs(self._residuals))

        ax.scatter(fitted, sqrt_abs_res, alpha=0.6, edgecolors="none")

        # Add smoothed trend line
        if len(self._residuals) > 10:
            try:
                from scipy.ndimage import uniform_filter1d

                sorted_idx = np.argsort(fitted)
                smoothed = uniform_filter1d(
                    sqrt_abs_res[sorted_idx], size=max(5, len(self._residuals) // 20)
                )
                ax.plot(fitted[sorted_idx], smoothed, "r-", alpha=0.5, linewidth=2)
            except ImportError as exc:
                logger.debug(
                    "uniform_filter1d unavailable",
                    widget=self.__class__.__name__,
                    error=str(exc),
                )

        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("√|Residuals|")
        ax.set_title("Scale-Location Plot")

    def _plot_residuals_vs_x(self) -> None:
        """Plot residuals vs X values."""
        ax = self._figure.add_subplot(111)

        if self._x_values is not None:
            x = self._x_values
        else:
            x = np.arange(len(self._residuals))

        ax.scatter(x, self._residuals, alpha=0.6, edgecolors="none")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1)

        ax.set_xlabel("X Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs X")

    def _plot_autocorr(self) -> None:
        """Plot residual autocorrelation."""
        ax = self._figure.add_subplot(111)

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
        autocorr = np.correlate(
            self._residuals - np.mean(self._residuals),
            self._residuals - np.mean(self._residuals),
            mode="full",
        )
        autocorr = autocorr[n - 1 : n - 1 + max_lag]
        autocorr = autocorr / autocorr[0]  # Normalize

        lags = np.arange(max_lag)
        ax.bar(lags, autocorr, width=0.8, alpha=0.7)

        # Confidence bounds (approximate)
        conf_bound = 1.96 / np.sqrt(n)
        ax.axhline(y=conf_bound, color="r", linestyle="--", linewidth=1)
        ax.axhline(y=-conf_bound, color="r", linestyle="--", linewidth=1)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

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
        idx = self._type_combo.findData(plot_type)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)

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
            Statistics dictionary with keys: mean, std, n, min, max
        """
        if self._residuals is None:
            return {}

        return {
            "n": len(self._residuals),
            "mean": float(np.mean(self._residuals)),
            "std": float(np.std(self._residuals)),
            "min": float(np.min(self._residuals)),
            "max": float(np.max(self._residuals)),
        }

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
