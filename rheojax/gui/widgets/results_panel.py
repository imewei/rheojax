"""
Results Panel Widget
====================

Displays summary metrics for the latest fit and Bayesian runs.
"""

from __future__ import annotations

from rheojax.gui.compat import QLabel, QTextEdit, QVBoxLayout, QWidget
from rheojax.gui.state.store import BayesianResult, FitResult
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ResultsPanel(QWidget):
    """Compact panel to show fit/Bayesian summary metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fit_label = QLabel("Fit Results")
        self.fit_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        self.fit_text = QTextEdit()
        self.fit_text.setReadOnly(True)
        self.fit_text.setMaximumHeight(120)
        self.fit_text.setStyleSheet("font-size: 11pt;")

        self.bayes_label = QLabel("Bayesian Results")
        self.bayes_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        self.bayes_text = QTextEdit()
        self.bayes_text.setReadOnly(True)
        self.bayes_text.setMaximumHeight(140)
        self.bayes_text.setStyleSheet("font-size: 11pt;")

        layout.addWidget(self.fit_label)
        layout.addWidget(self.fit_text)
        layout.addWidget(self.bayes_label)
        layout.addWidget(self.bayes_text)
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def set_fit_result(self, result: FitResult | None) -> None:
        """Render a fit result summary."""
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_fit_result",
            has_result=result is not None,
        )
        if result is None:
            self.fit_text.setText("No fit results yet.")
            return

        lines = [
            f"Model: {result.model_name}",
            f"Dataset: {result.dataset_id}",
            f"R²: {result.r_squared:.4f}",
            f"MPE: {result.mpe:.2f}%",
            f"χ²: {result.chi_squared:.4f}",
            f"Iterations: {result.num_iterations or 0}",
            f"Time: {result.fit_time:.2f} s",
        ]
        self.fit_text.setText("\n".join(lines))
        logger.debug(
            "Fit result displayed",
            widget=self.__class__.__name__,
            model=result.model_name,
            r_squared=result.r_squared,
        )

    def set_bayesian_result(self, result: BayesianResult | None) -> None:
        """Render a Bayesian result summary."""
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_bayesian_result",
            has_result=result is not None,
        )
        if result is None:
            self.bayes_text.setText("No Bayesian results yet.")
            return

        lines = [
            f"Model: {result.model_name}",
            f"Dataset: {result.dataset_id}",
            f"Samples: {result.num_samples} x {result.num_chains} chains",
            f"Time: {result.sampling_time:.2f} s",
            f"Divergences: {result.diagnostics.get('divergences', 0)}",
        ]
        if result.summary:
            lines.append("Summary:")
            for name, stats in result.summary.items():
                mean = stats.get("mean", 0.0)
                sd = stats.get("sd", 0.0)
                lines.append(f"  {name}: {mean:.4g} ± {sd:.4g}")
        self.bayes_text.setText("\n".join(lines))
        logger.debug(
            "Bayesian result displayed",
            widget=self.__class__.__name__,
            model=result.model_name,
            num_samples=result.num_samples,
            num_chains=result.num_chains,
        )
