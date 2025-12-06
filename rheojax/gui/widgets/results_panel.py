"""
Results Panel Widget
====================

Displays summary metrics for the latest fit and Bayesian runs.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from rheojax.gui.state.store import BayesianResult, FitResult


class ResultsPanel(QWidget):
    """Compact panel to show fit/Bayesian summary metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fit_label = QLabel("Fit Results")
        self.fit_text = QTextEdit()
        self.fit_text.setReadOnly(True)
        self.fit_text.setMaximumHeight(120)

        self.bayes_label = QLabel("Bayesian Results")
        self.bayes_text = QTextEdit()
        self.bayes_text.setReadOnly(True)
        self.bayes_text.setMaximumHeight(140)

        layout.addWidget(self.fit_label)
        layout.addWidget(self.fit_text)
        layout.addWidget(self.bayes_label)
        layout.addWidget(self.bayes_text)

    def set_fit_result(self, result: FitResult | None) -> None:
        """Render a fit result summary."""
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

    def set_bayesian_result(self, result: BayesianResult | None) -> None:
        """Render a Bayesian result summary."""
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
