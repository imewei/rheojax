"""
Bayesian Options Dialog
======================

Configure NUTS sampling parameters.
"""

import random
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)
import json


class BayesianOptionsDialog(QDialog):
    """Bayesian inference configuration.

    Options:
        - Sampler selection (NUTS)
        - Warmup and sampling parameters
        - Number of chains
        - NLSQ warm-start
        - Advanced NUTS parameters

    Example
    -------
    >>> dialog = BayesianOptionsDialog()  # doctest: +SKIP
    >>> if dialog.exec() == QDialog.DialogCode.Accepted:  # doctest: +SKIP
    ...     options = dialog.get_options()  # doctest: +SKIP
    """

    def __init__(
        self,
        current_options: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize Bayesian options dialog.

        Parameters
        ----------
        current_options : dict[str, Any], optional
            Current Bayesian options
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self.current_options = current_options or {}

        self.setWindowTitle("Bayesian Inference Options")
        self.setMinimumSize(550, 600)

        self._setup_ui()
        self._load_current_options()

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # Sampler selection
        sampler_group = QGroupBox("Sampler")
        sampler_layout = QFormLayout()

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(["NUTS (No U-Turn Sampler)"])
        sampler_layout.addRow("Sampling Method:", self.sampler_combo)

        sampler_group.setLayout(sampler_layout)
        layout.addWidget(sampler_group)

        # Sampling parameters
        sampling_group = QGroupBox("Sampling Parameters")
        sampling_layout = QFormLayout()

        # Warmup samples
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(100, 10000)
        self.warmup_spin.setValue(1000)
        self.warmup_spin.setSingleStep(100)
        sampling_layout.addRow("Warmup Samples:", self.warmup_spin)

        # Number of samples
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 20000)
        self.samples_spin.setValue(2000)
        self.samples_spin.setSingleStep(100)
        sampling_layout.addRow("Number of Samples:", self.samples_spin)

        # Number of chains
        self.chains_spin = QSpinBox()
        self.chains_spin.setRange(1, 8)
        self.chains_spin.setValue(4)
        sampling_layout.addRow("Number of Chains:", self.chains_spin)

        sampling_group.setLayout(sampling_layout)
        layout.addWidget(sampling_group)

        # Initialization options
        init_group = QGroupBox("Initialization")
        init_layout = QVBoxLayout()

        # Warm-start from NLSQ
        self.warmstart_check = QCheckBox(
            "Warm-start from NLSQ fit (recommended for faster convergence)"
        )
        self.warmstart_check.setChecked(True)
        init_layout.addWidget(self.warmstart_check)

        # Random seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random Seed:"))
        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("(Optional)")
        self.seed_edit.setMaximumWidth(150)
        seed_layout.addWidget(self.seed_edit)

        generate_seed_button = QPushButton("Generate")
        generate_seed_button.clicked.connect(self._generate_seed)
        seed_layout.addWidget(generate_seed_button)
        seed_layout.addStretch()

        init_layout.addLayout(seed_layout)
        init_group.setLayout(init_layout)
        layout.addWidget(init_group)

        # Advanced section (collapsible)
        self.advanced_group = QGroupBox("Advanced NUTS Parameters")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout()

        # Dense mass matrix
        self.dense_mass_check = QCheckBox("Use dense mass matrix")
        self.dense_mass_check.setChecked(False)
        advanced_layout.addWidget(self.dense_mass_check)

        # Target accept rate
        target_layout = QVBoxLayout()
        target_label_layout = QHBoxLayout()
        target_label_layout.addWidget(QLabel("Target Accept Rate:"))
        self.target_label = QLabel("0.80")
        target_label_layout.addWidget(self.target_label)
        target_label_layout.addStretch()
        target_layout.addLayout(target_label_layout)

        self.target_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_slider.setRange(65, 95)  # 0.65 to 0.95
        self.target_slider.setValue(80)  # 0.80
        self.target_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.target_slider.setTickInterval(5)
        self.target_slider.valueChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_slider)

        advanced_layout.addLayout(target_layout)

        # Max tree depth
        max_depth_layout = QFormLayout()
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(5, 15)
        self.max_depth_spin.setValue(10)
        max_depth_layout.addRow("Max Tree Depth:", self.max_depth_spin)
        advanced_layout.addLayout(max_depth_layout)

        self.advanced_group.setLayout(advanced_layout)
        layout.addWidget(self.advanced_group)

        # Priors (JSON editable)
        self.priors_group = QGroupBox("Priors (JSON editable)")
        self.priors_group.setCheckable(True)
        self.priors_group.setChecked(True)
        priors_layout = QVBoxLayout()
        self.priors_edit = QPlainTextEdit()
        self.priors_edit.setPlaceholderText("e.g. {\n  \"G_cage\": {\"dist\": \"lognormal\", \"loc\": 8.5, \"scale\": 1.0}\n}")
        self.priors_edit.setMinimumHeight(120)
        priors_layout.addWidget(self.priors_edit)
        self.priors_group.setLayout(priors_layout)
        layout.addWidget(self.priors_group)

        # Info label
        info_label = QLabel(
            "<b>Note:</b> NLSQ warm-start significantly improves convergence "
            "(R-hat < 1.01, ESS > 400)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info_label)

        layout.addStretch()

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _load_current_options(self) -> None:
        """Load current options into UI."""
        if not self.current_options:
            return

        # Sampler (currently only NUTS)
        if "sampler" in self.current_options:
            sampler = self.current_options["sampler"]
            idx = self.sampler_combo.findText(sampler, flags=Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.sampler_combo.setCurrentIndex(idx)

        # Warmup samples
        if "num_warmup" in self.current_options:
            self.warmup_spin.setValue(self.current_options["num_warmup"])

        # Number of samples
        if "num_samples" in self.current_options:
            self.samples_spin.setValue(self.current_options["num_samples"])

        # Number of chains
        if "num_chains" in self.current_options:
            self.chains_spin.setValue(self.current_options["num_chains"])

        # Warm-start
        if "warm_start" in self.current_options:
            self.warmstart_check.setChecked(self.current_options["warm_start"])

        # Seed
        if "seed" in self.current_options:
            seed = self.current_options["seed"]
            if seed is not None:
                self.seed_edit.setText(str(seed))

        # Advanced options
        if "dense_mass" in self.current_options:
            self.dense_mass_check.setChecked(self.current_options["dense_mass"])
            self.advanced_group.setChecked(True)

        if "target_accept_prob" in self.current_options:
            target = self.current_options["target_accept_prob"]
            self.target_slider.setValue(int(target * 100))
            self.advanced_group.setChecked(True)

        if "max_tree_depth" in self.current_options:
            self.max_depth_spin.setValue(self.current_options["max_tree_depth"])
            self.advanced_group.setChecked(True)

    def _on_target_changed(self, value: int) -> None:
        """Handle target accept rate slider change."""
        self.target_label.setText(f"{value / 100:.2f}")

    def _generate_seed(self) -> None:
        """Generate random seed."""
        seed = random.randint(0, 2**31 - 1)
        self.seed_edit.setText(str(seed))

    def get_options(self) -> dict[str, Any]:
        """Get Bayesian inference options.

        Returns
        -------
        dict[str, Any]
            Bayesian options with keys:
            - sampler: Sampling method (currently "NUTS")
            - num_warmup: Number of warmup samples
            - num_samples: Number of samples
            - num_chains: Number of chains
            - warm_start: Whether to warm-start from NLSQ
            - seed: Random seed (or None)
            - dense_mass: Whether to use dense mass matrix (if advanced enabled)
            - target_accept_prob: Target acceptance probability (if advanced enabled)
            - max_tree_depth: Maximum tree depth (if advanced enabled)
            - priors: JSON dict of priors (optional)
        """
        options = {
            "sampler": "NUTS",  # Currently only option
            "num_warmup": self.warmup_spin.value(),
            "num_samples": self.samples_spin.value(),
            "num_chains": self.chains_spin.value(),
            "warm_start": self.warmstart_check.isChecked(),
        }

        # Add seed if specified
        seed_text = self.seed_edit.text().strip()
        if seed_text:
            try:
                options["seed"] = int(seed_text)
            except ValueError:
                options["seed"] = None
        else:
            options["seed"] = None

        # Add advanced options if section is enabled
        if self.advanced_group.isChecked():
            options["dense_mass"] = self.dense_mass_check.isChecked()
            options["target_accept_prob"] = self.target_slider.value() / 100.0
            options["max_tree_depth"] = self.max_depth_spin.value()

        priors_text = self.priors_edit.toPlainText().strip()
        if priors_text:
            try:
                options["priors"] = json.loads(priors_text)
            except Exception:
                options["priors"] = None

        return options


# Alias for backward compatibility
BayesianOptions = BayesianOptionsDialog
