"""
Transform Page
=============

Transform application interface (mastercurve, FFT, SRFS, etc.).
"""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.plot_canvas import PlotCanvas


class TransformPage(QWidget):
    """Transform application page."""

    transform_selected = Signal(str)
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = StateStore()
        self._selected_transform: str | None = None
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Transform cards grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(15)

        transforms = [
            ("FFT", "Fast Fourier Transform for frequency analysis", "#FF5722"),
            ("Mastercurve", "Time-Temperature Superposition", "#2196F3"),
            ("SRFS", "Strain-Rate Frequency Superposition", "#4CAF50"),
            ("Mutation Number", "Calculate mutation number", "#9C27B0"),
            ("OW Chirp", "Optimally-windowed chirp analysis", "#FF9800"),
            ("Derivatives", "Calculate numerical derivatives", "#607D8B"),
        ]

        for i, (name, desc, color) in enumerate(transforms):
            card = self._create_transform_card(name, desc, color)
            grid_layout.addWidget(card, i // 3, i % 3)

        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, 1)

        # Config and preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_config_panel())
        splitter.addWidget(self._create_preview_panel())
        splitter.setSizes([300, 700])
        layout.addWidget(splitter, 2)

    def _create_transform_card(self, name: str, desc: str, color: str) -> QWidget:
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 8px;
                padding: 15px;
            }}
            QFrame:hover {{
                background-color: {self._darken(color)};
            }}
        """)
        card.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(card)
        name_label = QLabel(name)
        name_label.setStyleSheet("color: white; font-size: 13pt; font-weight: bold;")
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: white; font-size: 9pt;")
        desc_label.setWordWrap(True)

        layout.addWidget(name_label)
        layout.addWidget(desc_label)
        layout.addStretch()

        btn = QPushButton("Configure")
        btn.setStyleSheet("background-color: white; color: black; font-weight: bold; padding: 5px;")
        btn.clicked.connect(lambda: self._select_transform(name))
        layout.addWidget(btn)

        return card

    def _create_config_panel(self) -> QWidget:
        panel = QGroupBox("Configuration")
        layout = QVBoxLayout(panel)

        self._config_widget = QWidget()
        self._config_layout = QVBoxLayout(self._config_widget)
        self._config_layout.addWidget(QLabel("Select a transform to configure"))
        layout.addWidget(self._config_widget)

        layout.addStretch()

        btn_apply = QPushButton("Apply Transform")
        btn_apply.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_apply.clicked.connect(self._apply_transform)
        layout.addWidget(btn_apply)

        return panel

    def _create_preview_panel(self) -> QWidget:
        panel = QGroupBox("Preview")
        layout = QVBoxLayout(panel)

        # Before/after split
        splitter = QSplitter(Qt.Horizontal)

        before_widget = QWidget()
        before_layout = QVBoxLayout(before_widget)
        before_layout.addWidget(QLabel("Before", styleSheet="font-weight: bold; font-size: 11pt;"))
        self._before_canvas = PlotCanvas()
        before_layout.addWidget(self._before_canvas)

        after_widget = QWidget()
        after_layout = QVBoxLayout(after_widget)
        after_layout.addWidget(QLabel("After", styleSheet="font-weight: bold; font-size: 11pt;"))
        self._after_canvas = PlotCanvas()
        after_layout.addWidget(self._after_canvas)

        splitter.addWidget(before_widget)
        splitter.addWidget(after_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)

        return panel

    def _select_transform(self, name: str) -> None:
        self._selected_transform = name
        self.transform_selected.emit(name)

        # Clear and rebuild config
        while self._config_layout.count():
            child = self._config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._config_layout.addWidget(QLabel(f"{name} Parameters", styleSheet="font-weight: bold; font-size: 11pt;"))

        # Add transform-specific parameters
        if name == "Mastercurve":
            self._config_layout.addWidget(QLabel("Reference Temperature (°C):"))
            spin = QDoubleSpinBox()
            spin.setRange(-100, 300)
            spin.setValue(25)
            self._config_layout.addWidget(spin)

            check = QCheckBox("Auto-detect shift factors")
            check.setChecked(True)
            self._config_layout.addWidget(check)

        elif name == "SRFS":
            self._config_layout.addWidget(QLabel("Reference Shear Rate (1/s):"))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000)
            spin.setValue(1.0)
            self._config_layout.addWidget(spin)

        elif name == "FFT":
            self._config_layout.addWidget(QLabel("Window Function:"))
            combo = QComboBox()
            combo.addItems(["Hann", "Hamming", "Blackman", "None"])
            self._config_layout.addWidget(combo)

        self._config_layout.addStretch()

    def _apply_transform(self) -> None:
        if not self._selected_transform:
            return

        dataset = self._store.get_active_dataset()
        if dataset:
            self.transform_applied.emit(self._selected_transform, dataset.id)

    def _darken(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"#{int(r*0.9):02x}{int(g*0.9):02x}{int(b*0.9):02x}"

    def apply_transform(self, transform_name: str, dataset_ids: list[str], params: dict[str, Any] | None = None) -> None:
        pass  # Delegated to service

    def show_transform_preview(self, transform_name: str, params: dict[str, Any]) -> None:
        pass  # Updates preview canvases

    def get_available_transforms(self) -> list[dict[str, Any]]:
        return []
