"""
Quick Fit Strip Widget
======================

Compact toolbar for quick fitting actions with one-click model selection.
"""


from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from rheojax.gui.services.model_service import normalize_model_name

# Quick-access models with display names and categories
QUICK_MODELS = [
    ("maxwell", "Maxwell", "Classical viscoelastic model"),
    ("zener", "Zener", "Standard linear solid"),
    ("power_law", "Power Law", "Shear-thinning flow"),
    ("carreau", "Carreau", "Generalized Newtonian"),
    ("herschel_bulkley", "Herschel-Bulkley", "Yield stress fluid"),
    ("generalized_maxwell", "GMM", "Multi-mode Maxwell"),
]

# Test mode options
TEST_MODES = [
    ("relaxation", "Relaxation", "G(t) stress relaxation"),
    ("creep", "Creep", "J(t) creep compliance"),
    ("oscillation", "Oscillation", "G*(ω) dynamic moduli"),
    ("flow", "Flow", "η(γ̇) viscosity curve"),
]


class QuickFitStrip(QWidget):
    """Quick fit toolbar widget for rapid model fitting.

    Features:
        - One-click quick model buttons for common models
        - Test mode selector dropdown
        - Progress indicator during fitting
        - Cancel button for long-running fits
        - Fit/Auto-fit buttons

    Signals
    -------
    fit_requested : Signal(str, str)
        Emitted when fit is requested (model_name, test_mode)
    auto_fit_requested : Signal(str)
        Emitted when auto-fit is requested (test_mode)
    cancel_requested : Signal()
        Emitted when cancel is clicked

    Example
    -------
    >>> strip = QuickFitStrip()  # doctest: +SKIP
    >>> strip.fit_requested.connect(on_fit_requested)  # doctest: +SKIP
    >>> strip.show_progress(50)  # doctest: +SKIP
    """

    fit_requested = Signal(str, str)  # model_name, test_mode
    auto_fit_requested = Signal(str)  # test_mode
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize quick fit strip.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._is_fitting = False
        self._selected_model: str | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Quick model buttons
        quick_label = QLabel("Quick Fit:")
        quick_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(quick_label)

        self._model_buttons: dict[str, QToolButton] = {}
        for model_name, display_name, tooltip in QUICK_MODELS:
            btn = QToolButton()
            btn.setText(display_name)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumWidth(60)
            btn.clicked.connect(lambda checked, m=model_name: self._on_model_clicked(m))
            self._model_buttons[model_name] = btn
            layout.addWidget(btn)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep1)

        # Test mode selector
        mode_label = QLabel("Mode:")
        layout.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.setMinimumWidth(100)
        for mode_id, display_name, tooltip in TEST_MODES:
            self._mode_combo.addItem(display_name, mode_id)
            idx = self._mode_combo.count() - 1
            self._mode_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        layout.addWidget(self._mode_combo)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        # Fit button
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setToolTip("Fit selected model to data")
        self._fit_btn.setMinimumWidth(60)
        self._fit_btn.setEnabled(False)
        layout.addWidget(self._fit_btn)

        # Auto-fit button
        self._auto_fit_btn = QPushButton("Auto-Fit")
        self._auto_fit_btn.setToolTip("Automatically select and fit best model")
        self._auto_fit_btn.setMinimumWidth(70)
        layout.addWidget(self._auto_fit_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setMinimumWidth(100)
        self._progress.setMaximumWidth(150)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Cancel button
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip("Cancel current fitting operation")
        self._cancel_btn.setStyleSheet("QPushButton { color: #dc3545; }")
        self._cancel_btn.setVisible(False)
        layout.addWidget(self._cancel_btn)

        # Stretch to push items left
        layout.addStretch()

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._status_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._fit_btn.clicked.connect(self._on_fit_clicked)
        self._auto_fit_btn.clicked.connect(self._on_auto_fit_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

    def _on_model_clicked(self, model_name: str) -> None:
        """Handle quick model button click.

        Parameters
        ----------
        model_name : str
            Name of clicked model
        """
        self._selected_model = normalize_model_name(model_name)
        self._fit_btn.setEnabled(True)
        self._status_label.setText(f"Selected: {self._selected_model}")

    def _on_fit_clicked(self) -> None:
        """Handle fit button click."""
        if self._selected_model:
            test_mode = self._mode_combo.currentData()
            self.fit_requested.emit(self._selected_model, test_mode)

    def _on_auto_fit_clicked(self) -> None:
        """Handle auto-fit button click."""
        test_mode = self._mode_combo.currentData()
        self.auto_fit_requested.emit(test_mode)

    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self.cancel_requested.emit()

    def show_progress(self, value: int) -> None:
        """Update progress bar.

        Parameters
        ----------
        value : int
            Progress value (0-100)
        """
        self._progress.setValue(value)
        if not self._progress.isVisible():
            self.set_fitting(True)

    def set_fitting(self, is_fitting: bool) -> None:
        """Set fitting state.

        Parameters
        ----------
        is_fitting : bool
            Whether fitting is in progress
        """
        self._is_fitting = is_fitting

        # Toggle visibility
        self._progress.setVisible(is_fitting)
        self._cancel_btn.setVisible(is_fitting)

        # Disable controls during fitting
        self._fit_btn.setEnabled(not is_fitting and self._selected_model is not None)
        self._auto_fit_btn.setEnabled(not is_fitting)
        for btn in self._model_buttons.values():
            btn.setEnabled(not is_fitting)
        self._mode_combo.setEnabled(not is_fitting)

        if not is_fitting:
            self._progress.setValue(0)

    def set_status(self, message: str) -> None:
        """Set status message.

        Parameters
        ----------
        message : str
            Status message to display
        """
        self._status_label.setText(message)

    def set_test_mode(self, mode: str) -> None:
        """Set selected test mode.

        Parameters
        ----------
        mode : str
            Test mode identifier
        """
        idx = self._mode_combo.findData(mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)

    def get_test_mode(self) -> str:
        """Get selected test mode.

        Returns
        -------
        str
            Selected test mode identifier
        """
        return self._mode_combo.currentData()

    def select_model(self, model_name: str) -> bool:
        """Programmatically select a model.

        Parameters
        ----------
        model_name : str
            Model name to select

        Returns
        -------
        bool
            True if model was found and selected
        """
        normalized = normalize_model_name(model_name)
        if normalized in self._model_buttons:
            self._model_buttons[normalized].setChecked(True)
            self._selected_model = normalized
            self._fit_btn.setEnabled(True)
            return True
        return False

    def get_selected_model(self) -> str | None:
        """Get selected model name.

        Returns
        -------
        str or None
            Selected model name, or None if no selection
        """
        return self._selected_model

    def clear_selection(self) -> None:
        """Clear model selection."""
        for btn in self._model_buttons.values():
            btn.setChecked(False)
        self._selected_model = None
        self._fit_btn.setEnabled(False)
        self._status_label.setText("")

    def is_fitting(self) -> bool:
        """Check if fitting is in progress.

        Returns
        -------
        bool
            True if fitting is in progress
        """
        return self._is_fitting

    def reset(self) -> None:
        """Reset strip to initial state."""
        self.clear_selection()
        self.set_fitting(False)
        self._mode_combo.setCurrentIndex(0)
