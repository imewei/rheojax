"""
Toolbars
========

Main toolbar and related utilities.
"""


from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QLabel, QToolBar, QToolButton, QWidget

from rheojax.gui.services.model_service import normalize_model_name


class MainToolBar(QToolBar):
    """Main application toolbar with common actions.

    Actions:
        - File operations: Open, Save, Import
        - Fitting: Fit, Bayesian, Stop
        - View: Zoom In, Zoom Out, Reset
        - Settings icon

    Example
    -------
    >>> toolbar = MainToolBar()  # doctest: +SKIP
    >>> toolbar.open_action.triggered.connect(on_open)  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize main toolbar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__("Main Toolbar", parent)

        self.setObjectName("MainToolBar")
        self.setMovable(False)

        # File operations
        self.open_action = QAction("Open", self)
        self.open_action.setToolTip("Open file (Ctrl+O)")
        self.addAction(self.open_action)

        self.save_action = QAction("Save", self)
        self.save_action.setToolTip("Save file (Ctrl+S)")
        self.addAction(self.save_action)

        self.import_action = QAction("Import", self)
        self.import_action.setToolTip("Import data (Ctrl+I)")
        self.addAction(self.import_action)

        self.addSeparator()

        # Fitting operations
        self.fit_action = QAction("Fit", self)
        self.fit_action.setToolTip("Fit model (Ctrl+F)")
        self.addAction(self.fit_action)

        self.bayesian_action = QAction("Bayesian", self)
        self.bayesian_action.setToolTip("Bayesian inference (Ctrl+B)")
        self.addAction(self.bayesian_action)

        self.stop_action = QAction("Stop", self)
        self.stop_action.setToolTip("Stop current operation")
        self.stop_action.setEnabled(False)
        self.addAction(self.stop_action)

        self.addSeparator()

        # View controls
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.setToolTip("Zoom in (Ctrl++)")
        self.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.setToolTip("Zoom out (Ctrl+-)")
        self.addAction(self.zoom_out_action)

        self.reset_zoom_action = QAction("Reset", self)
        self.reset_zoom_action.setToolTip("Reset zoom (Ctrl+0)")
        self.addAction(self.reset_zoom_action)

        self.addSeparator()

        # Settings
        self.settings_action = QAction("Settings", self)
        self.settings_action.setToolTip("Settings")
        self.addAction(self.settings_action)
