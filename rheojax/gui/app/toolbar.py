"""
Toolbars
========

Main toolbar and related utilities.
"""

from pathlib import Path

from rheojax.gui.compat import QAction, QIcon, QSize, QToolBar, QWidget
from rheojax.gui.resources.styles import ColorPalette, Spacing
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Path to icons directory
ICONS_DIR = Path(__file__).parent.parent / "resources" / "icons"


def get_icon(name: str) -> QIcon:
    """Get an icon by name from the icons directory.

    Parameters
    ----------
    name : str
        Icon name (without extension)

    Returns
    -------
    QIcon
        The icon, or empty icon if not found
    """
    icon_path = ICONS_DIR / f"{name}.svg"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


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
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.setObjectName("MainToolBar")
        self.setMovable(False)
        self.setIconSize(QSize(20, 20))

        # Apply consistent styling
        self.setStyleSheet(f"""
            QToolBar {{
                background-color: {ColorPalette.BG_ELEVATED};
                border: none;
                border-bottom: 1px solid {ColorPalette.BORDER_DEFAULT};
                spacing: {Spacing.SM}px;
                padding: {Spacing.SM}px {Spacing.MD}px;
            }}
            QToolBar::separator {{
                background-color: {ColorPalette.BORDER_DEFAULT};
                width: 1px;
                margin: {Spacing.XS}px {Spacing.SM}px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: {Spacing.SM}px {Spacing.MD}px;
                font-size: 11pt;
                font-weight: 500;
                color: {ColorPalette.TEXT_PRIMARY};
            }}
            QToolButton:hover {{
                background-color: {ColorPalette.BG_HOVER};
            }}
            QToolButton:pressed {{
                background-color: {ColorPalette.BG_ACTIVE};
            }}
            QToolButton:disabled {{
                color: {ColorPalette.TEXT_DISABLED};
            }}
        """)

        # File operations
        self.open_action = QAction(get_icon("load"), "Open", self)
        self.open_action.setToolTip("Open file (Ctrl+O)")
        self.open_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="open")
        )
        self.addAction(self.open_action)

        self.save_action = QAction(get_icon("export"), "Save", self)
        self.save_action.setToolTip("Save file (Ctrl+S)")
        self.save_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="save")
        )
        self.addAction(self.save_action)

        self.import_action = QAction(get_icon("load"), "Import", self)
        self.import_action.setToolTip("Import data (Ctrl+I)")
        self.import_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="import")
        )
        self.addAction(self.import_action)

        self.addSeparator()

        # Fitting operations
        self.fit_action = QAction(get_icon("fit"), "Fit", self)
        self.fit_action.setToolTip("Fit model (Ctrl+F)")
        self.fit_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="fit")
        )
        self.addAction(self.fit_action)

        self.bayesian_action = QAction(get_icon("bayesian"), "Bayesian", self)
        self.bayesian_action.setToolTip("Bayesian inference (Ctrl+B)")
        self.bayesian_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="bayesian")
        )
        self.addAction(self.bayesian_action)

        self.stop_action = QAction(get_icon("close"), "Stop", self)
        self.stop_action.setToolTip("Stop current operation")
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="stop")
        )
        self.addAction(self.stop_action)

        self.addSeparator()

        # View controls
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.setToolTip("Zoom in (Ctrl++)")
        self.zoom_in_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="zoom_in")
        )
        self.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.setToolTip("Zoom out (Ctrl+-)")
        self.zoom_out_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="zoom_out")
        )
        self.addAction(self.zoom_out_action)

        self.reset_zoom_action = QAction("Reset", self)
        self.reset_zoom_action.setToolTip("Reset zoom (Ctrl+0)")
        self.reset_zoom_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="reset_zoom")
        )
        self.addAction(self.reset_zoom_action)

        self.addSeparator()

        # Settings
        self.settings_action = QAction("Settings", self)
        self.settings_action.setToolTip("Settings")
        self.settings_action.triggered.connect(
            lambda: logger.debug("Toolbar button clicked", button_id="settings")
        )
        self.addAction(self.settings_action)
