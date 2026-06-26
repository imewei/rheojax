"""Qt binding abstraction layer for PyQt/PySide compatibility.

This module re-exports the Qt classes RheoJAX uses through ``qtpy``, which
itself selects the installed binding (PySide6 by default, PyQt6 if present).

Usage:
    from rheojax.gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
"""

from __future__ import annotations

# Attempt imports in priority order
QT_BINDING: str = "none"
QT_VERSION: str = "0.0.0"

try:
    # Priority 1: qtpy abstraction layer (best compatibility)
    from qtpy import QT_VERSION as _qt_version
    from qtpy import QtCore, QtGui, QtWidgets
    from qtpy.QtCore import (
        QObject,
        QPoint,
        QRectF,
        QRunnable,
        QSettings,
        QSize,
        Qt,
        QThread,
        QThreadPool,
        QTimer,
        Signal,
        Slot,
    )
    from qtpy.QtGui import (
        QAction,
        QBrush,
        QCloseEvent,
        QColor,
        QDragEnterEvent,
        QDropEvent,
        QFont,
        QIcon,
        QImage,
        QKeySequence,
        QPalette,
        QPixmap,
        QShortcut,
    )
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDockWidget,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMenuBar,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QProgressDialog,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QStyle,
        QStyleFactory,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextBrowser,
        QTextEdit,
        QToolBar,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
        QWizard,
        QWizardPage,
    )

    QT_BINDING = "qtpy"
    QT_VERSION = str(_qt_version)

except ImportError as e:
    raise ImportError(
        "No Qt binding found. Install one of:\n"
        "  - pip install qtpy PySide6  (recommended)\n"
        "  - pip install PySide6\n"
        "  - pip install PyQt6"
    ) from e


def _is_qobject_alive(obj: object) -> bool:
    """Check whether a QObject's C++ counterpart is still alive.

    Works across PySide6 (shiboken6), PyQt6, and qtpy by trying the
    appropriate validity check for the active binding.  Returns ``True``
    if the binding does not expose a validity check (safe default).
    """
    try:
        import shiboken6  # PySide6

        return shiboken6.isValid(obj)  # type: ignore[arg-type]
    except ImportError:
        pass
    try:
        import sip  # PyQt6/PyQt5

        return not sip.isdeleted(obj)  # type: ignore[arg-type]
    except (ImportError, TypeError):
        pass
    return True


def get_qt_info() -> dict[str, str]:
    """Get information about the active Qt binding.

    Returns
    -------
    dict
        Dictionary with 'binding' and 'version' keys.
    """
    return {
        "binding": QT_BINDING,
        "version": QT_VERSION,
    }


# Convenience re-exports for common patterns
__all__ = [
    # Binding info
    "QT_BINDING",
    "QT_VERSION",
    "get_qt_info",
    "_is_qobject_alive",
    # Core modules
    "QtCore",
    "QtGui",
    "QtWidgets",
    # Core classes
    "Signal",
    "Slot",
    "Qt",
    "QObject",
    "QRectF",
    "QRunnable",
    "QSettings",
    "QThread",
    "QThreadPool",
    "QTimer",
    "QSize",
    "QPoint",
    # GUI classes
    "QAction",
    "QBrush",
    "QCloseEvent",
    "QColor",
    "QDragEnterEvent",
    "QDropEvent",
    "QFont",
    "QIcon",
    "QImage",
    "QKeySequence",
    "QPalette",
    "QPixmap",
    "QShortcut",
    # Widgets
    "QAbstractItemView",
    "QApplication",
    "QButtonGroup",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QDockWidget",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QInputDialog",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMenuBar",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QProgressDialog",
    "QPushButton",
    "QRadioButton",
    "QScrollArea",
    "QSizePolicy",
    "QSlider",
    "QSpinBox",
    "QSplitter",
    "QStackedWidget",
    "QStatusBar",
    "QStyle",
    "QStyleFactory",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTextBrowser",
    "QTextEdit",
    "QToolBar",
    "QToolButton",
    "QTreeWidget",
    "QTreeWidgetItem",
    "QVBoxLayout",
    "QWidget",
    "QWizard",
    "QWizardPage",
]
