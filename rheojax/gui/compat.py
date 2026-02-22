"""Qt binding abstraction layer for PyQt/PySide compatibility.

This module provides a unified import interface that works with either
PySide6 or PyQt6, following Technical Guidelines §6.1.

Usage:
    from rheojax.gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot

The module attempts imports in this order:
1. qtpy (if installed) - recommended for maximum compatibility
2. PySide6 (default for rheojax)
3. PyQt6 (fallback)

This allows users to choose their preferred Qt binding without code changes.
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

except ImportError:
    try:
        # Priority 2: PySide6 (default for rheojax)
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6 import __version__ as _qt_version
        from PySide6.QtCore import (
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
        from PySide6.QtGui import (
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
        from PySide6.QtWidgets import (
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

        QT_BINDING = "PySide6"
        QT_VERSION = str(_qt_version)

    except ImportError:
        try:
            # Priority 3: PyQt6 (fallback)
            from PyQt6 import QtCore, QtGui, QtWidgets
            from PyQt6.QtCore import PYQT_VERSION_STR as _qt_version
            from PyQt6.QtCore import (
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
            )
            from PyQt6.QtCore import pyqtSignal as Signal
            from PyQt6.QtCore import pyqtSlot as Slot
            from PyQt6.QtGui import (
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
            from PyQt6.QtWidgets import (
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

            QT_BINDING = "PyQt6"
            QT_VERSION = str(_qt_version)

        except ImportError as e:
            raise ImportError(
                "No Qt binding found. Install one of:\n"
                "  - pip install qtpy PySide6  (recommended)\n"
                "  - pip install PySide6\n"
                "  - pip install PyQt6"
            ) from e


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
