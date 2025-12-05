"""Fixtures and configuration for GUI tests.

Provides shared fixtures for GUI testing, including:
- QApplication lifecycle management
- Mock services and state
- Test data for GUI components
- Environment detection for display availability
"""

import os
from pathlib import Path
from typing import Generator

import pytest

# Check if PySide6 is available and display is present
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Check for display (relevant on CI systems)
HAS_DISPLAY = os.getenv("DISPLAY") is not None or os.getenv("QT_QPA_PLATFORM") is not None


@pytest.fixture(scope="session")
def gui_config() -> dict:
    """Provide GUI configuration for tests.

    Returns
    -------
    dict
        Configuration with display availability and test settings.
    """
    return {
        "has_pyside6": HAS_PYSIDE6,
        "has_display": HAS_DISPLAY,
        "headless_mode": not HAS_DISPLAY,
        "test_theme": "light",
    }


@pytest.fixture(scope="session")
def qapp() -> Generator:
    """Create and manage QApplication lifecycle for the test session.

    Only created if PySide6 is available. Uses offscreen surface format
    to avoid display requirements.

    Yields
    ------
    QApplication or None
        Shared QApplication instance for all tests in session.

    Notes
    -----
    - QApplication must be created only once per process
    - Fixture uses session scope to maintain single instance
    - Renders to offscreen buffer if no display is available
    - Cleaned up automatically after session
    """
    if not HAS_PYSIDE6:
        pytest.skip("PySide6 not installed")

    # Set offscreen rendering platform if no display
    if not HAS_DISPLAY:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    yield app

    # No explicit cleanup needed - Qt handles it


@pytest.fixture
def app_state_instance() -> dict:
    """Provide sample AppState data for testing without Qt.

    Returns
    -------
    dict
        Sample application state matching AppState structure.

    Notes
    -----
    This fixture provides test data without importing Qt components,
    useful for testing state management and serialization.
    """
    return {
        "project_path": None,
        "project_name": "Test Project",
        "is_modified": False,
        "datasets": {},
        "active_dataset_id": None,
        "active_model_name": None,
        "model_params": {},
        "fit_results": {},
        "bayesian_results": {},
        "current_tab": "home",
        "theme": "light",
        "auto_save_enabled": True,
        "current_seed": 42,
    }


@pytest.fixture
def service_config() -> dict:
    """Provide configuration for service testing.

    Returns
    -------
    dict
        Service configuration including paths and settings.
    """
    return {
        "data_dir": Path(__file__).parent / "data",
        "test_csv": Path(__file__).parent / "data" / "test_data.csv",
        "test_excel": Path(__file__).parent / "data" / "test_data.xlsx",
        "cache_enabled": False,
        "verbose": True,
    }


@pytest.fixture
def stylesheet_sample() -> str:
    """Provide sample QSS stylesheet for testing.

    Returns
    -------
    str
        Minimal but valid QSS stylesheet.
    """
    return """
    QMainWindow {
        background-color: #ffffff;
        color: #000000;
    }
    QLabel {
        color: #333333;
    }
    QPushButton {
        background-color: #0078d4;
        color: white;
        border-radius: 3px;
        padding: 5px;
    }
    """
