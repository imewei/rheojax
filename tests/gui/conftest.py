"""Fixtures and configuration for GUI tests.

Provides shared fixtures for GUI testing, including:
- QApplication lifecycle management
- Mock services and state
- Test data for GUI components
- Environment detection for display availability
- Subprocess-based crash detection utilities
"""

import os
import subprocess
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Check if PySide6 is available and display is present
try:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Check for display (relevant on CI systems)
HAS_DISPLAY = (
    os.getenv("DISPLAY") is not None or os.getenv("QT_QPA_PLATFORM") is not None
)


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


# =============================================================================
# Subprocess-based Crash Detection Utilities
# =============================================================================


@dataclass
class SubprocessResult:
    """Result from subprocess execution for crash detection.

    Attributes
    ----------
    return_code : int
        Process return code. 0=success, <0=signal, >0=exception
    stdout : str
        Standard output
    stderr : str
        Standard error
    timed_out : bool
        Whether the process timed out
    crashed : bool
        Whether the process crashed (signal or non-zero return)
    signal_name : str | None
        Name of signal that killed process, if any
    """

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    crashed: bool
    signal_name: str | None = None


def _get_signal_name(signum: int) -> str:
    """Get signal name from negative return code.

    Parameters
    ----------
    signum : int
        Negative signal number (e.g., -11 for SIGSEGV)

    Returns
    -------
    str
        Signal name or "UNKNOWN"
    """
    import signal

    abs_sig = abs(signum)
    signal_names = {
        signal.SIGABRT: "SIGABRT",
        signal.SIGBUS: "SIGBUS",
        signal.SIGFPE: "SIGFPE",
        signal.SIGILL: "SIGILL",
        signal.SIGSEGV: "SIGSEGV",
        signal.SIGTERM: "SIGTERM",
        signal.SIGKILL: "SIGKILL",
    }
    return signal_names.get(abs_sig, f"SIG{abs_sig}")


def run_gui_code_subprocess(
    code: str,
    timeout: float = 10.0,
    env_override: dict | None = None,
) -> SubprocessResult:
    """Run GUI code in a subprocess to detect crashes.

    This function runs Python code in an isolated subprocess with
    offscreen Qt rendering, allowing crash detection without
    affecting the main test process.

    Parameters
    ----------
    code : str
        Python code to execute
    timeout : float, default=10.0
        Maximum execution time in seconds
    env_override : dict, optional
        Additional environment variables

    Returns
    -------
    SubprocessResult
        Result containing return code, output, and crash status

    Examples
    --------
    >>> result = run_gui_code_subprocess('''
    ... from PySide6.QtWidgets import QApplication, QLabel
    ... app = QApplication([])
    ... label = QLabel("Test")
    ... label.show()
    ... print("SUCCESS")
    ... ''')
    >>> assert not result.crashed
    >>> assert "SUCCESS" in result.stdout
    """
    # Prepare environment with offscreen rendering
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_LOGGING_RULES"] = "*.debug=false"  # Reduce Qt noise
    if env_override:
        env.update(env_override)

    timed_out = False
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent),  # Project root
        )
        return_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        return_code = -9  # SIGKILL equivalent
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""

    # Determine if it crashed
    crashed = return_code != 0
    signal_name = None
    if return_code < 0:
        signal_name = _get_signal_name(return_code)

    return SubprocessResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        crashed=crashed,
        signal_name=signal_name,
    )


@pytest.fixture
def subprocess_runner():
    """Provide subprocess runner for crash detection tests.

    Returns
    -------
    Callable
        Function to run GUI code in subprocess

    Examples
    --------
    >>> def test_widget_no_crash(subprocess_runner):
    ...     result = subprocess_runner('''
    ...         from PySide6.QtWidgets import QApplication, QLabel
    ...         app = QApplication([])
    ...         label = QLabel("Test")
    ...         print("OK")
    ...     ''')
    ...     assert not result.crashed
    """
    return run_gui_code_subprocess


def is_ascii_safe(text: str) -> bool:
    """Check if text contains only ASCII characters.

    Parameters
    ----------
    text : str
        Text to check

    Returns
    -------
    bool
        True if all characters are ASCII (< 128)
    """
    return all(ord(c) < 128 for c in text)


def contains_emoji(text: str) -> bool:
    """Check if text contains emoji characters.

    Parameters
    ----------
    text : str
        Text to check

    Returns
    -------
    bool
        True if text contains emoji codepoints

    Notes
    -----
    Checks common emoji ranges:
    - U+1F300 to U+1FAFF (Miscellaneous Symbols and Pictographs, Emoticons, etc.)
    - U+2600 to U+26FF (Miscellaneous Symbols)
    - U+2700 to U+27BF (Dingbats)
    - U+FE00 to U+FE0F (Variation Selectors)
    - U+1F000 to U+1F02F (Mahjong/Domino tiles)
    """
    for char in text:
        code = ord(char)
        if (
            0x1F300 <= code <= 0x1FAFF  # Main emoji block
            or 0x2600 <= code <= 0x26FF  # Misc symbols
            or 0x2700 <= code <= 0x27BF  # Dingbats
            or 0xFE00 <= code <= 0xFE0F  # Variation selectors
            or 0x1F000 <= code <= 0x1F02F  # Mahjong/Domino
            or 0x231A <= code <= 0x23FF  # Misc technical
        ):
            return True
    return False


@pytest.fixture
def ascii_checker():
    """Provide ASCII safety checker function.

    Returns
    -------
    Callable
        Function to check if text is ASCII-safe
    """
    return is_ascii_safe


@pytest.fixture
def emoji_checker():
    """Provide emoji detection function.

    Returns
    -------
    Callable
        Function to check if text contains emoji
    """
    return contains_emoji


@pytest.fixture(autouse=True)
def reset_state_store():
    """Reset StateStore singleton between tests.

    This fixture ensures each test starts with a clean StateStore instance,
    preventing state leakage between tests that could cause flaky failures.

    The fixture:
    1. Resets the singleton before each test
    2. Yields control to the test
    3. Resets again after the test completes

    Notes
    -----
    This is an autouse fixture, meaning it runs automatically for every test
    in the gui test suite without needing to be explicitly requested.
    """
    # Reset before test
    try:
        from rheojax.gui.state.store import StateStore

        StateStore._instance = None
    except ImportError:
        pass  # StateStore not available, skip reset

    yield

    # Reset after test to clean up
    try:
        from rheojax.gui.state.store import StateStore

        StateStore._instance = None
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_plot_metrics():
    """Reset PlotMetrics between tests.

    This fixture ensures plot metrics don't accumulate across tests,
    providing clean performance data for each test run.
    """
    try:
        from rheojax.gui.widgets.base_arviz_widget import PlotMetrics

        PlotMetrics.reset()
    except ImportError:
        pass

    yield

    try:
        from rheojax.gui.widgets.base_arviz_widget import PlotMetrics

        PlotMetrics.reset()
    except ImportError:
        pass
