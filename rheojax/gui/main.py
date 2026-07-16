"""
RheoJAX GUI Application Entry Point.

This module provides the main entry point for the RheoJAX graphical user interface.
It handles application initialization, argument parsing, dependency checking, and the
main event loop.

Usage
-----
    rheojax-gui                    # Launch GUI
    rheojax-gui --project FILE     # Open project
    rheojax-gui --import FILE --protocol PROTOCOL   # Import data file on startup
    rheojax-gui --maximized        # Start window maximized
    rheojax-gui --verbose          # Enable verbose logging
    rheojax-gui --help             # Show help

Example
-------
    >>> from rheojax.gui.main import main
    >>> main()  # doctest: +SKIP
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress matplotlib warnings that are harmless in GUI context
# - Glyph warnings: ArviZ uses tab characters in labels that some fonts don't support
# - Layout warnings: Complex ArviZ plots may not fit tight/constrained layouts perfectly
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", message=".*layout not applied.*")
warnings.filterwarnings("ignore", message=".*constrained_layout.*collapsed.*")

# Suppress multiprocessing resource_tracker warnings at exit.
# On macOS + Python 3.13, PySide6's Cocoa platform plugin can create a
# C-level POSIX semaphore that Python's resource_tracker reports as leaked.
# The tracker cleans it up automatically — the warning is cosmetic.
# We use PYTHONWARNINGS env var because the resource_tracker runs in a child
# process where warnings.filterwarnings() from the main process has no effect.
warnings.filterwarnings("ignore", message="resource_tracker:.*leaked.*")
warnings.filterwarnings("ignore", message="resource_tracker:.*semaphore.*")
_pw = os.environ.get("PYTHONWARNINGS", "")
_rt_filter = "ignore::UserWarning:multiprocessing.resource_tracker"
if _rt_filter not in _pw:
    os.environ["PYTHONWARNINGS"] = f"{_rt_filter},{_pw}" if _pw else _rt_filter

# Imports after warnings configuration (intentional)
from rheojax import __version__  # noqa: E402
from rheojax.gui.utils.logging import install_gui_log_handler  # noqa: E402
from rheojax.logging import configure_logging, get_logger, is_configured  # noqa: E402

# Module-level logger
logger = get_logger(__name__)


def _create_workspace_window() -> "WorkspaceWindow":  # noqa: F821
    """Construct the workspace shell window (now the default entry point).

    Kept as a standalone import point so it can be unit-tested without
    entering the Qt event loop.
    """
    from rheojax.gui.foundation.state import AppState
    from rheojax.gui.workspace.window import WorkspaceWindow

    return WorkspaceWindow(AppState())


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging.

    Parameters
    ----------
    verbose : bool, optional
        Enable verbose (DEBUG) logging, by default False
    """
    # Use RheoJAX logging system if not already configured
    if not is_configured():
        level = "DEBUG" if verbose else "INFO"
        configure_logging(level=level, format="standard", colorize=True)
        logger.debug("Logging configured", log_level=level)


def check_dependencies() -> tuple[bool, list[str]]:
    """Verify all GUI dependencies are available.

    Returns
    -------
    tuple[bool, list[str]]
        (success, missing_dependencies)
    """

    missing: list[str] = []

    try:
        from rheojax.gui.compat import QT_BINDING  # noqa: F401
    except ImportError:
        missing.append("PySide6>=6.6.0 or PyQt6>=6.6.0")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib>=3.8.0")

    try:
        from rheojax.core.jax_config import safe_import_jax

        safe_import_jax()
    except ImportError as exc:  # pragma: no cover - environment dependent
        missing.append(f"rheojax-core ({exc})")

    try:
        import numpyro  # noqa: F401
    except ImportError:
        missing.append("numpyro>=0.19.0")

    try:
        import arviz  # noqa: F401
    except ImportError:
        missing.append("arviz>=1.2.0")

    return not missing, missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="rheojax-gui",
        description="RheoJAX - JAX-Accelerated Rheological Analysis GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rheojax-gui                              # Launch GUI
  rheojax-gui --project analysis.rheojax     # Open project file
  rheojax-gui --import data.xlsx --protocol relaxation  # Import data on startup
  rheojax-gui --maximized                 # Start maximized (per-window manager)
  rheojax-gui --verbose                   # Enable verbose logging

For more information, visit: https://github.com/imewei/rheojax
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--project",
        "-p",
        type=Path,
        metavar="FILE",
        help="Project file to open on startup (.rheojax)",
    )
    group.add_argument(
        "--import",
        "-i",
        dest="import_file",
        type=Path,
        metavar="FILE",
        help="Data file to import on startup (requires --protocol)",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        metavar="PROTOCOL",
        help="Protocol for --import (required together): flow_curve, creep, relaxation, "
        "startup, oscillation, laos",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    parser.add_argument(
        "--maximized",
        "-M",
        action="store_true",
        help="Start the main window maximized",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    if args.import_file and not args.protocol:
        parser.error("--protocol is required when --import is given")
    if args.protocol and not args.import_file:
        parser.error("--protocol is only valid together with --import")

    return args


def main(argv: list[str] | None = None) -> int:
    """Main entry point for RheoJAX GUI.

    Parameters
    ----------
    argv : list[str], optional
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)

    Raises
    ------
    SystemExit
        If critical dependencies are missing or initialization fails
    """
    # Parse arguments
    args = parse_args(argv)

    # Setup logging
    setup_logging(args.verbose)

    def _log_uncaught_exception(exc_type, exc_value, exc_tb) -> None:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _log_uncaught_exception

    logger.info("RheoJAX GUI starting", version=__version__)

    # Check dependencies
    logger.debug("Checking GUI dependencies")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        logger.error("Missing required dependencies", missing=missing)
        print("\nERROR: Missing required dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nInstall all GUI dependencies with:", file=sys.stderr)
        print("  pip install rheojax[gui]", file=sys.stderr)
        print("\nOr install the full development environment:", file=sys.stderr)
        print("  pip install rheojax[dev,gui]", file=sys.stderr)
        return 1

    logger.debug("All dependencies verified")

    # Validate file arguments
    if args.project and not args.project.exists():
        logger.error("Project file not found", path=str(args.project))
        print(
            f"ERROR: Project file not found: {args.project}",
            file=sys.stderr,
        )
        return 1

    if args.import_file and not args.import_file.exists():
        logger.error("Import file not found", path=str(args.import_file))
        print(
            f"ERROR: Import file not found: {args.import_file}",
            file=sys.stderr,
        )
        return 1

    # Import JAX early (configures float64) and Qt after dependency check
    logger.debug("Importing JAX runtime")
    try:
        from rheojax.core.jax_config import safe_import_jax

        safe_import_jax()
        logger.debug("JAX runtime initialized")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("Failed to import JAX", error=str(exc), exc_info=True)
        print(f"ERROR: Failed to import JAX: {exc}", file=sys.stderr)
        return 1

    logger.debug("Initializing Qt application")
    try:
        from rheojax.gui.compat import QApplication, QFont, QIcon, Qt, QTimer
    except ImportError as exc:
        logger.error("Failed to import Qt bindings", error=str(exc), exc_info=True)
        print(f"ERROR: Failed to import Qt bindings: {exc}", file=sys.stderr)
        return 1

    # Configure Qt application attributes before instantiation
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    def _adaptive_font_size(app: QApplication) -> float:
        """Compute a font size that scales with screen DPI."""

        screen = app.primaryScreen()
        if not screen:
            return 12.0

        # 96 DPI is the typical desktop baseline; scale relative to it.
        base_size = 12.0
        dpi_scale = max(screen.logicalDotsPerInch() / 96.0, 1.0)
        # Cap scaling to avoid excessively large fonts on very high DPI setups.
        return min(base_size * dpi_scale, 16.0)

    def _show_main_window(window: "WorkspaceWindow", maximize: bool) -> None:  # noqa: F821
        """Show the main window with cross-platform maximize handling."""

        if maximize:
            window.setWindowState(window.windowState() | Qt.WindowMaximized)
            window.show()
            window.showMaximized()
            window.raise_()
            window.activateWindow()

            def _ensure_maximized_fullscreen_fallback() -> None:
                if window.isMaximized():
                    return
                # Last-resort nudge for stubborn compositors: fullscreen then back to maximized
                window.showFullScreen()
                QTimer.singleShot(0, window.showMaximized)

            # Re-apply after event loop starts; Wayland often needs a later nudge
            QTimer.singleShot(0, window.showMaximized)
            QTimer.singleShot(75, _ensure_maximized_fullscreen_fallback)
            return

        window.show()

    # Create Qt application
    if argv is None:
        app = QApplication(sys.argv)
    else:
        app = QApplication(argv or ["rheojax-gui"])

    app.setApplicationName("RheoJAX")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("RheoJAX")
    app.setOrganizationDomain("github.com/imewei/rheojax")

    # Apply stylesheet from resources
    from rheojax.gui.resources import get_icon_path, load_stylesheet

    app.setStyle("Fusion")

    # Map missing font aliases to installed families to avoid fallback scans
    QFont.insertSubstitution("Sans-serif", "DejaVu Sans")
    QFont.insertSubstitution("Sans Serif", "DejaVu Sans")
    QFont.insertSubstitution("sans-serif", "DejaVu Sans")

    # Apply adaptive base font before the stylesheet so the theme inherits it
    # Bump all fonts by +2pt for better readability.
    base_font_size = _adaptive_font_size(app) + 2.0
    base_font = QFont()
    base_font.setFamilies(
        [
            "Inter",
            "Segoe UI",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ]
    )
    base_font.setPointSizeF(base_font_size)
    app.setFont(base_font)

    # Append a global rule to bump widget font sizes while keeping the theme
    stylesheet = load_stylesheet("light")
    stylesheet += f"\n* {{ font-size: {base_font_size:.1f}pt; }}\n"
    app.setStyleSheet(stylesheet)

    logger.debug("Qt application initialized", font_size=base_font_size)

    logger.debug("Launching workspace shell (now the default)")
    try:
        workspace_window = _create_workspace_window()
    except Exception as e:
        logger.error(
            "Failed to create workspace window", error=str(e), exc_info=True
        )
        print(f"ERROR: Failed to create workspace window: {e}", file=sys.stderr)
        return 1

    gui_handler = install_gui_log_handler(
        workspace_window.log_dock.append_record,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    workspace_window.destroyed.connect(
        lambda *_: logging.getLogger().removeHandler(gui_handler)
    )

    app.setWindowIcon(QIcon(str(get_icon_path("rheojax"))))
    _show_main_window(workspace_window, args.maximized)
    logger.info("RheoJAX GUI workspace shell ready", version=__version__)

    if args.project:
        from rheojax.gui.foundation.project_codec import load_project_v2

        try:
            loaded_state = load_project_v2(args.project)
        except (ValueError, FileNotFoundError) as e:
            logger.error(
                "Failed to load project", path=str(args.project), error=str(e)
            )
            print(f"ERROR: Failed to load project: {e}", file=sys.stderr)
            return 1
        workspace_window._rebuild(loaded_state)
        workspace_window._state.project.path = str(args.project)

    if args.import_file:
        from rheojax.gui.foundation.import_service import import_dataset

        try:
            ref, data = import_dataset(args.import_file, args.protocol)
        except (ValueError, FileNotFoundError) as e:
            logger.error(
                "Failed to import data", path=str(args.import_file), error=str(e)
            )
            print(f"ERROR: Failed to import data: {e}", file=sys.stderr)
            return 1
        workspace_window._commit_dataset(ref, data, overwrite=False)

    exit_code = app.exec()
    logger.info("Application exiting", exit_code=exit_code)
    try:
        from multiprocessing import resource_tracker

        tracker = resource_tracker._resource_tracker
        if tracker._pid is not None:
            tracker._stop()
    except Exception:
        pass
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
