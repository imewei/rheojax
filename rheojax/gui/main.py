"""
RheoJAX GUI Application Entry Point.

This module provides the main entry point for the RheoJAX graphical user interface.
It handles application initialization, argument parsing, dependency checking, and the
main event loop.

Usage
-----
    rheojax-gui                    # Launch GUI
    rheojax-gui --project FILE     # Open project
    rheojax-gui --import FILE      # Import data file on startup
    rheojax-gui --verbose          # Enable verbose logging
    rheojax-gui --help             # Show help

Example
-------
    >>> from rheojax.gui.main import main
    >>> main()  # doctest: +SKIP
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging.

    Parameters
    ----------
    verbose : bool, optional
        Enable verbose (DEBUG) logging, by default False
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def check_dependencies() -> tuple[bool, list[str]]:
    """Verify all required dependencies are available.

    Returns
    -------
    tuple[bool, list[str]]
        (success, missing_dependencies)
        success is True if all dependencies are available
        missing_dependencies is a list of missing package names
    """
    missing = []

    # Check PySide6
    try:
        import PySide6  # noqa: F401
    except ImportError:
        missing.append("PySide6>=6.7.0")

    # Check matplotlib
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib>=3.8.0")

    # Check rheojax core (includes JAX)
    try:
        from rheojax.core.jax_config import safe_import_jax

        safe_import_jax()
    except ImportError as e:
        missing.append(f"rheojax-core ({e})")

    # Check NumPyro for Bayesian
    try:
        import numpyro  # noqa: F401
    except ImportError:
        missing.append("numpyro>=0.19.0")

    # Check ArviZ for diagnostics
    try:
        import arviz  # noqa: F401
    except ImportError:
        missing.append("arviz>=0.22.0")

    return len(missing) == 0, missing


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
  rheojax-gui --project analysis.rheo     # Open project file
  rheojax-gui --import data.xlsx          # Import data on startup
  rheojax-gui --verbose                    # Enable verbose logging

For more information, visit: https://github.com/imewei/rheojax
        """,
    )

    parser.add_argument(
        "--project",
        "-p",
        type=Path,
        metavar="FILE",
        help="Project file to open on startup (.rheo)",
    )

    parser.add_argument(
        "--import",
        "-i",
        dest="import_file",
        type=Path,
        metavar="FILE",
        help="Data file to import on startup (TRIOS, Excel, CSV, etc.)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.5.0",
    )

    return parser.parse_args(argv)


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
    logger = logging.getLogger(__name__)
    logger.info("Starting RheoJAX GUI v0.5.0...")

    # Check dependencies
    logger.debug("Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(
            "\nERROR: Missing required dependencies:",
            file=sys.stderr,
        )
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print(
            "\nInstall all GUI dependencies with:",
            file=sys.stderr,
        )
        print("  pip install rheojax[gui]", file=sys.stderr)
        print(
            "\nOr install the full development environment:",
            file=sys.stderr,
        )
        print("  pip install rheojax[dev,gui]", file=sys.stderr)
        return 1

    logger.debug("All dependencies found")

    # Validate file arguments
    if args.project and not args.project.exists():
        logger.error(f"Project file not found: {args.project}")
        print(
            f"ERROR: Project file not found: {args.project}",
            file=sys.stderr,
        )
        return 1

    if args.import_file and not args.import_file.exists():
        logger.error(f"Import file not found: {args.import_file}")
        print(
            f"ERROR: Import file not found: {args.import_file}",
            file=sys.stderr,
        )
        return 1

    # Import Qt after dependency check
    logger.debug("Initializing Qt application...")
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        logger.error(f"Failed to import PySide6: {e}")
        print(
            f"ERROR: Failed to import PySide6: {e}",
            file=sys.stderr,
        )
        return 1

    # Configure Qt application attributes before instantiation
    # Note: These must be set before QApplication is created
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # Create Qt application
    if argv is None:
        app = QApplication(sys.argv)
    else:
        app = QApplication(argv or ["rheojax-gui"])

    app.setApplicationName("RheoJAX")
    app.setApplicationVersion("0.5.0")
    app.setOrganizationName("RheoJAX")
    app.setOrganizationDomain("github.com/imewei/rheojax")

    # Set application style (Fusion for cross-platform consistency)
    app.setStyle("Fusion")

    logger.debug("Qt application initialized")

    # Import main window after Qt app is created
    try:
        from rheojax.gui.app.main_window import RheoJAXMainWindow
    except ImportError as e:
        logger.error(f"Failed to import main window: {e}")
        print(
            f"ERROR: Failed to import main window: {e}",
            file=sys.stderr,
        )
        return 1

    # Create main window
    try:
        logger.debug("Creating main window...")
        window = RheoJAXMainWindow()
        window.show()
        logger.info("RheoJAX GUI ready")
    except Exception as e:
        logger.exception(f"Failed to create main window: {e}")
        print(
            f"ERROR: Failed to create main window: {e}",
            file=sys.stderr,
        )
        return 1

    # Handle startup arguments
    if args.project:
        logger.info(f"Loading project: {args.project}")
        try:
            # Project loading will be implemented via file dialog in GUI
            # For now, just log the request
            window.log(f"Project file specified: {args.project}")
            window.log("Note: Project loading from command line not yet implemented")
            logger.debug("Project loading action logged")
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            window.log(f"ERROR: Failed to load project: {e}")

    if args.import_file:
        logger.info(f"Importing data file: {args.import_file}")
        try:
            # This will be implemented when the data loading service is ready
            # For now, just log it
            window.log(f"Data import requested: {args.import_file}")
            window.log("Note: Data import from command line not yet implemented")
            logger.debug("Data import action logged")
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            window.log(f"ERROR: Failed to import data: {e}")

    # Run event loop
    logger.debug("Entering Qt event loop...")
    exit_code: int = app.exec()
    logger.info(f"Application exiting with code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
