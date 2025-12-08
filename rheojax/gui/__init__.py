"""
RheoJAX GUI Package
==================

Qt-based graphical user interface for RheoJAX rheological analysis.

Features:
    - Interactive data loading and visualization
    - Model fitting with real-time parameter updates
    - Bayesian inference with ArviZ diagnostics
    - Transform pipelines (mastercurve, FFT, SRFS)
    - Multi-view plotting with publication-ready exports
    - GPU acceleration status monitoring

Architecture:
    - Redux-inspired state management with Qt signals
    - Service layer for RheoJAX API integration
    - Background workers for long-running computations
    - Page-based navigation with Material Design widgets

Requirements:
    - PySide6 >= 6.7.0
    - matplotlib >= 3.8.0
    - Additional dependencies via `pip install rheojax[gui]`
"""

__version__ = "0.6.0"

__all__ = [
    "__version__",
    "main",
]


def main() -> int:
    """Launch the RheoJAX GUI application.

    This is a convenience wrapper that imports and calls the actual main function
    from rheojax.gui.main. It is the entry point for the 'rheojax-gui' console script.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)

    Raises
    ------
    SystemExit
        If critical dependencies are missing or initialization fails

    Example
    -------
    >>> from rheojax.gui import main
    >>> main()  # doctest: +SKIP

    Notes
    -----
    This function performs the following initialization steps:
        1. Parse command line arguments (--project, --import, --verbose)
        2. Check for required dependencies (PySide6, matplotlib, JAX, etc.)
        3. Configure logging based on verbosity
        4. Initialize Qt application with proper settings
        5. Create and show the main window
        6. Handle startup file loading if specified
        7. Run the Qt event loop
    """
    from rheojax.gui.main import main as gui_main

    return gui_main()


if __name__ == "__main__":
    import sys

    sys.exit(main())
