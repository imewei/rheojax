"""Helper script to capture GUI golden screenshots locally.

Usage (run locally with a display or offscreen Qt):
    QT_QPA_PLATFORM=offscreen python tests/gui/visual/capture_golden_screens.py

Screenshots are written to `tests/gui/visual/golden_images/` for:
- home (light/dark)
- fit, transform, bayesian, export tabs

Note: This script does not mock data; captures reflect the current UI state.
Populate datasets/models beforehand if you need data-driven renders.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QApplication

from rheojax.gui.app.main_window import RheoJAXMainWindow


def _ensure_offscreen() -> None:
    # Allow running without a visible display when possible
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _capture(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    pixmap.save(str(path))


def main() -> None:
    _ensure_offscreen()
    app = QApplication([])

    window = RheoJAXMainWindow()
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    golden_dir = Path(__file__).parent / "golden_images"

    # Home light
    window._apply_theme("light")
    window.navigate_to("home")
    app.processEvents()
    _capture(window, golden_dir / "home_page_light.png")

    # Home dark
    window._apply_theme("dark")
    window.navigate_to("home")
    app.processEvents()
    _capture(window, golden_dir / "home_page_dark.png")

    # Fit tab
    window._apply_theme("light")
    window.navigate_to("fit")
    app.processEvents()
    _capture(window, golden_dir / "fit_page_with_results.png")

    # Transform tab
    window.navigate_to("transform")
    app.processEvents()
    _capture(window, golden_dir / "transform_page.png")

    # Bayesian tab
    window.navigate_to("bayesian")
    app.processEvents()
    _capture(window, golden_dir / "bayesian_page.png")

    # Export tab
    window.navigate_to("export")
    app.processEvents()
    _capture(window, golden_dir / "export_page.png")

    window.close()
    app.quit()


if __name__ == "__main__":
    main()
