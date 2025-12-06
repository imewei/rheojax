"""Placeholder visual test to ensure golden assets exist.

Replace with real GUI rendering and image diff once Qt rendering is wired.
"""

import pathlib
import pytest


@pytest.mark.gui
@pytest.mark.skip(reason="Full-page goldens are placeholders until real GUI renders are captured")
def test_fit_page_golden_exists():
    golden_dir = pathlib.Path(__file__).parent / "golden_images"
    assert (golden_dir / "fit_page_with_results.png").exists()


@pytest.mark.gui
@pytest.mark.skip(reason="Full-page goldens are placeholders until real GUI renders are captured")
def test_transform_page_golden_exists():
    golden_dir = pathlib.Path(__file__).parent / "golden_images"
    assert (golden_dir / "transform_page.png").exists()
