"""Visual regression scaffold aligned to GUI SPEC.md golden images."""

import pytest


@pytest.mark.gui
def test_home_page_visual_golden():
    """Smoke check golden files exist (replace with real renders when available)."""
    import pathlib

    golden_dir = pathlib.Path(__file__).parent / "golden_images"
    assert (golden_dir / "home_page_light.png").exists()
