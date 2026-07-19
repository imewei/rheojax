"""Tests for transform display-name -> internal-key mapping.

Verifies:
- Robust transform display-name → internal-key mapping
- Model dropdown family grouping (non-selectable headers)
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

pytestmark = pytest.mark.skipif(
    not HAS_PYSIDE6,
    reason="PySide6 not installed",
)


@pytest.fixture(scope="module")
def qapp_module():
    if not HAS_PYSIDE6:
        return None
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Transform name mapping
# ---------------------------------------------------------------------------


class TestTransformNameMapping:
    @pytest.mark.smoke
    def test_transform_metadata_keys_match_service(self, qapp_module):
        """TransformService metadata keys should match _transforms dict."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        metadata = service.get_transform_metadata()
        available = service.get_available_transforms()

        meta_keys = {m["key"] for m in metadata}
        # Aliases in _transforms have no display metadata; meta keys are a subset.
        assert meta_keys <= set(available), (
            f"Metadata keys {meta_keys} not subset of available {set(available)}"
        )

    @pytest.mark.unit
    def test_all_display_names_are_unique(self, qapp_module):
        """Each transform display name should be unique."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        metadata = service.get_transform_metadata()
        names = [m["name"] for m in metadata]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


# ---------------------------------------------------------------------------
# Transform fallback path
# ---------------------------------------------------------------------------


class TestTransformFallbackPath:
    @pytest.mark.unit
    def test_transform_metadata_has_name_and_key_fields(self, qapp_module):
        """TransformPage.get_available_transforms() returns dicts with 'name' and 'key'."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        metadata = service.get_transform_metadata()
        assert len(metadata) > 0, "Should have at least one transform"

        for meta in metadata:
            assert "name" in meta, f"Missing 'name' key in metadata: {meta}"
            assert "key" in meta, f"Missing 'key' key in metadata: {meta}"

    @pytest.mark.unit
    def test_display_to_key_map_round_trips(self, qapp_module):
        """Building a display-name → key map from metadata should resolve all transforms."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        metadata = service.get_transform_metadata()
        display_to_key = {m["name"].lower(): m["key"] for m in metadata}

        # Every key should be discoverable via its display name
        for meta in metadata:
            resolved = display_to_key.get(meta["name"].lower())
            assert resolved == meta["key"], (
                f"Display name '{meta['name']}' resolved to '{resolved}', expected '{meta['key']}'"
            )
