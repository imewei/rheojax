"""Tests for pipeline step config auto-population and sync.

Verifies:
- GUI-016: Auto-populate step configs when steps are added
- GUI-016: Sync fit/transform step configs on selection changes
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


@pytest.fixture(autouse=True)
def reset_store():
    from rheojax.gui.state.store import StateStore

    StateStore.reset()
    yield
    StateStore.reset()


# ---------------------------------------------------------------------------
# Model dropdown grouping
# ---------------------------------------------------------------------------


class TestModelDropdownGrouping:
    @pytest.mark.smoke
    def test_model_combo_has_category_headers(self, qapp_module):
        """Model dropdown should contain non-selectable category headers."""
        from rheojax.gui.pages.fit_page import FitPage

        page = FitPage()
        combo = page._quick_model_combo
        assert combo.count() > 2

        assert combo.itemData(0) is None

        found_header = False
        found_model = False
        for i in range(1, combo.count()):
            data = combo.itemData(i)
            if data is None:
                text = combo.itemText(i)
                assert "──" in text, f"Header should have separators: {text}"
                found_header = True
            else:
                found_model = True

        assert found_header, "Should have at least one category header"
        assert found_model, "Should have at least one selectable model"

    @pytest.mark.unit
    def test_header_items_are_not_selectable(self, qapp_module):
        """Category header items should be disabled (not selectable)."""
        from rheojax.gui.pages.fit_page import FitPage

        page = FitPage()
        combo = page._quick_model_combo
        model = combo.model()
        assert model is not None

        for i in range(1, combo.count()):
            data = combo.itemData(i)
            if data is None and i > 0:
                item = model.item(i)
                assert item is not None
                assert not item.isEnabled(), f"Header at index {i} should be disabled"
                break

    @pytest.mark.unit
    def test_model_selection_skips_headers(self, qapp_module):
        """Selecting a header item (None data) should not trigger model change."""
        from rheojax.gui.pages.fit_page import FitPage

        page = FitPage()
        page._on_quick_model_changed(0)
        assert page._current_model is None or page._current_model == ""

    @pytest.mark.unit
    def test_find_data_still_works_with_grouped_combo(self, qapp_module):
        """findData should still find models despite indented display text."""
        from rheojax.gui.pages.fit_page import FitPage

        page = FitPage()
        combo = page._quick_model_combo
        idx = combo.findData("maxwell")
        assert idx >= 0, "Should find 'maxwell' model by data"


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
        assert meta_keys == set(available), (
            f"Metadata keys {meta_keys} != available {set(available)}"
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
# Pipeline step config auto-population (state-level, no MainWindow)
# ---------------------------------------------------------------------------


class TestPipelineStepAutoPopulate:
    @pytest.mark.smoke
    def test_fit_step_config_can_be_populated_with_model(self, qapp_module):
        """update_step_config should store model name in step config."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id

        step_id = add_pipeline_step("fit", "Fit Model")
        update_step_config(step_id, {"model": "maxwell"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["model"] == "maxwell"

    @pytest.mark.unit
    def test_transform_step_config_can_be_populated_with_name(self, qapp_module):
        """update_step_config should store transform name in step config."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id

        step_id = add_pipeline_step("transform", "Transform")
        update_step_config(step_id, {"name": "fft", "window": "hann"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["name"] == "fft"
        assert step.config["window"] == "hann"

    @pytest.mark.unit
    def test_load_step_config_stores_file_path(self, qapp_module):
        """update_step_config should store file path in load step config."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id

        step_id = add_pipeline_step("load", "Load Data")
        update_step_config(step_id, {"file": "/tmp/test.csv", "test_mode": "oscillation"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["file"] == "/tmp/test.csv"
        assert step.config["test_mode"] == "oscillation"

    @pytest.mark.unit
    def test_export_step_config_defaults(self, qapp_module):
        """update_step_config should store export format/output."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id

        step_id = add_pipeline_step("export", "Export Results")
        update_step_config(step_id, {"format": "directory", "output": "results/"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["format"] == "directory"
        assert step.config["output"] == "results/"

    @pytest.mark.unit
    def test_update_step_config_merges(self, qapp_module):
        """Successive update_step_config calls should merge, not replace."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id

        step_id = add_pipeline_step("fit", "Fit Model")
        update_step_config(step_id, {"model": "maxwell"})
        update_step_config(step_id, {"test_mode": "oscillation"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["model"] == "maxwell"
        assert step.config["test_mode"] == "oscillation"

    @pytest.mark.unit
    def test_update_step_config_invalidates_downstream(self, qapp_module):
        """Updating a step's config should invalidate downstream steps."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            update_step_config,
            update_step_status,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id
        from rheojax.gui.state.store import StepStatus

        step1_id = add_pipeline_step("load", "Load")
        step2_id = add_pipeline_step("fit", "Fit")

        # Mark both as complete
        update_step_status(step1_id, StepStatus.COMPLETE)
        update_step_status(step2_id, StepStatus.COMPLETE)

        # Update load step config → fit step should be reset to PENDING
        update_step_config(step1_id, {"file": "new_data.csv"})

        step2 = get_pipeline_step_by_id(step2_id)
        assert step2 is not None
        assert step2.status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# Step config sync logic (without MainWindow)
# ---------------------------------------------------------------------------


class TestStepConfigSync:
    @pytest.mark.unit
    def test_sync_fit_step_on_model_change(self, qapp_module):
        """Simulates model change → selected fit step config updates."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            select_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import (
            get_pipeline_step_by_id,
            get_selected_pipeline_step,
        )
        from rheojax.gui.state.store import StateStore

        store = StateStore()
        step_id = add_pipeline_step("fit", "Fit Model")
        select_pipeline_step(step_id)

        # Verify step is selected
        selected = get_selected_pipeline_step()
        assert selected is not None
        assert selected.step_type == "fit"

        # Simulate what _sync_fit_step_config does
        store.dispatch("SET_ACTIVE_MODEL", {"model_name": "zener"})
        state = store.get_state()
        if state.active_model_name:
            update_step_config(step_id, {"model": state.active_model_name})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config.get("model") == "zener"

    @pytest.mark.unit
    def test_sync_does_not_affect_non_fit_steps(self, qapp_module):
        """Sync should not modify load step config when model changes."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            select_pipeline_step,
        )
        from rheojax.gui.state.selectors import (
            get_pipeline_step_by_id,
            get_selected_pipeline_step,
        )

        step_id = add_pipeline_step("load", "Load Data")
        select_pipeline_step(step_id)

        selected = get_selected_pipeline_step()
        assert selected is not None
        assert selected.step_type == "load"

        # _sync_fit_step_config would check step_type != "fit" → early return
        # Verify step config remains empty
        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert "model" not in step.config

    @pytest.mark.unit
    def test_sync_transform_step_config(self, qapp_module):
        """Transform step config updates when transform is selected."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            select_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import (
            get_pipeline_step_by_id,
            get_selected_pipeline_step,
        )

        step_id = add_pipeline_step("transform", "Transform")
        select_pipeline_step(step_id)

        selected = get_selected_pipeline_step()
        assert selected is not None
        assert selected.step_type == "transform"

        # Simulate what _sync_transform_step_config does
        update_step_config(step_id, {"name": "derivative"})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config.get("name") == "derivative"

    @pytest.mark.unit
    def test_bayesian_step_receives_model_from_sync(self, qapp_module):
        """Bayesian steps should also receive model name from sync."""
        from rheojax.gui.state.actions import (
            add_pipeline_step,
            select_pipeline_step,
            update_step_config,
        )
        from rheojax.gui.state.selectors import get_pipeline_step_by_id
        from rheojax.gui.state.store import StateStore

        store = StateStore()
        step_id = add_pipeline_step("bayesian", "Bayesian Inference")
        select_pipeline_step(step_id)

        store.dispatch("SET_ACTIVE_MODEL", {"model_name": "springpot"})
        state = store.get_state()

        # Simulate _sync_fit_step_config logic for bayesian
        update_step_config(step_id, {"model": state.active_model_name})

        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config.get("model") == "springpot"


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


# ---------------------------------------------------------------------------
# Defensive exception handling
# ---------------------------------------------------------------------------


class TestAutoPopulateDefensiveHandling:
    @pytest.mark.unit
    def test_update_step_config_with_nonexistent_id_does_not_crash(self, qapp_module):
        """update_step_config with a bad step_id should not raise."""
        from rheojax.gui.state.actions import update_step_config

        # Should not raise — the reducer simply won't find the step
        update_step_config("nonexistent-uuid", {"model": "maxwell"})

    @pytest.mark.unit
    def test_auto_populate_pattern_handles_missing_step(self, qapp_module):
        """Simulates the try/except guard in _auto_populate_step_config."""
        from rheojax.gui.state import selectors

        # get_pipeline_step_by_id returns None for unknown IDs
        step = selectors.get_pipeline_step_by_id("does-not-exist")
        assert step is None
