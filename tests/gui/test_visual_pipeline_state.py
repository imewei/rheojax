"""Tests for visual pipeline state management.

Covers action creators, reducers, and selectors for the VisualPipelineState
slice of the AppState singleton.  All tests are self-contained and reset
the StateStore singleton before and after each test via the autouse fixture.
"""

import pytest

from rheojax.gui.state.actions import (
    add_pipeline_step,
    cache_step_result,
    clear_pipeline,
    load_pipeline,
    remove_pipeline_step,
    reorder_pipeline_step,
    select_pipeline_step,
    set_pipeline_name,
    set_pipeline_running,
    update_step_config,
    update_step_status,
)
from rheojax.gui.state.selectors import (
    get_pipeline_name,
    get_pipeline_step_by_id,
    get_pipeline_step_result,
    get_selected_pipeline_step,
    get_visual_pipeline,
    get_visual_pipeline_progress,
    get_visual_pipeline_steps,
    is_pipeline_running,
)
from rheojax.gui.state.store import (
    PipelineStepConfig,
    StateStore,
    StepStatus,
    VisualPipelineState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_load_step(name: str = "Load Data") -> str:
    """Add a load step and return its ID."""
    return add_pipeline_step("load", name, config={"file": "/tmp/data.csv"})


def _add_fit_step(name: str = "Fit Maxwell") -> str:
    """Add a fit step and return its ID."""
    return add_pipeline_step("fit", name, config={"model": "Maxwell"})


def _add_transform_step(name: str = "FFT") -> str:
    """Add a transform step and return its ID."""
    return add_pipeline_step("transform", name, config={"name": "fft"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the StateStore singleton before and after every test."""
    StateStore.reset()
    yield
    StateStore.reset()


# ---------------------------------------------------------------------------
# TestAddPipelineStep
# ---------------------------------------------------------------------------


class TestAddPipelineStep:
    @pytest.mark.smoke
    def test_add_step_returns_id(self):
        """add_pipeline_step must return a non-empty string UUID."""
        step_id = _add_load_step()
        assert isinstance(step_id, str)
        assert len(step_id) > 0

    @pytest.mark.smoke
    def test_add_step_appears_in_selectors(self):
        """Step added via action must be retrievable via selectors."""
        step_id = _add_load_step("My Load")
        steps = get_visual_pipeline_steps()
        assert len(steps) == 1
        assert steps[0].id == step_id
        assert steps[0].name == "My Load"

    @pytest.mark.smoke
    def test_add_multiple_steps_ordered(self):
        """Multiple steps must appear in insertion order with correct positions."""
        id0 = _add_load_step("Load")
        id1 = _add_fit_step("Fit")
        id2 = _add_transform_step("Transform")
        steps = get_visual_pipeline_steps()
        assert len(steps) == 3
        assert steps[0].id == id0
        assert steps[1].id == id1
        assert steps[2].id == id2
        assert steps[0].position == 0
        assert steps[1].position == 1
        assert steps[2].position == 2

    def test_add_step_with_config(self):
        """Config dict passed to add_pipeline_step must be stored on the step."""
        step_id = add_pipeline_step(
            "fit", "Fit Giesekus", config={"model": "Giesekus", "max_iter": 2000}
        )
        step = get_pipeline_step_by_id(step_id)
        assert step is not None
        assert step.config["model"] == "Giesekus"
        assert step.config["max_iter"] == 2000

    def test_add_step_default_status_is_pending(self):
        """Newly added steps must start with PENDING status."""
        step_id = _add_load_step()
        step = get_pipeline_step_by_id(step_id)
        assert step.status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# TestRemovePipelineStep
# ---------------------------------------------------------------------------


class TestRemovePipelineStep:
    @pytest.mark.smoke
    def test_remove_step(self):
        """Removing a step must eliminate it from the step list."""
        step_id = _add_load_step()
        assert len(get_visual_pipeline_steps()) == 1
        remove_pipeline_step(step_id)
        assert len(get_visual_pipeline_steps()) == 0

    def test_remove_nonexistent_step_no_error(self):
        """Removing a step with an unknown ID must not raise an exception."""
        _add_load_step()
        remove_pipeline_step("nonexistent-uuid-does-not-exist")
        assert len(get_visual_pipeline_steps()) == 1

    def test_remove_selected_step_clears_selection(self):
        """Removing the currently selected step must clear selected_step_id."""
        step_id = _add_load_step()
        select_pipeline_step(step_id)
        assert get_selected_pipeline_step() is not None
        remove_pipeline_step(step_id)
        assert get_selected_pipeline_step() is None

    def test_positions_reindexed_after_remove(self):
        """After removing a middle step positions must be consecutive from 0."""
        _add_load_step("Load")
        mid_id = _add_fit_step("Fit")
        _add_transform_step("Export")
        remove_pipeline_step(mid_id)
        steps = get_visual_pipeline_steps()
        assert len(steps) == 2
        for idx, step in enumerate(steps):
            assert step.position == idx

    def test_remove_step_clears_cached_result(self):
        """Removing a step must also remove its cached result."""
        step_id = _add_load_step()
        cache_step_result(step_id, {"data": [1, 2, 3]})
        assert get_pipeline_step_result(step_id) is not None
        remove_pipeline_step(step_id)
        assert get_pipeline_step_result(step_id) is None


# ---------------------------------------------------------------------------
# TestReorderPipelineStep
# ---------------------------------------------------------------------------


class TestReorderPipelineStep:
    @pytest.mark.smoke
    def test_reorder_step(self):
        """Moving the last step to position 0 must make it first."""
        id0 = _add_load_step("Load")
        id1 = _add_fit_step("Fit")
        id2 = _add_transform_step("Transform")
        reorder_pipeline_step(id2, 0)
        steps = get_visual_pipeline_steps()
        assert steps[0].id == id2
        assert steps[1].id == id0
        assert steps[2].id == id1

    def test_reorder_clears_results(self):
        """Reordering any step must invalidate all cached step results."""
        id0 = _add_load_step("Load")
        id1 = _add_fit_step("Fit")
        cache_step_result(id0, "data_object")
        cache_step_result(id1, "fit_object")
        reorder_pipeline_step(id1, 0)
        assert get_pipeline_step_result(id0) is None
        assert get_pipeline_step_result(id1) is None

    def test_reorder_positions_updated(self):
        """Positions must reflect the new order after reordering."""
        id0 = _add_load_step("A")
        id1 = _add_fit_step("B")
        id2 = _add_transform_step("C")
        # Move last to first
        reorder_pipeline_step(id2, 0)
        steps = get_visual_pipeline_steps()
        assert steps[0].position == 0
        assert steps[1].position == 1
        assert steps[2].position == 2


# ---------------------------------------------------------------------------
# TestSelectPipelineStep
# ---------------------------------------------------------------------------


class TestSelectPipelineStep:
    @pytest.mark.smoke
    def test_select_step(self):
        """select_pipeline_step must set the selected step."""
        step_id = _add_load_step()
        select_pipeline_step(step_id)
        selected = get_selected_pipeline_step()
        assert selected is not None
        assert selected.id == step_id

    def test_deselect_step(self):
        """Passing None to select_pipeline_step must clear the selection."""
        step_id = _add_load_step()
        select_pipeline_step(step_id)
        assert get_selected_pipeline_step() is not None
        select_pipeline_step(None)
        assert get_selected_pipeline_step() is None

    def test_selected_step_returned_by_selector(self):
        """get_selected_pipeline_step must return the full PipelineStepConfig."""
        step_id = _add_fit_step("Fit Maxwell")
        select_pipeline_step(step_id)
        selected = get_selected_pipeline_step()
        assert isinstance(selected, PipelineStepConfig)
        assert selected.step_type == "fit"
        assert selected.name == "Fit Maxwell"

    def test_select_nonexistent_step_returns_none(self):
        """Selecting an ID that does not exist must return None from the selector."""
        select_pipeline_step("ghost-id")
        # No steps exist, so selected_step_id is set but step not found.
        assert get_selected_pipeline_step() is None


# ---------------------------------------------------------------------------
# TestUpdateStepConfig
# ---------------------------------------------------------------------------


class TestUpdateStepConfig:
    @pytest.mark.smoke
    def test_update_config(self):
        """update_step_config must merge new keys into the existing config."""
        step_id = add_pipeline_step("fit", "Fit", config={"model": "Maxwell"})
        update_step_config(step_id, {"max_iter": 3000, "ftol": 1e-8})
        step = get_pipeline_step_by_id(step_id)
        assert step.config["model"] == "Maxwell"
        assert step.config["max_iter"] == 3000
        assert step.config["ftol"] == 1e-8

    def test_update_config_invalidates_downstream(self):
        """Updating a step config must clear its cached result and reset downstream steps."""
        id0 = _add_load_step("Load")
        id1 = _add_fit_step("Fit")
        # Mark both as COMPLETE with cached results.
        update_step_status(id0, StepStatus.COMPLETE)
        update_step_status(id1, StepStatus.COMPLETE)
        cache_step_result(id0, "data")
        cache_step_result(id1, "fit")
        # Update config of the first step — this must invalidate downstream.
        update_step_config(id0, {"new_option": True})
        step0 = get_pipeline_step_by_id(id0)
        step1 = get_pipeline_step_by_id(id1)
        # The edited step keeps its status but loses its cached result.
        assert step0.status == StepStatus.COMPLETE
        assert get_pipeline_step_result(id0) is None
        # Downstream steps are fully reset.
        assert step1.status == StepStatus.PENDING
        assert get_pipeline_step_result(id1) is None

    def test_update_config_overwrite_existing_key(self):
        """update_step_config must overwrite an existing config key."""
        step_id = add_pipeline_step("load", "Load", config={"file": "/old/path.csv"})
        update_step_config(step_id, {"file": "/new/path.csv"})
        step = get_pipeline_step_by_id(step_id)
        assert step.config["file"] == "/new/path.csv"


# ---------------------------------------------------------------------------
# TestUpdateStepStatus
# ---------------------------------------------------------------------------


class TestUpdateStepStatus:
    @pytest.mark.smoke
    def test_update_status(self):
        """update_step_status must change the step's status field."""
        step_id = _add_load_step()
        update_step_status(step_id, StepStatus.ACTIVE)
        step = get_pipeline_step_by_id(step_id)
        assert step.status == StepStatus.ACTIVE

    def test_error_message_stored(self):
        """An error_message must be stored when status is ERROR."""
        step_id = _add_fit_step()
        update_step_status(step_id, StepStatus.ERROR, error_message="Convergence failed")
        step = get_pipeline_step_by_id(step_id)
        assert step.status == StepStatus.ERROR
        assert step.error_message == "Convergence failed"

    def test_update_to_complete(self):
        """update_step_status to COMPLETE must be reflected by the selector."""
        step_id = _add_load_step()
        update_step_status(step_id, StepStatus.COMPLETE)
        step = get_pipeline_step_by_id(step_id)
        assert step.status == StepStatus.COMPLETE

    def test_clear_error_message_on_re_run(self):
        """Resetting to ACTIVE after ERROR must allow error_message to be None."""
        step_id = _add_load_step()
        update_step_status(step_id, StepStatus.ERROR, error_message="oops")
        update_step_status(step_id, StepStatus.ACTIVE, error_message=None)
        step = get_pipeline_step_by_id(step_id)
        assert step.status == StepStatus.ACTIVE
        assert step.error_message is None


# ---------------------------------------------------------------------------
# TestCacheStepResult
# ---------------------------------------------------------------------------


class TestCacheStepResult:
    @pytest.mark.smoke
    def test_cache_and_retrieve_result(self):
        """cache_step_result must persist the result so it is readable.

        The state store deep-copies step_results on every state snapshot, so
        identity (``is``) cannot be asserted — equality is used instead.
        """
        step_id = _add_load_step()
        payload = {"omega": [1, 2, 3], "Gp": [10, 20, 30]}
        cache_step_result(step_id, payload)
        retrieved = get_pipeline_step_result(step_id)
        assert retrieved == payload

    def test_cache_overwrite_result(self):
        """A second cache_step_result for the same step_id must replace the first."""
        step_id = _add_load_step()
        cache_step_result(step_id, {"v": "first"})
        cache_step_result(step_id, {"v": "second"})
        assert get_pipeline_step_result(step_id) == {"v": "second"}

    def test_uncached_step_returns_none(self):
        """get_pipeline_step_result for a step with no cached result must be None."""
        step_id = _add_load_step()
        assert get_pipeline_step_result(step_id) is None

    def test_cache_none_value(self):
        """Caching None explicitly must store None (not absent)."""
        step_id = _add_load_step()
        cache_step_result(step_id, None)
        vp = get_visual_pipeline()
        assert step_id in vp.step_results


# ---------------------------------------------------------------------------
# TestPipelineExecution
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    @pytest.mark.smoke
    def test_set_running(self):
        """set_pipeline_running(True) must be reflected by is_pipeline_running()."""
        step_id = _add_load_step()
        set_pipeline_running(True, current_step_id=step_id)
        assert is_pipeline_running() is True

    def test_set_not_running(self):
        """set_pipeline_running(False) must stop the running flag."""
        step_id = _add_load_step()
        set_pipeline_running(True, current_step_id=step_id)
        set_pipeline_running(False)
        assert is_pipeline_running() is False

    def test_current_running_step_tracked(self):
        """set_pipeline_running must store the current_running_step_id."""
        step_id = _add_load_step()
        set_pipeline_running(True, current_step_id=step_id)
        vp = get_visual_pipeline()
        assert vp.current_running_step_id == step_id

    def test_not_running_by_default(self):
        """A freshly reset store must have is_running=False."""
        assert is_pipeline_running() is False


# ---------------------------------------------------------------------------
# TestPipelineName
# ---------------------------------------------------------------------------


class TestPipelineName:
    @pytest.mark.smoke
    def test_set_name(self):
        """set_pipeline_name must update the pipeline name."""
        set_pipeline_name("My Analysis Pipeline")
        assert get_pipeline_name() == "My Analysis Pipeline"

    def test_default_name(self):
        """A freshly reset store must use the default pipeline name."""
        assert get_pipeline_name() == "Untitled Pipeline"

    def test_rename_pipeline(self):
        """Setting name twice must keep the latest value."""
        set_pipeline_name("First")
        set_pipeline_name("Second")
        assert get_pipeline_name() == "Second"


# ---------------------------------------------------------------------------
# TestClearPipeline
# ---------------------------------------------------------------------------


class TestClearPipeline:
    @pytest.mark.smoke
    def test_clear_pipeline(self):
        """clear_pipeline must remove all steps, results, and selection."""
        id0 = _add_load_step()
        id1 = _add_fit_step()
        select_pipeline_step(id0)
        cache_step_result(id0, "data")
        cache_step_result(id1, "fit")
        set_pipeline_running(True, current_step_id=id0)

        clear_pipeline()

        assert len(get_visual_pipeline_steps()) == 0
        assert get_selected_pipeline_step() is None
        assert get_pipeline_step_result(id0) is None
        assert get_pipeline_step_result(id1) is None

    def test_clear_empty_pipeline_is_idempotent(self):
        """Clearing an already-empty pipeline must not raise."""
        clear_pipeline()
        clear_pipeline()
        assert len(get_visual_pipeline_steps()) == 0


# ---------------------------------------------------------------------------
# TestLoadPipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    @pytest.mark.smoke
    def test_load_pipeline(self):
        """load_pipeline must replace the entire visual pipeline state."""
        # Create a VisualPipelineState to load.
        new_step = PipelineStepConfig(
            id="abc-123",
            step_type="fit",
            name="Restored Fit",
            config={"model": "Maxwell"},
            status=StepStatus.COMPLETE,
            position=0,
        )
        incoming = VisualPipelineState(
            steps=[new_step],
            pipeline_name="Restored Pipeline",
        )
        # Pre-populate with a different step so we can verify replacement.
        _add_load_step("Old Step")
        assert len(get_visual_pipeline_steps()) == 1

        load_pipeline(incoming)

        steps = get_visual_pipeline_steps()
        assert len(steps) == 1
        assert steps[0].id == "abc-123"
        assert steps[0].name == "Restored Fit"
        assert get_pipeline_name() == "Restored Pipeline"

    def test_load_pipeline_clears_previous_results(self):
        """load_pipeline must not carry over step_results from the prior state."""
        id0 = _add_load_step()
        cache_step_result(id0, "old_result")
        incoming = VisualPipelineState(steps=[], pipeline_name="Fresh")
        load_pipeline(incoming)
        # No steps, so no results either.
        vp = get_visual_pipeline()
        assert len(vp.step_results) == 0


# ---------------------------------------------------------------------------
# TestVisualPipelineProgress
# ---------------------------------------------------------------------------


class TestVisualPipelineProgress:
    @pytest.mark.smoke
    def test_empty_pipeline_zero_progress(self):
        """An empty pipeline must report 0.0 progress."""
        assert get_visual_pipeline_progress() == 0.0

    def test_partial_progress(self):
        """Two of three steps complete must report 2/3 progress."""
        id0 = _add_load_step()
        id1 = _add_transform_step()
        id2 = _add_fit_step()
        update_step_status(id0, StepStatus.COMPLETE)
        update_step_status(id1, StepStatus.COMPLETE)
        update_step_status(id2, StepStatus.PENDING)
        progress = get_visual_pipeline_progress()
        assert abs(progress - 2 / 3) < 1e-9

    def test_full_progress(self):
        """All steps complete must report 1.0 progress."""
        id0 = _add_load_step()
        id1 = _add_fit_step()
        update_step_status(id0, StepStatus.COMPLETE)
        update_step_status(id1, StepStatus.COMPLETE)
        assert get_visual_pipeline_progress() == 1.0

    def test_single_step_complete(self):
        """A single completed step must report 1.0 progress."""
        step_id = _add_load_step()
        update_step_status(step_id, StepStatus.COMPLETE)
        assert get_visual_pipeline_progress() == 1.0

    def test_error_status_does_not_count_as_complete(self):
        """Steps with ERROR status must not contribute to progress."""
        id0 = _add_load_step()
        id1 = _add_fit_step()
        update_step_status(id0, StepStatus.COMPLETE)
        update_step_status(id1, StepStatus.ERROR)
        progress = get_visual_pipeline_progress()
        assert abs(progress - 0.5) < 1e-9
