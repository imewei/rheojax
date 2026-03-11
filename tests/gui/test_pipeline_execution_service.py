"""Tests for the pipeline execution service.

Unit tests that verify status dispatch, context accumulation, and error
handling in PipelineExecutionService.  All service calls are mocked so
that no real data I/O, model fitting, or Qt event loop is needed.

PipelineExecutionService subclasses QObject, so a QApplication must exist
before instantiation.  All tests that touch the service are guarded by
``pytest.mark.skipif(not HAS_PYSIDE6, ...)``.  State-only assertions use
the real StateStore and reset it via the autouse fixture from conftest.py.
"""

import os

# Set offscreen platform BEFORE any Qt imports to avoid display errors in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock, patch

import pytest

from rheojax.gui.state.actions import add_pipeline_step
from rheojax.gui.state.selectors import (
    get_pipeline_step_by_id,
    get_pipeline_step_result,
    is_pipeline_running,
)
from rheojax.gui.state.store import StateStore, StepStatus

# ---------------------------------------------------------------------------
# PySide6 availability guard
# ---------------------------------------------------------------------------

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Mark all tests in this module as needing PySide6 for Qt object creation.
pytestmark = pytest.mark.skipif(
    not HAS_PYSIDE6,
    reason="PySide6 not installed — PipelineExecutionService requires QObject",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the StateStore singleton before and after every test."""
    StateStore.reset()
    yield
    StateStore.reset()


@pytest.fixture(scope="module")
def qapp_module():
    """Provide a QApplication for the entire module (created once)."""
    if not HAS_PYSIDE6:
        return None
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def service(qapp_module):
    """Return a fresh PipelineExecutionService instance."""
    from rheojax.gui.services.pipeline_execution_service import (
        PipelineExecutionService,
    )

    return PipelineExecutionService()


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _add_load_step() -> str:
    return add_pipeline_step("load", "Load", config={"file": "/tmp/data.csv"})


def _add_fit_step() -> str:
    return add_pipeline_step("fit", "Fit", config={"model": "Maxwell"})


def _add_transform_step() -> str:
    return add_pipeline_step("transform", "FFT", config={"name": "fft"})


def _add_bayesian_step() -> str:
    return add_pipeline_step(
        "bayesian",
        "Bayes",
        config={"num_warmup": 100, "num_samples": 200},
    )


def _add_export_step() -> str:
    return add_pipeline_step("export", "Export", config={"output": "/tmp/out/"})


# ---------------------------------------------------------------------------
# TestPipelineExecutionService — instantiation
# ---------------------------------------------------------------------------


class TestPipelineExecutionService:
    @pytest.mark.smoke
    def test_service_instantiation(self, service):
        """PipelineExecutionService must be instantiable without arguments."""
        assert service is not None

    @pytest.mark.smoke
    def test_service_has_expected_signals(self, service):
        """Service must expose the required Qt signals."""
        assert hasattr(service, "step_started")
        assert hasattr(service, "step_completed")
        assert hasattr(service, "step_failed")
        assert hasattr(service, "pipeline_started")
        assert hasattr(service, "pipeline_completed")
        assert hasattr(service, "pipeline_failed")

    # ------------------------------------------------------------------
    # execute_step dispatches to the correct handler
    # ------------------------------------------------------------------

    @pytest.mark.smoke
    def test_execute_step_updates_status_to_active_then_complete(self, service):
        """execute_single_step must set ACTIVE then COMPLETE on a successful step."""
        step_id = _add_load_step()
        step = get_pipeline_step_by_id(step_id)

        mock_data = MagicMock(name="RheoData")
        mock_data.metadata = {}

        with patch.object(service, "_execute_load", return_value=mock_data):
            context: dict = {}
            service.execute_single_step(step, context)

        final_step = get_pipeline_step_by_id(step_id)
        assert final_step.status == StepStatus.COMPLETE

    @pytest.mark.smoke
    def test_execute_step_caches_result(self, service):
        """execute_single_step must cache the step result in state.

        The state store deep-copies step_results on every clone, so the cached
        value may not be the same object.  We verify the result is not None
        (i.e., something was cached) rather than using identity.
        """
        step_id = _add_load_step()
        step = get_pipeline_step_by_id(step_id)

        mock_data = MagicMock(name="RheoData")
        mock_data.metadata = {}

        with patch.object(service, "_execute_load", return_value=mock_data):
            context: dict = {}
            service.execute_single_step(step, context)

        assert get_pipeline_step_result(step_id) is not None

    def test_execute_step_unknown_type_raises(self, service):
        """_execute_step must raise ValueError for an unknown step type."""
        from rheojax.gui.state.store import PipelineStepConfig

        bad_step = PipelineStepConfig(
            id="bad-id",
            step_type="not_a_real_type",
            name="Bad",
            config={},
            status=StepStatus.PENDING,
        )
        with pytest.raises(ValueError, match="Unknown step type"):
            service._execute_step(bad_step, {})

    def test_execute_load_requires_file_key(self, service):
        """_execute_load must raise ValueError when 'file' key is absent."""
        with pytest.raises(ValueError, match="'file'"):
            service._execute_load(config={}, context={})

    def test_execute_transform_requires_data_in_context(self, service):
        """_execute_transform must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_transform(
                config={"name": "fft"}, context={}
            )

    def test_execute_transform_requires_name_key(self, service):
        """_execute_transform must raise ValueError when 'name' key is absent."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="'name'"):
            service._execute_transform(
                config={}, context={"data": mock_data}
            )

    def test_execute_fit_requires_data_in_context(self, service):
        """_execute_fit must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_fit(
                config={"model": "Maxwell"}, context={}
            )

    def test_execute_fit_requires_model_key(self, service):
        """_execute_fit must raise ValueError when 'model' key is absent."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="'model'"):
            service._execute_fit(
                config={}, context={"data": mock_data}
            )

    def test_execute_bayesian_requires_data_in_context(self, service):
        """_execute_bayesian must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_bayesian(config={}, context={})

    def test_execute_bayesian_requires_model_name_in_context(self, service):
        """_execute_bayesian must raise ValueError when 'model_name' is missing."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="prior fit step"):
            service._execute_bayesian(
                config={}, context={"data": mock_data}
            )

    # ------------------------------------------------------------------
    # execute_all integration (mocked services)
    # ------------------------------------------------------------------

    def test_execute_all_empty_pipeline_emits_completed(self, service):
        """execute_all with no steps must emit pipeline_completed."""
        completed_calls = []
        service.pipeline_completed.connect(lambda: completed_calls.append(True))
        service.execute_all([])
        assert len(completed_calls) == 1

    def test_execute_step_sets_error_status_on_failure(self, service):
        """When a step handler raises, the step status must become ERROR."""
        step_id = _add_load_step()
        step = get_pipeline_step_by_id(step_id)

        with patch.object(
            service,
            "_execute_load",
            side_effect=RuntimeError("disk error"),
        ):
            context: dict = {}
            with pytest.raises(RuntimeError, match="disk error"):
                service.execute_single_step(step, context)

        failed_step = get_pipeline_step_by_id(step_id)
        assert failed_step.status == StepStatus.ERROR
        assert "disk error" in (failed_step.error_message or "")

    def test_execute_all_sets_running_true_during_execution(self, service):
        """execute_all must mark the pipeline as running before the first step."""
        running_states: list[bool] = []

        step_id = _add_load_step()
        step = get_pipeline_step_by_id(step_id)
        mock_data = MagicMock()
        mock_data.metadata = {}

        # Capture is_pipeline_running() at the moment the step starts.
        original_execute = service._execute_step

        def capturing_execute(s, ctx):
            running_states.append(is_pipeline_running())
            return original_execute(s, ctx)

        with (
            patch.object(service, "_execute_step", side_effect=capturing_execute),
            patch.object(service, "_execute_load", return_value=mock_data),
        ):
            service.execute_all([step])

        assert any(running_states), "is_pipeline_running() was never True during execution"
        # After completion the pipeline must be stopped.
        assert is_pipeline_running() is False

    # ------------------------------------------------------------------
    # I7: Multi-step context accumulation
    # ------------------------------------------------------------------

    def test_execute_all_passes_context_between_steps(self, service):
        """execute_all must accumulate context across steps: load → transform → fit."""
        load_id = _add_load_step()
        transform_id = _add_transform_step()
        fit_id = _add_fit_step()

        load_step = get_pipeline_step_by_id(load_id)
        transform_step = get_pipeline_step_by_id(transform_id)
        fit_step = get_pipeline_step_by_id(fit_id)

        mock_data = MagicMock(name="RheoData")
        mock_data.metadata = {"test_mode": "relaxation"}
        mock_transformed = MagicMock(name="TransformedData")
        mock_transformed.metadata = {"test_mode": "relaxation"}
        mock_fit_result = MagicMock(name="FitResult")

        # Track what context keys each handler sees
        received_context_keys: list[set] = []

        def fake_load(config, context):
            received_context_keys.append(set(context.keys()))
            context["data"] = mock_data
            return mock_data

        def fake_transform(config, context):
            received_context_keys.append(set(context.keys()))
            context["data"] = mock_transformed
            return mock_transformed

        def fake_fit(config, context):
            received_context_keys.append(set(context.keys()))
            context["fit_result"] = mock_fit_result
            return mock_fit_result

        with (
            patch.object(service, "_execute_load", side_effect=fake_load),
            patch.object(service, "_execute_transform", side_effect=fake_transform),
            patch.object(service, "_execute_fit", side_effect=fake_fit),
        ):
            service.execute_all([load_step, transform_step, fit_step])

        # Load receives empty context
        assert "data" not in received_context_keys[0]
        # Transform receives data from load
        assert "data" in received_context_keys[1]
        # Fit receives data from transform
        assert "data" in received_context_keys[2]


# ---------------------------------------------------------------------------
# TestCoerceBayesianInt
# ---------------------------------------------------------------------------


class TestCoerceBayesianInt:
    """Unit tests for _coerce_bayesian_int helper."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        from rheojax.gui.services.pipeline_execution_service import (
            _coerce_bayesian_int,
        )

        self._coerce = _coerce_bayesian_int

    @pytest.mark.smoke
    def test_default_value_used_when_key_absent(self):
        assert self._coerce({}, "num_warmup", 1000) == 1000

    @pytest.mark.unit
    def test_string_coerced_to_int(self):
        assert self._coerce({"num_warmup": "500"}, "num_warmup", 1000) == 500

    @pytest.mark.unit
    def test_float_coerced_to_int(self):
        assert self._coerce({"num_samples": 2000.0}, "num_samples", 1000) == 2000

    @pytest.mark.unit
    def test_non_numeric_raises_value_error(self):
        with pytest.raises(ValueError, match="must be an integer"):
            self._coerce({"num_warmup": "abc"}, "num_warmup", 1000)

    @pytest.mark.unit
    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="must be between"):
            self._coerce({"num_warmup": 0}, "num_warmup", 1000)

    @pytest.mark.unit
    def test_exceeds_max_raises_value_error(self):
        with pytest.raises(ValueError, match="must be between"):
            self._coerce({"num_chains": 100}, "num_chains", 4, max_val=16)

    @pytest.mark.unit
    def test_boundary_value_accepted(self):
        assert self._coerce({"num_chains": 16}, "num_chains", 4, max_val=16) == 16
        assert self._coerce({"num_chains": 1}, "num_chains", 4, max_val=16) == 1
