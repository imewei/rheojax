"""Tests for the pipeline execution service.

Unit tests that verify signal emission, context accumulation, and error
handling in PipelineExecutionService.  All service calls are mocked so
that no real data I/O, model fitting, or Qt event loop is needed.

PipelineExecutionService subclasses QObject, so a QApplication must exist
before instantiation.  All tests that touch the service are guarded by
``pytest.mark.skipif(not HAS_PYSIDE6, ...)``.  Step/pipeline status is
signal-based in production (step_started/step_completed/step_failed/
pipeline_started/pipeline_completed/pipeline_failed) -- window.py connects
to these signals directly, so status-related assertions here spy on the
same signals rather than reading a shared store.
"""

import os

# Set offscreen platform BEFORE any Qt imports to avoid display errors in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock, patch

import pytest

from rheojax.gui.foundation.state import PipelineStepConfig

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
# PipelineExecutionService only ever reads .id/.step_type/.config off a step
# (confirmed against batch_runner.py's real construction) -- these mirror
# that shape directly, no add_pipeline_step()/state-store round-trip needed.


def _load_step() -> PipelineStepConfig:
    return PipelineStepConfig(
        id="load-1", step_type="load", config={"file": "/tmp/data.csv"}
    )


def _fit_step() -> PipelineStepConfig:
    return PipelineStepConfig(id="fit-1", step_type="fit", config={"model": "Maxwell"})


def _transform_step() -> PipelineStepConfig:
    return PipelineStepConfig(
        id="transform-1", step_type="transform", config={"name": "fft"}
    )


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
        """execute_single_step must emit step_started then step_completed on success."""
        step = _load_step()
        completed_ids = []
        service.step_completed.connect(completed_ids.append)

        mock_data = MagicMock(name="RheoData")
        mock_data.metadata = {}

        with patch.object(service, "_execute_load", return_value=mock_data):
            context: dict = {}
            service.execute_single_step(step, context)

        assert completed_ids == [step.id]

    @pytest.mark.smoke
    def test_execute_step_caches_result(self, service):
        """execute_single_step must return the step result to the caller."""
        step = _load_step()

        mock_data = MagicMock(name="RheoData")
        mock_data.metadata = {}

        with patch.object(service, "_execute_load", return_value=mock_data):
            context: dict = {}
            result = service.execute_single_step(step, context)

        assert result is mock_data

    def test_execute_step_unknown_type_raises(self, service):
        """_execute_step must raise ValueError for an unknown step type."""
        bad_step = PipelineStepConfig(id="bad-id", step_type="not_a_real_type", config={})
        with pytest.raises(ValueError, match="Unknown step type"):
            service._execute_step(bad_step, {})

    def test_execute_load_requires_file_key(self, service):
        """_execute_load must raise ValueError when 'file' key is absent."""
        with pytest.raises(ValueError, match="'file'"):
            service._execute_load(config={}, context={})

    def test_execute_transform_requires_data_in_context(self, service):
        """_execute_transform must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_transform(config={"name": "fft"}, context={})

    def test_execute_transform_requires_name_key(self, service):
        """_execute_transform must raise ValueError when 'name' key is absent."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="'name'"):
            service._execute_transform(config={}, context={"data": mock_data})

    def test_execute_fit_requires_data_in_context(self, service):
        """_execute_fit must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_fit(config={"model": "Maxwell"}, context={})

    def test_execute_fit_requires_model_key(self, service):
        """_execute_fit must raise ValueError when 'model' key is absent."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="'model'"):
            service._execute_fit(config={}, context={"data": mock_data})

    def test_execute_fit_raises_on_unsuccessful_fit_result(self, service):
        """_execute_fit must raise when ModelService.fit() reports success=False.

        Regression: ModelService.fit() catches its own exceptions and returns
        FitResult(success=False, ...) instead of raising. execute_all()/
        execute_single_step() only mark a step ERROR when the handler raises,
        so a diverged/failed fit was silently dispatched as StepStatus.COMPLETE
        and the pipeline reported success.
        """
        mock_data = MagicMock()
        failed_result = MagicMock(success=False, message="Fit diverged")
        service._model_service = MagicMock()
        service._model_service.fit.return_value = failed_result

        with pytest.raises(RuntimeError, match="Fit diverged"):
            service._execute_fit(
                config={"model": "Maxwell"}, context={"data": mock_data}
            )

    def test_execute_single_step_marks_error_on_failed_fit(self, service):
        """A failed fit step must emit step_failed, not step_completed."""
        step = _fit_step()
        failed_ids = []
        completed_ids = []
        service.step_failed.connect(lambda step_id, msg: failed_ids.append(step_id))
        service.step_completed.connect(completed_ids.append)

        mock_data = MagicMock()
        failed_result = MagicMock(success=False, message="Fit diverged")
        service._model_service = MagicMock()
        service._model_service.fit.return_value = failed_result

        with pytest.raises(RuntimeError, match="Fit diverged"):
            service.execute_single_step(step, {"data": mock_data})

        assert failed_ids == [step.id]
        assert completed_ids == []

    def test_execute_bayesian_requires_data_in_context(self, service):
        """_execute_bayesian must raise ValueError when context has no 'data'."""
        with pytest.raises(ValueError, match="prior load step"):
            service._execute_bayesian(config={}, context={})

    def test_execute_bayesian_requires_model_name_in_context(self, service):
        """_execute_bayesian must raise ValueError when 'model_name' is missing."""
        mock_data = MagicMock()
        with pytest.raises(ValueError, match="prior fit step"):
            service._execute_bayesian(config={}, context={"data": mock_data})

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
        """When a step handler raises, the service must emit step_failed with the error."""
        step = _load_step()
        failed = []
        service.step_failed.connect(lambda step_id, msg: failed.append((step_id, msg)))

        with patch.object(
            service,
            "_execute_load",
            side_effect=RuntimeError("disk error"),
        ):
            context: dict = {}
            with pytest.raises(RuntimeError, match="disk error"):
                service.execute_single_step(step, context)

        assert len(failed) == 1
        assert failed[0][0] == step.id
        assert "disk error" in failed[0][1]

    def test_execute_all_sets_running_true_during_execution(self, service):
        """execute_all must emit pipeline_started before the first step, and
        pipeline_completed only after the last step finishes."""
        events: list[str] = []

        step = _load_step()
        mock_data = MagicMock()
        mock_data.metadata = {}

        service.pipeline_started.connect(lambda: events.append("started"))
        service.pipeline_completed.connect(lambda: events.append("completed"))

        original_execute = service._execute_step

        def capturing_execute(s, ctx):
            events.append("step")
            return original_execute(s, ctx)

        with (
            patch.object(service, "_execute_step", side_effect=capturing_execute),
            patch.object(service, "_execute_load", return_value=mock_data),
        ):
            service.execute_all([step])

        assert events == ["started", "step", "completed"], (
            "pipeline_started must fire before the step runs, "
            "pipeline_completed only after it finishes"
        )

    # ------------------------------------------------------------------
    # I7: Multi-step context accumulation
    # ------------------------------------------------------------------

    def test_execute_all_passes_context_between_steps(self, service):
        """execute_all must accumulate context across steps: load → transform → fit."""
        load_step = _load_step()
        transform_step = _transform_step()
        fit_step = _fit_step()

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
