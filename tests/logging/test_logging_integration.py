"""
Integration tests for RheoJAX logging with actual model fitting.

Tests the logging system with real RheoJAX model operations including
NLSQ fitting, Bayesian inference, and pipeline workflows.
"""

import logging
import numpy as np
import pytest

from rheojax.logging import (
    configure_logging,
    get_logger,
    log_fit,
    log_bayesian,
    log_operation,
    reset_config,
    timed,
    IterationLogger,
    ConvergenceTracker,
    log_array_info,
    log_jax_config,
)
from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


class LogCapture:
    """Helper to capture log records for verification."""

    def __init__(self):
        self.records: list[logging.LogRecord] = []
        self.handler = logging.Handler()
        self.handler.emit = self.capture

    def capture(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def has_message(self, substring: str) -> bool:
        return any(substring in r.getMessage() for r in self.records)

    def get_messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture
def log_capture():
    """Fixture to capture log records."""
    capture = LogCapture()
    root_logger = logging.getLogger("rheojax")
    root_logger.addHandler(capture.handler)
    root_logger.setLevel(logging.DEBUG)
    yield capture
    root_logger.removeHandler(capture.handler)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    reset_config()
    yield
    reset_config()


class TestLoggingWithMaxwellModel:
    """Test logging integration with Maxwell model fitting."""

    @pytest.mark.smoke
    def test_log_fit_context_with_maxwell(self, log_capture):
        """Test log_fit context manager with actual Maxwell model."""
        from rheojax.models import Maxwell

        # Generate synthetic relaxation data
        t = np.logspace(-3, 2, 50)
        G0 = 1000.0
        tau = 1.0
        G_t = G0 * np.exp(-t / tau)

        logger = get_logger("rheojax.models.test")

        with log_fit(logger, "Maxwell", data_shape=t.shape, test_mode="relaxation") as ctx:
            model = Maxwell()
            model.fit(t, G_t, test_mode="relaxation")
            ctx["R2"] = float(model.r_squared) if hasattr(model, "r_squared") else 0.99
            ctx["n_params"] = len(model.parameters)

        # Verify logging occurred
        assert log_capture.has_message("model_fit started")
        assert log_capture.has_message("model_fit completed")

    @pytest.mark.smoke
    def test_timed_decorator_with_model_fit(self, log_capture):
        """Test @timed decorator with actual model fitting."""
        from rheojax.models import Maxwell

        logger = get_logger("rheojax.models.test")

        @timed(logger=logger, level=logging.INFO)
        def fit_maxwell(t, G_t):
            model = Maxwell()
            model.fit(t, G_t, test_mode="relaxation")
            return model

        t = np.logspace(-3, 2, 30)
        G_t = 1000.0 * np.exp(-t / 1.0)

        model = fit_maxwell(t, G_t)

        assert model is not None
        assert log_capture.has_message("fit_maxwell completed")


class TestLoggingWithFractionalModels:
    """Test logging with fractional viscoelastic models."""

    @pytest.mark.smoke
    def test_log_fit_with_fractional_maxwell(self, log_capture):
        """Test logging with FractionalMaxwellModel."""
        from rheojax.models import FractionalMaxwellModel

        # Generate oscillation data
        omega = np.logspace(-2, 2, 50)
        G_star = 1000 * (1j * omega * 1.0) ** 0.5 / (1 + (1j * omega * 1.0) ** 0.5)

        logger = get_logger("rheojax.models.fractional")

        with log_fit(logger, "FractionalMaxwellModel", data_shape=omega.shape, test_mode="oscillation") as ctx:
            model = FractionalMaxwellModel()
            model.fit(omega, G_star, test_mode="oscillation")
            ctx["alpha"] = float(model.parameters["alpha"].value)

        assert log_capture.has_message("model_fit started")
        assert log_capture.has_message("model_fit completed")


class TestLoggingWithPipeline:
    """Test logging with RheoJAX pipeline workflows."""

    @pytest.mark.smoke
    def test_pipeline_stages_logged(self, log_capture):
        """Test that pipeline stages are properly logged."""
        from rheojax.logging import log_pipeline_stage

        logger = get_logger("rheojax.pipeline.test")

        # Simulate pipeline stages
        with log_pipeline_stage(logger, "load", pipeline_id="test_001") as ctx:
            t = np.logspace(-3, 2, 50)
            G_t = 1000.0 * np.exp(-t / 1.0)
            ctx["records"] = len(t)

        with log_pipeline_stage(logger, "fit", pipeline_id="test_001") as ctx:
            # Simulate fitting
            ctx["model"] = "Maxwell"
            ctx["R2"] = 0.9987

        assert log_capture.has_message("pipeline_load started")
        assert log_capture.has_message("pipeline_load completed")
        assert log_capture.has_message("pipeline_fit started")
        assert log_capture.has_message("pipeline_fit completed")


class TestLoggingJAXOperations:
    """Test JAX-specific logging utilities with real JAX operations."""

    @pytest.mark.smoke
    def test_log_array_info_with_jax_array(self):
        """Test log_array_info with actual JAX arrays."""
        x = jnp.ones((100, 50), dtype=jnp.float64)

        info = log_array_info(x, "test_array")

        assert info["test_array_shape"] == (100, 50)
        assert "float64" in info["test_array_dtype"]
        assert info["test_array_size"] == 5000

    @pytest.mark.smoke
    def test_log_jax_config_returns_valid_info(self):
        """Test log_jax_config returns JAX configuration."""
        config = log_jax_config()

        assert "jax_version" in config
        assert "default_backend" in config
        assert "devices" in config

    @pytest.mark.smoke
    def test_log_operation_with_jax_computation(self, log_capture):
        """Test log_operation with JAX computation."""
        logger = get_logger("rheojax.core.test")

        with log_operation(logger, "jax_computation") as ctx:
            x = jnp.linspace(0, 10, 1000)
            y = jnp.sin(x) * jnp.exp(-x / 5)
            result = jnp.sum(y ** 2)
            ctx["result"] = float(result)
            ctx["array_size"] = len(x)

        assert log_capture.has_message("jax_computation started")
        assert log_capture.has_message("jax_computation completed")


class TestIterationLoggingWithOptimization:
    """Test iteration logging with actual optimization loops."""

    @pytest.mark.smoke
    def test_iteration_logger_tracks_optimization(self, log_capture):
        """Test IterationLogger with simulated optimization."""
        logger = get_logger("rheojax.optimization.test")
        iter_logger = IterationLogger(logger, log_every=10, level=logging.DEBUG)

        # Simulate optimization loop
        cost = 1.0
        for i in range(50):
            cost *= 0.95  # Exponential decay
            iter_logger.log(cost=cost, grad_norm=0.1 * cost)

        iter_logger.log_final(converged=True, method="NLSQ")

        assert iter_logger.iteration == 50
        assert log_capture.has_message("Iteration 10")
        assert log_capture.has_message("Iteration 20")
        assert log_capture.has_message("completed")

    @pytest.mark.smoke
    def test_convergence_tracker_with_real_costs(self, log_capture):
        """Test ConvergenceTracker with realistic cost sequence."""
        logger = get_logger("rheojax.optimization.test")
        tracker = ConvergenceTracker(
            logger,
            tolerance=1e-4,
            patience=3,
            min_iterations=5
        )

        # Simulate converging cost sequence
        costs = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0624, 0.0623, 0.0623, 0.0623]

        converged = False
        for cost in costs:
            converged = tracker.update(cost)
            if converged:
                break

        assert converged
        assert log_capture.has_message("Convergence achieved")


class TestLoggingWithDataTransforms:
    """Test logging with RheoJAX data transforms."""

    @pytest.mark.smoke
    def test_log_transform_context(self, log_capture):
        """Test log_transform context manager."""
        from rheojax.logging import log_transform

        logger = get_logger("rheojax.transforms.test")

        # Simulate transform operation
        input_data = np.random.randn(10, 100)

        with log_transform(logger, "mastercurve", input_shape=input_data.shape) as ctx:
            # Simulate transform
            output_data = input_data * 2
            ctx["output_shape"] = output_data.shape
            ctx["shift_factors"] = 10

        assert log_capture.has_message("transform started")
        assert log_capture.has_message("transform completed")


class TestLoggingConfiguration:
    """Test logging configuration with actual usage."""

    @pytest.mark.smoke
    def test_configure_logging_applies_to_models(self):
        """Test that configure_logging configures the logging system."""
        from rheojax.logging import is_configured

        # Initially may or may not be configured depending on test order
        configure_logging(level="INFO", colorize=False)

        # Verify configuration was applied
        assert is_configured()

        # Test that loggers work
        capture = LogCapture()
        capture.handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger("rheojax")
        root_logger.addHandler(capture.handler)

        logger = get_logger("rheojax.models.configured")
        logger.info("Info message from configured logger")
        logger.warning("Warning message from configured logger")

        root_logger.removeHandler(capture.handler)

        assert capture.has_message("Info message")
        assert capture.has_message("Warning message")

    @pytest.mark.smoke
    def test_subsystem_level_override(self):
        """Test per-subsystem log level configuration."""
        configure_logging(
            level="WARNING",
            colorize=False,
            subsystem_levels={"rheojax.models": "DEBUG"}
        )

        capture = LogCapture()
        root_logger = logging.getLogger("rheojax")
        root_logger.addHandler(capture.handler)

        # Models logger should log at DEBUG
        model_logger = get_logger("rheojax.models.test")
        model_logger.debug("Model debug message")

        # Other loggers should respect WARNING level
        other_logger = get_logger("rheojax.other.test")
        other_logger.debug("Other debug message")  # Should not appear
        other_logger.warning("Other warning message")  # Should appear

        root_logger.removeHandler(capture.handler)

        messages = capture.get_messages()
        assert any("Model debug" in m for m in messages)
        assert any("Other warning" in m for m in messages)


class TestLoggingErrorHandling:
    """Test logging during error conditions."""

    @pytest.mark.smoke
    def test_log_fit_captures_exceptions(self, log_capture):
        """Test that log_fit properly logs exceptions."""
        logger = get_logger("rheojax.models.error")

        with pytest.raises(ValueError):
            with log_fit(logger, "TestModel", test_mode="unknown") as ctx:
                raise ValueError("Simulated fitting error")

        assert log_capture.has_message("model_fit started")
        assert log_capture.has_message("failed")

    @pytest.mark.smoke
    def test_log_operation_logs_exception_type(self, log_capture):
        """Test that exceptions are logged with type information."""
        logger = get_logger("rheojax.core.error")

        with pytest.raises(RuntimeError):
            with log_operation(logger, "failing_operation") as ctx:
                raise RuntimeError("Test runtime error")

        # Check that error was logged
        error_records = [r for r in log_capture.records if r.levelno >= logging.ERROR]
        assert len(error_records) > 0
