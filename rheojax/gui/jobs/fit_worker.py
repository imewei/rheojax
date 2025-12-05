"""
Fit Worker
=========

Background worker for NLSQ model fitting operations.
"""

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from PySide6.QtCore import QObject, QRunnable, Signal
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    class QObject:  # type: ignore
        pass
    class QRunnable:  # type: ignore
        pass
    class Signal:  # type: ignore
        def __init__(self, *args): pass

from rheojax.core.jax_config import safe_import_jax
from rheojax.gui.jobs.cancellation import CancellationError, CancellationToken

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Results from NLSQ model fitting.

    Attributes
    ----------
    model_name : str
        Name of the fitted model
    parameters : dict
        Fitted parameter values
    r_squared : float
        R-squared goodness of fit
    mpe : float
        Mean percentage error
    chi_squared : float
        Chi-squared statistic
    fit_time : float
        Fitting duration in seconds
    timestamp : datetime
        When the fit was completed
    n_iterations : int, optional
        Number of optimization iterations
    success : bool, optional
        Whether optimization converged successfully
    """

    model_name: str
    parameters: dict[str, float]
    r_squared: float
    mpe: float
    chi_squared: float
    fit_time: float
    timestamp: datetime
    n_iterations: int | None = None
    success: bool = True


class FitWorkerSignals(QObject):
    """Signals for FitWorker.

    Signals
    -------
    progress : Signal(int, float, str)
        Progress update: (iteration, loss, message)
    completed : Signal(FitResult)
        Fitting completed successfully with result
    failed : Signal(str)
        Fitting failed with error message
    cancelled : Signal()
        Fitting was cancelled
    """

    progress = Signal(int, float, str)  # iteration, loss, message
    completed = Signal(object)  # FitResult
    failed = Signal(str)  # error message
    cancelled = Signal()


class FitWorker(QRunnable):
    """Worker for running NLSQ fitting in background.

    Features:
        - NLSQ optimization with progress callbacks
        - Cancellation support via token
        - Multi-start optimization
        - Automatic parameter initialization
        - Comprehensive error handling

    Example
    -------
    >>> token = CancellationToken()  # doctest: +SKIP
    >>> worker = FitWorker(  # doctest: +SKIP
    ...     model_name='maxwell',
    ...     data=rheo_data,
    ...     initial_params={'G0': 1e6, 'tau': 1.0},
    ...     options={'max_iter': 5000},
    ...     cancel_token=token
    ... )
    >>> pool.submit(worker)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str,
        data: Any,  # RheoData
        initial_params: dict[str, float] | None = None,
        options: dict[str, Any] | None = None,
        cancel_token: CancellationToken | None = None,
    ):
        """Initialize fit worker.

        Parameters
        ----------
        model_name : str
            Name of model to fit (e.g., 'maxwell', 'springpot')
        data : RheoData
            Rheological data to fit
        initial_params : dict, optional
            Initial parameter values (if None, uses model defaults)
        options : dict, optional
            Fitting options (max_iter, ftol, xtol, etc.)
        cancel_token : CancellationToken, optional
            Token for cancellation support
        """
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for FitWorker. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self.signals = FitWorkerSignals()
        self.cancel_token = cancel_token or CancellationToken()

        self._model_name = model_name
        self._data = data
        self._initial_params = initial_params or {}
        self._options = options or {}

        # Track progress
        self._last_iteration = 0
        self._last_loss = float('inf')

    def run(self) -> None:
        """Execute fitting in background thread.

        This method runs in a separate thread and should not
        directly manipulate UI elements. Use signals to communicate
        with the main thread.
        """
        try:
            logger.info(f"Starting fit for model: {self._model_name}")
            start_time = time.perf_counter()

            # Import model from registry
            # Import inside run() to avoid JAX initialization issues
            from rheojax.models import ModelRegistry

            model_class = ModelRegistry.get(self._model_name)
            if model_class is None:
                raise ValueError(f"Model '{self._model_name}' not found in registry")

            model = model_class()

            # Set initial parameters if provided
            for name, value in self._initial_params.items():
                if name in model.parameters:
                    model.parameters[name].value = value
                else:
                    logger.warning(
                        f"Parameter '{name}' not found in model {self._model_name}"
                    )

            # Prepare fitting options
            fit_kwargs = self._options.copy()

            # Add progress callback that checks cancellation
            def progress_callback(iteration: int, loss: float, **kwargs):
                """Progress callback for NLSQ optimization."""
                # Check for cancellation
                self.cancel_token.check()

                # Update progress (emit signal)
                self._last_iteration = iteration
                self._last_loss = loss
                message = f"Iteration {iteration}: loss = {loss:.6e}"
                self.signals.progress.emit(iteration, loss, message)

            fit_kwargs['callback'] = progress_callback

            # Get test mode from data
            test_mode = getattr(self._data, 'test_mode', None)
            if test_mode is None:
                # Try to detect from metadata
                if hasattr(self._data, 'metadata'):
                    test_mode = self._data.metadata.get('test_mode', 'oscillation')
                else:
                    test_mode = 'oscillation'
                    logger.warning(f"No test_mode found, defaulting to {test_mode}")

            # Execute fitting
            logger.debug(f"Fitting {self._model_name} with test_mode={test_mode}")
            model.fit(
                self._data.x,
                self._data.y,
                test_mode=test_mode,
                **fit_kwargs
            )

            fit_time = time.perf_counter() - start_time

            # Extract fitted parameters
            fitted_params = {}
            for name, param in model.parameters.items():
                fitted_params[name] = float(param.value)

            # Get goodness of fit metrics
            r_squared = getattr(model, 'r_squared', 0.0)
            mpe = getattr(model, 'mpe', 0.0)
            chi_squared = getattr(model, 'chi_squared', 0.0)

            # Get iteration count from NLSQ result if available
            n_iterations = None
            success = True
            if hasattr(model, '_nlsq_result') and model._nlsq_result is not None:
                nlsq_result = model._nlsq_result
                n_iterations = getattr(nlsq_result, 'nit', self._last_iteration)
                success = getattr(nlsq_result, 'success', True)

            # Create result
            result = FitResult(
                model_name=self._model_name,
                parameters=fitted_params,
                r_squared=float(r_squared),
                mpe=float(mpe),
                chi_squared=float(chi_squared),
                fit_time=fit_time,
                timestamp=datetime.now(),
                n_iterations=n_iterations,
                success=success,
            )

            logger.info(
                f"Fit completed in {fit_time:.2f}s: "
                f"R²={r_squared:.4f}, MPE={mpe:.2f}%"
            )

            # Emit completion signal
            self.signals.completed.emit(result)

        except CancellationError:
            logger.info(f"Fit for {self._model_name} cancelled")
            self.signals.cancelled.emit()

        except Exception as e:
            error_msg = f"Fit failed for {self._model_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())

            # Store error in token
            self.cancel_token.set_error(e)

            # Emit failure signal
            self.signals.failed.emit(error_msg)

    def check_cancellation(self) -> None:
        """Check if job should be cancelled.

        Raises
        ------
        CancellationError
            If cancellation requested

        Notes
        -----
        This is called automatically in the progress callback.
        Can also be called manually in long-running operations.
        """
        self.cancel_token.check()
