"""
Fit Worker
=========

Background worker for NLSQ model fitting operations.
"""

import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from rheojax.gui.compat import QObject, QRunnable, Signal

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

    class QObject:  # type: ignore
        pass

    class QRunnable:  # type: ignore
        pass

    class Signal:  # type: ignore
        def __init__(self, *args):
            pass


from rheojax.core.jax_config import safe_import_jax
from rheojax.gui.jobs.cancellation import CancellationError, CancellationToken
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

logger = get_logger(__name__)


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
    x_fit: Any | None = None
    y_fit: Any | None = None
    residuals: Any | None = None


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
        self._last_loss = float("inf")

    def run(self) -> None:
        """Execute fitting in background thread.

        This method runs in a separate thread and should not
        directly manipulate UI elements. Use signals to communicate
        with the main thread.
        """
        try:
            logger.info("Fit worker started", model=self._model_name)
            start_time = time.perf_counter()

            # Prepare fitting options
            fit_kwargs = self._options.copy()
            max_iter = int(fit_kwargs.get("max_iter", 100)) or 100

            # Add progress callback that checks cancellation and emits percentage
            def progress_callback(iteration: int, loss: float, **kwargs):
                """Progress callback for NLSQ optimization."""
                self.cancel_token.check()
                self._last_iteration = iteration
                self._last_loss = loss
                percent = min(int(iteration / max_iter * 100), 100)
                message = f"Iteration {iteration}: loss = {loss:.6e}"

                # Log iteration timing at DEBUG level
                elapsed = time.perf_counter() - start_time
                logger.debug(
                    "Iteration complete",
                    iteration=iteration,
                    elapsed=elapsed,
                    loss=loss,
                    percent=percent,
                )

                self.signals.progress.emit(percent, 100, message)

            fit_kwargs["callback"] = progress_callback

            # Delegate to ModelService for fitting
            from rheojax.gui.services.model_service import ModelService

            service = ModelService()
            logger.debug(
                "Fitting with ModelService",
                model=self._model_name,
                initial_params=self._initial_params,
                options=self._options,
            )
            service_result = service.fit(
                self._model_name,
                self._data,
                params=self._initial_params,
                progress_callback=progress_callback,
                **fit_kwargs,
            )

            fit_time = time.perf_counter() - start_time

            # Check for service-level failure
            if not getattr(service_result, "success", True):
                error_msg = getattr(service_result, "message", "Fit failed")
                logger.error(
                    "Fit service reported failure",
                    model=self._model_name,
                    message=error_msg,
                )
                self.signals.failed.emit(
                    f"Fit failed for {self._model_name}: {error_msg}"
                )
                return

            # Extract fitted parameters and metrics from service result
            fitted_params = service_result.parameters
            metadata = getattr(service_result, "metadata", {}) or {}
            r_squared = metadata.get("r_squared", 0.0)
            mpe = metadata.get("mpe", 0.0)
            chi_squared = float(getattr(service_result, "chi_squared", 0.0))
            n_iterations = metadata.get("n_iterations", self._last_iteration)

            # Create worker-level result
            result = FitResult(
                model_name=self._model_name,
                parameters=fitted_params,
                r_squared=float(r_squared),
                mpe=float(mpe),
                chi_squared=float(chi_squared),
                fit_time=fit_time,
                timestamp=datetime.now(),
                n_iterations=n_iterations,
                success=True,
                x_fit=getattr(service_result, "x_fit", None),
                y_fit=getattr(service_result, "y_fit", None),
                residuals=getattr(service_result, "residuals", None),
            )

            logger.info(
                "Fit worker complete",
                model=self._model_name,
                total_time=fit_time,
                r_squared=r_squared,
                mpe=mpe,
                n_iterations=n_iterations,
            )

            # Emit completion signal
            self.signals.completed.emit(result)

        except CancellationError:
            logger.info("Fit cancelled", model=self._model_name)
            self.signals.cancelled.emit()

        except Exception as e:
            error_msg = f"Fit failed for {self._model_name}: {str(e)}"
            logger.error(
                "Fit worker failed",
                model=self._model_name,
                error=str(e),
                exc_info=True,
            )
            logger.debug("Traceback: %s", traceback.format_exc())

            # Store error in token
            self.cancel_token.set_error(e)

            # Emit failure signal
            self.signals.failed.emit(error_msg)

        finally:
            # Release JAX compilation caches to avoid memory buildup
            import gc

            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass

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
