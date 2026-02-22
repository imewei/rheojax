"""
Fit Worker
=========

Background worker for NLSQ model fitting operations.
"""

import time
import traceback
from datetime import datetime
from typing import Any

from rheojax.gui.jobs._cleanup import fit_cleanup_lock

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


# Import canonical FitResult from store (single source of truth)
from rheojax.gui.state.store import FitResult


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

    progress = Signal(int, int, str)  # percent, total, message
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

            # GUI-008 fix: Track last NLSQ callback time so the elapsed timer
            # thread only emits when NLSQ hasn't reported progress recently,
            # avoiding progress bar flickering from dual update sources.
            _last_nlsq_progress = time.perf_counter()

            # Add progress callback that checks cancellation and emits percentage
            def progress_callback(iteration: int, loss: float, **kwargs):
                """Progress callback for NLSQ optimization."""
                nonlocal _last_nlsq_progress
                _last_nlsq_progress = time.perf_counter()
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

            # GUI-003 fix: Do NOT inject callback into fit_kwargs here.
            # ModelService.fit() accepts progress_callback as a named parameter
            # (line 204) and injects it into fit_kwargs internally (line 765-766).
            # Injecting here would create a duplicate/conflicting "callback" key.

            # Delegate to ModelService for fitting
            from rheojax.gui.services.model_service import ModelService

            service = ModelService()
            logger.debug(
                "Fitting with ModelService",
                model=self._model_name,
                initial_params=self._initial_params,
                options=self._options,
            )

            # F-004 fix: Emit periodic elapsed-time updates for slow fits
            # (e.g., EPM lattice simulations that take 5-30 min per NLSQ).
            import threading

            _fit_done = threading.Event()
            _fit_start = time.perf_counter()

            def _elapsed_timer():
                while not _fit_done.wait(timeout=5.0):
                    # GUI-008 fix: Only emit elapsed time if the NLSQ callback
                    # hasn't reported progress in the last 4 seconds, to avoid
                    # both sources fighting over the progress bar.
                    if time.perf_counter() - _last_nlsq_progress > 4.0:
                        elapsed = time.perf_counter() - _fit_start
                        self.signals.progress.emit(
                            0, 0, f"Fitting {self._model_name}... ({elapsed:.0f}s elapsed)"
                        )

            timer_thread = threading.Thread(target=_elapsed_timer, daemon=True)
            timer_thread.start()

            # F-MCT-008 fix: Pre-compile JIT kernels for ITT-MCT models
            # to avoid 30-90s apparent freeze on first prediction.
            if "itt_mct" in self._model_name:
                self.signals.progress.emit(0, 100, "Compiling JIT kernels...")
                try:
                    from rheojax.core.registry import ModelRegistry

                    model_cls = ModelRegistry.get(self._model_name)
                    if model_cls is not None:
                        tmp_model = model_cls()
                        if hasattr(tmp_model, "precompile"):
                            tmp_model.precompile()
                        del tmp_model
                except Exception:
                    pass  # Don't block fitting if precompile fails

            try:
                service_result = service.fit(
                    self._model_name,
                    self._data,
                    params=self._initial_params,
                    progress_callback=progress_callback,
                    **fit_kwargs,
                )
            finally:
                _fit_done.set()
                timer_thread.join(timeout=1.0)

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
                num_iterations=n_iterations,
                success=getattr(service_result, "success", False),
                message=getattr(service_result, "message", ""),
                x_fit=getattr(service_result, "x_fit", None),
                y_fit=getattr(service_result, "y_fit", None),
                residuals=getattr(service_result, "residuals", None),
                pcov=getattr(service_result, "pcov", None),
                rmse=float(metadata.get("rmse", 0.0)) if metadata.get("rmse") is not None else None,
                mae=float(metadata.get("mae", 0.0)) if metadata.get("mae") is not None else None,
                aic=float(metadata.get("aic", 0.0)) if metadata.get("aic") is not None else None,
                bic=float(metadata.get("bic", 0.0)) if metadata.get("bic") is not None else None,
                metadata=metadata,
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
            # Serialize cleanup across workers to avoid concurrent gc/JIT issues
            import gc

            with fit_cleanup_lock:
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
