"""
Transform Worker
===============

Background worker for transform operations (T-009).
Follows FitWorker pattern for WorkerPool integration.
"""

import time
import traceback
from dataclasses import dataclass, field
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

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


@dataclass
class TransformResult:
    """Results from a transform operation.

    Attributes
    ----------
    transform_name : str
        Name of the transform applied
    data : Any
        Transformed RheoData (or tuple with extras)
    extras : dict
        Additional results (shift_factors, spp_results, etc.)
    transform_time : float
        Wall-clock time in seconds
    timestamp : datetime
        When the transform completed
    success : bool
        Whether the transform succeeded
    message : str
        Status or error message
    warnings : list[str]
        Validation warnings
    """

    transform_name: str
    data: Any = None
    extras: dict = field(default_factory=dict)
    transform_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    message: str = ""
    warnings: list[str] = field(default_factory=list)


class TransformWorkerSignals(QObject):
    """Signals for TransformWorker.

    Signals
    -------
    progress : Signal(int, int, str)
        Progress update: (percent, total, message)
    completed : Signal(TransformResult)
        Transform completed successfully
    failed : Signal(str)
        Transform failed with error message
    cancelled : Signal()
        Transform was cancelled
    """

    progress = Signal(int, int, str)
    completed = Signal(object)  # TransformResult
    failed = Signal(str)
    cancelled = Signal()


class TransformWorker(QRunnable):
    """Background worker for transform operations.

    Runs transform execution off the main UI thread to prevent
    freezing during long transforms (Mastercurve TTS, SPP, etc.).

    Parameters
    ----------
    transform_id : str
        Transform name (e.g., 'fft', 'mastercurve', 'spp')
    data : Any
        RheoData or list[RheoData] input
    params : dict
        Transform parameters
    cancel_token : CancellationToken, optional
        Token for cancellation support
    """

    def __init__(
        self,
        transform_id: str,
        data: Any,
        params: dict[str, Any] | None = None,
        cancel_token: CancellationToken | None = None,
    ):
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for TransformWorker. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self.signals = TransformWorkerSignals()
        self.cancel_token = cancel_token or CancellationToken()

        self._transform_id = transform_id
        self._data = data
        self._params = params or {}

    def run(self) -> None:
        """Execute transform in background thread."""
        try:
            logger.info("Transform worker started", transform=self._transform_id)
            start_time = time.perf_counter()

            self.signals.progress.emit(0, 100, f"Starting {self._transform_id}...")

            # Check cancellation before starting
            self.cancel_token.check()

            from rheojax.gui.services.transform_service import TransformService

            service = TransformService()

            self.signals.progress.emit(10, 100, "Validating input...")

            # Apply transform
            self.signals.progress.emit(20, 100, f"Running {self._transform_id}...")

            result = service.apply_transform(
                self._transform_id, self._data, params=self._params
            )

            # Check cancellation after compute
            self.cancel_token.check()

            # Handle tuple return (data, extras)
            extras = {}
            if isinstance(result, tuple):
                transformed, extras = result
            else:
                transformed = result

            transform_time = time.perf_counter() - start_time

            self.signals.progress.emit(100, 100, "Complete")

            transform_result = TransformResult(
                transform_name=self._transform_id,
                data=transformed,
                extras=extras,
                transform_time=transform_time,
                timestamp=datetime.now(),
                success=True,
                message=f"Transform completed in {transform_time:.2f}s",
            )

            logger.info(
                "Transform worker complete",
                transform=self._transform_id,
                total_time=transform_time,
            )

            self.signals.completed.emit(transform_result)

        except CancellationError:
            logger.info("Transform cancelled", transform=self._transform_id)
            self.signals.cancelled.emit()

        except Exception as e:
            error_msg = f"Transform failed for {self._transform_id}: {e}"
            logger.error(
                "Transform worker failed",
                transform=self._transform_id,
                error=str(e),
                exc_info=True,
            )
            logger.debug("Traceback: %s", traceback.format_exc())

            self.cancel_token.set_error(e)
            self.signals.failed.emit(error_msg)

        finally:
            import gc

            with fit_cleanup_lock:
                gc.collect()
                try:
                    jax.clear_caches()
                except Exception:
                    pass
