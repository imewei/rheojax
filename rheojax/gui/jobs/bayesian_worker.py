"""
Bayesian Worker
==============

Background worker for Bayesian inference with NUTS sampling.
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
        def __init__(self, *args):
            pass


from rheojax.core.jax_config import safe_import_jax
from rheojax.gui.jobs.cancellation import CancellationError, CancellationToken

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Results from Bayesian inference with NUTS sampling.

    Attributes
    ----------
    model_name : str
        Name of the fitted model
    posterior_samples : dict
        Dictionary mapping parameter names to posterior samples
    summary : dict
        Summary statistics (mean, std, quantiles) for each parameter
    diagnostics : dict
        Convergence diagnostics (R-hat, ESS, divergences)
    num_samples : int
        Number of posterior samples per chain
    num_chains : int
        Number of MCMC chains
    sampling_time : float
        Total sampling duration in seconds
    timestamp : datetime
        When the sampling was completed
    credible_intervals : dict, optional
        Credible intervals for each parameter
    inference_data : Any, optional
        Full ArviZ InferenceData with sample_stats for energy plots
    """

    model_name: str
    posterior_samples: dict[str, Any]
    summary: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]
    num_samples: int
    num_chains: int
    sampling_time: float
    timestamp: datetime
    credible_intervals: dict[str, tuple[float, float]] | None = None
    inference_data: Any | None = None


class BayesianWorkerSignals(QObject):
    """Signals for BayesianWorker.

    Signals
    -------
    progress : Signal(int, int, str)
        Progress update: (percent, total, message)
    stage_changed : Signal(str)
        Sampling stage changed: ('warmup' or 'sampling')
    completed : Signal(BayesianResult)
        Sampling completed successfully with result
    failed : Signal(str)
        Sampling failed with error message
    cancelled : Signal()
        Sampling was cancelled
    divergence_detected : Signal(int)
        Divergence detected: (count)
    """

    progress = Signal(int, int, str)  # percent, total, message
    stage_changed = Signal(str)  # 'warmup' or 'sampling'
    completed = Signal(object)  # BayesianResult
    failed = Signal(str)  # error message
    cancelled = Signal()
    divergence_detected = Signal(int)  # count


class BayesianWorker(QRunnable):
    """Worker for running MCMC sampling in background.

    Features:
        - NUTS sampling with NumPyro
        - Progress tracking via warmup/sampling stages
        - NLSQ warm-start integration
        - ArviZ diagnostics computation
        - Divergence detection and reporting
        - Cancellation support

    Example
    -------
    >>> token = CancellationToken()  # doctest: +SKIP
    >>> worker = BayesianWorker(  # doctest: +SKIP
    ...     model_name='maxwell',
    ...     data=rheo_data,
    ...     num_warmup=1000,
    ...     num_samples=2000,
    ...     num_chains=4,
    ...     warm_start={'G0': 1e6, 'tau': 1.0},
    ...     priors={},
    ...     seed=42,
    ...     cancel_token=token
    ... )
    >>> pool.submit(worker)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str,
        data: Any,  # RheoData
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        warm_start: dict[str, float] | None = None,
        priors: dict[str, Any] | None = None,
        seed: int = 0,
        cancel_token: CancellationToken | None = None,
    ):
        """Initialize Bayesian worker.

        Parameters
        ----------
        model_name : str
            Name of model to fit (e.g., 'maxwell', 'springpot')
        data : RheoData
            Rheological data to fit
        num_warmup : int, default=1000
            Number of warmup iterations per chain
        num_samples : int, default=2000
            Number of posterior samples per chain
        num_chains : int, default=4
            Number of MCMC chains to run
        warm_start : dict, optional
            Initial parameter values from NLSQ fit
        priors : dict, optional
            Custom prior distributions for parameters
        seed : int, default=0
            Random seed for reproducibility
        cancel_token : CancellationToken, optional
            Token for cancellation support
        """
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required for BayesianWorker. "
                "Install with: pip install PySide6"
            )

        super().__init__()
        self.signals = BayesianWorkerSignals()
        self.cancel_token = cancel_token or CancellationToken()

        self._model_name = model_name
        self._data = data
        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
        self._warm_start = warm_start
        self._priors = priors or {}
        self._seed = seed

        # Track progress
        self._current_stage = "warmup"
        self._total_iterations = (num_warmup + num_samples) * num_chains
        self._completed_iterations = 0

    def run(self) -> None:
        """Execute MCMC in background thread.

        This method runs in a separate thread and should not
        directly manipulate UI elements. Use signals to communicate
        with the main thread.
        """
        try:
            logger.info(
                f"Starting Bayesian inference for model: {self._model_name} "
                f"({self._num_chains} chains, {self._num_samples} samples)"
            )
            start_time = time.perf_counter()

            # Import model from registry (core registry is authoritative)
            # Import inside run() to avoid JAX initialization issues
            from rheojax.core.registry import ModelRegistry

            try:
                model = ModelRegistry.create(self._model_name)
            except KeyError:
                raise ValueError(f"Model '{self._model_name}' not found in registry")

            # Apply warm start if provided
            if self._warm_start:
                logger.info(f"Using warm-start from NLSQ: {self._warm_start}")
                for name, value in self._warm_start.items():
                    if name in model.parameters:
                        model.parameters[name].value = value
                    else:
                        logger.warning(
                            f"Warm-start parameter '{name}' not found in model"
                        )

            # Apply custom priors if provided
            if self._priors:
                logger.info(f"Using custom priors: {list(self._priors.keys())}")
                for name, prior in self._priors.items():
                    if name in model.parameters:
                        model.parameters[name].prior = prior
                    else:
                        logger.warning(f"Prior parameter '{name}' not found in model")

            # Get test mode from data
            test_mode = getattr(self._data, "test_mode", None)
            if test_mode is None:
                if hasattr(self._data, "metadata"):
                    test_mode = self._data.metadata.get("test_mode", "oscillation")
                else:
                    test_mode = "oscillation"
                    logger.warning(f"No test_mode found, defaulting to {test_mode}")

            # Create progress callback
            def progress_callback(stage: str, chain: int, iteration: int, total: int):
                """Progress callback for MCMC sampling."""
                self.cancel_token.check()

                # Update stage if changed
                if stage != self._current_stage:
                    self._current_stage = stage
                    self.signals.stage_changed.emit(stage)
                    logger.debug(f"MCMC stage: {stage}")

                # Calculate progress percent
                samples_per_chain = self._num_warmup + self._num_samples
                total_samples = max(self._num_chains * samples_per_chain, 1)
                percent = min(int(iteration / total_samples * 100), 100)
                message = (
                    f"{stage.capitalize()} chain {chain}: {iteration}/{total_samples}"
                )

                # Emit progress signal normalized to percent
                self.signals.progress.emit(percent, 100, message)

            # Emit warmup stage start
            self.signals.stage_changed.emit("warmup")

            # Execute Bayesian inference via service to honor progress/cancel
            logger.debug(
                f"Running NUTS sampling: "
                f"warmup={self._num_warmup}, samples={self._num_samples}, "
                f"chains={self._num_chains}, test_mode={test_mode}"
            )

            from rheojax.gui.services.bayesian_service import BayesianService

            svc = BayesianService()
            bayesian_result = svc.run_mcmc(
                self._model_name,
                self._data,
                num_warmup=self._num_warmup,
                num_samples=self._num_samples,
                num_chains=self._num_chains,
                warm_start=self._warm_start,
                test_mode=test_mode,
                progress_callback=progress_callback,
                seed=self._seed,
            )

            sampling_time = time.perf_counter() - start_time

            # Emit sampling stage (in case callback wasn't called)
            self.signals.stage_changed.emit("sampling")

            # Extract diagnostics
            diagnostics = bayesian_result.diagnostics

            # Check for divergences
            num_divergences = diagnostics.get("divergences", 0)
            if num_divergences > 0:
                self.signals.divergence_detected.emit(num_divergences)
                logger.warning(f"Detected {num_divergences} divergent transitions")

            # Compute credible intervals (95% HDI)
            credible_intervals = {}
            for param_name, samples in bayesian_result.posterior_samples.items():
                # Compute 95% credible interval
                lower = float(jnp.percentile(samples, 2.5))
                upper = float(jnp.percentile(samples, 97.5))
                credible_intervals[param_name] = (lower, upper)

            # Create result
            result = BayesianResult(
                model_name=self._model_name,
                posterior_samples=bayesian_result.posterior_samples,
                summary=bayesian_result.summary,
                diagnostics=diagnostics,
                num_samples=self._num_samples,
                num_chains=self._num_chains,
                sampling_time=sampling_time,
                timestamp=datetime.now(),
                credible_intervals=credible_intervals,
                inference_data=getattr(bayesian_result, "inference_data", None),
            )

            # Log diagnostics
            logger.info(f"Bayesian inference completed in {sampling_time:.2f}s")
            logger.info("Diagnostics:")
            rhat_dict = diagnostics.get("r_hat") or diagnostics.get("rhat") or {}
            ess_dict = diagnostics.get("ess", {})
            for param_name in bayesian_result.posterior_samples.keys():
                r_hat = rhat_dict.get(param_name, float("nan"))
                ess = ess_dict.get(param_name, float("nan"))
                logger.info(f"  {param_name}: R-hat={r_hat:.4f}, ESS={ess:.0f}")

            # Emit completion signal
            self.signals.completed.emit(result)

        except CancellationError:
            logger.info(f"Bayesian inference for {self._model_name} cancelled")
            self.signals.cancelled.emit()

        except Exception as e:
            error_msg = f"Bayesian inference failed for {self._model_name}: {str(e)}"
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
