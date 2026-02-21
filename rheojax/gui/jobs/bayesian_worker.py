"""
Bayesian Worker
==============

Background worker for Bayesian inference with NUTS sampling.
"""

import time
import traceback
from datetime import datetime
from typing import Any

from rheojax.gui.jobs._cleanup import bayesian_cleanup_lock

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
from rheojax.gui.state.store import BayesianResult
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

logger = get_logger(__name__)


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
        seed: int = 42,
        cancel_token: CancellationToken | None = None,
        deformation_mode: str | None = None,
        poisson_ratio: float | None = None,
        fitted_model_state: dict[str, Any] | None = None,
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
        deformation_mode : str, optional
            Deformation mode for DMTA (e.g., 'tension', 'shear')
        poisson_ratio : float, optional
            Poisson ratio for E*-G* conversion
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
        self._deformation_mode = deformation_mode
        self._poisson_ratio = poisson_ratio
        self._fitted_model_state = fitted_model_state

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
                "Bayesian worker started",
                model=self._model_name,
                num_samples=self._num_samples,
                num_warmup=self._num_warmup,
                num_chains=self._num_chains,
            )
            start_time = time.perf_counter()

            # Import model from registry (core registry is authoritative)
            # Import inside run() to avoid JAX initialization issues
            from rheojax.core.registry import ModelRegistry

            # F-GMM-001: Infer model kwargs (e.g., n_modes) from warm-start
            from rheojax.gui.services.model_service import infer_model_kwargs

            model_kwargs = (
                infer_model_kwargs(
                    self._model_name, list(self._warm_start.keys())
                )
                if self._warm_start
                else {}
            )

            try:
                model = ModelRegistry.create(self._model_name, **model_kwargs)
                logger.debug(
                    "Model created from registry",
                    model=self._model_name,
                    model_kwargs=model_kwargs or None,
                )
            except KeyError:
                raise ValueError(
                    f"Model '{self._model_name}' not found in registry"
                ) from None

            # Apply warm start if provided
            if self._warm_start:
                logger.info(
                    "Using warm-start from NLSQ",
                    parameters=list(self._warm_start.keys()),
                )
                for name, value in self._warm_start.items():
                    if name in model.parameters:
                        model.parameters[name].value = value
                        logger.debug(
                            "Applied warm-start parameter",
                            parameter=name,
                            value=value,
                        )
                    else:
                        logger.warning(
                            "Warm-start parameter not found in model",
                            parameter=name,
                        )

            # GUI-004/GUI-006 fix: Apply custom priors if provided.
            # TODO(GUI-004): This is currently a dead code path. The priors
            # are set on `model.parameters[name].prior` here, but
            # BayesianService.run_mcmc() creates a NEW model instance
            # internally, discarding these assignments. Furthermore,
            # BayesianMixin.fit_bayesian() does not read `param.prior`.
            # Fixing the full prior pipeline requires changes to
            # BayesianMixin (core/bayesian.py), which is outside GUI scope.
            if self._priors:
                # Warn the user that configured priors will not propagate
                if any(
                    name in model.parameters for name in self._priors
                ):
                    logger.warning(
                        "User-configured priors are not yet propagated to "
                        "NUTS inference. BayesianService creates a fresh "
                        "model instance, discarding prior assignments.",
                        parameters=list(self._priors.keys()),
                    )
                for name, prior in self._priors.items():
                    if name in model.parameters:
                        model.parameters[name].prior = prior
                        logger.debug(
                            "Applied custom prior (will be discarded by service)",
                            parameter=name,
                        )
                    else:
                        logger.warning(
                            "Prior parameter not found in model",
                            parameter=name,
                        )

            # Get test mode from data.
            # GUI-014 fix: DatasetState.test_mode is a required str field,
            # so the None/missing branch is effectively dead code. Keep the
            # metadata fallback for robustness but replace the dead warning
            # with a debug assertion.
            test_mode = getattr(self._data, "test_mode", None)
            if test_mode is None:
                if hasattr(self._data, "metadata"):
                    test_mode = self._data.metadata.get("test_mode", "oscillation")
                else:
                    test_mode = "oscillation"
                logger.debug(
                    "test_mode not found on data object, using fallback",
                    resolved_mode=test_mode,
                    data_type=type(self._data).__name__,
                )

            # Track sampling start time for progress logging
            sampling_start_time = time.perf_counter()
            last_progress_log_time = sampling_start_time

            # Create progress callback
            def progress_callback(stage: str, chain: int, iteration: int, total: int):
                """Progress callback for MCMC sampling."""
                nonlocal last_progress_log_time
                self.cancel_token.check()

                # Update stage if changed
                if stage != self._current_stage:
                    self._current_stage = stage
                    self.signals.stage_changed.emit(stage)
                    logger.debug(
                        "MCMC stage changed",
                        stage=stage,
                        chain=chain,
                    )

                # Calculate progress percent
                samples_per_chain = self._num_warmup + self._num_samples
                total_samples = max(self._num_chains * samples_per_chain, 1)
                percent = min(int(iteration / total_samples * 100), 100)
                message = (
                    f"{stage.capitalize()} chain {chain}: {iteration}/{total_samples}"
                )

                # Log sampling progress at DEBUG level (throttled to every 5 seconds)
                current_time = time.perf_counter()
                if current_time - last_progress_log_time >= 5.0:
                    elapsed = current_time - sampling_start_time
                    logger.debug(
                        "Sampling progress",
                        samples=iteration,
                        total=total_samples,
                        percent=percent,
                        elapsed=f"{elapsed:.1f}s",
                        stage=stage,
                        chain=chain,
                    )
                    last_progress_log_time = current_time

                # Emit progress signal normalized to percent
                self.signals.progress.emit(percent, 100, message)

            # Emit warmup stage start
            self.signals.stage_changed.emit("warmup")

            # Execute Bayesian inference via service to honor progress/cancel
            logger.debug(
                "Running NUTS sampling",
                warmup=self._num_warmup,
                samples=self._num_samples,
                chains=self._num_chains,
                test_mode=test_mode,
                seed=self._seed,
            )

            from rheojax.gui.services.bayesian_service import BayesianService

            svc = BayesianService()
            mcmc_kwargs: dict[str, Any] = {"seed": self._seed}
            if self._deformation_mode is not None:
                mcmc_kwargs["deformation_mode"] = self._deformation_mode
            if self._poisson_ratio is not None:
                mcmc_kwargs["poisson_ratio"] = self._poisson_ratio
            # F-HL-005 fix: Pass fitted model state for stateful models
            if self._fitted_model_state:
                mcmc_kwargs["fitted_model_state"] = self._fitted_model_state

            # F-009 fix: Emit periodic elapsed-time updates during NUTS sampling
            # since NumPyro NUTS does not support progress callbacks.
            import threading

            _nuts_done = threading.Event()
            _nuts_start = time.perf_counter()

            def _elapsed_timer():
                while not _nuts_done.wait(timeout=5.0):
                    elapsed = time.perf_counter() - _nuts_start
                    self.signals.progress.emit(
                        0, 0, f"NUTS sampling... ({elapsed:.0f}s elapsed)"
                    )

            timer_thread = threading.Thread(target=_elapsed_timer, daemon=True)
            timer_thread.start()

            try:
                bayesian_result = svc.run_mcmc(
                    self._model_name,
                    self._data,
                    num_warmup=self._num_warmup,
                    num_samples=self._num_samples,
                    num_chains=self._num_chains,
                    warm_start=self._warm_start,
                    test_mode=test_mode,
                    progress_callback=progress_callback,
                    **mcmc_kwargs,
                )
            finally:
                _nuts_done.set()
                timer_thread.join(timeout=1.0)

            sampling_time = time.perf_counter() - start_time

            # Emit sampling stage and 100% progress on completion
            self.signals.stage_changed.emit("sampling")
            self.signals.progress.emit(100, 100, "Sampling complete")

            # Extract diagnostics
            diagnostics = bayesian_result.diagnostics

            # Check for divergences
            num_divergences = diagnostics.get("divergences", 0)
            if num_divergences > 0:
                self.signals.divergence_detected.emit(num_divergences)
                logger.warning(
                    "Detected divergent transitions",
                    divergences=num_divergences,
                    model=self._model_name,
                )

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
                dataset_id="",  # Filled by BayesianPage._on_finished()
                posterior_samples=bayesian_result.posterior_samples,
                summary=bayesian_result.summary,
                r_hat=diagnostics.get("r_hat") or diagnostics.get("rhat") or {},
                ess=diagnostics.get("ess", {}),
                divergences=int(diagnostics.get("divergences", 0) or 0),
                credible_intervals=credible_intervals,
                mcmc_time=sampling_time,
                timestamp=datetime.now(),
                num_warmup=self._num_warmup,
                num_samples=self._num_samples,
                num_chains=self._num_chains,
                inference_data=getattr(bayesian_result, "inference_data", None),
            )

            # Log convergence diagnostics at DEBUG level
            rhat_dict = diagnostics.get("r_hat") or diagnostics.get("rhat") or {}
            ess_dict = diagnostics.get("ess", {})
            for param_name in bayesian_result.posterior_samples.keys():
                r_hat = rhat_dict.get(param_name, float("nan"))
                ess = ess_dict.get(param_name, float("nan"))
                logger.debug(
                    "Parameter convergence diagnostics",
                    parameter=param_name,
                    r_hat=f"{r_hat:.4f}",
                    ess=f"{ess:.0f}",
                )

            # Log completion at INFO level
            logger.info(
                "Bayesian worker complete",
                model=self._model_name,
                total_time=f"{sampling_time:.2f}s",
                num_samples=self._num_samples,
                num_chains=self._num_chains,
                divergences=num_divergences,
            )

            # Emit completion signal
            self.signals.completed.emit(result)

        except CancellationError:
            elapsed = (
                time.perf_counter() - start_time if "start_time" in locals() else 0
            )
            logger.info(
                "Bayesian worker cancelled",
                model=self._model_name,
                elapsed=f"{elapsed:.2f}s",
            )
            self.signals.cancelled.emit()

        except Exception as e:
            error_msg = f"Bayesian inference failed for {self._model_name}: {str(e)}"
            logger.error(
                "Bayesian worker failed",
                model=self._model_name,
                error=str(e),
                exc_info=True,
            )
            logger.debug(traceback.format_exc())

            # Store error in token
            self.cancel_token.set_error(e)

            # Emit failure signal
            self.signals.failed.emit(error_msg)

        finally:
            # Release JAX compilation caches to avoid memory buildup
            # Serialize cleanup across workers to avoid concurrent gc/JIT issues
            import gc

            with bayesian_cleanup_lock:
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
