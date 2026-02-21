"""
Background Jobs
==============

Worker pool and job management for long-running computations.

This module provides a complete background job system for running JAX
operations off the main UI thread. It includes:

- CancellationToken: Thread-safe cancellation mechanism
- WorkerPool: PySide6 QThreadPool-based job manager
- FitWorker: NLSQ model fitting worker
- BayesianWorker: MCMC sampling worker
- FitResult: Results from model fitting
- BayesianResult: Results from Bayesian inference

Example
-------
>>> # Create worker pool
>>> pool = WorkerPool(max_threads=4)  # doctest: +SKIP
>>>
>>> # Submit NLSQ fit job
>>> token = CancellationToken()  # doctest: +SKIP
>>> fit_worker = FitWorker(  # doctest: +SKIP
...     model_name='maxwell',
...     data=rheo_data,
...     cancel_token=token
... )
>>> job_id = pool.submit(fit_worker)  # doctest: +SKIP
>>>
>>> # Connect signals
>>> pool.job_completed.connect(on_fit_completed)  # doctest: +SKIP
>>> pool.job_failed.connect(on_fit_failed)  # doctest: +SKIP
>>>
>>> # Cancel if needed
>>> pool.cancel(job_id)  # doctest: +SKIP
"""

__all__ = [
    "CancellationToken",
    "CancellationError",
    "WorkerPool",
    "FitWorker",
    "FitWorkerSignals",
    "FitResult",
    "BayesianWorker",
    "BayesianWorkerSignals",
    "BayesianResult",
    "TransformWorker",
    "TransformWorkerSignals",
    "TransformResult",
]


def __getattr__(name: str):
    """Lazy import for job components.

    This defers imports until actually needed, avoiding circular
    dependencies and reducing startup time.
    """
    if name == "CancellationToken":
        from rheojax.gui.jobs.cancellation import CancellationToken

        return CancellationToken
    elif name == "CancellationError":
        from rheojax.gui.jobs.cancellation import CancellationError

        return CancellationError
    elif name == "WorkerPool":
        from rheojax.gui.jobs.worker_pool import WorkerPool

        return WorkerPool
    elif name == "FitWorker":
        from rheojax.gui.jobs.fit_worker import FitWorker

        return FitWorker
    elif name == "FitWorkerSignals":
        from rheojax.gui.jobs.fit_worker import FitWorkerSignals

        return FitWorkerSignals
    elif name == "FitResult":
        from rheojax.gui.jobs.fit_worker import FitResult

        return FitResult
    elif name == "BayesianWorker":
        from rheojax.gui.jobs.bayesian_worker import BayesianWorker

        return BayesianWorker
    elif name == "BayesianWorkerSignals":
        from rheojax.gui.jobs.bayesian_worker import BayesianWorkerSignals

        return BayesianWorkerSignals
    elif name == "BayesianResult":
        from rheojax.gui.state.store import BayesianResult

        return BayesianResult
    elif name == "TransformWorker":
        from rheojax.gui.jobs.transform_worker import TransformWorker

        return TransformWorker
    elif name == "TransformWorkerSignals":
        from rheojax.gui.jobs.transform_worker import TransformWorkerSignals

        return TransformWorkerSignals
    elif name == "TransformResult":
        from rheojax.gui.jobs.transform_worker import TransformResult

        return TransformResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
