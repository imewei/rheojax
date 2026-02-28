"""Parallel execution layer for RheoJAX.

Provides process-based parallelism for model fitting, Bayesian inference,
and transforms. Thread parallelism for I/O only.

Configuration:
    RHEOJAX_PARALLEL_WORKERS=N  -- worker count (default: auto)
    RHEOJAX_SEQUENTIAL=1        -- disable all parallelism
    RHEOJAX_WORKER_ISOLATION=subprocess|thread
    RHEOJAX_WARM_POOL=1         -- pre-initialize workers
"""
