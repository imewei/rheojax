"""Shared cleanup utilities for worker threads.

FitWorker and BayesianWorker call gc.collect() / jax.clear_caches()
in their finally blocks.  Each worker type uses its own lock so that
a fit cleanup does not block a Bayesian cleanup and vice-versa.
"""

import threading

# Per-worker-type locks — avoids unnecessarily serializing independent workers
fit_cleanup_lock = threading.Lock()
bayesian_cleanup_lock = threading.Lock()

# Backward-compatible alias (external code may reference cleanup_lock)
cleanup_lock = fit_cleanup_lock
