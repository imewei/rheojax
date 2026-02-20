"""Shared cleanup utilities for worker threads.

FitWorker and BayesianWorker call gc.collect() / jax.clear_caches()
in their finally blocks.  A single shared lock serializes all cleanup
to prevent concurrent jax.clear_caches() calls (which is not thread-safe).
"""

import threading

# Single shared lock for all worker cleanup (jax.clear_caches is not thread-safe)
cleanup_lock = threading.Lock()

# Both worker types share the same lock to prevent concurrent jax.clear_caches()
fit_cleanup_lock = cleanup_lock
bayesian_cleanup_lock = cleanup_lock
