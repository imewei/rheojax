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

# NOTE: cleanup_lock serializes jax.clear_caches() between worker threads
# but does NOT synchronize with main-thread predict() calls. The main
# thread should not call model.predict() while a worker's finally block
# is running. In practice, this is safe because the completion signal
# triggers main-thread predict only after the worker fully returns.
