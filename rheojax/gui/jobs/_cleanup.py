"""Shared cleanup lock for worker threads.

Both FitWorker and BayesianWorker serialize gc.collect() / jax.clear_caches()
through this single lock to prevent concurrent cleanup from different worker
types.
"""

import threading

cleanup_lock = threading.Lock()
