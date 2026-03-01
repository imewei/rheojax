"""Adaptive parallelism configuration.

Auto-detects optimal worker count based on CPU cores, GPU count, and RAM.
All settings overridable via environment variables or configure() API.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Module-level overrides (set via configure()), guarded by _config_lock
_overrides: dict[str, Any] = {}
_config_lock = threading.Lock()


def get_default_workers() -> int:
    """Optimal worker count for the current system.

    Priority: sequential mode (returns 1) > configure() override >
    env var > auto-detection.
    """
    # Sequential mode takes absolute priority
    if is_sequential_mode():
        return 1

    with _config_lock:
        override = _overrides.get("n_workers")
    if override is not None:
        return int(override)

    env_val = os.environ.get("RHEOJAX_PARALLEL_WORKERS", "").strip()
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            logger.warning(
                "Invalid RHEOJAX_PARALLEL_WORKERS=%r, falling back to auto-detection",
                env_val,
            )

    cpu_count = multiprocessing.cpu_count() or 1

    # GPU-aware: each worker needs ~2GB GPU RAM
    try:
        from rheojax.core.jax_config import safe_import_jax

        jax, _ = safe_import_jax()
        devices = jax.devices()
        gpu_count = sum(1 for d in devices if d.platform != "cpu")
        if gpu_count > 0:
            return min(gpu_count, cpu_count, 4)
    except (ImportError, RuntimeError, AttributeError):
        # ImportError: JAX not installed
        # RuntimeError: JAX initialization failed
        # AttributeError: API mismatch
        pass

    # CPU: half of cores, min 1, max 8 for practical memory limits
    return max(1, min(cpu_count // 2, 8))


def is_sequential_mode() -> bool:
    """Check if all parallelism is disabled."""
    return os.environ.get("RHEOJAX_SEQUENTIAL", "0") == "1"


def get_worker_isolation() -> str:
    """Get worker isolation mode: 'subprocess' or 'thread'."""
    with _config_lock:
        override = _overrides.get("isolation")
    if override is not None:
        return str(override)
    return os.environ.get("RHEOJAX_WORKER_ISOLATION", "subprocess")


def get_parallel_config() -> dict[str, Any]:
    """Get full parallel configuration as dict.

    Takes a snapshot of _overrides under the lock for a consistent view.
    """
    with _config_lock:
        overrides_snapshot = dict(_overrides)
    return {
        "n_workers": get_default_workers(),
        "isolation": overrides_snapshot.get("isolation")
        or os.environ.get("RHEOJAX_WORKER_ISOLATION", "subprocess"),
        "sequential": is_sequential_mode(),
        "warm_pool": overrides_snapshot.get("warm_pool", False)
        or os.environ.get("RHEOJAX_WARM_POOL", "0") == "1",
    }


def configure(
    n_workers: int | None = None,
    warm_pool: bool = False,
    isolation: str | None = None,
) -> None:
    """Override default parallel configuration.

    Call once at application startup. Thread-safe.
    Pass no arguments to reset to auto-detection.
    """
    global _overrides
    new_overrides: dict[str, Any] = {}
    if n_workers is not None:
        new_overrides["n_workers"] = max(1, n_workers)
    if warm_pool:
        new_overrides["warm_pool"] = True
    if isolation is not None:
        new_overrides["isolation"] = isolation
    with _config_lock:
        _overrides = new_overrides
