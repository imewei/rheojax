"""Adaptive parallelism configuration.

Auto-detects optimal worker count based on CPU cores, GPU count, and RAM.
All settings overridable via environment variables or configure() API.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

# Module-level overrides (set via configure())
_overrides: dict[str, Any] = {}


def get_default_workers() -> int:
    """Optimal worker count for the current system.

    Priority: configure() override > env var > auto-detection.
    Auto-detection: GPU-aware (min(gpu_count, cpu_count, 4)) or
    CPU-only (half of cores, capped at 8).
    """
    override = _overrides.get("n_workers")
    if override is not None:
        return int(override)

    env_val = os.environ.get("RHEOJAX_PARALLEL_WORKERS")
    if env_val:
        return max(1, int(env_val))

    if is_sequential_mode():
        return 1

    cpu_count = multiprocessing.cpu_count() or 1

    # GPU-aware: each worker needs ~2GB GPU RAM
    try:
        import jax

        devices = jax.devices()
        gpu_count = sum(1 for d in devices if d.platform != "cpu")
        if gpu_count > 0:
            return min(gpu_count, cpu_count, 4)
    except Exception:
        pass

    # CPU: half of cores, min 1, max 8 for practical memory limits
    return max(1, min(cpu_count // 2, 8))


def is_sequential_mode() -> bool:
    """Check if all parallelism is disabled."""
    return os.environ.get("RHEOJAX_SEQUENTIAL", "0") == "1"


def get_worker_isolation() -> str:
    """Get worker isolation mode: 'subprocess' or 'thread'."""
    override = _overrides.get("isolation")
    if override is not None:
        return str(override)
    return os.environ.get("RHEOJAX_WORKER_ISOLATION", "subprocess")


def get_parallel_config() -> dict[str, Any]:
    """Get full parallel configuration as dict."""
    return {
        "n_workers": get_default_workers(),
        "isolation": get_worker_isolation(),
        "sequential": is_sequential_mode(),
        "warm_pool": _overrides.get("warm_pool", False)
        or os.environ.get("RHEOJAX_WARM_POOL", "0") == "1",
    }


def configure(
    n_workers: int | None = None,
    warm_pool: bool = False,
    isolation: str | None = None,
    warmup_models: list[str] | None = None,
) -> None:
    """Override default parallel configuration.

    Call once at application startup. Pass None to reset to auto-detection.
    """
    global _overrides
    _overrides = {}
    if n_workers is not None:
        _overrides["n_workers"] = max(1, n_workers)
    if warm_pool:
        _overrides["warm_pool"] = True
    if isolation is not None:
        _overrides["isolation"] = isolation
    if warmup_models:
        _overrides["warmup_models"] = warmup_models
