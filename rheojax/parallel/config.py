"""Adaptive parallelism configuration.

Auto-detects optimal worker count based on CPU cores, GPU count, and RAM.
All settings overridable via environment variables or configure() API.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Module-level overrides (set via configure()), guarded by _config_lock
_overrides: dict[str, Any] = {}
# RLock so get_parallel_config() can call get_default_workers() (which also
# acquires this lock) while holding it, giving a consistent snapshot.
_config_lock = threading.RLock()

# SYS-04: Cache the JAX GPU device count so jax.devices() is called at most
# once per process.  Initialised to None (sentinel = "not yet queried").
_cached_gpu_count: int | None = None
_gpu_count_lock = threading.Lock()


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _available_cpu_count() -> int:
    """Host CPU count, honoring CPU affinity restrictions when available.

    ``os.cpu_count()`` reports the physical host's core count even when
    this process is pinned to a subset of cores (``taskset``,
    ``--cpuset-cpus``, Kubernetes' static CPU-manager policy on Guaranteed
    pods), which over-caps worker/thread sizing there.
    ``os.sched_getaffinity(0)`` (Linux only) reports the affinity-restricted
    set instead; fall back to ``os.cpu_count()`` on platforms without it
    (macOS, Windows) or if the call itself is unsupported in this sandbox.
    Note this does NOT reflect CFS-quota-based limits (Docker ``--cpus``,
    k8s ``resources.limits.cpu`` without an exclusive-core policy) --
    those throttle scheduling without restricting affinity, so this still
    returns the full core count under a pure CPU quota.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def cap_thread_env_vars(n_workers: int) -> None:
    """Cap BLAS/XLA thread counts to a fair per-worker share.

    Without this, each of ``n_workers`` processes sizes its thread pool
    to the full host core count, oversubscribing the host by ~n_workers x.
    Uses setdefault() so an explicit caller/environment override wins.
    Call this before any BLAS/JAX import in the target process/thread.
    """
    threads_per_worker = str(max(1, _available_cpu_count() // max(1, n_workers)))
    resolved = {}
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, threads_per_worker)
        resolved[var] = os.environ[var]
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    logger.debug(
        "Capped worker thread env vars: n_workers=%d, resolved=%s, xla_flags=%r",
        n_workers,
        resolved,
        os.environ["XLA_FLAGS"],
    )


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

    cpu_count = _available_cpu_count()

    # GPU-aware: each worker needs ~2GB GPU RAM.
    gpu_count = get_gpu_count()

    if gpu_count > 0:
        return min(gpu_count, cpu_count, 4)

    # CPU: half of cores, min 1, max 8 for practical memory limits
    return max(1, min(cpu_count // 2, 8))


def get_gpu_count() -> int:
    """Number of non-CPU JAX devices visible to this process.

    SYS-04: Cache the device-count query — jax.devices() triggers JAX
    initialisation and is expensive when called on every invocation.
    Shared by get_default_workers() (sizing) and PersistentProcessPool
    (per-worker CUDA_VISIBLE_DEVICES pinning).
    """
    global _cached_gpu_count
    with _gpu_count_lock:
        if _cached_gpu_count is None:
            try:
                from rheojax.core.jax_config import safe_import_jax

                jax, _ = safe_import_jax()
                devices = jax.devices()
                _cached_gpu_count = sum(1 for d in devices if d.platform != "cpu")
            except (ImportError, RuntimeError, AttributeError):
                # ImportError: JAX not installed
                # RuntimeError: JAX initialization failed
                # AttributeError: API mismatch
                _cached_gpu_count = 0
        return _cached_gpu_count


def is_sequential_mode() -> bool:
    """Check if all parallelism is disabled.

    Automatically enabled when running under pytest-xdist to prevent
    subprocess multiplication (xdist workers x pool workers -> OOM).
    """
    if os.environ.get("RHEOJAX_SEQUENTIAL", "0") == "1":
        return True
    # pytest-xdist sets PYTEST_XDIST_WORKER in each worker subprocess
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return True
    return False


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
        n_workers_val = get_default_workers()
        return {
            "n_workers": n_workers_val,
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
