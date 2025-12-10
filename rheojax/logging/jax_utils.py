"""
RheoJAX JAX-Safe Logging Utilities.

Utilities for logging JAX arrays and operations without interfering
with JIT compilation or causing expensive device transfers.
"""

import logging
from typing import Any


def log_array_info(
    arr: Any,
    name: str = "array",
    include_device: bool = True
) -> dict[str, Any]:
    """Extract loggable info from JAX/NumPy array without device transfer.

    This function extracts metadata from arrays (shape, dtype, device)
    without transferring array data from GPU to CPU, making it safe
    to use in performance-critical code.

    Args:
        arr: JAX or NumPy array.
        name: Name for the array in log output.
        include_device: Include device information for JAX arrays.

    Returns:
        Dictionary with array metadata.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((100, 50))
        >>> info = log_array_info(x, "input_data")
        >>> logger.debug("Processing data", **info)
        DEBUG | rheojax | Processing data | name=input_data | shape=(100, 50) | dtype=float32
    """
    info = {
        f"{name}_shape": getattr(arr, "shape", "unknown"),
        f"{name}_dtype": str(getattr(arr, "dtype", "unknown")),
    }

    # Add size information
    if hasattr(arr, "size"):
        info[f"{name}_size"] = arr.size

    # Add device info for JAX arrays (without transferring data)
    if include_device and hasattr(arr, "devices"):
        try:
            devices = arr.devices()
            if devices:
                info[f"{name}_device"] = str(list(devices)[0])
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Could not read device info for %s: %s", name, exc
            )

    return info


def log_array_stats(
    arr: Any,
    name: str = "array",
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG
) -> dict[str, Any]:
    """Compute and log full array statistics.

    WARNING: This function forces a device-to-host transfer for JAX arrays.
    Use only for debugging at DEBUG level.

    Args:
        arr: JAX or NumPy array.
        name: Name for the array.
        logger: Logger to use (optional, for immediate logging).
        level: Log level (default DEBUG).

    Returns:
        Dictionary with array metadata and statistics.

    Example:
        >>> stats = log_array_stats(residuals, "residuals", logger)
        DEBUG | rheojax | Array statistics | name=residuals | min=0.001 | max=0.234 | mean=0.045
    """
    import numpy as np

    # Get basic info first
    info = log_array_info(arr, name, include_device=True)

    try:
        # Convert to numpy (forces transfer)
        arr_np = np.asarray(arr)

        info.update({
            f"{name}_min": float(np.min(arr_np)),
            f"{name}_max": float(np.max(arr_np)),
            f"{name}_mean": float(np.mean(arr_np)),
            f"{name}_std": float(np.std(arr_np)),
            f"{name}_has_nan": bool(np.any(np.isnan(arr_np))),
            f"{name}_has_inf": bool(np.any(np.isinf(arr_np))),
        })
    except Exception as e:
        info[f"{name}_stats_error"] = str(e)

    # Log immediately if logger provided
    if logger is not None:
        logger.log(level, f"Array statistics for {name}", extra=info)

    return info


def jax_safe_log(
    logger: logging.Logger,
    level: int,
    msg: str,
    **kwargs
) -> None:
    """Log only if not inside JAX JIT tracing.

    This function checks if we're currently being traced by JAX JIT
    and skips logging if so, preventing tracing issues.

    Args:
        logger: Logger instance.
        level: Log level.
        msg: Log message.
        **kwargs: Extra context to log.

    Example:
        >>> @jax.jit
        ... def my_function(x):
        ...     jax_safe_log(logger, logging.DEBUG, "Inside JIT", value=x.shape)
        ...     return x * 2
    """
    try:
        import jax.core

        # Check if we're being traced
        # Try cur_sublevel() if available, otherwise use fallback
        cur_sublevel_fn = getattr(jax.core, "cur_sublevel", None)
        if cur_sublevel_fn is not None:
            try:
                sublevel = cur_sublevel_fn()
                if hasattr(sublevel, "level") and sublevel.level > 0:
                    return  # Skip logging during tracing
            except (RuntimeError, AttributeError) as exc:
                logging.getLogger(__name__).debug(
                    "cur_sublevel unavailable during jax tracing check: %s", exc
                )
    except ImportError:
        logging.getLogger(__name__).debug("JAX not available; proceeding with standard logging")

    logger.log(level, msg, **kwargs)


def jax_debug_log(
    logger: logging.Logger,
    msg: str,
    *values: Any,
    level: int = logging.DEBUG
) -> None:
    """Use jax.debug.callback for logging inside JIT-compiled functions.

    This allows logging from within JIT-compiled code using JAX's
    debug callback mechanism.

    Args:
        logger: Logger instance.
        msg: Log message (can include {} placeholders for values).
        *values: Values to log (will be passed through debug.callback).
        level: Log level (default DEBUG).

    Example:
        >>> @jax.jit
        ... def my_function(x):
        ...     y = x * 2
        ...     jax_debug_log(logger, "Computed y with shape {}", y.shape)
        ...     return y
    """
    try:
        import jax

        def _log_callback(*args):
            formatted_msg = msg.format(*args) if args else msg
            logger.log(level, formatted_msg)

        jax.debug.callback(_log_callback, *values)
    except ImportError:
        # JAX not available, fall back to regular logging
        formatted_msg = msg.format(*values) if values else msg
        logger.log(level, formatted_msg)


def log_jax_config(logger: logging.Logger | None = None) -> dict[str, Any]:
    """Log JAX configuration state.

    Logs JAX version, available devices, default backend, and
    float64 configuration.

    Args:
        logger: Logger to use. If provided, logs immediately.

    Returns:
        Dictionary with JAX configuration.

    Example:
        >>> log_jax_config(logger)
        INFO | rheojax | JAX Configuration | jax_version=0.8.0 | devices=['gpu:0'] | float64_enabled=True
    """
    config_info = {}

    try:
        import jax

        config_info = {
            "jax_version": jax.__version__,
            "default_backend": jax.default_backend(),
            "float64_enabled": getattr(jax.config, "jax_enable_x64", None),
        }

        # Get device info
        try:
            devices = jax.devices()
            config_info["devices"] = [str(d) for d in devices]
            config_info["device_count"] = len(devices)
        except Exception:
            config_info["devices"] = ["unavailable"]

        # Get platform info (using non-deprecated API)
        try:
            config_info["platform"] = str(jax.devices()[0].platform)
        except Exception as exc:
            logging.getLogger(__name__).debug("JAX platform lookup failed: %s", exc)

    except ImportError:
        config_info["jax_available"] = False

    if logger is not None:
        logger.info("JAX Configuration", extra=config_info)

    return config_info


def log_numerical_issue(
    logger: logging.Logger,
    arr: Any,
    name: str = "array",
    context: str = ""
) -> bool:
    """Check for and log numerical issues (NaN, Inf) in arrays.

    Args:
        logger: Logger instance.
        arr: Array to check.
        name: Name for the array in log output.
        context: Additional context about where the issue occurred.

    Returns:
        True if numerical issues were found, False otherwise.

    Example:
        >>> if log_numerical_issue(logger, residuals, "residuals", "during fitting"):
        ...     raise ValueError("Numerical instability detected")
    """
    import numpy as np

    try:
        arr_np = np.asarray(arr)
        has_nan = bool(np.any(np.isnan(arr_np)))
        has_inf = bool(np.any(np.isinf(arr_np)))

        if has_nan or has_inf:
            issues = []
            if has_nan:
                nan_count = int(np.sum(np.isnan(arr_np)))
                issues.append(f"NaN ({nan_count} values)")
            if has_inf:
                inf_count = int(np.sum(np.isinf(arr_np)))
                issues.append(f"Inf ({inf_count} values)")

            logger.warning(
                f"Numerical issue detected in {name}",
                extra={
                    "array_name": name,
                    "issues": ", ".join(issues),
                    "shape": arr_np.shape,
                    "context": context,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                }
            )
            return True

    except Exception as e:
        logger.debug(f"Could not check {name} for numerical issues: {e}")

    return False


def log_device_transfer(
    logger: logging.Logger,
    arr: Any,
    name: str = "array",
    target: str = "host"
) -> None:
    """Log when a device transfer occurs.

    Useful for debugging performance issues related to GPU-CPU transfers.

    Args:
        logger: Logger instance.
        arr: Array being transferred.
        name: Name for the array.
        target: Target of transfer (e.g., "host", "gpu:0").

    Example:
        >>> log_device_transfer(logger, result, "model_output", "host")
        DEBUG | rheojax | Device transfer | array=model_output | target=host | size_mb=45.2
    """
    size_bytes = getattr(arr, "nbytes", 0)
    size_mb = size_bytes / (1024 * 1024)

    source = "unknown"
    if hasattr(arr, "devices"):
        try:
            devices = arr.devices()
            if devices:
                source = str(list(devices)[0])
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Could not resolve device source for %s: %s", name, exc
            )

    logger.debug(
        "Device transfer",
        extra={
            "array_name": name,
            "source": source,
            "target": target,
            "size_mb": round(size_mb, 2),
            "shape": getattr(arr, "shape", "unknown"),
        }
    )
