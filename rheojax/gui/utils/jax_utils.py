"""
JAX Utilities
============

JAX device detection and configuration helpers for GUI.
"""

from typing import Any

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


class JaxUtils:
    """JAX utility functions for GUI.

    Features:
        - Device detection
        - Memory monitoring
        - Backend configuration
        - Float64 verification

    Example
    -------
    >>> utils = JaxUtils()
    >>> devices = utils.get_devices()
    >>> memory = utils.get_memory_usage()
    """

    @staticmethod
    def get_devices() -> list[dict[str, Any]]:
        """Get available JAX devices.

        Returns
        -------
        list[dict]
            Device information (type, name, memory)
        """
        logger.debug("Entering get_devices")
        devices = []
        try:
            for device in jax.devices():
                device_info = {
                    "id": device.id,
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "name": str(device),
                }
                logger.debug(
                    "Found device",
                    device_id=device.id,
                    platform=device.platform,
                    device_kind=device.device_kind,
                )

                # Try to get memory info for GPU devices
                if device.platform == "gpu":
                    try:
                        stats = device.memory_stats()
                        if stats:
                            device_info["memory_total_mb"] = stats.get(
                                "bytes_limit", 0
                            ) / (1024 * 1024)
                            device_info["memory_used_mb"] = stats.get(
                                "bytes_in_use", 0
                            ) / (1024 * 1024)
                            logger.debug(
                                "GPU memory stats retrieved",
                                device_id=device.id,
                                memory_total_mb=device_info["memory_total_mb"],
                                memory_used_mb=device_info["memory_used_mb"],
                            )
                    except Exception as exc:
                        logger.debug("GPU memory stats unavailable: %s", exc)

                devices.append(device_info)
        except Exception as e:
            logger.error("Failed to enumerate devices", exc_info=True)
            devices.append(
                {
                    "id": 0,
                    "platform": "cpu",
                    "device_kind": "cpu",
                    "name": "CPU (default)",
                }
            )

        logger.debug("get_devices complete", device_count=len(devices))
        return devices

    @staticmethod
    def get_default_device() -> dict[str, Any]:
        """Get default device info.

        Returns
        -------
        dict
            Device information
        """
        logger.debug("Entering get_default_device")
        try:
            device = jax.devices()[0]
            result = {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "name": str(device),
            }
            logger.debug(
                "Default device retrieved",
                device_id=result["id"],
                platform=result["platform"],
            )
            return result
        except Exception as e:
            logger.error("Failed to get default device", exc_info=True)
            return {
                "id": 0,
                "platform": "cpu",
                "device_kind": "cpu",
                "name": "CPU (default)",
            }

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        """Get device memory usage.

        Returns
        -------
        dict
            Memory stats (used_mb, total_mb, percent)
        """
        logger.debug("Entering get_memory_usage")
        result = {
            "used_mb": 0.0,
            "total_mb": 0.0,
            "percent": 0.0,
        }

        try:
            device = jax.devices()[0]
            logger.debug("Checking memory for device", platform=device.platform)

            if device.platform == "gpu":
                try:
                    stats = device.memory_stats()
                    if stats:
                        total = stats.get("bytes_limit", 0)
                        used = stats.get("bytes_in_use", 0)
                        result["total_mb"] = total / (1024 * 1024)
                        result["used_mb"] = used / (1024 * 1024)
                        if total > 0:
                            result["percent"] = (used / total) * 100
                        logger.debug(
                            "GPU memory usage retrieved",
                            used_mb=result["used_mb"],
                            total_mb=result["total_mb"],
                            percent=result["percent"],
                        )
                except Exception as exc:
                    logger.debug("GPU memory stats unavailable: %s", exc)
            else:
                # For CPU, try to get system memory info
                try:
                    import psutil

                    mem = psutil.virtual_memory()
                    result["total_mb"] = mem.total / (1024 * 1024)
                    result["used_mb"] = mem.used / (1024 * 1024)
                    result["percent"] = mem.percent
                    logger.debug(
                        "CPU memory usage retrieved via psutil",
                        used_mb=result["used_mb"],
                        total_mb=result["total_mb"],
                        percent=result["percent"],
                    )
                except ImportError:
                    logger.debug("psutil not installed; skipping CPU memory stats")

        except Exception as e:
            logger.error("Failed to get memory usage", exc_info=True)

        logger.debug("get_memory_usage complete", result=result)
        return result

    @staticmethod
    def verify_float64() -> bool:
        """Verify float64 is enabled.

        Returns
        -------
        bool
            True if float64 enabled
        """
        logger.debug("Entering verify_float64")
        try:
            x = jnp.array([1.0])
            is_float64 = x.dtype == jnp.float64
            logger.debug("Float64 verification complete", enabled=is_float64, dtype=str(x.dtype))
            return is_float64
        except Exception as e:
            logger.error("Failed to verify float64", exc_info=True)
            return False

    @staticmethod
    def get_backend_info() -> dict[str, str]:
        """Get JAX backend information.

        Returns
        -------
        dict
            Backend details (name, version, platform)
        """
        logger.debug("Entering get_backend_info")
        result = {
            "name": "unknown",
            "version": "unknown",
            "platform": "unknown",
        }

        try:
            result["version"] = jax.__version__

            device = jax.devices()[0]
            result["platform"] = device.platform

            if device.platform == "gpu":
                result["name"] = "JAX (GPU)"
            elif device.platform == "tpu":
                result["name"] = "JAX (TPU)"
            else:
                result["name"] = "JAX (CPU)"

            logger.debug(
                "Backend info retrieved",
                name=result["name"],
                version=result["version"],
                platform=result["platform"],
            )

        except Exception as e:
            logger.error("Failed to get backend info", exc_info=True)
            result["name"] = "JAX"

        return result


# Module-level convenience functions
def get_jax_device_info() -> dict[str, Any]:
    """Get information about the default JAX device.

    Returns
    -------
    dict
        Device information including platform, name, and memory stats
    """
    logger.debug("Entering get_jax_device_info")
    utils = JaxUtils()
    device = utils.get_default_device()
    memory = utils.get_memory_usage()
    backend = utils.get_backend_info()

    result = {
        **device,
        "memory": memory,
        "backend": backend,
        "float64_enabled": utils.verify_float64(),
    }
    logger.debug("get_jax_device_info complete", platform=result.get("platform"))
    return result


def get_jax_info() -> dict[str, Any]:
    """Get JAX configuration info for HomePage display.

    This function returns a flat structure with keys expected by
    the HomePage._update_jax_status() method.

    Returns
    -------
    dict[str, Any]
        JAX configuration with keys:
        - devices: List of device name strings
        - default_device: Current default device name
        - memory_used_mb: Memory used in MB (float)
        - memory_total_mb: Total memory in MB (float)
        - float64_enabled: Whether float64 is enabled
        - jit_cache_count: Number of JIT-compiled functions (always 0, not exposed by JAX)

    Example
    -------
    >>> info = get_jax_info()
    >>> print(info["default_device"])
    'TFRT_CPU_0'
    """
    logger.debug("Entering get_jax_info")
    utils = JaxUtils()
    devices = utils.get_devices()
    memory = utils.get_memory_usage()

    # Build device name list
    device_names = [d.get("name", "cpu") for d in devices] if devices else ["cpu"]

    # Get default device name
    default_device = devices[0].get("name", "cpu") if devices else "cpu"

    result = {
        "devices": device_names,
        "default_device": default_device,
        "memory_used_mb": memory.get("used_mb", 0.0),
        "memory_total_mb": memory.get("total_mb", 0.0),
        "float64_enabled": utils.verify_float64(),
        "jit_cache_count": 0,  # JAX doesn't expose JIT cache count directly
    }
    logger.debug(
        "get_jax_info complete",
        device_count=len(device_names),
        default_device=default_device,
        float64_enabled=result["float64_enabled"],
    )
    return result


def format_memory_usage(memory: dict[str, float]) -> str:
    """Format memory usage for display.

    Parameters
    ----------
    memory : dict
        Memory stats from get_memory_usage()

    Returns
    -------
    str
        Formatted memory string (e.g., "1.2 GB / 8.0 GB (15%)")
    """
    logger.debug("Entering format_memory_usage", memory=memory)
    used = memory.get("used_mb", 0)
    total = memory.get("total_mb", 0)
    percent = memory.get("percent", 0)

    if total == 0:
        logger.debug("Total memory is 0, returning N/A")
        return "N/A"

    # Convert to GB if large
    if total >= 1024:
        formatted = f"{used / 1024:.1f} GB / {total / 1024:.1f} GB ({percent:.0f}%)"
    else:
        formatted = f"{used:.0f} MB / {total:.0f} MB ({percent:.0f}%)"
    logger.debug("Memory formatted", result=formatted)
    return formatted
