"""
JAX Utilities
============

JAX device detection and configuration helpers for GUI.
"""

import logging
from typing import Any

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


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
        devices = []
        try:
            for device in jax.devices():
                device_info = {
                    "id": device.id,
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "name": str(device),
                }

                # Try to get memory info for GPU devices
                if device.platform == "gpu":
                    try:
                        stats = device.memory_stats()
                        if stats:
                            device_info["memory_total_mb"] = (
                                stats.get("bytes_limit", 0) / (1024 * 1024)
                            )
                            device_info["memory_used_mb"] = (
                                stats.get("bytes_in_use", 0) / (1024 * 1024)
                            )
                    except Exception:
                        pass

                devices.append(device_info)
        except Exception as e:
            logger.warning(f"Failed to enumerate devices: {e}")
            devices.append({
                "id": 0,
                "platform": "cpu",
                "device_kind": "cpu",
                "name": "CPU (default)",
            })

        return devices

    @staticmethod
    def get_default_device() -> dict[str, Any]:
        """Get default device info.

        Returns
        -------
        dict
            Device information
        """
        try:
            device = jax.devices()[0]
            return {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "name": str(device),
            }
        except Exception as e:
            logger.warning(f"Failed to get default device: {e}")
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
        result = {
            "used_mb": 0.0,
            "total_mb": 0.0,
            "percent": 0.0,
        }

        try:
            device = jax.devices()[0]

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
                except Exception:
                    pass
            else:
                # For CPU, try to get system memory info
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    result["total_mb"] = mem.total / (1024 * 1024)
                    result["used_mb"] = mem.used / (1024 * 1024)
                    result["percent"] = mem.percent
                except ImportError:
                    pass

        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")

        return result

    @staticmethod
    def verify_float64() -> bool:
        """Verify float64 is enabled.

        Returns
        -------
        bool
            True if float64 enabled
        """
        try:
            x = jnp.array([1.0])
            return x.dtype == jnp.float64
        except Exception:
            return False

    @staticmethod
    def get_backend_info() -> dict[str, str]:
        """Get JAX backend information.

        Returns
        -------
        dict
            Backend details (name, version, platform)
        """
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

        except Exception as e:
            logger.warning(f"Failed to get backend info: {e}")
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
    utils = JaxUtils()
    device = utils.get_default_device()
    memory = utils.get_memory_usage()
    backend = utils.get_backend_info()

    return {
        **device,
        "memory": memory,
        "backend": backend,
        "float64_enabled": utils.verify_float64(),
    }


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
    used = memory.get("used_mb", 0)
    total = memory.get("total_mb", 0)
    percent = memory.get("percent", 0)

    if total == 0:
        return "N/A"

    # Convert to GB if large
    if total >= 1024:
        return f"{used / 1024:.1f} GB / {total / 1024:.1f} GB ({percent:.0f}%)"
    else:
        return f"{used:.0f} MB / {total:.0f} MB ({percent:.0f}%)"
