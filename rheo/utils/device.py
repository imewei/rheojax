"""GPU detection and warning utilities for Rheo.

This module provides utilities to detect GPU availability and warn users
when they have GPU hardware available but are using CPU-only JAX.
"""

import subprocess
from typing import List, Tuple


def check_gpu_availability() -> None:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if:
    - NVIDIA GPU hardware is detected (nvidia-smi works)
    - But JAX is running in CPU-only mode

    This helps users realize they can enable GPU acceleration for 20-100x speedup.

    Examples
    --------
    Call this at package initialization or in CLI entry points:

    >>> from rheo.utils.device import check_gpu_availability
    >>> check_gpu_availability()  # Prints warning if GPU detected but not used
    """
    try:
        # Check if nvidia-smi detects GPU hardware
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()

            # Check if JAX is using GPU
            import jax

            devices = jax.devices()
            using_gpu = any(
                "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
            )

            if not using_gpu:
                print("\n⚠️  GPU ACCELERATION AVAILABLE")
                print("═══════════════════════════════")
                print(f"NVIDIA GPU detected: {gpu_name}")
                print(f"JAX is currently using: CPU-only")
                print("\nEnable 20-100x speedup with GPU acceleration:")
                print("  make install-jax-gpu")
                print("\nOr manually:")
                print("  pip uninstall -y jax jaxlib")
                print("  pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0")
                print("\nSee README.md GPU Installation section for details.\n")

    except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
        # nvidia-smi not found or JAX not installed - silently skip
        pass
    except Exception:
        # Unexpected error - silently skip to avoid disrupting workflow
        pass


def get_device_info() -> Tuple[List[str], bool]:
    """Get information about available JAX devices.

    Returns
    -------
    devices : List[str]
        List of device descriptions (e.g., ['cpu', 'gpu:0'])
    has_gpu : bool
        True if at least one GPU device is available

    Examples
    --------
    >>> from rheo.utils.device import get_device_info
    >>> devices, has_gpu = get_device_info()
    >>> print(f"Devices: {devices}, Has GPU: {has_gpu}")
    Devices: ['cpu'], Has GPU: False
    """
    try:
        import jax

        devices = jax.devices()
        device_strs = [str(d) for d in devices]
        has_gpu = any(
            "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
        )

        return device_strs, has_gpu

    except ImportError:
        # JAX not installed
        return ["unknown"], False


def get_gpu_memory_info() -> dict:
    """Get GPU memory information using nvidia-smi.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'total_mb': Total GPU memory in MB
        - 'used_mb': Used GPU memory in MB
        - 'free_mb': Free GPU memory in MB
        - 'utilization_percent': GPU utilization percentage

    Returns empty dict if nvidia-smi is not available.

    Examples
    --------
    >>> from rheo.utils.device import get_gpu_memory_info
    >>> info = get_gpu_memory_info()
    >>> if info:
    ...     print(f"GPU Memory: {info['used_mb']}/{info['total_mb']} MB")
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            values = result.stdout.strip().split(",")
            if len(values) >= 4:
                return {
                    "total_mb": int(values[0].strip()),
                    "used_mb": int(values[1].strip()),
                    "free_mb": int(values[2].strip()),
                    "utilization_percent": int(values[3].strip()),
                }

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return {}


def print_device_summary() -> None:
    """Print a summary of available compute devices.

    Displays:
    - JAX version
    - Available devices (CPU/GPU)
    - GPU memory info (if available)
    - Warning if GPU hardware is detected but not being used

    Examples
    --------
    >>> from rheo.utils.device import print_device_summary
    >>> print_device_summary()
    JAX Device Summary
    ==================
    JAX version: 0.8.0
    Devices: [CpuDevice(id=0)]
    Using: CPU-only
    """
    print("\nJAX Device Summary")
    print("==================")

    try:
        import jax

        print(f"JAX version: {jax.__version__}")

        devices = jax.devices()
        print(f"Devices: {devices}")

        has_gpu = any(
            "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
        )

        if has_gpu:
            print("Using: GPU acceleration ✓")

            # Try to get GPU memory info
            mem_info = get_gpu_memory_info()
            if mem_info:
                print(
                    f"GPU Memory: {mem_info['used_mb']}/{mem_info['total_mb']} MB "
                    f"({mem_info['utilization_percent']}% utilized)"
                )
        else:
            print("Using: CPU-only")

            # Check if GPU hardware is available but not being used
            check_gpu_availability()

    except ImportError:
        print("JAX not installed")

    print()
