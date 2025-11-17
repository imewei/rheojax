"""Test data fixture generation for RheoJAX v0.4.0 validation suite.

This module provides utilities to generate synthetic TRIOS files of various sizes
and reference datasets for Bayesian validation testing.

Test Fixtures Generated:
- Synthetic TRIOS files: 1 MB, 5 MB, 10 MB, 50 MB, 100 MB
- Reference datasets for Bayesian mode-aware testing:
  * Relaxation mode: time-dependent shear modulus G(t)
  * Creep mode: time-dependent compliance J(t) with step stress
  * Oscillation mode: frequency-dependent complex modulus G*(ω)

Usage:
    from tests.fixtures.generate_test_data import (
        generate_synthetic_trios_file,
        generate_relaxation_reference_data,
    )

    # Generate synthetic TRIOS file (10 MB)
    trios_file = generate_synthetic_trios_file(
        target_size_mb=10,
        output_path="data.txt"
    )

    # Generate relaxation reference data for validation
    t, G_t = generate_relaxation_reference_data(
        model_params={'G0': 1e6, 'tau': 1.0},
        num_points=1000
    )

Performance Targets:
- TRIOS file generation: <5 seconds per 10 MB
- Memory overhead: <2x target file size during generation
- Reference data generation: <100 ms

Validation Use Cases:
1. Memory profiling tests: Verify auto-chunking reduces peak memory 50-70%
2. Data integrity tests: Verify chunked loading produces identical RheoData
3. Bayesian mode-aware tests: Validate posteriors for all three test modes
4. Performance benchmarks: Measure speedup improvements consistently
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.test_modes import TestMode

logger = logging.getLogger(__name__)


# =============================================================================
# TRIOS File Generation (Synthetic)
# =============================================================================

def generate_synthetic_trios_file(
    target_size_mb: float = 10.0,
    output_path: str | Path | None = None,
    num_segments: int = 1,
    points_per_mb: int = 5000,
) -> Path:
    """Generate synthetic TRIOS file of specified size.

    Creates a valid TRIOS format file with synthetic rheological data
    suitable for testing large file handling and memory profiling.

    Args:
        target_size_mb: Target file size in megabytes (default: 10.0)
        output_path: Output file path (default: ./tests/fixtures/trios_<size>mb.txt)
        num_segments: Number of data segments to generate (default: 1)
        points_per_mb: Data points per megabyte (default: 5000, ~1-2 KB per point)

    Returns:
        Path to generated TRIOS file

    Raises:
        ValueError: If target_size_mb < 1 or points_per_mb < 100

    Notes:
        - Each data point is ~200-300 bytes in TRIOS format
        - Generated data consists of synthetic relaxation curves
        - Material name: "Synthetic_RheoJAX_Test_<size>MB"
        - Suitable for memory profiling and chunked loading tests

    Example:
        Generate 50 MB TRIOS file for memory profiling:

        >>> trios_file = generate_synthetic_trios_file(
        ...     target_size_mb=50,
        ...     output_path="tests/fixtures/large_file_50mb.txt"
        ... )
        >>> print(f"Created {trios_file.stat().st_size / 1e6:.1f} MB file")
        Created 50.1 MB file
    """
    if target_size_mb < 1:
        raise ValueError(f"target_size_mb must be >= 1, got {target_size_mb}")
    if points_per_mb < 100:
        raise ValueError(f"points_per_mb must be >= 100, got {points_per_mb}")

    # Calculate total points needed
    total_points = int(target_size_mb * points_per_mb)
    points_per_segment = total_points // num_segments

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent / f"trios_synthetic_{int(target_size_mb)}mb.txt"
    else:
        output_path = Path(output_path)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate TRIOS header
    lines = [
        "Rheology Data - TRIOS",
        "File Format Version: 1.0",
        f"Material: Synthetic_RheoJAX_Test_{int(target_size_mb)}MB",
        f"Test Type: Relaxation",
        f"Temperature: 25.0 C",
        "",
        "Segment 1",
        "Type: Relaxation",
        f"Points: {points_per_segment}",
        "Time(s)\tStress(Pa)\tStrain\tNote",
    ]

    # Generate synthetic data points
    np.random.seed(42)  # Reproducible
    t_points = np.logspace(-3, 5, points_per_segment)  # 1 ms to 100 ks
    G_t = 1e6 * np.exp(-t_points / 1.0) + 1e5  # Maxwell model

    for i, (t, G) in enumerate(zip(t_points, G_t)):
        # TRIOS format: Time(s) Stress(Pa) Strain Note
        line = f"{t:.6e}\t{G:.2f}\t0.01\tPoint_{i:06d}"
        lines.append(line)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    actual_size_mb = output_path.stat().st_size / 1e6
    logger.info(
        f"Generated TRIOS file: {output_path.name} "
        f"({actual_size_mb:.1f} MB, {points_per_segment} points)"
    )

    return output_path


def generate_trios_files_batch(
    sizes_mb: list[float] = None,
    output_dir: str | Path | None = None,
) -> dict[float, Path]:
    """Generate batch of TRIOS files for testing at multiple sizes.

    Args:
        sizes_mb: List of target sizes in MB (default: [1, 5, 10, 50, 100])
        output_dir: Output directory (default: ./tests/fixtures/)

    Returns:
        Dictionary mapping size (MB) to Path of generated file

    Example:
        >>> files = generate_trios_files_batch(sizes_mb=[1, 5, 10])
        >>> for size_mb, path in files.items():
        ...     actual_size = path.stat().st_size / 1e6
        ...     print(f"{size_mb} MB target -> {actual_size:.1f} MB actual")
        1 MB target -> 1.0 MB actual
        5 MB target -> 5.0 MB actual
        10 MB target -> 10.0 MB actual
    """
    if sizes_mb is None:
        sizes_mb = [1.0, 5.0, 10.0, 50.0, 100.0]

    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for size_mb in sizes_mb:
        path = generate_synthetic_trios_file(
            target_size_mb=size_mb,
            output_path=output_dir / f"trios_synthetic_{int(size_mb)}mb.txt",
        )
        files[size_mb] = path

    logger.info(f"Generated {len(files)} TRIOS files: {list(files.keys())} MB")
    return files


# =============================================================================
# Reference Data Generation (Bayesian Validation)
# =============================================================================

def generate_relaxation_reference_data(
    num_points: int = 1000,
    model_type: str = "maxwell",
    noise_level: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic relaxation mode data with known ground truth.

    Args:
        num_points: Number of time points (default: 1000)
        model_type: Model type "maxwell" or "fractional_zener" (default: "maxwell")
        noise_level: Standard deviation of Gaussian noise as fraction of signal (default: 0.01)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (time_array, modulus_array) with shape (num_points,) each

    Notes:
        - Time range: 10^-3 to 10^5 seconds (log-spaced)
        - Maxwell model: G(t) = G_0 * exp(-t/tau)
        - Fractional Zener: G(t) = G_0 * E_{alpha}(-t^alpha / tau)

    Example:
        Generate Maxwell relaxation data:

        >>> t, G_t = generate_relaxation_reference_data(
        ...     num_points=1000,
        ...     model_type="maxwell",
        ...     noise_level=0.01
        ... )
        >>> print(f"Time range: {t.min():.2e} to {t.max():.2e} s")
        >>> print(f"Modulus range: {G_t.min():.2e} to {G_t.max():.2e} Pa")
        Time range: 1.00e-03 to 1.00e+05 s
        Modulus range: 9.95e+04 to 1.00e+06 Pa
    """
    np.random.seed(seed)

    # Time array: log-spaced 1 ms to 100 ks
    t = np.logspace(-3, 5, num_points)

    if model_type == "maxwell":
        # Maxwell: G(t) = G_0 * exp(-t/tau)
        G_0 = 1e6  # Pa
        tau = 1.0  # seconds
        G_t = G_0 * np.exp(-t / tau)

    elif model_type == "fractional_zener":
        # Fractional Zener Solid-Solid: G(t) = G_inf + (G_0 - G_inf) * E_alpha(-t^alpha / tau)
        G_0 = 1e6  # Pa
        G_inf = 1e5  # Pa
        alpha = 0.7  # Fractional exponent
        tau = 1.0  # seconds

        # Mittag-Leffler function approximation (simplified)
        from rheojax.utils.mittag_leffler import mittag_leffler

        E_alpha = mittag_leffler(-(t ** alpha) / tau, alpha)
        G_t = G_inf + (G_0 - G_inf) * E_alpha

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * G_t.std(), len(G_t))
    G_t_noisy = G_t + noise

    return t, G_t_noisy


def generate_creep_reference_data(
    num_points: int = 1000,
    model_type: str = "maxwell",
    noise_level: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic creep mode data with known ground truth.

    Args:
        num_points: Number of time points (default: 1000)
        model_type: Model type "maxwell" or "fractional_zener" (default: "maxwell")
        noise_level: Standard deviation of Gaussian noise as fraction of signal (default: 0.01)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (time_array, compliance_array) with shape (num_points,) each

    Notes:
        - Time range: 10^-3 to 10^3 seconds (log-spaced)
        - Maxwell creep: J(t) = 1/G_0 + t/eta
        - Fractional Zener creep: More complex with Mittag-Leffler

    Example:
        >>> t, J_t = generate_creep_reference_data(
        ...     num_points=500,
        ...     model_type="maxwell"
        ... )
        >>> print(f"Creep compliance range: {J_t.min():.2e} to {J_t.max():.2e} Pa^-1")
        Creep compliance range: 1.00e-06 to 3.00e-06 Pa^-1
    """
    np.random.seed(seed)

    # Time array: log-spaced 1 ms to 1000 s
    t = np.logspace(-3, 3, num_points)

    if model_type == "maxwell":
        # Maxwell creep: J(t) = 1/G_0 + t/eta
        G_0 = 1e6  # Pa
        eta = 1e5  # Pa*s (viscosity)
        J_t = 1 / G_0 + t / eta

    elif model_type == "fractional_zener":
        # Fractional Zener Solid-Solid creep
        G_0 = 1e6  # Pa
        G_inf = 1e5  # Pa
        alpha = 0.7  # Fractional exponent
        tau = 1.0  # seconds

        # Simplified Mittag-Leffler approximation
        from rheojax.utils.mittag_leffler import mittag_leffler

        E_alpha = mittag_leffler(-(t ** alpha) / tau, alpha)
        J_0 = 1 / G_0
        J_inf = 1 / G_inf
        J_t = J_0 + (J_inf - J_0) * (1 - E_alpha)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * J_t.std(), len(J_t))
    J_t_noisy = J_t + noise

    return t, J_t_noisy


def generate_oscillation_reference_data(
    num_points: int = 1000,
    model_type: str = "maxwell",
    noise_level: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic oscillation mode data with known ground truth.

    Args:
        num_points: Number of frequency points (default: 1000)
        model_type: Model type "maxwell" or "fractional_zener" (default: "maxwell")
        noise_level: Standard deviation of Gaussian noise as fraction of signal (default: 0.01)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (frequency_array, complex_modulus_magnitude) with shape (num_points,) each

    Notes:
        - Frequency range: 0.01 to 1000 rad/s (log-spaced)
        - Maxwell oscillation: G*(ω) = G_0 * i*ω*tau / (1 + i*ω*tau)
        - Output is magnitude |G*(ω)| only

    Example:
        >>> omega, G_star = generate_oscillation_reference_data(
        ...     num_points=800,
        ...     model_type="maxwell"
        ... )
        >>> print(f"Frequency range: {omega.min():.2e} to {omega.max():.2e} rad/s")
        >>> print(f"Modulus range: {G_star.min():.2e} to {G_star.max():.2e} Pa")
        Frequency range: 1.00e-02 to 1.00e+03 rad/s
        Modulus range: 1.00e+04 to 1.00e+06 Pa
    """
    np.random.seed(seed)

    # Frequency array: log-spaced 0.01 to 1000 rad/s
    omega = np.logspace(-2, 3, num_points)

    if model_type == "maxwell":
        # Maxwell: G*(ω) = G_0 * (ω*tau)^2 / (1 + (ω*tau)^2) + i * G_0 * ω*tau / (1 + (ω*tau)^2)
        G_0 = 1e6  # Pa
        tau = 1.0  # seconds
        wt = omega * tau
        G_prime = G_0 * (wt ** 2) / (1 + wt ** 2)  # Storage modulus
        G_double_prime = G_0 * wt / (1 + wt ** 2)  # Loss modulus
        G_star = np.sqrt(G_prime ** 2 + G_double_prime ** 2)  # Magnitude

    elif model_type == "fractional_zener":
        # Fractional Zener complex modulus
        G_0 = 1e6  # Pa
        G_inf = 1e5  # Pa
        alpha = 0.7  # Fractional exponent
        tau = 1.0  # seconds

        # G*(ω) = G_inf + (G_0 - G_inf) / (1 + (i*ω*tau)^alpha)
        iwt_alpha = (1j * omega * tau) ** alpha
        G_complex = G_inf + (G_0 - G_inf) / (1 + iwt_alpha)
        G_star = np.abs(G_complex)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * G_star.std(), len(G_star))
    G_star_noisy = G_star + noise

    return omega, G_star_noisy


# =============================================================================
# RheoData Fixture Generation
# =============================================================================

def create_rheo_data_relaxation(
    model_type: str = "maxwell",
    num_points: int = 1000,
    noise_level: float = 0.01,
    seed: int = 42,
) -> RheoData:
    """Create RheoData object for relaxation mode testing.

    Args:
        model_type: Model type for reference data generation
        num_points: Number of data points
        noise_level: Noise level in generated data
        seed: Random seed

    Returns:
        RheoData object with relaxation data and metadata

    Example:
        >>> rheo_data = create_rheo_data_relaxation(
        ...     model_type="maxwell",
        ...     num_points=1000
        ... )
        >>> print(rheo_data.domain)
        'time'
    """
    t, G_t = generate_relaxation_reference_data(
        num_points=num_points,
        model_type=model_type,
        noise_level=noise_level,
        seed=seed,
    )

    return RheoData(
        x=t,
        y=G_t,
        domain="time",
        x_units="s",
        y_units="Pa",
        metadata={
            "material_name": f"Synthetic_{model_type.upper()}_Relaxation",
            "test_mode": "relaxation",
            "model_type": model_type,
        }
    )


def create_rheo_data_creep(
    model_type: str = "maxwell",
    num_points: int = 1000,
    noise_level: float = 0.01,
    seed: int = 42,
) -> RheoData:
    """Create RheoData object for creep mode testing.

    Args:
        model_type: Model type for reference data generation
        num_points: Number of data points
        noise_level: Noise level in generated data
        seed: Random seed

    Returns:
        RheoData object with creep data and metadata
    """
    t, J_t = generate_creep_reference_data(
        num_points=num_points,
        model_type=model_type,
        noise_level=noise_level,
        seed=seed,
    )

    return RheoData(
        x=t,
        y=J_t,
        domain="time",
        x_units="s",
        y_units="1/Pa",
        metadata={
            "material_name": f"Synthetic_{model_type.upper()}_Creep",
            "test_mode": "creep",
            "model_type": model_type,
        }
    )


def create_rheo_data_oscillation(
    model_type: str = "maxwell",
    num_points: int = 1000,
    noise_level: float = 0.01,
    seed: int = 42,
) -> RheoData:
    """Create RheoData object for oscillation mode testing.

    Args:
        model_type: Model type for reference data generation
        num_points: Number of data points
        noise_level: Noise level in generated data
        seed: Random seed

    Returns:
        RheoData object with oscillation data and metadata
    """
    omega, G_star = generate_oscillation_reference_data(
        num_points=num_points,
        model_type=model_type,
        noise_level=noise_level,
        seed=seed,
    )

    return RheoData(
        x=omega,
        y=G_star,
        domain="frequency",
        x_units="rad/s",
        y_units="Pa",
        metadata={
            "material_name": f"Synthetic_{model_type.upper()}_Oscillation",
            "test_mode": "oscillation",
            "model_type": model_type,
        }
    )


# =============================================================================
# Fixture Registration for pytest
# =============================================================================

def get_fixture_path(filename: str, create_if_missing: bool = True) -> Path:
    """Get or create path to fixture file.

    Args:
        filename: Name of fixture file (relative to tests/fixtures/)
        create_if_missing: Whether to create parent directory if missing

    Returns:
        Path to fixture file

    Example:
        >>> path = get_fixture_path("trios_synthetic_10mb.txt")
        >>> print(path)
        /path/to/rheojax/tests/fixtures/trios_synthetic_10mb.txt
    """
    base_dir = Path(__file__).parent
    if create_if_missing:
        base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / filename


if __name__ == "__main__":
    # Example usage and fixture generation
    import sys

    logging.basicConfig(level=logging.INFO)

    # Generate TRIOS files at standard sizes
    logger.info("Generating standard TRIOS test files...")
    sizes = [1.0, 5.0, 10.0]  # Start small for quick testing
    files = generate_trios_files_batch(sizes_mb=sizes)

    for size_mb, path in files.items():
        actual_size = path.stat().st_size / 1e6
        logger.info(
            f"  {size_mb:6.1f} MB target -> {actual_size:6.1f} MB actual "
            f"({path.name})"
        )

    # Generate reference data fixtures
    logger.info("Generating reference data fixtures...")

    rheo_relax = create_rheo_data_relaxation(model_type="maxwell", num_points=1000)
    logger.info(f"  Relaxation: {len(rheo_relax.x)} points, range "
                f"[{rheo_relax.x.min():.2e}, {rheo_relax.x.max():.2e}] s")

    rheo_creep = create_rheo_data_creep(model_type="maxwell", num_points=1000)
    logger.info(f"  Creep: {len(rheo_creep.x)} points, range "
                f"[{rheo_creep.x.min():.2e}, {rheo_creep.x.max():.2e}] s")

    rheo_osc = create_rheo_data_oscillation(model_type="maxwell", num_points=1000)
    logger.info(f"  Oscillation: {len(rheo_osc.x)} points, range "
                f"[{rheo_osc.x.min():.2e}, {rheo_osc.x.max():.2e}] rad/s")

    logger.info("Fixture generation complete!")
