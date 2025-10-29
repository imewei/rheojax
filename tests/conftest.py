"""Shared fixtures and configuration for rheo test suite.

This module provides:
- Shared test data fixtures for common rheological patterns
- RheoData fixtures for different test modes (oscillation, relaxation, creep)
- Mock file data for I/O testing
- Parameters fixtures for model testing
- Registry fixtures for model/transform discovery testing
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheo.core.data import RheoData
from rheo.core.jax_config import safe_import_jax
from rheo.core.parameters import Parameter, ParameterSet
from rheo.core.test_modes import TestMode

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# =============================================================================
# OSCILLATORY TEST DATA (SAOS - Small Amplitude Oscillatory Shear)
# =============================================================================


@pytest.fixture
def oscillation_data_simple():
    """Simple oscillatory data for basic testing.

    Returns RheoData with frequency sweep data typical of SAOS experiments.
    - 10 frequency points from 0.1 to 10 rad/s
    - Realistic moduli values (G' and G'' combined as complex modulus)
    - Linear viscoelastic behavior
    """
    frequency = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

    # Simulated Maxwell model response: G(ω) = Gₛωη_s² / (1 + (ωη_s)²)
    # With Gₛ = 1e5 Pa, η_s = 10 Pa.s
    G_s = 1e5
    eta_s = 10.0
    omega = 2 * np.pi * frequency

    # Storage modulus G'
    G_prime = G_s * (omega * eta_s) ** 2 / (1 + (omega * eta_s) ** 2)

    # Loss modulus G''
    G_double_prime = G_s * omega * eta_s / (1 + (omega * eta_s) ** 2)

    # Combine into complex modulus
    G_complex = G_prime + 1j * G_double_prime

    metadata = {
        "test_mode": "oscillation",
        "test_type": "SAOS",
        "temperature": 25.0,
        "temperature_unit": "C",
        "strain_amplitude": 0.01,
        "frequency_range": (0.1, 100),
        "sample_name": "test_sample",
        "instrument": "test_instrument",
    }

    return RheoData(
        x=frequency,
        y=G_complex,
        x_units="Hz",
        y_units="Pa",
        domain="frequency",
        metadata=metadata,
    )


@pytest.fixture
def oscillation_data_large():
    """Large oscillatory dataset with more frequency points.

    Useful for testing FFT-based transforms and performance.
    - 100 logarithmically spaced frequency points
    - Broader frequency range: 1e-3 to 1e3 rad/s
    """
    frequency = np.logspace(-3, 3, 100)

    # Zener model response (more complex than Maxwell)
    G_s = 1e5  # Spring modulus
    G_p = 5e4  # Parallel modulus
    eta_p = 100.0  # Parallel viscosity

    omega = 2 * np.pi * frequency

    # Complex modulus for Zener (standard linear solid)
    numerator = G_s * (1j * omega * eta_p + (G_s + G_p))
    denominator = 1j * omega * eta_p + G_p
    G_complex = numerator / denominator

    metadata = {
        "test_mode": "oscillation",
        "test_type": "SAOS",
        "temperature": 20.0,
        "temperature_unit": "C",
        "strain_amplitude": 0.02,
        "sample_name": "large_dataset",
        "instrument": "rheometer_v2",
    }

    return RheoData(
        x=frequency,
        y=G_complex,
        x_units="Hz",
        y_units="Pa",
        domain="frequency",
        metadata=metadata,
    )


# =============================================================================
# RELAXATION TEST DATA
# =============================================================================


@pytest.fixture
def relaxation_data_simple():
    """Simple stress relaxation data.

    Returns RheoData with time-domain relaxation curves.
    - Exponential relaxation response
    - 50 time points from 0.1 to 100 seconds
    """
    time = np.logspace(-1, 2, 50)

    # Single exponential relaxation: σ(t) = σ₀ * exp(-t/τ)
    tau = 1.0  # Relaxation time
    stress = 1e5 * np.exp(-time / tau)

    metadata = {
        "test_mode": "relaxation",
        "test_type": "Stress Relaxation",
        "temperature": 25.0,
        "temperature_unit": "C",
        "strain_amplitude": 0.01,
        "sample_name": "relaxation_sample",
        "instrument": "test_instrument",
    }

    return RheoData(
        x=time, y=stress, x_units="s", y_units="Pa", domain="time", metadata=metadata
    )


@pytest.fixture
def relaxation_data_multi_mode():
    """Multi-mode (Generalized Maxwell) relaxation data.

    Multiple relaxation times simulating a polymer melt.
    """
    time = np.logspace(-2, 3, 100)

    # Generalized Maxwell with 3 modes
    # σ(t) = Σ Gᵢ * exp(-t/τᵢ)
    G = np.array([5e4, 3e4, 2e4])
    tau = np.array([0.01, 0.1, 1.0])

    stress = np.zeros_like(time)
    for g, t in zip(G, tau):
        stress += g * np.exp(-time / t)

    metadata = {
        "test_mode": "relaxation",
        "test_type": "Stress Relaxation",
        "temperature": 25.0,
        "temperature_unit": "C",
        "sample_name": "multi_mode_relaxation",
        "num_modes": 3,
    }

    return RheoData(
        x=time, y=stress, x_units="s", y_units="Pa", domain="time", metadata=metadata
    )


# =============================================================================
# CREEP TEST DATA
# =============================================================================


@pytest.fixture
def creep_data_simple():
    """Simple creep compliance data.

    Returns RheoData with creep response.
    - Power-law creep: J(t) = 1/G₀ + (t/K)^n
    """
    time = np.logspace(-1, 3, 50)

    # Creep compliance: combination of elastic and viscous response
    # J(t) = 1/G₀ + t/η
    G_0 = 1e5
    eta = 1e3

    compliance = 1.0 / G_0 + time / eta

    metadata = {
        "test_mode": "creep",
        "test_type": "Creep Compliance",
        "temperature": 25.0,
        "temperature_unit": "C",
        "applied_stress": 1000.0,
        "sample_name": "creep_sample",
    }

    return RheoData(
        x=time,
        y=compliance,
        x_units="s",
        y_units="Pa^-1",
        domain="time",
        metadata=metadata,
    )


# =============================================================================
# STEADY SHEAR / FLOW DATA
# =============================================================================


@pytest.fixture
def flow_data_power_law():
    """Power-law flow behavior (shear thinning).

    Viscosity data as function of shear rate.
    η(γ̇) = K * γ̇^(n-1)
    """
    shear_rate = np.logspace(-2, 2, 50)

    # Power-law parameters
    K = 1000.0  # Consistency index
    n = 0.5  # Flow index (shear thinning: n < 1)

    viscosity = K * shear_rate ** (n - 1)

    metadata = {
        "test_mode": "rotation",
        "test_type": "Steady Shear",
        "temperature": 25.0,
        "temperature_unit": "C",
        "sample_name": "power_law_fluid",
        "model_params": {"K": K, "n": n},
    }

    return RheoData(
        x=shear_rate,
        y=viscosity,
        x_units="1/s",
        y_units="Pa.s",
        domain="time",  # Shear rate is quasi-static
        metadata=metadata,
    )


@pytest.fixture
def flow_data_bingham():
    """Bingham plastic flow behavior.

    τ = τ₀ + η_p * γ̇ (for τ > τ₀)
    """
    shear_rate = np.logspace(-1, 2, 50)

    # Bingham parameters
    tau_0 = 100.0  # Yield stress
    eta_p = 1.0  # Plastic viscosity

    # Avoid zero shear rate region
    shear_rate = shear_rate[shear_rate > 0.1]

    stress = tau_0 + eta_p * shear_rate
    viscosity = stress / shear_rate

    metadata = {
        "test_mode": "rotation",
        "test_type": "Steady Shear",
        "temperature": 25.0,
        "temperature_unit": "C",
        "sample_name": "bingham_fluid",
    }

    return RheoData(
        x=shear_rate,
        y=viscosity,
        x_units="1/s",
        y_units="Pa.s",
        domain="time",
        metadata=metadata,
    )


# =============================================================================
# PARAMETER FIXTURES
# =============================================================================


@pytest.fixture
def maxwell_parameters():
    """Parameters for Maxwell model.

    Returns ParameterSet with Gₛ and η_s.
    """
    params = ParameterSet()
    params.add(
        "G_s", value=1e5, bounds=(1e3, 1e8), units="Pa", description="Spring modulus"
    )
    params.add(
        "eta_s",
        value=10.0,
        bounds=(0.01, 1e6),
        units="Pa.s",
        description="Dashpot viscosity",
    )
    return params


@pytest.fixture
def zener_parameters():
    """Parameters for Zener (standard linear solid) model."""
    params = ParameterSet()
    params.add(
        "G_s", value=1e5, bounds=(1e3, 1e8), units="Pa", description="Spring modulus"
    )
    params.add(
        "G_p",
        value=5e4,
        bounds=(1e3, 1e8),
        units="Pa",
        description="Parallel spring modulus",
    )
    params.add(
        "eta_p",
        value=100.0,
        bounds=(0.1, 1e6),
        units="Pa.s",
        description="Parallel damper viscosity",
    )
    return params


@pytest.fixture
def power_law_parameters():
    """Parameters for power-law flow model."""
    params = ParameterSet()
    params.add(
        "K",
        value=1000.0,
        bounds=(0.1, 1e6),
        units="Pa.s^n",
        description="Consistency index",
    )
    params.add(
        "n",
        value=0.5,
        bounds=(0.0, 1.0),
        units="dimensionless",
        description="Flow index",
    )
    return params


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================


@pytest.fixture
def synthetic_noisy_data():
    """Generate synthetic noisy data for robustness testing.

    Returns (clean_data, noisy_data) tuple.
    """
    x = np.logspace(-1, 1, 20)

    # Clean exponential decay
    y_clean = 1e5 * np.exp(-x / 1.0)

    # Add Gaussian noise (5% of signal)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05 * y_clean, y_clean.shape)
    y_noisy = y_clean + noise

    clean = RheoData(x=x, y=y_clean, x_units="s", y_units="Pa", domain="time")
    noisy = RheoData(
        x=x,
        y=y_noisy,
        x_units="s",
        y_units="Pa",
        domain="time",
        metadata={"noise_level": 0.05},
    )

    return clean, noisy


@pytest.fixture
def synthetic_multi_temperature_data():
    """Generate synthetic data at multiple temperatures.

    Useful for testing mastercurve and multi-technique fitting.
    """
    temperatures = np.array([20.0, 30.0, 40.0, 50.0])
    frequency = np.logspace(-2, 2, 30)

    datasets = []
    for T in temperatures:
        # Temperature-dependent shift factor (WLF-like)
        log_aT = -0.1 * (T - 30.0)  # Reference at 30°C
        shifted_freq = frequency * 10**log_aT

        # Create complex modulus data
        G_s = 1e5
        eta_s = 10.0 * 10 ** (log_aT / 2)  # Temperature-dependent viscosity
        omega = 2 * np.pi * shifted_freq

        G_prime = G_s * (omega * eta_s) ** 2 / (1 + (omega * eta_s) ** 2)
        G_double_prime = G_s * omega * eta_s / (1 + (omega * eta_s) ** 2)

        G_complex = G_prime + 1j * G_double_prime

        metadata = {
            "test_mode": "oscillation",
            "temperature": T,
            "temperature_unit": "C",
        }

        data = RheoData(
            x=frequency,
            y=G_complex,
            x_units="Hz",
            y_units="Pa",
            domain="frequency",
            metadata=metadata,
        )
        datasets.append(data)

    return datasets


# =============================================================================
# FILE I/O TEST DATA
# =============================================================================


@pytest.fixture
def csv_file_data():
    """Create a temporary CSV file for testing readers."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Write CSV with headers
        f.write("frequency,G_prime,G_double_prime\n")
        frequency = np.logspace(-1, 1, 10)
        G_s = 1e5
        eta_s = 10.0
        omega = 2 * np.pi * frequency

        G_prime = G_s * (omega * eta_s) ** 2 / (1 + (omega * eta_s) ** 2)
        G_double_prime = G_s * omega * eta_s / (1 + (omega * eta_s) ** 2)

        for freq, gp, gdp in zip(frequency, G_prime, G_double_prime):
            f.write(f"{freq},{gp},{gdp}\n")

        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def json_file_data():
    """Create a temporary JSON file for testing serialization."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "x": [0.1, 1.0, 10.0],
            "y": [100.0, 1000.0, 10000.0],
            "x_units": "Hz",
            "y_units": "Pa",
            "domain": "frequency",
            "metadata": {"test_mode": "oscillation", "temperature": 25.0},
        }
        json.dump(data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# REGISTRY AND PLUGIN FIXTURES
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def load_all_plugins():
    """Load all models and transforms at the start of test session.

    This ensures that all plugins are registered before any tests run,
    preventing registry isolation issues in the full test suite.
    """
    # Import all model modules to trigger registration
    import rheo.models

    # Import all transform modules to trigger registration
    import rheo.transforms

    yield


@pytest.fixture
def clean_registries():
    """Provide clean model and transform registries for testing.

    Useful for testing registry functionality without side effects.
    """
    from rheo.core.registry import ModelRegistry, TransformRegistry

    # Save original registries
    original_models = ModelRegistry._models.copy()
    original_transforms = TransformRegistry._transforms.copy()

    # Clear registries
    ModelRegistry._models.clear()
    TransformRegistry._transforms.clear()

    yield ModelRegistry, TransformRegistry

    # Restore original registries (update dict in place to preserve singleton)
    ModelRegistry._models.clear()
    ModelRegistry._models.update(original_models)
    TransformRegistry._transforms.clear()
    TransformRegistry._transforms.update(original_transforms)


# =============================================================================
# NUMPY VS JAX COMPARISON FIXTURES
# =============================================================================


@pytest.fixture
def array_pair_numpy_jax():
    """Provide matching numpy and JAX arrays for comparison testing."""
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    jax_array = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return {"numpy": np_array, "jax": jax_array}


@pytest.fixture
def complex_array_pair_numpy_jax():
    """Complex arrays for frequency domain testing."""
    np_array = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
    jax_array = jnp.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])

    return {"numpy": np_array, "jax": jax_array}


# =============================================================================
# PYTEST CONFIGURATION AND MARKERS
# =============================================================================


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "edge_case: mark test as testing edge cases")
    config.addinivalue_line("markers", "io: mark test as I/O related")
    config.addinivalue_line(
        "markers", "performance: mark test as performance/benchmark"
    )
    config.addinivalue_line("markers", "jax: mark test as JAX-specific")


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_jax_config():
    """Reset JAX configuration after each test to avoid state leakage."""
    yield
    # Optional: Add any JAX state reset if needed
    # For now, this is a placeholder for future use
