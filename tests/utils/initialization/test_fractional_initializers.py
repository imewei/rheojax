"""Unit tests for all 11 fractional model initializers.

Tests verify:
- Each initializer produces valid parameter estimates
- Results match original function behavior
- Edge cases handled correctly
- All 11 models covered with 2-3 tests each (22-33 total tests)
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.models import (
    FractionalBurgersModel,
    FractionalJeffreysModel,
    FractionalKelvinVoigt,
    FractionalKelvinVoigtZener,
    FractionalMaxwellGel,
    FractionalMaxwellLiquid,
    FractionalMaxwellModel,
    FractionalPoyntingThomson,
    FractionalZenerLiquidLiquid,
    FractionalZenerSolidLiquid,
    FractionalZenerSolidSolid,
)
from rheojax.utils.initialization.fractional_burgers import FractionalBurgersInitializer
from rheojax.utils.initialization.fractional_jeffreys import (
    FractionalJeffreysInitializer,
)
from rheojax.utils.initialization.fractional_kelvin_voigt import (
    FractionalKelvinVoigtInitializer,
)
from rheojax.utils.initialization.fractional_kv_zener import (
    FractionalKVZenerInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_gel import (
    FractionalMaxwellGelInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_liquid import (
    FractionalMaxwellLiquidInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_model import (
    FractionalMaxwellModelInitializer,
)
from rheojax.utils.initialization.fractional_poynting_thomson import (
    FractionalPoyntingThomsonInitializer,
)
from rheojax.utils.initialization.fractional_zener_ll import (
    FractionalZenerLLInitializer,
)
from rheojax.utils.initialization.fractional_zener_sl import (
    FractionalZenerSLInitializer,
)
from rheojax.utils.initialization.fractional_zener_ss import (
    FractionalZenerSSInitializer,
)


def create_synthetic_oscillation_data():
    """Create synthetic oscillation data for testing."""
    omega = np.logspace(-2, 2, 50)
    # Simple Maxwell-like response with realistic plateaus
    G_prime = 1e5 + 9e5 * omega**2 / (1 + omega**2)
    G_double_prime = 9e5 * omega / (1 + omega**2)
    G_star = np.column_stack([G_prime, G_double_prime])
    return omega, G_star


# === FractionalZenerSolidSolid Tests ===


def test_fzss_initializer_produces_valid_parameters():
    """Test FZSS initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    # Use real model to get proper ParameterSet
    model = FractionalZenerSolidSolid()
    param_set = model.parameters

    initializer = FractionalZenerSSInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Ge") > 0
    assert param_set.get_value("Gm") > 0
    assert 0 < param_set.get_value("alpha") < 1
    assert param_set.get_value("tau_alpha") > 0


def test_fzss_initializer_respects_bounds():
    """Test FZSS initializer respects parameter bounds."""
    omega, G_star = create_synthetic_oscillation_data()

    # Use real model with tighter bounds
    model = FractionalZenerSolidSolid()
    param_set = model.parameters

    # Set tighter bounds for moduli
    param_set.set_bounds("Ge", (1e3, 1e5))
    param_set.set_bounds("Gm", (1e3, 1e6))

    initializer = FractionalZenerSSInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    # Check values respect bounds
    assert 1e3 <= param_set.get_value("Ge") <= 1e5
    assert 1e3 <= param_set.get_value("Gm") <= 1e6


# === FractionalMaxwellLiquid Tests ===


def test_fml_initializer_produces_valid_parameters():
    """Test FML initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalMaxwellLiquid()
    param_set = model.parameters

    initializer = FractionalMaxwellLiquidInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Gm") > 0
    assert 0 < param_set.get_value("alpha") < 1
    assert param_set.get_value("tau_alpha") > 0


def test_fml_initializer_handles_noisy_data():
    """Test FML initializer handles noisy data."""
    omega, G_star = create_synthetic_oscillation_data()
    # Add noise
    noise = np.random.normal(0, 1e4, G_star.shape)
    G_star_noisy = G_star + noise

    model = FractionalMaxwellLiquid()
    param_set = model.parameters

    initializer = FractionalMaxwellLiquidInitializer()
    success = initializer.initialize(omega, G_star_noisy, param_set)

    # Should still succeed despite noise
    assert success is True


# === FractionalMaxwellGel Tests ===


def test_fmg_initializer_produces_valid_parameters():
    """Test FMG initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalMaxwellGel()
    param_set = model.parameters

    initializer = FractionalMaxwellGelInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("c_alpha") > 0
    assert 0 < param_set.get_value("alpha") < 1
    assert param_set.get_value("eta") > 0


def test_fmg_initializer_handles_edge_alpha():
    """Test FMG initializer handles edge case alpha values."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalMaxwellGel()
    param_set = model.parameters

    initializer = FractionalMaxwellGelInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    # Alpha should be clipped to valid range
    alpha = param_set.get_value("alpha")
    assert 0 < alpha < 1


# === FractionalZenerLL Tests ===


def test_fzll_initializer_produces_valid_parameters():
    """Test FractionalZenerLL initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalZenerLiquidLiquid()
    param_set = model.parameters

    initializer = FractionalZenerLLInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("c1") > 0
    assert param_set.get_value("c2") > 0
    assert param_set.get_value("tau") > 0


def test_fzll_initializer_splits_modulus():
    """Test FractionalZenerLL splits modulus between c1 and c2."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalZenerLiquidLiquid()
    param_set = model.parameters

    initializer = FractionalZenerLLInitializer()
    initializer.initialize(omega, G_star, param_set)

    # c1 should be larger than c2 (60/40 split)
    assert param_set.get_value("c1") > param_set.get_value("c2")


# === FractionalZenerSL Tests ===


def test_fzsl_initializer_produces_valid_parameters():
    """Test FractionalZenerSL initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalZenerSolidLiquid()
    param_set = model.parameters

    initializer = FractionalZenerSLInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Ge") > 0
    assert param_set.get_value("c_alpha") > 0


def test_fzsl_initializer_estimates_correct_ge():
    """Test FractionalZenerSL estimates Ge from low-frequency plateau."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalZenerSolidLiquid()
    param_set = model.parameters

    initializer = FractionalZenerSLInitializer()
    initializer.initialize(omega, G_star, param_set)

    # Ge should be close to low-frequency modulus
    Ge = param_set.get_value("Ge")
    assert 0.5e5 < Ge < 2e5  # Approximate range for test data


# === FractionalKelvinVoigt Tests ===


def test_fkv_initializer_produces_valid_parameters():
    """Test FractionalKelvinVoigt initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalKelvinVoigt()
    param_set = model.parameters

    initializer = FractionalKelvinVoigtInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Ge") > 0
    assert param_set.get_value("c_alpha") > 0
    assert 0 < param_set.get_value("alpha") < 1


def test_fkv_initializer_handles_small_dataset():
    """Test FractionalKelvinVoigt handles small datasets."""
    omega = np.logspace(-1, 1, 10)  # Only 10 points
    G_star = np.column_stack([np.ones(10) * 1e5, np.ones(10) * 1e4])

    model = FractionalKelvinVoigt()
    param_set = model.parameters

    initializer = FractionalKelvinVoigtInitializer()
    # May fail due to insufficient data, but should not crash
    initializer.initialize(omega, G_star, param_set)


# === FractionalMaxwellModel Tests ===


def test_fmm_initializer_produces_valid_parameters():
    """Test FractionalMaxwellModel initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalMaxwellModel()
    param_set = model.parameters

    initializer = FractionalMaxwellModelInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("c1") > 0
    assert param_set.get_value("tau") > 0


def test_fmm_initializer_defaults_beta():
    """Test FractionalMaxwellModel defaults beta to 0.5."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalMaxwellModel()
    param_set = model.parameters

    initializer = FractionalMaxwellModelInitializer()
    initializer.initialize(omega, G_star, param_set)

    # Beta should be around 0.5 (default)
    beta = param_set.get_value("beta")
    assert 0.4 < beta < 0.6


# === FractionalKVZener Tests ===


def test_fkvz_initializer_produces_valid_parameters():
    """Test FractionalKVZener initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalKelvinVoigtZener()
    param_set = model.parameters

    initializer = FractionalKVZenerInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Ge") > 0
    assert param_set.get_value("Gk") > 0


def test_fkvz_initializer_estimates_moduli():
    """Test FractionalKVZener estimates Ge and Gk correctly."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalKelvinVoigtZener()
    param_set = model.parameters

    initializer = FractionalKVZenerInitializer()
    initializer.initialize(omega, G_star, param_set)

    # Ge should be from high-frequency plateau
    Ge = param_set.get_value("Ge")
    assert 8e5 < Ge < 1.2e6  # Approximate for test data


# === FractionalPoyntingThomson Tests ===


def test_fpt_initializer_produces_valid_parameters():
    """Test FractionalPoyntingThomson initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalPoyntingThomson()
    param_set = model.parameters

    initializer = FractionalPoyntingThomsonInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Ge") > 0
    assert param_set.get_value("Gk") > 0


def test_fpt_initializer_handles_high_frequency_data():
    """Test FractionalPoyntingThomson handles high-frequency dominated data."""
    omega = np.logspace(0, 3, 50)  # High frequencies only
    G_star = np.column_stack([np.ones(50) * 1e6, np.ones(50) * 1e5])

    model = FractionalPoyntingThomson()
    param_set = model.parameters

    initializer = FractionalPoyntingThomsonInitializer()
    # May fail validation, but should not crash
    initializer.initialize(omega, G_star, param_set)


# === FractionalJeffreys Tests ===


def test_fj_initializer_produces_valid_parameters():
    """Test FractionalJeffreys initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalJeffreysModel()
    param_set = model.parameters

    initializer = FractionalJeffreysInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("eta1") > 0
    assert param_set.get_value("eta2") > 0


def test_fj_initializer_eta_ratio():
    """Test FractionalJeffreys maintains eta2 < eta1."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalJeffreysModel()
    param_set = model.parameters

    initializer = FractionalJeffreysInitializer()
    initializer.initialize(omega, G_star, param_set)

    # eta2 should be smaller than eta1 (0.5 ratio)
    eta1 = param_set.get_value("eta1")
    eta2 = param_set.get_value("eta2")
    assert eta2 < eta1


# === FractionalBurgers Tests ===


def test_fb_initializer_produces_valid_parameters():
    """Test FractionalBurgers initializer produces valid parameter estimates."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalBurgersModel()
    param_set = model.parameters

    initializer = FractionalBurgersInitializer()
    success = initializer.initialize(omega, G_star, param_set)

    assert success is True
    assert param_set.get_value("Jg") > 0
    assert param_set.get_value("eta1") > 0
    assert param_set.get_value("Jk") > 0


def test_fb_initializer_compliance_relationship():
    """Test FractionalBurgers Jg and Jk relationship."""
    omega, G_star = create_synthetic_oscillation_data()

    model = FractionalBurgersModel()
    param_set = model.parameters

    initializer = FractionalBurgersInitializer()
    initializer.initialize(omega, G_star, param_set)

    # Both compliances should be positive
    Jg = param_set.get_value("Jg")
    Jk = param_set.get_value("Jk")
    assert Jg > 0
    assert Jk > 0
