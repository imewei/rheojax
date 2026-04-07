"""Tests for untested fractional model variants.

Covers 8 fractional models:
1. FractionalBurgersModel (FBM)
2. FractionalJeffreysModel (FJM)
3. FractionalKelvinVoigt (FKV)
4. FractionalKelvinVoigtZener (FKVZ)
5. FractionalMaxwellGel (FMG)
6. FractionalPoyntingThomson (FPT)
7. FractionalZenerLiquidLiquid (FZLL)
8. FractionalZenerSolidSolid (FZSS)
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry

jax, jnp = safe_import_jax()

# Lazy imports to ensure registrations happen
import rheojax.models  # noqa: F401
from rheojax.models.fractional import (
    FBM,
    FKVZ,
    FPT,
    FZLL,
    FZSS,
    FractionalBurgersModel,
    FractionalJeffreysModel,
    FractionalKelvinVoigt,
    FractionalKelvinVoigtZener,
    FractionalMaxwellGel,
    FractionalPoyntingThomson,
    FractionalZenerLiquidLiquid,
    FractionalZenerSolidSolid,
)

# ============================================================================
# Fractional Burgers Model
# ============================================================================


class TestFractionalBurgersModel:
    """Tests for FractionalBurgersModel."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalBurgersModel is not None
        assert FBM is FractionalBurgersModel

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalBurgersModel()
        assert model is not None
        assert not model.fitted_

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalBurgersModel()
        expected = {"Jg", "eta1", "Jk", "alpha", "tau_k"}
        assert set(model.parameters.keys()) == expected
        # Check alpha bounds
        assert model.parameters["alpha"].bounds[0] >= 0.0
        assert model.parameters["alpha"].bounds[1] <= 1.0

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_burgers" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalBurgersModel()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        assert np.all(np.isfinite(result))
        assert len(result) == 20
        # Creep compliance should be positive
        assert np.all(result > 0)

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalBurgersModel()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalBurgersModel()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        assert np.all(np.isfinite(result))
        # Oscillation returns (N, 2) for G', G''
        assert result.shape == (20, 2)


# ============================================================================
# Fractional Jeffreys Model
# ============================================================================


class TestFractionalJeffreysModel:
    """Tests for FractionalJeffreysModel."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalJeffreysModel is not None

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalJeffreysModel()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalJeffreysModel()
        expected = {"eta1", "eta2", "alpha", "tau1"}
        assert set(model.parameters.keys()) == expected

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_jeffreys" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalJeffreysModel()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalJeffreysModel()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        assert np.all(np.isfinite(result))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalJeffreysModel()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        assert np.all(np.isfinite(result))
        assert result.shape == (20, 2)


# ============================================================================
# Fractional Kelvin-Voigt
# ============================================================================


class TestFractionalKelvinVoigt:
    """Tests for FractionalKelvinVoigt."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalKelvinVoigt is not None

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalKelvinVoigt()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalKelvinVoigt()
        expected = {"Ge", "c_alpha", "alpha"}
        assert set(model.parameters.keys()) == expected
        assert model.parameters.get_value("Ge") == 1e6

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_kelvin_voigt" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))
        assert np.all(result_arr > 0)

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalKelvinVoigt()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalKelvinVoigt()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))


# ============================================================================
# Fractional Kelvin-Voigt Zener
# ============================================================================


class TestFractionalKelvinVoigtZener:
    """Tests for FractionalKelvinVoigtZener."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalKelvinVoigtZener is not None
        assert FKVZ is FractionalKelvinVoigtZener

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalKelvinVoigtZener()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalKelvinVoigtZener()
        expected = {"Ge", "Gk", "alpha", "tau"}
        assert set(model.parameters.keys()) == expected
        assert model.parameters.get_value("Ge") == 1000.0
        assert model.parameters.get_value("Gk") == 500.0

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_kv_zener" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalKelvinVoigtZener()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalKelvinVoigtZener()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalKelvinVoigtZener()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        assert np.all(np.isfinite(result))
        assert result.shape == (20, 2)


# ============================================================================
# Fractional Maxwell Gel
# ============================================================================


class TestFractionalMaxwellGel:
    """Tests for FractionalMaxwellGel."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalMaxwellGel is not None

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalMaxwellGel()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalMaxwellGel()
        expected = {"c_alpha", "alpha", "eta"}
        assert set(model.parameters.keys()) == expected
        assert model.parameters["alpha"].bounds[0] >= 0.0

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_maxwell_gel" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalMaxwellGel()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalMaxwellGel()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))


# ============================================================================
# Fractional Poynting-Thomson
# ============================================================================


class TestFractionalPoyntingThomson:
    """Tests for FractionalPoyntingThomson."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalPoyntingThomson is not None
        assert FPT is FractionalPoyntingThomson

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalPoyntingThomson()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalPoyntingThomson()
        expected = {"Ge", "Gk", "alpha", "tau"}
        assert set(model.parameters.keys()) == expected
        assert model.parameters.get_value("Ge") == 1500.0

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_poynting_thomson" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalPoyntingThomson()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        assert np.all(np.isfinite(result))

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalPoyntingThomson()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalPoyntingThomson()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        assert np.all(np.isfinite(result))
        assert result.shape == (20, 2)


# ============================================================================
# Fractional Zener Liquid-Liquid
# ============================================================================


class TestFractionalZenerLiquidLiquid:
    """Tests for FractionalZenerLiquidLiquid."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalZenerLiquidLiquid is not None
        assert FZLL is FractionalZenerLiquidLiquid

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalZenerLiquidLiquid()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalZenerLiquidLiquid()
        expected = {"c1", "c2", "alpha", "beta", "gamma", "tau"}
        assert set(model.parameters.keys()) == expected

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_zener_ll" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalZenerLiquidLiquid()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        assert np.all(np.isfinite(result))
        assert result.shape == (20, 2)

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalZenerLiquidLiquid()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        assert np.all(np.isfinite(result))


# ============================================================================
# Fractional Zener Solid-Solid
# ============================================================================


class TestFractionalZenerSolidSolid:
    """Tests for FractionalZenerSolidSolid."""

    @pytest.mark.smoke
    def test_import(self):
        assert FractionalZenerSolidSolid is not None
        assert FZSS is FractionalZenerSolidSolid

    @pytest.mark.smoke
    def test_instantiation(self):
        model = FractionalZenerSolidSolid()
        assert model is not None

    @pytest.mark.smoke
    def test_parameters(self):
        model = FractionalZenerSolidSolid()
        expected = {"Ge", "Gm", "alpha", "tau_alpha"}
        assert set(model.parameters.keys()) == expected
        assert model.parameters.get_value("Ge") == 1000.0

    @pytest.mark.smoke
    def test_registry(self):
        assert "fractional_zener_ss" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_predict_relaxation(self):
        model = FractionalZenerSolidSolid()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="relaxation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))
        assert np.all(result_arr > 0)

    @pytest.mark.smoke
    def test_predict_creep(self):
        model = FractionalZenerSolidSolid()
        t = np.logspace(-2, 2, 20)
        result = model.predict(t, test_mode="creep")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))

    @pytest.mark.smoke
    def test_predict_oscillation(self):
        model = FractionalZenerSolidSolid()
        omega = np.logspace(-1, 2, 20)
        result = model.predict(omega, test_mode="oscillation")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))

    def test_relaxation_limits(self):
        """Test that G(t=0) ~ Ge + Gm and G(t->inf) ~ Ge."""
        model = FractionalZenerSolidSolid()
        model.parameters.set_value("Ge", 100.0)
        model.parameters.set_value("Gm", 200.0)
        model.parameters.set_value("alpha", 0.5)
        model.parameters.set_value("tau_alpha", 1.0)

        t_short = np.array([1e-4])
        t_long = np.array([1e4])

        G_short = np.asarray(model.predict(t_short, test_mode="relaxation"))
        G_long = np.asarray(model.predict(t_long, test_mode="relaxation"))

        # Short time should be near Ge + Gm = 300
        assert G_short[0] > 200.0
        # Long time should approach Ge = 100
        assert G_long[0] < 200.0
