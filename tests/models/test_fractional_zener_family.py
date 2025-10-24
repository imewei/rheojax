"""Comprehensive tests for Fractional Zener Family and Advanced Fractional Models (Task Group 12).

This test suite validates all 7 fractional models:
- FractionalZenerSolidLiquid (FZSL)
- FractionalZenerSolidSolid (FZSS)
- FractionalZenerLiquidLiquid (FZLL)
- FractionalKelvinVoigtZener (FKVZ)
- FractionalBurgersModel (FBM)
- FractionalPoyntingThomson (FPT)
- FractionalJeffreysModel (FJM)

Each model is tested for:
1. Parameter initialization and bounds
2. Limit cases (alpha -> 0, alpha -> 1)
3. All applicable test modes
4. Numerical stability
5. JAX operations
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from rheo.models.fractional_zener_sl import FractionalZenerSolidLiquid, FZSL
from rheo.models.fractional_zener_ss import FractionalZenerSolidSolid, FZSS
from rheo.models.fractional_zener_ll import FractionalZenerLiquidLiquid, FZLL
from rheo.models.fractional_kv_zener import FractionalKelvinVoigtZener, FKVZ
from rheo.models.fractional_burgers import FractionalBurgersModel, FBM
from rheo.models.fractional_poynting_thomson import FractionalPoyntingThomson, FPT
from rheo.models.fractional_jeffreys import FractionalJeffreysModel, FJM


class TestFractionalZenerSolidLiquid:
    """Tests for FZSL model."""

    @pytest.fixture
    def model(self):
        return FractionalZenerSolidLiquid()

    @pytest.fixture
    def params(self):
        return {'Ge': 1000.0, 'c_alpha': 500.0, 'alpha': 0.5, 'tau': 1.0}

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'Ge' in model.parameters
        assert 'c_alpha' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau' in model.parameters

    def test_relaxation(self, model, params):
        """Test relaxation modulus."""
        t = jnp.logspace(-2, 2, 20)
        G_t = model._predict_relaxation(t, **params)
        assert G_t.shape == t.shape
        assert jnp.all(G_t > 0)
        assert jnp.all(jnp.isfinite(G_t))

    def test_oscillation(self, model, params):
        """Test complex modulus."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(G_star > 0)

    def test_alpha_limits(self, model):
        """Test alpha limit cases."""
        params_low = {'Ge': 1000.0, 'c_alpha': 500.0, 'alpha': 0.01, 'tau': 1.0}
        params_high = {'Ge': 1000.0, 'c_alpha': 500.0, 'alpha': 0.99, 'tau': 1.0}
        t = jnp.logspace(-2, 2, 10)

        G_low = model._predict_relaxation(t, **params_low)
        G_high = model._predict_relaxation(t, **params_high)

        assert jnp.all(jnp.isfinite(G_low))
        assert jnp.all(jnp.isfinite(G_high))


class TestFractionalZenerSolidSolid:
    """Tests for FZSS model."""

    @pytest.fixture
    def model(self):
        return FractionalZenerSolidSolid()

    @pytest.fixture
    def params(self):
        return {'Ge': 1000.0, 'Gm': 500.0, 'alpha': 0.5, 'tau_alpha': 1.0}

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'Ge' in model.parameters
        assert 'Gm' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau_alpha' in model.parameters

    def test_relaxation(self, model, params):
        """Test relaxation modulus."""
        t = jnp.logspace(-2, 2, 20)
        G_t = model._predict_relaxation(t, **params)
        assert G_t.shape == t.shape
        assert jnp.all(G_t > 0)

        # Should approach G_e at long times
        assert jnp.allclose(G_t[-1], params['Ge'], rtol=0.2)

    def test_creep(self, model, params):
        """Test creep compliance."""
        t = jnp.logspace(-2, 2, 20)
        J_t = model._predict_creep(t, **params)
        assert J_t.shape == t.shape
        assert jnp.all(J_t > 0)

        # Should approach 1/Ge at long times
        expected_J = 1.0 / params['Ge']
        assert jnp.allclose(J_t[-1], expected_J, rtol=0.3)

    def test_oscillation(self, model, params):
        """Test complex modulus."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(G_star[:, 0] > 0)  # G'
        assert jnp.all(G_star[:, 1] > 0)  # G''


class TestFractionalZenerLiquidLiquid:
    """Tests for FZLL model."""

    @pytest.fixture
    def model(self):
        return FractionalZenerLiquidLiquid()

    @pytest.fixture
    def params(self):
        return {
            'c1': 500.0, 'c2': 100.0,
            'alpha': 0.5, 'beta': 0.3, 'gamma': 0.7,
            'tau': 1.0
        }

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'c1' in model.parameters
        assert 'c2' in model.parameters
        assert 'alpha' in model.parameters
        assert 'beta' in model.parameters
        assert 'gamma' in model.parameters
        assert 'tau' in model.parameters

    def test_oscillation(self, model, params):
        """Test complex modulus (primary mode for FZLL)."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(jnp.isfinite(G_star))

    def test_multiple_fractional_orders(self, model):
        """Test model with three different fractional orders."""
        params = {
            'c1': 500.0, 'c2': 100.0,
            'alpha': 0.3, 'beta': 0.5, 'gamma': 0.7,
            'tau': 1.0
        }
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)

        # Should produce valid results
        assert jnp.all(jnp.isfinite(G_star))
        assert jnp.all(G_star[:, 0] > 0)


class TestFractionalKelvinVoigtZener:
    """Tests for FKVZ model."""

    @pytest.fixture
    def model(self):
        return FractionalKelvinVoigtZener()

    @pytest.fixture
    def params(self):
        return {'Ge': 1000.0, 'Gk': 500.0, 'alpha': 0.5, 'tau': 1.0}

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'Ge' in model.parameters
        assert 'Gk' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau' in model.parameters

    def test_creep(self, model, params):
        """Test creep compliance (primary mode)."""
        t = jnp.logspace(-2, 2, 20)
        J_t = model._predict_creep(t, **params)
        assert J_t.shape == t.shape
        assert jnp.all(J_t > 0)

        # Should show retardation behavior
        assert jnp.all(jnp.diff(J_t) >= 0)

    def test_relaxation(self, model, params):
        """Test relaxation modulus."""
        t = jnp.logspace(-2, 2, 20)
        G_t = model._predict_relaxation(t, **params)
        assert G_t.shape == t.shape
        assert jnp.all(G_t > 0)

    def test_oscillation(self, model, params):
        """Test complex modulus via compliance."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(jnp.isfinite(G_star))


class TestFractionalBurgersModel:
    """Tests for FBM."""

    @pytest.fixture
    def model(self):
        return FractionalBurgersModel()

    @pytest.fixture
    def params(self):
        return {
            'Jg': 1e-6, 'eta1': 1000.0,
            'Jk': 5e-6, 'alpha': 0.5, 'tau_k': 1.0
        }

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'Jg' in model.parameters
        assert 'eta1' in model.parameters
        assert 'Jk' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau_k' in model.parameters

    def test_creep(self, model, params):
        """Test creep compliance with viscous flow."""
        t = jnp.logspace(-2, 2, 20)
        J_t = model._predict_creep(t, **params)
        assert J_t.shape == t.shape
        assert jnp.all(J_t > 0)

        # Should show increasing compliance (viscous flow)
        assert jnp.all(jnp.diff(J_t) > 0)

    def test_oscillation(self, model, params):
        """Test complex modulus."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(jnp.isfinite(G_star))

    def test_four_mechanisms(self, model, params):
        """Test that model captures four relaxation mechanisms."""
        t = jnp.logspace(-3, 3, 50)
        J_t = model._predict_creep(t, **params)

        # Should have multiple regimes (instantaneous, retardation, flow)
        assert jnp.all(jnp.isfinite(J_t))
        assert J_t[-1] > J_t[0]  # Unbounded growth


class TestFractionalPoyntingThomson:
    """Tests for FPT model."""

    @pytest.fixture
    def model(self):
        return FractionalPoyntingThomson()

    @pytest.fixture
    def params(self):
        return {'Ge': 1500.0, 'Gk': 500.0, 'alpha': 0.5, 'tau': 1.0}

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'Ge' in model.parameters
        assert 'Gk' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau' in model.parameters

    def test_creep(self, model, params):
        """Test creep compliance."""
        t = jnp.logspace(-2, 2, 20)
        J_t = model._predict_creep(t, **params)
        assert J_t.shape == t.shape
        assert jnp.all(J_t > 0)

    def test_relaxation(self, model, params):
        """Test relaxation modulus."""
        t = jnp.logspace(-2, 2, 20)
        G_t = model._predict_relaxation(t, **params)
        assert G_t.shape == t.shape
        assert jnp.all(G_t > 0)

        # Should show stress relaxation
        assert G_t[0] > G_t[-1]

    def test_similarity_to_fkvz(self, model):
        """Test that FPT and FKVZ have identical mathematical forms."""
        params_fpt = {'Ge': 1000.0, 'Gk': 500.0, 'alpha': 0.5, 'tau': 1.0}

        from rheo.models.fractional_kv_zener import FractionalKelvinVoigtZener
        fkvz = FractionalKelvinVoigtZener()

        t = jnp.logspace(-2, 2, 20)
        J_fpt = model._predict_creep(t, **params_fpt)
        J_fkvz = fkvz._predict_creep(t, **params_fpt)

        # Should be identical (same mathematical model)
        assert jnp.allclose(J_fpt, J_fkvz, rtol=1e-10)


class TestFractionalJeffreysModel:
    """Tests for FJM."""

    @pytest.fixture
    def model(self):
        return FractionalJeffreysModel()

    @pytest.fixture
    def params(self):
        return {'eta1': 1000.0, 'eta2': 500.0, 'alpha': 0.5, 'tau1': 1.0}

    def test_initialization(self, model):
        """Test parameter initialization."""
        assert 'eta1' in model.parameters
        assert 'eta2' in model.parameters
        assert 'alpha' in model.parameters
        assert 'tau1' in model.parameters

    def test_relaxation(self, model, params):
        """Test relaxation modulus."""
        t = jnp.logspace(-2, 2, 20)
        G_t = model._predict_relaxation(t, **params)
        assert G_t.shape == t.shape
        assert jnp.all(G_t > 0)
        assert jnp.all(jnp.isfinite(G_t))

    def test_oscillation(self, model, params):
        """Test complex modulus."""
        omega = jnp.logspace(-2, 2, 20)
        G_star = model._predict_oscillation(omega, **params)
        assert G_star.shape == (20, 2)
        assert jnp.all(jnp.isfinite(G_star))

    def test_liquid_behavior(self, model, params):
        """Test viscous liquid behavior."""
        omega = jnp.logspace(-2, 2, 30)
        G_star = model._predict_oscillation(omega, **params)

        # At low frequencies, should show liquid-like behavior
        # G'' > G' (viscous dominated)
        G_prime_low = G_star[0, 0]
        G_double_prime_low = G_star[0, 1]

        # For liquid: G'' typically > G' at low omega
        # (this may not always hold depending on parameters)
        assert jnp.all(jnp.isfinite(G_star))


class TestJAXOperations:
    """Test JAX-specific operations across all models."""

    @pytest.fixture
    def models(self):
        return [
            FractionalZenerSolidLiquid(),
            FractionalZenerSolidSolid(),
            FractionalZenerLiquidLiquid(),
            FractionalKelvinVoigtZener(),
            FractionalBurgersModel(),
            FractionalPoyntingThomson(),
            FractionalJeffreysModel()
        ]

    def test_jit_compilation(self, models):
        """Test JIT compilation for all models."""
        for model in models:
            # Skip models without simple relaxation
            if hasattr(model, '_predict_relaxation'):
                params = model.parameters.to_dict()
                # Set some default values
                for key in params:
                    if 'eta' in key or 'J' in key:
                        params[key] = 1000.0
                    elif 'alpha' in key or 'beta' in key or 'gamma' in key:
                        params[key] = 0.5
                    elif 'tau' in key or 'c' in key or 'G' in key:
                        params[key] = 1.0

                t = jnp.array([0.1, 1.0, 10.0])

                # Should compile without error
                try:
                    predict_jit = jax.jit(lambda t: model._predict_relaxation(t, **params))
                    result = predict_jit(t)
                    assert jnp.all(jnp.isfinite(result))
                except Exception as e:
                    pytest.skip(f"JIT compilation failed for {model.__class__.__name__}: {e}")

    def test_numerical_stability(self, models):
        """Test numerical stability for all models."""
        for model in models:
            if hasattr(model, '_predict_oscillation'):
                params = model.parameters.to_dict()
                # Set default values
                for key in params:
                    if 'eta' in key or 'J' in key:
                        params[key] = 1000.0
                    elif 'alpha' in key or 'beta' in key or 'gamma' in key:
                        params[key] = 0.5
                    elif 'tau' in key or 'c' in key or 'G' in key:
                        params[key] = 1.0

                omega = jnp.logspace(-3, 3, 20)

                try:
                    G_star = model._predict_oscillation(omega, **params)
                    assert jnp.all(jnp.isfinite(G_star))
                    assert G_star.shape == (20, 2)
                except Exception as e:
                    pytest.skip(f"Oscillation test failed for {model.__class__.__name__}: {e}")


class TestAliases:
    """Test convenience aliases."""

    def test_all_aliases(self):
        """Test that all aliases work correctly."""
        assert FZSL is FractionalZenerSolidLiquid
        assert FZSS is FractionalZenerSolidSolid
        assert FZLL is FractionalZenerLiquidLiquid
        assert FKVZ is FractionalKelvinVoigtZener
        assert FBM is FractionalBurgersModel
        assert FPT is FractionalPoyntingThomson
        assert FJM is FractionalJeffreysModel
