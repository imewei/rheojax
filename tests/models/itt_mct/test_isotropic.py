"""Tests for ITTMCTIsotropic (ISM) model.

Tests cover:
- Percus-Yevick S(k) computation
- User-provided S(k) interface
- k-resolved correlator basics
- Protocol predictions
"""

import numpy as np
import pytest

from rheojax.models.itt_mct import ITTMCTIsotropic
from rheojax.utils.structure_factor import hard_sphere_properties, percus_yevick_sk


class TestISMInitialization:
    """Tests for ISM model initialization."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test default initialization."""
        model = ITTMCTIsotropic()

        assert model.parameters.get_value("phi") == 0.55
        assert model.parameters.get_value("sigma_d") == 1e-6
        assert model.parameters.get_value("D0") == 1e-12
        assert hasattr(model, "k_grid")
        assert hasattr(model, "S_k")

    @pytest.mark.smoke
    def test_initialization_with_phi(self):
        """Test initialization with volume fraction."""
        model = ITTMCTIsotropic(phi=0.52)

        assert model.parameters.get_value("phi") == 0.52
        info = model.get_glass_transition_info()
        assert info["phi"] == 0.52

    def test_initialization_with_n_k(self):
        """Test initialization with custom k-grid size."""
        model = ITTMCTIsotropic(n_k=50)

        assert len(model.k_grid) == 50
        assert len(model.S_k) == 50

    def test_glass_transition_detection(self):
        """Test glass transition detection based on phi."""
        # Below MCT transition
        model_fluid = ITTMCTIsotropic(phi=0.50)
        assert not model_fluid.get_glass_transition_info()["is_glass"]

        # Above MCT transition
        model_glass = ITTMCTIsotropic(phi=0.55)
        assert model_glass.get_glass_transition_info()["is_glass"]


class TestPercusYevickSk:
    """Tests for Percus-Yevick S(k) computation."""

    @pytest.mark.smoke
    def test_sk_positive(self):
        """Test S(k) is always positive."""
        model = ITTMCTIsotropic(phi=0.55)
        assert np.all(model.S_k > 0)

    @pytest.mark.smoke
    def test_sk_peak(self):
        """Test S(k) has characteristic peak."""
        model = ITTMCTIsotropic(phi=0.55)

        # Peak should be around k*σ ≈ 7
        k_peak = model.k_grid[model.S_k.argmax()]
        sigma_d = model.parameters.get_value("sigma_d")
        k_peak_scaled = k_peak * sigma_d

        # Peak position should be in reasonable range
        assert 5.0 < k_peak_scaled < 10.0

    def test_sk_increases_with_phi(self):
        """Test S(k) peak increases with volume fraction."""
        model_low = ITTMCTIsotropic(phi=0.40)
        model_high = ITTMCTIsotropic(phi=0.55)

        assert model_high.S_k.max() > model_low.S_k.max()

    def test_sk_properties_function(self):
        """Test hard_sphere_properties utility."""
        props = hard_sphere_properties(0.55)

        assert "phi" in props
        assert "S_max" in props
        assert "k_max_position" in props
        assert "is_glassy" in props
        assert props["is_glassy"]  # φ=0.55 > φ_MCT


class TestUserProvidedSk:
    """Tests for user-provided S(k) interface."""

    @pytest.mark.smoke
    def test_user_provided_sk(self):
        """Test initialization with user-provided S(k)."""
        # Create synthetic S(k) data
        k_data = np.linspace(0.1, 50, 100)
        sk_data = 1.0 + 2.0 * np.exp(-((k_data - 7.0) ** 2) / 2)

        model = ITTMCTIsotropic(
            sk_source="user_provided",
            k_data=k_data,
            sk_data=sk_data,
            n_k=50,
        )

        assert model._sk_source == "user_provided"
        assert len(model.S_k) == 50

    def test_user_provided_sk_missing_data_raises(self):
        """Test that missing S(k) data raises error."""
        with pytest.raises(ValueError, match="Must provide k_data and sk_data"):
            ITTMCTIsotropic(sk_source="user_provided")

    def test_update_structure_factor(self):
        """Test updating S(k) after initialization."""
        model = ITTMCTIsotropic(phi=0.50)
        original_sk_max = model.S_k.max()

        # Update to higher phi
        model.update_structure_factor(phi=0.55)

        assert model.S_k.max() > original_sk_max


class TestSkInfo:
    """Tests for S(k) information methods."""

    def test_get_sk_info(self):
        """Test get_sk_info method."""
        model = ITTMCTIsotropic(phi=0.55, n_k=100)
        info = model.get_sk_info()

        assert info["source"] == "percus_yevick"
        assert info["n_k"] == 100
        assert "k_range" in info
        assert "S_max" in info
        assert "S_max_position" in info


class TestISMProtocols:
    """Tests for ISM protocol predictions."""

    @pytest.mark.smoke
    def test_flow_curve(self):
        """Test ISM flow curve prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        gamma_dot = np.logspace(-2, 2, 10)

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma >= 0)

    @pytest.mark.smoke
    def test_oscillation(self):
        """Test ISM oscillation prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        omega = np.logspace(-1, 2, 10)

        G_star = model.predict(omega, test_mode="oscillation")

        assert G_star.shape == omega.shape
        assert np.all(G_star >= 0)

    def test_oscillation_components(self):
        """Test ISM oscillation with components."""
        model = ITTMCTIsotropic(phi=0.55)
        omega = np.logspace(-1, 2, 10)

        G_components = model.predict(
            omega, test_mode="oscillation", return_components=True
        )

        assert G_components.shape == (len(omega), 2)
        G_prime = G_components[:, 0]
        G_double_prime = G_components[:, 1]

        # Both should be positive
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    @pytest.mark.smoke
    def test_startup(self):
        """Test ISM startup prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        t = np.linspace(0, 10, 50)

        sigma = model.predict(t, test_mode="startup", gamma_dot=1.0)

        assert sigma.shape == t.shape
        assert np.all(sigma >= 0)

    @pytest.mark.smoke
    def test_creep(self):
        """Test ISM creep prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        t = np.linspace(0.1, 100, 50)

        J = model.predict(t, test_mode="creep", sigma_applied=100.0)

        assert J.shape == t.shape
        assert np.all(J >= 0)

    @pytest.mark.smoke
    def test_relaxation(self):
        """Test ISM relaxation prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        t = np.linspace(0, 50, 50)

        sigma = model.predict(t, test_mode="relaxation", gamma_pre=0.05)

        assert sigma.shape == t.shape
        assert np.all(sigma >= 0)

    @pytest.mark.smoke
    def test_laos(self):
        """Test ISM LAOS prediction."""
        model = ITTMCTIsotropic(phi=0.55)
        t = np.linspace(0, 10, 100)

        sigma = model.predict(t, test_mode="laos", gamma_0=0.1, omega=1.0)

        assert sigma.shape == t.shape


class TestISMFluidVsGlass:
    """Tests comparing fluid and glass behavior in ISM."""

    def test_fluid_no_yield_stress(self):
        """Test fluid state has no yield stress."""
        model = ITTMCTIsotropic(phi=0.40)  # Well below MCT
        gamma_dot = np.array([1e-4, 1e-3, 1e-2])

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Fluid: σ → 0 as γ̇ → 0
        assert sigma[0] < sigma[-1]

    def test_glass_yield_stress(self):
        """Test glass state has yield stress."""
        model = ITTMCTIsotropic(phi=0.58)  # Above MCT
        gamma_dot = np.array([1e-4, 1e-3, 1e-2])

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Glass: σ > 0 even at low γ̇
        assert sigma[0] > 0


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test repr method."""
        model = ITTMCTIsotropic(phi=0.55, n_k=100)
        repr_str = repr(model)

        assert "ITTMCTIsotropic" in repr_str
        assert "φ=" in repr_str
        assert "n_k=" in repr_str
        assert "glass" in repr_str  # φ=0.55 is glass
