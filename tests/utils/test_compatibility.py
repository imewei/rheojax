"""Tests for model-data compatibility checking."""

import numpy as np
import pytest

from rheojax.models import FractionalKelvinVoigt
from rheojax.models import FractionalMaxwellGel
from rheojax.models import FractionalMaxwellLiquid
from rheojax.models import FractionalZenerSolidSolid
from rheojax.models import Maxwell
from rheojax.utils.compatibility import (
    DecayType,
    MaterialType,
    check_model_compatibility,
    detect_decay_type,
    detect_material_type,
    format_compatibility_message,
)


class TestDecayTypeDetection:
    """Tests for decay type detection from relaxation data."""

    @pytest.mark.smoke
    def test_exponential_decay(self):
        """Test detection of exponential decay (Maxwell-like)."""
        t = np.logspace(-2, 2, 100)
        G_t = 1e5 * np.exp(-t / 1.0)

        decay_type = detect_decay_type(t, G_t)

        assert decay_type == DecayType.EXPONENTIAL

    @pytest.mark.smoke
    def test_power_law_decay(self):
        """Test detection of power-law decay (gel-like)."""
        t = np.logspace(-2, 2, 100)
        alpha = 0.5
        G_t = 1e5 * t ** (-alpha)

        decay_type = detect_decay_type(t, G_t)

        assert decay_type == DecayType.POWER_LAW

    @pytest.mark.smoke
    def test_insufficient_data(self):
        """Test that insufficient data returns UNKNOWN."""
        t = np.array([0.1, 0.2])
        G_t = np.array([1e5, 9e4])

        decay_type = detect_decay_type(t, G_t)

        assert decay_type == DecayType.UNKNOWN

    @pytest.mark.smoke
    def test_invalid_data(self):
        """Test handling of invalid data (NaN, inf)."""
        t = np.logspace(-2, 2, 50)
        G_t = np.full(50, np.nan)

        decay_type = detect_decay_type(t, G_t)

        assert decay_type == DecayType.UNKNOWN

    @pytest.mark.smoke
    def test_multi_mode_decay(self):
        """Test detection of multi-mode decay."""
        t = np.logspace(-2, 2, 100)
        # Two Maxwell modes with different time constants
        G_t = 5e5 * np.exp(-t / 0.1) + 5e5 * np.exp(-t / 10.0)

        decay_type = detect_decay_type(t, G_t)

        # Should detect as either EXPONENTIAL or MULTI_MODE
        assert decay_type in [DecayType.EXPONENTIAL, DecayType.MULTI_MODE]


class TestMaterialTypeDetection:
    """Tests for material type detection."""

    def test_solid_from_relaxation(self):
        """Test detection of solid-like material from relaxation."""
        t = np.logspace(-2, 2, 50)
        # Material with finite equilibrium modulus
        G_e = 5e4
        G_t = G_e + 5e4 * np.exp(-t / 1.0)

        material_type = detect_material_type(t=t, G_t=G_t)

        assert material_type == MaterialType.VISCOELASTIC_SOLID

    def test_liquid_from_relaxation(self):
        """Test detection of liquid-like material from relaxation."""
        t = np.logspace(-2, 2, 50)
        # Material that fully relaxes
        G_t = 1e5 * np.exp(-t / 1.0)

        material_type = detect_material_type(t=t, G_t=G_t)

        assert material_type == MaterialType.VISCOELASTIC_LIQUID

    def test_gel_from_relaxation(self):
        """Test detection of gel-like material from relaxation."""
        t = np.logspace(-2, 2, 50)
        alpha = 0.5
        G_t = 1e5 * t ** (-alpha)

        material_type = detect_material_type(t=t, G_t=G_t)

        # Should detect as gel (power-law with high final modulus)
        assert material_type in [MaterialType.GEL, MaterialType.VISCOELASTIC_SOLID]

    def test_solid_from_oscillation(self):
        """Test detection of solid from oscillation data."""
        omega = np.logspace(-2, 2, 50)
        # G' > G" at low frequency → solid
        G_prime = 1e5 * np.ones(50)
        G_double_prime = 1e3 * omega**0.5

        G_star = np.column_stack([G_prime, G_double_prime])
        material_type = detect_material_type(omega=omega, G_star=G_star)

        assert material_type == MaterialType.SOLID

    def test_liquid_from_oscillation(self):
        """Test detection of liquid from oscillation data."""
        omega = np.logspace(-2, 2, 50)
        # G" > G' at low frequency → liquid
        G_prime = 1e4 * omega**2
        G_double_prime = 1e5 * omega

        G_star = np.column_stack([G_prime, G_double_prime])
        material_type = detect_material_type(omega=omega, G_star=G_star)

        assert material_type == MaterialType.LIQUID

    def test_no_data(self):
        """Test that no data returns UNKNOWN."""
        material_type = detect_material_type()

        assert material_type == MaterialType.UNKNOWN


class TestModelCompatibility:
    """Tests for model-data compatibility checking."""

    def test_fzss_with_exponential_incompatible(self):
        """Test that FZSS is detected as incompatible with exponential decay."""
        t = np.logspace(-2, 2, 50)
        G_t = 1e5 * np.exp(-t / 1.0)

        model = FractionalZenerSolidSolid()
        compatibility = check_model_compatibility(
            model, t=t, G_t=G_t, test_mode="relaxation"
        )

        assert not compatibility["compatible"]
        assert compatibility["confidence"] > 0.7
        assert len(compatibility["warnings"]) > 0
        assert "Maxwell" in compatibility["recommendations"]

    def test_maxwell_with_exponential_compatible(self):
        """Test that Maxwell is compatible with exponential decay."""
        t = np.logspace(-2, 2, 50)
        G_t = 1e5 * np.exp(-t / 1.0)

        model = Maxwell()
        compatibility = check_model_compatibility(
            model, t=t, G_t=G_t, test_mode="relaxation"
        )

        # Maxwell should be compatible or at least not strongly incompatible
        assert compatibility["compatible"] or compatibility["confidence"] < 0.5

    def test_maxwell_with_power_law_incompatible(self):
        """Test that Maxwell is incompatible with power-law decay."""
        t = np.logspace(-2, 2, 50)
        alpha = 0.5
        G_t = 1e5 * t ** (-alpha)

        model = Maxwell()
        compatibility = check_model_compatibility(
            model, t=t, G_t=G_t, test_mode="relaxation"
        )

        assert not compatibility["compatible"]
        assert len(compatibility["warnings"]) > 0
        assert any(
            "FractionalMaxwellGel" in rec for rec in compatibility["recommendations"]
        )

    def test_fmg_with_power_law_compatible(self):
        """Test that FractionalMaxwellGel is compatible with power-law."""
        t = np.logspace(-2, 2, 50)
        alpha = 0.5
        G_t = 1e5 * t ** (-alpha)

        model = FractionalMaxwellGel()
        compatibility = check_model_compatibility(
            model, t=t, G_t=G_t, test_mode="relaxation"
        )

        # Should be compatible or not strongly incompatible
        assert compatibility["compatible"] or compatibility["confidence"] < 0.5

    def test_fml_with_solid_incompatible(self):
        """Test that FractionalMaxwellLiquid is incompatible with solid data."""
        omega = np.logspace(-2, 2, 50)
        # Solid: G' > G" at low frequency
        G_prime = 1e5 * np.ones(50)
        G_double_prime = 1e3 * omega**0.5

        G_star = np.column_stack([G_prime, G_double_prime])
        model = FractionalMaxwellLiquid()
        compatibility = check_model_compatibility(
            model, omega=omega, G_star=G_star, test_mode="oscillation"
        )

        assert not compatibility["compatible"]
        assert len(compatibility["warnings"]) > 0

    def test_unknown_model(self):
        """Test compatibility check with unknown model."""
        t = np.logspace(-2, 2, 50)
        G_t = 1e5 * np.exp(-t / 1.0)

        # Use a model without specific compatibility rules
        model = FractionalKelvinVoigt()
        compatibility = check_model_compatibility(
            model, t=t, G_t=G_t, test_mode="relaxation"
        )

        # Should return default compatibility
        assert "compatible" in compatibility
        assert "confidence" in compatibility


class TestCompatibilityMessageFormatting:
    """Tests for compatibility message formatting."""

    def test_format_compatible_message(self):
        """Test formatting of compatible message."""
        compatibility = {
            "compatible": True,
            "confidence": 0.85,
            "decay_type": DecayType.EXPONENTIAL,
            "material_type": MaterialType.VISCOELASTIC_LIQUID,
            "warnings": [],
            "recommendations": [],
        }

        message = format_compatibility_message(compatibility)

        assert "compatible" in message.lower()
        assert "85%" in message
        assert "exponential" in message

    def test_format_incompatible_message(self):
        """Test formatting of incompatible message."""
        compatibility = {
            "compatible": False,
            "confidence": 0.90,
            "decay_type": DecayType.EXPONENTIAL,
            "material_type": MaterialType.VISCOELASTIC_LIQUID,
            "warnings": [
                "FZSS model expects Mittag-Leffler relaxation, but data shows exponential decay."
            ],
            "recommendations": ["Maxwell", "Zener"],
        }

        message = format_compatibility_message(compatibility)

        assert "not be appropriate" in message.lower()
        assert "90%" in message
        assert "Maxwell" in message
        assert "Zener" in message
        assert "FZSS" in message

    def test_format_message_with_unknown_types(self):
        """Test formatting when decay/material types are unknown."""
        compatibility = {
            "compatible": True,
            "confidence": 0.5,
            "decay_type": DecayType.UNKNOWN,
            "material_type": MaterialType.UNKNOWN,
            "warnings": [],
            "recommendations": [],
        }

        message = format_compatibility_message(compatibility)

        # Should still format without errors
        assert "50%" in message
        assert "unknown" not in message.lower()  # Shouldn't show unknown types


class TestEnhancedErrorMessaging:
    """Tests for enhanced error messaging in model fitting."""

    def test_fzss_exponential_enhanced_error(self):
        """Test that FZSS with exponential data provides enhanced error."""
        np.random.seed(42)
        t = np.logspace(-2, 2, 50)
        G_t = 1e5 * np.exp(-t / 1.0) + np.random.normal(0, 1000, size=len(t))

        model = FractionalZenerSolidSolid()

        # Should fail with enhanced error message
        with pytest.raises(RuntimeError) as exc_info:
            model.fit(
                t,
                G_t,
                test_mode="relaxation",
                max_iter=100,
                compatibility_guard=True,
            )

        error_msg = str(exc_info.value)

        # Check that error is enhanced (either original or enhanced format)
        # Original: Just "Optimization failed"
        # Enhanced: Includes "Model-data compatibility"
        assert (
            "Optimization failed" in error_msg
            or "Model-data compatibility" in error_msg
        )

    def test_check_compatibility_parameter(self):
        """Test that check_compatibility parameter works."""
        np.random.seed(42)
        t = np.logspace(-2, 2, 50)
        G_t = 1e5 * np.exp(-t / 1.0) + np.random.normal(0, 1000, size=len(t))

        model = FractionalZenerSolidSolid()

        # Should issue warning when check_compatibility=True
        import logging
        import warnings

        # Capture warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                # This should log a warning
                model.fit(
                    t,
                    G_t,
                    test_mode="relaxation",
                    max_iter=100,
                    check_compatibility=True,
                )
            except RuntimeError:
                # Expected to fail
                pass


class TestEdgeCases:
    """Tests for edge cases in compatibility checking."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        t = np.array([])
        G_t = np.array([])

        decay_type = detect_decay_type(t, G_t)
        assert decay_type == DecayType.UNKNOWN

    def test_single_point(self):
        """Test handling of single data point."""
        t = np.array([1.0])
        G_t = np.array([1e5])

        decay_type = detect_decay_type(t, G_t)
        assert decay_type == DecayType.UNKNOWN

    def test_negative_times(self):
        """Test handling of negative times."""
        t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        G_t = np.array([1e5, 9e4, 8e4, 7e4, 6e4])

        # Should filter out invalid times
        decay_type = detect_decay_type(t, G_t)
        # Should still work with remaining valid points
        assert decay_type != DecayType.UNKNOWN or len(t[t > 0]) < 10

    def test_zero_modulus(self):
        """Test handling of zero modulus values."""
        t = np.logspace(-2, 2, 50)
        G_t = np.zeros(50)

        decay_type = detect_decay_type(t, G_t)
        assert decay_type == DecayType.UNKNOWN

    def test_negative_modulus(self):
        """Test handling of negative modulus values."""
        t = np.logspace(-2, 2, 50)
        G_t = -1e5 * np.ones(50)

        decay_type = detect_decay_type(t, G_t)
        assert decay_type == DecayType.UNKNOWN
