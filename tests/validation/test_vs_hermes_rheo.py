"""
Validation Against hermes-rheo (Task 16.5)

Compare all 5 transforms against corresponding hermes-rheo implementations.
Verify numerical equivalence within tolerance.

NOTE: This requires hermes-rheo to be installed and accessible.
Path: /Users/b80985/Documents/GitHub/hermes-rheo/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add hermes-rheo to path
HERMES_PATH = Path("/Users/b80985/Documents/GitHub/hermes-rheo/")
if HERMES_PATH.exists():
    sys.path.insert(0, str(HERMES_PATH))

from rheojax.core.data import RheoData
from rheojax.transforms.fft_analysis import FFTAnalysis
from rheojax.transforms.mastercurve import Mastercurve
from rheojax.transforms.mutation_number import MutationNumber
from rheojax.transforms.owchirp import OWChirp
from rheojax.transforms.smooth_derivative import SmoothDerivative


class TestFFTAnalysisVsHermes:
    """Validate FFTAnalysis transform against hermes-rheo."""

    @pytest.mark.validation
    def test_fft_transform_comparison(self):
        """Compare FFT transform output with hermes-rheo."""
        pytest.skip("Requires hermes-rheo installation and validation framework")

        # Test data: Oscillation data
        omega = np.logspace(-2, 2, 50)
        G_star = 1e6 * (1 + 1j * omega)

        data = RheoData(x=omega, y=G_star, domain="frequency")

        # rheo transform
        fft_rheo = FFTAnalysis()
        result_rheo = fft_rheo.fit_transform(data)

        # hermes-rheo transform (would require import)
        # import hermes.transforms
        # result_hermes = hermes.transforms.fft_analysis(data)

        # Validation
        # assert np.allclose(result_rheo.y, result_hermes.y, rtol=1e-6)


class TestMastercurveVsHermes:
    """Validate Mastercurve transform against hermes-rheo."""

    @pytest.mark.validation
    def test_mastercurve_shifting(self):
        """Compare mastercurve time-temperature superposition with hermes-rheo."""
        pytest.skip("Requires hermes-rheo installation and validation framework")

        # Multi-temperature data would be loaded here
        # Compare shift factors and mastercurve construction


class TestMutationNumberVsHermes:
    """Validate MutationNumber transform against hermes-rheo."""

    @pytest.mark.validation
    def test_mutation_number_calculation(self):
        """Compare mutation number calculation with hermes-rheo."""
        pytest.skip("Requires hermes-rheo installation and validation framework")


class TestOWChirpVsHermes:
    """Validate OWChirp transform against hermes-rheo."""

    @pytest.mark.validation
    def test_owchirp_transform(self):
        """Compare OW-Chirp transform with hermes-rheo."""
        pytest.skip("Requires hermes-rheo installation and validation framework")


class TestSmoothDerivativeVsHermes:
    """Validate SmoothDerivative transform against hermes-rheo."""

    @pytest.mark.validation
    def test_smooth_derivative_comparison(self):
        """Compare Savitzky-Golay derivative with hermes-rheo."""
        pytest.skip("Requires hermes-rheo installation and validation framework")

        # Test data
        x = np.linspace(0, 10, 100)
        y = x**2  # True derivative = 2x

        data = RheoData(x=x, y=y, domain="time")

        # rheo transform
        deriv_rheo = SmoothDerivative()
        result_rheo = deriv_rheo.fit_transform(data, derivative_order=1)

        # hermes-rheo transform (would require import)
        # result_hermes = hermes.transforms.smooth_derivative(data, order=1)

        # Validation
        # assert np.allclose(result_rheo.y, result_hermes.y, rtol=1e-6)


# Validation report generation
def generate_transform_validation_report():
    """
    Generate validation report comparing all 5 transforms.

    Creates: docs/validation_report.md (transforms section)

    Format:
    | Transform Name | Validated | Tolerance | Notes |
    |----------------|-----------|-----------|-------|
    | FFTAnalysis    | ✓         | 1e-6      | Minor numerical differences |
    | ...            | ...       | ...       | ... |
    """
    report = """# Transform Validation Report - rheo vs hermes-rheo

**Date:** 2025-10-24
**Status:** PENDING - Awaiting validation framework implementation

## Transform Validation Results

| Transform | Validated | Tolerance | Status | Notes |
|-----------|-----------|-----------|--------|-------|
| FFTAnalysis | ⏸️ | 1e-6 | PENDING | Edge cases need review (inverse FFT correlation 0.08) |
| Mastercurve | ⏸️ | 1e-6 | PENDING | Overlap error calculation returns Inf - needs investigation |
| MutationNumber | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| OWChirp | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| SmoothDerivative | ⏸️ | 1e-4 | PENDING | Numerical precision issues with noisy data and second derivatives |

## Summary

- **Total Transforms:** 5
- **Validated:** 0
- **Blocked:** 0
- **Pending:** 5

## Known Issues

### FFTAnalysis
- `test_inverse_fft`: Correlation 0.08 (expected >0.95)
- `test_characteristic_time`: Returns NaN

### Mastercurve
- `test_overlap_error_calculation`: Returns Inf (should be finite)

### SmoothDerivative
- `test_second_derivative`: Numerical precision issues
- `test_noisy_data_smoothing`: Insufficient noise reduction (std 6.7, expected <0.5)
- `test_non_uniform_spacing`: Returns NaN

## Algorithmic Differences

Some transforms may have intentional algorithmic improvements over hermes-rheo:

1. **Improved numerical stability** - Better handling of edge cases
2. **JAX-optimized implementations** - Different numerical precision characteristics
3. **Enhanced error handling** - More robust validation

These should be documented in the final validation report after comparison.

## Next Steps

1. Install and configure hermes-rheo
2. Implement validation test framework
3. Fix identified edge cases in transforms
4. Run full validation suite
5. Document any intentional algorithmic differences

**Estimated Time to Complete:** 2-3 hours
"""

    return report


if __name__ == "__main__":
    # Generate validation report
    report = generate_transform_validation_report()
    print(report)
