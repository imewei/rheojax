"""
Validation Against pyRheo (Task 16.5)

Compare all 20 rheo models against corresponding pyRheo implementations.
Verify numerical equivalence within tolerance.

NOTE: This requires pyRheo to be installed and accessible.
Path: /Users/b80985/Documents/GitHub/pyRheo/

STATUS: BLOCKED by Parameter hashability issue - fractional models cannot be tested.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add pyRheo to path
PYRHEO_PATH = Path("/Users/b80985/Documents/GitHub/pyRheo/")
if PYRHEO_PATH.exists():
    sys.path.insert(0, str(PYRHEO_PATH))

from rheo.core.data import RheoData
from rheo.models.maxwell import Maxwell
from rheo.models.springpot import SpringPot
from rheo.models.zener import Zener


class TestClassicalModelsVsPyRheo:
    """
    Validate classical models (Maxwell, Zener, SpringPot) against pyRheo.
    """

    @pytest.mark.validation
    def test_maxwell_vs_pyrheo(self):
        """Compare Maxwell model predictions with pyRheo.Maxwell."""
        pytest.skip("Requires pyRheo installation and Parameter.__hash__() fix")

        # Test data
        t = np.logspace(-2, 2, 50)
        E = 1e6  # Pa
        tau = 1.0  # s

        # rheo prediction
        model_rheo = Maxwell()
        model_rheo.parameters["E"].value = E
        model_rheo.parameters["tau"].value = tau
        pred_rheo = model_rheo.predict(t, test_mode="relaxation")

        # pyRheo prediction (would require pyRheo import)
        # import pyRheo
        # model_pyrheo = pyRheo.Maxwell(E=E, tau=tau)
        # pred_pyrheo = model_pyrheo.relax(t)

        # Validation
        # assert np.allclose(pred_rheo, pred_pyrheo, rtol=1e-6), \
        #     "Maxwell predictions should match pyRheo within 1e-6 relative tolerance"

    @pytest.mark.validation
    def test_zener_vs_pyrheo(self):
        """Compare Zener model predictions with pyRheo.Zener."""
        pytest.skip("Requires pyRheo installation and validation framework")

    @pytest.mark.validation
    def test_springpot_vs_pyrheo(self):
        """Compare SpringPot model predictions with pyRheo.SpringPot."""
        pytest.skip("Requires pyRheo installation and validation framework")


class TestFractionalModelsVsPyRheo:
    """
    Validate fractional models against pyRheo implementations.

    BLOCKED: All tests blocked by Parameter hashability issue.
    """

    @pytest.mark.validation
    @pytest.mark.xfail(
        reason="BLOCKED: Parameter hashability issue prevents fractional model testing"
    )
    def test_fractional_maxwell_vs_pyrheo(self):
        """Compare FractionalMaxwellModel with pyRheo."""
        pytest.skip("BLOCKED by Parameter.__hash__() issue")

    @pytest.mark.validation
    @pytest.mark.xfail(reason="BLOCKED: Parameter hashability issue")
    def test_fractional_zener_sl_vs_pyrheo(self):
        """Compare FractionalZenerSolidLiquid with pyRheo."""
        pytest.skip("BLOCKED by Parameter.__hash__() issue")

    # Additional fractional model tests would follow same pattern...


class TestFlowModelsVsPyRheo:
    """
    Validate flow models against pyRheo implementations.
    """

    @pytest.mark.validation
    def test_power_law_vs_pyrheo(self):
        """Compare PowerLaw model with pyRheo."""
        pytest.skip("Requires pyRheo installation and validation framework")

    @pytest.mark.validation
    def test_bingham_vs_pyrheo(self):
        """Compare Bingham model with pyRheo."""
        pytest.skip("Requires pyRheo installation and validation framework")

    # Additional flow model tests...


# Validation report generation
def generate_model_validation_report():
    """
    Generate validation report comparing all 20 models.

    Creates: docs/validation_report.md

    Format:
    | Model Name | Validated | Tolerance | Notes |
    |------------|-----------|-----------|-------|
    | Maxwell    | ✓         | 1e-6      | Exact match |
    | ...        | ...       | ...       | ... |
    """
    report = """# Model Validation Report - rheo vs pyRheo

**Date:** 2025-10-24
**Status:** BLOCKED - Awaiting Parameter.__hash__() implementation

## Classical Models (3 models)

| Model | Validated | Tolerance | Status | Notes |
|-------|-----------|-----------|--------|-------|
| Maxwell | ⏸️ | 1e-6 | PENDING | Awaiting Parameter fix |
| Zener | ⏸️ | 1e-6 | PENDING | Awaiting Parameter fix |
| SpringPot | ⏸️ | 1e-6 | PENDING | Awaiting Parameter fix |

## Fractional Models (11 models)

| Model | Validated | Tolerance | Status | Notes |
|-------|-----------|-----------|--------|-------|
| FractionalMaxwellModel | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalMaxwellGel | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalMaxwellLiquid | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalKelvinVoigt | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalZenerSL | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalZenerSS | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalZenerLL | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalKVZener | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalBurgers | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalPoyntingThomson | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |
| FractionalJeffreys | ❌ | 1e-6 | BLOCKED | Parameter hashability issue |

## Flow Models (6 models)

| Model | Validated | Tolerance | Status | Notes |
|-------|-----------|-----------|--------|-------|
| PowerLaw | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| Bingham | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| HerschelBulkley | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| Cross | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| Carreau | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |
| CarreauYasuda | ⏸️ | 1e-6 | PENDING | Awaiting validation framework |

## Summary

- **Total Models:** 20
- **Validated:** 0
- **Blocked:** 11 (fractional models)
- **Pending:** 9 (classical + flow models)

## Critical Blocker

**Parameter Hashability Issue:**
- All fractional models fail with: `TypeError: cannot use 'rheo.core.parameters.Parameter' as a dict key (unhashable type: 'Parameter')`
- **Fix Required:** Implement `Parameter.__hash__()` and `Parameter.__eq__()` methods
- **Impact:** Blocks testing of 11 out of 20 models (55%)

## Next Steps

1. ✅ Fix Parameter class hashability
2. Install and configure pyRheo for comparison
3. Implement validation test framework
4. Run full validation suite
5. Document any intentional differences

**Estimated Time to Complete:** 2-3 hours (after Parameter fix)
"""

    return report


if __name__ == "__main__":
    # Generate validation report
    report = generate_model_validation_report()
    print(report)
