"""Tests for post-fit physics validation."""

import numpy as np
import pytest

from rheojax.utils.physics_checks import PhysicsViolation, check_fit_physics


@pytest.mark.smoke
class TestPhysicsChecks:
    """Tests for check_fit_physics function."""

    def test_clean_maxwell_no_violations(self):
        """A well-fitted Maxwell model should have no physics violations."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        t = np.logspace(-2, 2, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)
        model.fit(t, G_t, test_mode="relaxation")

        violations = check_fit_physics(model)
        assert isinstance(violations, list)
        # All violations should be PhysicsViolation instances
        for v in violations:
            assert isinstance(v, PhysicsViolation)
        # A clean fit should have few/no error-severity violations
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_returns_list(self):
        """Should always return a list, even for unfitted models."""
        from rheojax.models.classical import Maxwell

        model = Maxwell()
        violations = check_fit_physics(model)
        assert isinstance(violations, list)

    def test_violation_structure(self):
        """PhysicsViolation should have required fields."""
        v = PhysicsViolation(
            parameter="G0",
            value=-100.0,
            check="positive_moduli",
            message="Negative modulus",
            severity="error",
        )
        assert v.parameter == "G0"
        assert v.value == -100.0
        assert v.check == "positive_moduli"
        assert v.severity == "error"
