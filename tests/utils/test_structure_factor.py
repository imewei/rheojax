"""Tests for rheojax.utils.structure_factor."""

import numpy as np
import pytest

from rheojax.utils.structure_factor import percus_yevick_sk


@pytest.mark.smoke
class TestPercusYevickSk:
    """Tests for Percus-Yevick structure factor."""

    def test_basic_computation(self):
        k = np.linspace(0.1, 30.0, 100)
        phi = 0.40
        Sk = percus_yevick_sk(k, phi)
        assert Sk.shape == (100,)
        assert np.all(np.isfinite(Sk))

    def test_sk_positive(self):
        """S(k) should be positive for all k."""
        k = np.linspace(0.1, 50.0, 200)
        phi = 0.30
        Sk = percus_yevick_sk(k, phi)
        assert np.all(Sk > 0)

    def test_low_phi_approaches_unity(self):
        """At low volume fraction, S(k) -> 1 for all k."""
        k = np.linspace(1.0, 30.0, 50)
        phi = 0.01
        Sk = percus_yevick_sk(k, phi)
        np.testing.assert_allclose(Sk, 1.0, atol=0.1)

    def test_sk_has_peak(self):
        """At moderate phi, S(k) should show a peak near k*sigma ~ 2*pi."""
        k = np.linspace(0.1, 30.0, 500)
        phi = 0.45
        Sk = percus_yevick_sk(k, phi)
        # Peak should exist (max > 1)
        assert np.max(Sk) > 1.0

    def test_custom_sigma(self):
        k = np.linspace(0.01, 10.0, 50)
        phi = 0.30
        Sk = percus_yevick_sk(k, phi, sigma=2.0)
        assert np.all(np.isfinite(Sk))
        assert np.all(Sk > 0)
