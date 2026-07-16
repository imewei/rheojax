"""Regression tests for STZ creep elastic-offset initial condition.

Background
----------
Under creep (constant applied stress sigma_app), the STZ ODE evolves total
strain via d(gamma)/dt = gamma_dot_plastic, since the stress is held constant
and the elastic strain rate vanishes. The elastic component therefore has to
enter the simulation through the *initial condition*: gamma(0+) = sigma_app / G0.

Prior to the fix, conventional.py initialised y0_val = 0.0, which silently
dropped the elastic contribution. For soft materials (low G0) the elastic
strain dominates the early-time creep signal, so this caused NLSQ on real
creep data (e.g. mucus) to collapse with R^2 < 0.

These tests pin the invariant so the regression cannot recur.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.stz import STZConventional

jax, jnp = safe_import_jax()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("sigma_applied", "G0"),
    [
        (1.0, 1.0),  # Soft material — gamma_e ~ O(1), dominant
        (1.2e6, 1e9),  # Glassy material — gamma_e small but non-zero
        (1e3, 5e3),  # Intermediate
    ],
)
def test_creep_initial_strain_matches_elastic_response(sigma_applied, G0):
    """gamma(t=0+) for creep must equal sigma_applied / G0."""
    model = STZConventional(variant="standard")
    sigma_y_val = min(0.5 * sigma_applied, 1e6)
    # Set bounds before values so the test sweeps soft and stiff regimes
    model.parameters.set_bounds("G0", (min(G0, 0.1), max(G0, 1e10)))
    model.parameters.set_bounds("sigma_y", (sigma_y_val * 0.5, sigma_y_val * 2.0))
    model.parameters.set_bounds("tau0", (1e-12, 1e0))
    model.parameters.set_value("G0", G0)
    model.parameters.set_value("sigma_y", sigma_y_val)
    model.parameters.set_value("chi_inf", 0.15)
    model.parameters.set_value("tau0", 1e-3)
    model.parameters.set_value("epsilon0", 0.1)
    model.parameters.set_value("c0", 1.0)

    t = jnp.linspace(0.0, 1e-3, 50)
    p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}

    strain = np.asarray(
        model._simulate_transient_jit(
            t,
            p_values,
            "creep",
            None,
            sigma_applied,
            None,
            model.variant,
        )
    )

    expected = sigma_applied / G0
    assert np.isfinite(strain).all()
    assert np.isclose(strain[0], expected, rtol=1e-6), (
        f"creep gamma(0+) should equal sigma/G0 = {expected:.6e}, got {strain[0]:.6e}"
    )


@pytest.mark.smoke
def test_creep_strain_monotone_increasing_above_yield():
    """When sigma_applied >> sigma_y, creep strain must accumulate past the elastic offset.

    Uses a 4x stress overdrive with chi_inf=1.0 so plastic flow accumulates
    on the t~10 s timescale (STZ plastic rate is exponentially suppressed at
    low chi_inf — overdriving the stress unlocks measurable flow).
    """
    model = STZConventional(variant="standard")
    model.parameters.set_bounds("G0", (100.0, 1e5))
    model.parameters.set_bounds("sigma_y", (1.0, 200.0))
    model.parameters.set_bounds("tau0", (1e-8, 1.0))
    model.parameters.set_bounds("chi_inf", (0.05, 2.0))
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("sigma_y", 50.0)
    model.parameters.set_value("chi_inf", 1.0)
    model.parameters.set_value("tau0", 1e-4)
    model.parameters.set_value("epsilon0", 0.1)
    model.parameters.set_value("c0", 1.0)

    sigma_applied = 200.0
    t = jnp.linspace(0.0, 10.0, 80)
    p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}
    strain = np.asarray(
        model._simulate_transient_jit(
            t, p_values, "creep", None, sigma_applied, None, model.variant
        )
    )

    gamma_e = sigma_applied / 1000.0
    assert np.isclose(strain[0], gamma_e, rtol=1e-6)
    assert np.all(np.diff(strain) >= -1e-9), "strain must be monotonic non-decreasing"
    assert strain[-1] > gamma_e * 2.0, (
        "above-yield plastic flow must dominate at late time"
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_creep_roundtrip_synthetic_high_R2():
    """STZ creep should fit its own forward simulation with R^2 > 0.99.

    Generate synthetic creep using known parameters, then refit. This is a
    sanity check that the elastic-offset IC plus the plastic ODE together
    are self-consistent under NLSQ. Prior to the IC fix, this test would
    fail with the optimizer stuck at R^2 < 0.

    ODE-per-residual-evaluation NLSQ measures ~176s (genuine, non-reducible
    compute cost -- see 0ac133e6's HVM demo fix for the same pattern), just
    past the project's default 120s pytest-timeout. 300s gives ~1.7x margin.
    """
    true_params = {
        "G0": 1000.0,
        "sigma_y": 50.0,
        "chi_inf": 0.5,
        "tau0": 1e-4,
        "epsilon0": 0.1,
        "c0": 1.0,
        "ez": 1.0,
        "tau_beta": 1.0,
    }
    sigma_applied = 200.0  # 4x overdrive over sigma_y
    t_data = np.linspace(0.1, 5.0, 40)

    # --- Forward: generate noiseless synthetic data ---
    gen = STZConventional(variant="standard")
    gen.parameters.set_bounds("G0", (100.0, 1e6))
    gen.parameters.set_bounds("sigma_y", (1.0, 200.0))
    gen.parameters.set_bounds("tau0", (1e-8, 1.0))
    gen.parameters.set_bounds("chi_inf", (0.05, 2.0))
    for k, v in true_params.items():
        gen.parameters.set_value(k, v)
    p_values = {k: gen.parameters.get_value(k) for k in gen.parameters.keys()}
    strain_true = np.asarray(
        gen._simulate_transient_jit(
            jnp.asarray(t_data),
            p_values,
            "creep",
            None,
            sigma_applied,
            None,
            gen.variant,
        )
    )
    assert np.isclose(strain_true[0], sigma_applied / true_params["G0"], rtol=1e-6)
    assert strain_true[-1] > strain_true[0], "synthetic creep must increase"

    # --- Inverse: fit to the synthetic data starting from perturbed init ---
    fit_model = STZConventional(variant="standard")
    fit_model.parameters.set_bounds("G0", (100.0, 1e6))
    fit_model.parameters.set_bounds("sigma_y", (1.0, 200.0))
    fit_model.parameters.set_bounds("tau0", (1e-8, 1.0))
    fit_model.parameters.set_bounds("chi_inf", (0.05, 2.0))
    fit_model.parameters.set_value("G0", 1500.0)  # 50% perturbation
    fit_model.parameters.set_value("sigma_y", 40.0)
    fit_model.parameters.set_value("chi_inf", 0.4)
    fit_model.parameters.set_value("tau0", 2e-4)
    fit_model.parameters.set_value("epsilon0", 0.12)
    fit_model.parameters.set_value("c0", 1.0)
    fit_model.parameters.set_value("ez", 1.0)
    fit_model.parameters.set_value("tau_beta", 0.5)

    fit_model.fit(
        t_data,
        strain_true,
        test_mode="creep",
        sigma_applied=sigma_applied,
        method="scipy",
    )
    strain_pred = np.asarray(fit_model.predict(t_data))
    ss_res = float(np.sum((strain_true - strain_pred) ** 2))
    ss_tot = float(np.sum((strain_true - strain_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"STZ creep self-fit should give R^2 > 0.99, got {r2:.4f}"
