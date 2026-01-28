#!/usr/bin/env python
"""Test script to validate MIKH notebooks 01-06 core functionality.

This script extracts and tests the critical code paths from each notebook
without requiring full notebook execution.
"""

import sys
import os
import time
import traceback

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import numpy as np
from rheojax.models.ikh import MIKH
from ikh_tutorial_utils import (
    load_ml_ikh_flow_curve,
    load_pnas_startup,
    load_ml_ikh_creep,
    load_pnas_laos,
    load_ikh_parameters,
    set_model_parameters,
    generate_synthetic_relaxation,
    generate_synthetic_saos,
    get_mikh_param_names,
    compute_fit_quality,
)


def test_01_flow_curve():
    """Test NB01: MIKH Flow Curve fitting."""
    print("\n" + "="*60)
    print("Testing NB01: MIKH Flow Curve")
    print("="*60)

    try:
        # Load data
        gamma_dot, stress = load_ml_ikh_flow_curve('ARES_up')
        print(f"✓ Data loaded: {len(gamma_dot)} points")

        # NLSQ fit
        model = MIKH()
        model.fit(gamma_dot, stress, test_mode='flow_curve')
        print(f"✓ NLSQ fit completed")

        # Predict
        stress_pred = model.predict_flow_curve(gamma_dot)
        metrics = compute_fit_quality(stress, stress_pred)
        print(f"✓ Prediction R² = {metrics['R2']:.4f}")

        # Bayesian (minimal for speed)
        param_names = get_mikh_param_names()
        initial_values = {name: model.parameters.get_value(name) for name in param_names}
        result = model.fit_bayesian(
            gamma_dot, stress,
            test_mode='flow_curve',
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            initial_values=initial_values,
            seed=42,
        )
        print(f"✓ Bayesian inference completed")
        print(f"✓ NB01 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB01 FAILED: {e}")
        traceback.print_exc()
        return False


def test_02_startup_shear():
    """Test NB02: MIKH Startup Shear fitting."""
    print("\n" + "="*60)
    print("Testing NB02: MIKH Startup Shear")
    print("="*60)

    try:
        # Load data
        t, stress = load_pnas_startup(gamma_dot=1.0)
        print(f"✓ Data loaded: {len(t)} points")

        # NLSQ fit
        model = MIKH()
        model.fit(t, stress, test_mode='startup', gamma_dot=1.0)
        print(f"✓ NLSQ fit completed")

        # Predict
        stress_pred = model.predict_startup(t, gamma_dot=1.0)
        metrics = compute_fit_quality(stress, stress_pred)
        print(f"✓ Prediction R² = {metrics['R2']:.4f}")

        # Bayesian (minimal)
        param_names = get_mikh_param_names()
        initial_values = {name: model.parameters.get_value(name) for name in param_names}
        result = model.fit_bayesian(
            t, stress,
            test_mode='startup',
            gamma_dot=1.0,
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            initial_values=initial_values,
            seed=42,
        )
        print(f"✓ Bayesian inference completed")
        print(f"✓ NB02 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB02 FAILED: {e}")
        traceback.print_exc()
        return False


def test_03_relaxation():
    """Test NB03: MIKH Stress Relaxation with synthetic data."""
    print("\n" + "="*60)
    print("Testing NB03: MIKH Stress Relaxation")
    print("="*60)

    try:
        # Load calibrated parameters or use defaults
        try:
            calibrated_params = load_ikh_parameters("mikh", "flow_curve")
            print(f"✓ Loaded calibrated parameters from NB01")
        except FileNotFoundError:
            print("⚠ NB01 results not found, using default parameters")
            calibrated_params = None

        # Create model with parameters
        model = MIKH()
        if calibrated_params:
            set_model_parameters(model, calibrated_params)

        # Generate synthetic data
        t, stress = generate_synthetic_relaxation(
            model, sigma_0=100.0, t_end=500.0, n_points=100, noise_level=0.02, seed=42
        )
        print(f"✓ Synthetic data generated: {len(t)} points")

        # Fit
        model_fit = MIKH()
        model_fit.fit(t, stress, test_mode='relaxation', sigma_0=100.0)
        print(f"✓ NLSQ fit completed")

        # Predict
        stress_pred = model_fit.predict_relaxation(t, sigma_0=100.0)
        metrics = compute_fit_quality(stress, stress_pred)
        print(f"✓ Prediction R² = {metrics['R2']:.4f}")

        print(f"✓ NB03 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB03 FAILED: {e}")
        traceback.print_exc()
        return False


def test_04_creep():
    """Test NB04: MIKH Creep fitting."""
    print("\n" + "="*60)
    print("Testing NB04: MIKH Creep")
    print("="*60)

    try:
        # Load data
        t, gamma_dot, sigma_i, sigma_f = load_ml_ikh_creep(stress_pair_index=1)
        print(f"✓ Data loaded: {len(t)} points, stress {sigma_i} → {sigma_f} Pa")

        # Fit (Note: creep output is shear rate, not stress)
        model = MIKH()
        model.fit(t, gamma_dot, test_mode='creep', sigma_applied=sigma_f)
        print(f"✓ NLSQ fit completed")

        # Predict (returns strain, compute derivative for rate)
        gamma_pred = model.predict_creep(t, sigma_applied=sigma_f)
        gamma_dot_pred = np.gradient(np.array(gamma_pred), np.array(t))
        metrics = compute_fit_quality(gamma_dot, gamma_dot_pred)
        print(f"✓ Prediction R² = {metrics['R2']:.4f}")

        print(f"✓ NB04 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB04 FAILED: {e}")
        traceback.print_exc()
        return False


def test_05_saos():
    """Test NB05: MIKH SAOS with synthetic data."""
    print("\n" + "="*60)
    print("Testing NB05: MIKH SAOS")
    print("="*60)

    try:
        # Load calibrated parameters or use defaults
        try:
            calibrated_params = load_ikh_parameters("mikh", "flow_curve")
            print(f"✓ Loaded calibrated parameters from NB01")
        except FileNotFoundError:
            print("⚠ NB01 results not found, using default parameters")
            calibrated_params = None

        # Create model with parameters
        model = MIKH()
        if calibrated_params:
            set_model_parameters(model, calibrated_params)

        # Generate synthetic SAOS data
        omega, G_prime, G_double_prime = generate_synthetic_saos(
            model, omega_range=(0.01, 100.0), n_points=20, noise_level=0.02, seed=42
        )
        print(f"✓ Synthetic SAOS data generated: {len(omega)} points")

        # Fit to complex modulus magnitude
        G_star = G_prime + 1j * G_double_prime
        model_fit = MIKH()
        model_fit.fit(omega, np.abs(G_star), test_mode='oscillation')
        print(f"✓ NLSQ fit completed")

        # Verify Maxwell parameters recovered
        G_fit = model_fit.parameters.get_value("G")
        eta_fit = model_fit.parameters.get_value("eta")
        print(f"✓ Fitted G={G_fit:.4g} Pa, eta={eta_fit:.4g} Pa·s")

        print(f"✓ NB05 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB05 FAILED: {e}")
        traceback.print_exc()
        return False


def test_06_laos():
    """Test NB06: MIKH LAOS fitting."""
    print("\n" + "="*60)
    print("Testing NB06: MIKH LAOS")
    print("="*60)

    try:
        # Load data
        t, strain, stress = load_pnas_laos(omega=1.0, strain_amplitude_index=8)
        gamma_0 = np.max(np.abs(strain))
        print(f"✓ Data loaded: {len(t)} points, γ₀={gamma_0:.4f}")

        # Fit
        model = MIKH()
        model.fit(t, stress, test_mode='laos', gamma_0=gamma_0, omega=1.0)
        print(f"✓ NLSQ fit completed")

        # Predict
        stress_pred = model.predict_laos(t, gamma_0=gamma_0, omega=1.0)
        metrics = compute_fit_quality(stress, stress_pred)
        print(f"✓ Prediction R² = {metrics['R2']:.4f}")

        print(f"✓ NB06 PASSED")
        return True

    except Exception as e:
        print(f"✗ NB06 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all notebook tests."""
    print("="*60)
    print("MIKH Notebooks (01-06) Validation Test Suite")
    print("="*60)

    start_time = time.time()

    results = {
        "NB01_flow_curve": test_01_flow_curve(),
        "NB02_startup_shear": test_02_startup_shear(),
        "NB03_relaxation": test_03_relaxation(),
        "NB04_creep": test_04_creep(),
        "NB05_saos": test_05_saos(),
        "NB06_laos": test_06_laos(),
    }

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:25s} {status_str}")

    print(f"\nTotal: {passed}/{total} passed in {elapsed:.1f}s")

    if passed == total:
        print("\n🎉 All MIKH notebooks validated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - passed} notebook(s) failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
