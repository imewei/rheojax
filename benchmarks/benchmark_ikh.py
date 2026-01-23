
import time
import numpy as np
from rheojax.models.ikh.mikh import MIKH
from rheojax.core.jax_config import safe_import_jax

# Ensure JAX uses float64
jax, jnp = safe_import_jax()

def python_ikh_kernel(times, strains, params):
    """Pure Python/NumPy implementation of MIKH kernel for benchmarking."""
    # Unpack parameters
    G = params['G']
    C = params['C']
    gamma_dyn = params['gamma_dyn']
    sigma_y0 = params['sigma_y0']
    delta_sigma_y = params['delta_sigma_y']
    tau_thix = params['tau_thix']
    Gamma = params['Gamma']
    eta_inf = params['eta_inf']

    # Initialize state
    sigma = 0.0
    alpha = 0.0
    lam = 1.0

    stress_history = np.zeros(len(times))

    # Pre-calculate increments
    dts = np.diff(times, prepend=times[0])
    d_gammas = np.diff(strains, prepend=strains[0])

    for i in range(len(times)):
        dt = dts[i]
        d_gamma = d_gammas[i]

        # Avoid division by zero
        safe_dt = max(dt, 1e-9)
        gamma_dot = d_gamma / safe_dt
        gamma_dot_abs = abs(gamma_dot)

        # 1. Update structure (Explicit Euler)
        # d(lambda)/dt = (1 - lambda)/tau_thix - Gamma * lambda * gamma_dot
        build_up = (1.0 - lam) / max(tau_thix, 1e-6)
        break_down = Gamma * lam * gamma_dot_abs
        d_lambda = (build_up - break_down) * dt

        lam = np.clip(lam + d_lambda, 0.0, 1.0)

        # 2. Update Yield Stress
        sigma_y_current = sigma_y0 + delta_sigma_y * lam

        # 3. Elastic Predictor
        sigma_trial = sigma + G * d_gamma
        xi_trial = sigma_trial - alpha
        norm_xi_trial = abs(xi_trial)

        # 4. Yield Condition
        f_yield = norm_xi_trial - sigma_y_current

        # 5. Plastic Corrector
        if f_yield > 0:
            # Plastic step
            denom = G + C
            d_gamma_p = f_yield / denom
            sign_xi = np.sign(xi_trial)

            # Stress update
            sigma = sigma_trial - G * d_gamma_p * sign_xi

            # Backstress update
            d_alpha = (C * sign_xi - gamma_dyn * alpha) * d_gamma_p
            alpha = alpha + d_alpha
        else:
            # Elastic step
            sigma = sigma_trial
            # alpha remains unchanged

        # 6. Total Stress
        stress_history[i] = sigma + eta_inf * gamma_dot

    return stress_history

def run_benchmark():
    print("=" * 60)
    print("MIKH Benchmark: JAX (Scan) vs Pure Python (Loop)")
    print("=" * 60)

    # 1. Setup
    N = 10000
    print(f"Data points: {N}")

    t = np.linspace(0, 10.0, N)
    # Oscillatory shear input
    gamma = 5.0 * np.sin(2.0 * np.pi * 1.0 * t)

    # Initialize model
    model = MIKH()
    params = model.parameters.get_values()
    param_dict = {name: val for name, val in zip(model.parameters.keys(), params)}

    # 2. Python Benchmark
    print("\nRunning Python implementation...")
    start_py = time.perf_counter()
    res_py = python_ikh_kernel(t, gamma, param_dict)
    end_py = time.perf_counter()
    time_py = end_py - start_py
    print(f"Python time: {time_py:.4f} s")

    # 3. JAX Benchmark
    print("\nRunning JAX implementation...")

    # Warmup (Compilation)
    print("Compiling JAX kernel...")
    start_compile = time.perf_counter()
    _ = model.predict(t, strain=gamma)
    end_compile = time.perf_counter()
    print(f"Compilation time: {end_compile - start_compile:.4f} s")

    # Execution (Compiled)
    print("Executing compiled kernel...")
    start_jax = time.perf_counter()
    res_jax = model.predict(t, strain=gamma)
    res_jax.block_until_ready() # Ensure async execution finishes
    end_jax = time.perf_counter()
    time_jax = end_jax - start_jax
    print(f"JAX time:    {time_jax:.4f} s")

    # 4. Comparison
    speedup = time_py / time_jax
    print("-" * 60)
    print(f"Speedup: {speedup:.1f}x")

    # Verification
    # Note: Small numerical differences expected due to float precision handling and clip differences
    diff = np.abs(res_py - res_jax)
    max_diff = np.max(diff)
    print(f"\nMax difference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("Verification: PASSED")
    else:
        print("Verification: WARNING (differences detected)")

if __name__ == "__main__":
    run_benchmark()
