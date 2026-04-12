"""Base class for Elasto-Plastic Models (EPM).

This module provides the abstract base class for all EPM variants (scalar lattice,
tensorial, etc.), extracting common parameters, initialization logic, and protocol
runner templates.
"""

import time as time_module
from abc import abstractmethod
from functools import partial

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# JIT-compiled EPM core functions for Bayesian inference
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6), static_argnames=("fluidity_form",))
def _jit_flow_curve_single(
    gdot: float,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    dt: float,
    n_steps: int,
    L: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled flow curve for a single shear rate.

    Args:
        gdot: Shear rate
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width]
        dt: Time step
        n_steps: Number of simulation steps
        L: Lattice size
        fluidity_form: Constitutive law — "linear", "power", or "overstress" (default).
            See `compute_plastic_strain_rate` for the mathematical forms.

    Returns:
        Steady-state stress
    """
    # Unpack params (order matches EPMBase parameter declaration)
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    n_fluid = params_array[4]
    smoothing_width = params_array[5]

    fluidity = 1.0 / tau_pl
    inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)

    # Initialize state. The stress field is warm-started near the expected
    # power-law-fluidity steady state so that the scan only needs to relax
    # residual transients. Starting from zero would require elastic loading
    # time = sigma_c_mean / (mu * gdot) which exceeds the (n_steps * dt)
    # window at low rates and misses the plateau entirely.
    #
    # The analytical steady state with power-law fluidity is
    #     stress ~ sigma_c_mean + sigma_c_mean * (|gdot| * tau_pl / sigma_c_mean)^(1/n_fluid)
    # At n_fluid=1 this reduces to the classical Bingham estimate
    # sigma_c_mean + |gdot|*tau_pl. At n_fluid>1 the second term grows much
    # more slowly, which is critical: the old linear warm-start overshoots
    # by orders of magnitude at high gdot and causes kernel overflow when
    # raised to the fractional power.
    k1, k2 = jax.random.split(key)
    gdot_abs = jnp.abs(gdot)
    warm_excess = sigma_c_mean * (gdot_abs * tau_pl * inv_scm) ** (1.0 / n_fluid)
    sigma_warm = jnp.sign(gdot) * (sigma_c_mean + warm_excess)
    stress = sigma_warm * jnp.ones((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Smooth plastic strain rate, dispatched by `fluidity_form` (static arg).
        # See `compute_plastic_strain_rate` in rheojax.utils.epm_kernels for the
        # mathematical description of each form.
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )

        if fluidity_form == "linear":
            # Classical Bingham: plastic_rate = activation * sigma * fluidity
            plastic_strain_rate = activation * stress_curr * fluidity
        elif fluidity_form == "power":
            # Power-law fluidity (no additive yield-stress baseline).
            stress_mag_safe = jnp.maximum(stress_mag, 1e-8)
            power_term = (
                jnp.sign(stress_curr)
                * (stress_mag_safe * inv_scm) ** n_fluid
                * sigma_c_mean
            )
            plastic_strain_rate = activation * power_term * fluidity
        elif fluidity_form == "overstress":
            # Herschel-Bulkley: only overstress (|sigma| - sigma_c_mean)_+ drives flow.
            # Steady-state high-rate asymptote: sigma = sigma_c_mean + sigma_c_mean *
            # (gamma_dot * tau_pl / sigma_c_mean)^(1 / n_fluid) — full HB form.
            eps = 1e-6
            overstress = jnp.maximum(stress_mag - sigma_c_mean, eps)
            overstress_power = overstress**n_fluid * (sigma_c_mean ** (1.0 - n_fluid))
            plastic_strain_rate = (
                activation * jnp.sign(stress_curr) * overstress_power * fluidity
            )
        else:
            raise ValueError(
                f"Unknown fluidity_form={fluidity_form!r}; "
                "must be 'linear', 'power', or 'overstress'."
            )

        # Stress evolution
        loading_rate = mu * gdot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = (
            stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        )
        new_strain = strain_curr + gdot * dt

        return (new_stress, thresholds_curr, new_strain, key_curr), jnp.mean(new_stress)

    _, history = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    # Average second half for steady state
    steady_stress = jnp.mean(history[n_steps // 2 :])
    return steady_stress


@partial(jax.jit, static_argnums=(5, 6, 7), static_argnames=("fluidity_form",))
def _jit_flow_curve_batch(
    shear_rates: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    dt: float,
    n_steps: int,
    L: int,
    n_rates: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled flow curve for batch of shear rates.

    Args:
        shear_rates: Array of shear rates
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width]
        dt: Time step
        n_steps: Number of simulation steps
        L: Lattice size
        n_rates: Number of shear rates (static for JIT)
        fluidity_form: Constitutive law ("linear", "power", or "overstress").

    Returns:
        Array of steady-state stresses
    """
    # Use different keys for each shear rate
    keys = jax.random.split(key, n_rates)

    def single_rate(gdot_key):
        gdot, k = gdot_key
        return _jit_flow_curve_single(
            gdot, k, propagator_q, params_array, dt, n_steps, L,
            fluidity_form=fluidity_form,
        )

    return jax.vmap(single_rate)((shear_rates, keys))


@partial(jax.jit, static_argnames=("n_steps", "L", "fluidity_form"))
def _jit_startup_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    gamma_dot: float,
    dt: float,
    n_steps: int,
    L: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled startup shear simulation.

    ``fluidity_form`` selects the plastic-strain-rate constitutive law,
    matching the relaxation/creep kernels and the Python ``_run_startup``
    path so that fit ≡ predict for all constitutive forms (P0-2 fix).

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width]
        gamma_dot: Applied shear rate
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size
        fluidity_form: Constitutive law for the plastic strain rate.

    Returns:
        Array of stress over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    n_fluid = params_array[4]
    smoothing_width = params_array[5]

    fluidity = 1.0 / tau_pl

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Smooth activation (matches epm_kernels.plastic_strain_rate)
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )

        # Plastic strain rate — fluidity_form is static, only one branch
        # is emitted into the compiled graph.
        if fluidity_form == "linear":
            plastic_strain_rate = activation * stress_curr * fluidity
        elif fluidity_form == "power":
            inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
            stress_mag_safe = jnp.maximum(stress_mag, 1e-8)
            power_term = (
                jnp.sign(stress_curr)
                * (stress_mag_safe * inv_scm) ** n_fluid
                * sigma_c_mean
            )
            plastic_strain_rate = activation * power_term * fluidity
        elif fluidity_form == "overstress":
            eps = 1e-6
            overstress = jnp.maximum(stress_mag - sigma_c_mean, eps)
            overstress_power = overstress**n_fluid * (
                sigma_c_mean ** (1.0 - n_fluid)
            )
            plastic_strain_rate = (
                activation * jnp.sign(stress_curr) * overstress_power * fluidity
            )
        else:
            raise ValueError(
                f"Unknown fluidity_form={fluidity_form!r}; "
                "must be 'linear', 'power', or 'overstress'."
            )

        # Stress evolution
        loading_rate = mu * gamma_dot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = (
            stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        )
        new_strain = strain_curr + gamma_dot * dt

        return (new_stress, thresholds_curr, new_strain, key_curr), jnp.mean(new_stress)

    initial_stress = jnp.mean(stress)
    _, stresses_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([initial_stress]), stresses_scan])


@partial(jax.jit, static_argnames=("n_steps", "L", "fluidity_form"))
def _jit_relaxation_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    strain_step: float,
    dt: float,
    n_steps: int,
    L: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled stress relaxation simulation.

    ``fluidity_form`` selects the plastic-strain-rate constitutive law,
    matching ``rheojax.utils.epm_kernels.plastic_strain_rate`` so the fit
    path (this JIT kernel) and the predict path (Python ``_run_relaxation``
    → ``epm_step``) produce identical trajectories for the same parameters.
    Without this port, NLSQ round-trip fits are impossible on above-yield
    step strains because the JIT kernel would use the linear Bingham form
    while the predict path uses the HB overstress default.

    Note: the kernel assumes a *uniformly-spaced* time grid. The caller
    passes ``dt = time[1] - time[0]`` and the scan runs ``n_steps`` fixed
    Euler steps of size ``dt``. Log-spaced or piecewise-uniform data will
    silently be simulated only up to ``n_steps * dt`` physical seconds and
    mapped back onto the non-uniform ``time`` array incorrectly. Resample
    to uniform spacing before fitting.

    Args:
        time: Time array (only its shape is used).
        key: PRNG key.
        propagator_q: Propagator in Fourier space.
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width].
        strain_step: Applied step strain.
        dt: Time step (must equal ``time[1] - time[0]`` for correct mapping).
        n_steps: Number of time points minus 1.
        L: Lattice size.
        fluidity_form: Constitutive law for the plastic strain rate.

    Returns:
        Array of modulus G(t) over time.
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    n_fluid = params_array[4]
    smoothing_width = params_array[5]

    fluidity = 1.0 / tau_pl

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    # Apply step strain at t=0
    stress = stress + mu * strain_step
    strain = strain + strain_step

    g_0 = jnp.mean(stress) / strain_step

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Smooth activation (matches epm_kernels.plastic_strain_rate)
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )

        # Plastic strain rate — fluidity_form is static, only one branch
        # is emitted into the compiled graph.
        if fluidity_form == "linear":
            plastic_strain_rate = activation * stress_curr * fluidity
        elif fluidity_form == "power":
            inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
            stress_mag_safe = jnp.maximum(stress_mag, 1e-8)
            power_term = (
                jnp.sign(stress_curr)
                * (stress_mag_safe * inv_scm) ** n_fluid
                * sigma_c_mean
            )
            plastic_strain_rate = activation * power_term * fluidity
        elif fluidity_form == "overstress":
            eps = 1e-6
            overstress = jnp.maximum(stress_mag - sigma_c_mean, eps)
            overstress_power = overstress**n_fluid * (
                sigma_c_mean ** (1.0 - n_fluid)
            )
            plastic_strain_rate = (
                activation * jnp.sign(stress_curr) * overstress_power * fluidity
            )
        else:
            raise ValueError(
                f"Unknown fluidity_form={fluidity_form!r}; "
                "must be 'linear', 'power', or 'overstress'."
            )

        # Stress evolution (no loading - relaxation only)
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update (no loading)
        new_stress = stress_curr + (relaxation_rate + redistribution_rate) * dt

        return (new_stress, thresholds_curr, strain_curr, key_curr), jnp.mean(
            new_stress
        ) / strain_step

    _, moduli_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([g_0]), moduli_scan])


@partial(jax.jit, static_argnames=("n_steps", "L", "n_sub", "fluidity_form"))
def _jit_creep_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    target_stress: float,
    dt_sub: float,
    n_steps: int,
    L: int,
    n_sub: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled creep simulation with a substepped P-controller.

    The controller/integrator run at ``dt_sub`` (≤ ``self.dt``). Between
    adjacent data points the kernel takes ``n_sub`` substeps so the ODE step
    stays stable and the controller has enough iterations to drive the lattice
    stress to the target even when the data cadence is coarse.

    ``fluidity_form`` selects the plastic-strain-rate constitutive law, matching
    ``rheojax.utils.epm_kernels.plastic_strain_rate``:
      * ``"linear"`` — Bingham (plastic_rate = activation · σ · fluidity)
      * ``"power"``  — soft-glassy power law
      * ``"overstress"`` — Herschel-Bulkley (DEFAULT), needed for yield-stress
        creep where bounded/unbounded regimes depend on an additive yield
        baseline. Must match the Python ``_run_creep`` predict path so that
        round-trip NLSQ fits converge.

    Args:
        time: Time array (only its shape is used; n_steps = len(time)-1).
        key: PRNG key.
        propagator_q: Propagator in Fourier space.
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width].
        target_stress: Target stress for creep.
        dt_sub: ODE substep (stable step for the explicit integrator).
        n_steps: Number of outer steps = len(time) - 1.
        L: Lattice size.
        n_sub: Number of substeps between adjacent data points.
        fluidity_form: Constitutive law for the plastic strain rate.

    Returns:
        Strain sampled at each data point (length n_steps + 1).
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    n_fluid = params_array[4]
    smoothing_width = params_array[5]

    fluidity = 1.0 / tau_pl

    # P-controller parameters
    Kp_base = 0.01
    alpha = 10.0

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    initial_strain = strain

    def sub_step(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr, gdot = carrier

        # P-controller: adjust shear rate to maintain target stress
        curr_stress = jnp.mean(stress_curr)
        error = target_stress - curr_stress
        rel_error = jnp.abs(error) / (jnp.abs(target_stress) + 1e-6)
        Kp = Kp_base * (1.0 + alpha * rel_error)
        gdot_new = jnp.maximum(gdot + Kp * error, 0.0)

        # Smooth activation (matches epm_kernels.plastic_strain_rate)
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )

        # Plastic strain rate — fluidity_form is static, so only one branch
        # is emitted into the compiled graph.
        if fluidity_form == "linear":
            plastic_strain_rate = activation * stress_curr * fluidity
        elif fluidity_form == "power":
            inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
            stress_mag_safe = jnp.maximum(stress_mag, 1e-8)
            power_term = (
                jnp.sign(stress_curr)
                * (stress_mag_safe * inv_scm) ** n_fluid
                * sigma_c_mean
            )
            plastic_strain_rate = activation * power_term * fluidity
        elif fluidity_form == "overstress":
            eps = 1e-6
            overstress = jnp.maximum(stress_mag - sigma_c_mean, eps)
            overstress_power = overstress**n_fluid * (
                sigma_c_mean ** (1.0 - n_fluid)
            )
            plastic_strain_rate = (
                activation * jnp.sign(stress_curr) * overstress_power * fluidity
            )
        else:
            raise ValueError(
                f"Unknown fluidity_form={fluidity_form!r}; "
                "must be 'linear', 'power', or 'overstress'."
            )

        # Stress evolution at dt_sub
        loading_rate = mu * gdot_new
        relaxation_rate = -mu * plastic_strain_rate
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        new_stress = (
            stress_curr
            + (loading_rate + relaxation_rate + redistribution_rate) * dt_sub
        )
        new_strain = strain_curr + gdot_new * dt_sub

        return (new_stress, thresholds_curr, new_strain, key_curr, gdot_new), None

    def outer_step(carrier, _):
        carrier_next, _ = jax.lax.scan(sub_step, carrier, None, length=n_sub)
        strain_sampled = carrier_next[2]
        return carrier_next, strain_sampled

    _, strains_scan = jax.lax.scan(
        outer_step, (stress, thresholds, strain, k2, 0.0), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([initial_strain]), strains_scan])


@partial(jax.jit, static_argnames=("n_steps", "L", "fluidity_form"))
def _jit_oscillation_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    gamma0: float,
    omega: float,
    dt: float,
    n_steps: int,
    L: int,
    fluidity_form: str = "overstress",
) -> jax.Array:
    """JIT-compiled oscillatory shear simulation.

    ``fluidity_form`` selects the plastic-strain-rate constitutive law,
    matching the relaxation/creep kernels and the Python ``_run_oscillation``
    path so that fit ≡ predict for all constitutive forms (P0-2 fix).

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, n_fluid, smoothing_width]
        gamma0: Strain amplitude
        omega: Angular frequency
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size
        fluidity_form: Constitutive law for the plastic strain rate.

    Returns:
        Array of stress over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    n_fluid = params_array[4]
    smoothing_width = params_array[5]

    fluidity = 1.0 / tau_pl

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    initial_stress = jnp.mean(stress)
    scan_time = time[:-1]

    def body_fn(carrier, t):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Time-varying shear rate
        gdot = gamma0 * omega * jnp.cos(omega * t)

        # Smooth activation (matches epm_kernels.plastic_strain_rate)
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )

        # Plastic strain rate — fluidity_form is static, only one branch
        # is emitted into the compiled graph.
        if fluidity_form == "linear":
            plastic_strain_rate = activation * stress_curr * fluidity
        elif fluidity_form == "power":
            inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
            stress_mag_safe = jnp.maximum(stress_mag, 1e-8)
            power_term = (
                jnp.sign(stress_curr)
                * (stress_mag_safe * inv_scm) ** n_fluid
                * sigma_c_mean
            )
            plastic_strain_rate = activation * power_term * fluidity
        elif fluidity_form == "overstress":
            eps = 1e-6
            overstress = jnp.maximum(stress_mag - sigma_c_mean, eps)
            overstress_power = overstress**n_fluid * (
                sigma_c_mean ** (1.0 - n_fluid)
            )
            plastic_strain_rate = (
                activation * jnp.sign(stress_curr) * overstress_power * fluidity
            )
        else:
            raise ValueError(
                f"Unknown fluidity_form={fluidity_form!r}; "
                "must be 'linear', 'power', or 'overstress'."
            )

        # Stress evolution
        loading_rate = mu * gdot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = (
            stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        )
        new_strain = strain_curr + gdot * dt

        return (new_stress, thresholds_curr, new_strain, key_curr), jnp.mean(new_stress)

    _, stresses_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), scan_time, length=n_steps
    )

    return jnp.concatenate([jnp.array([initial_stress]), stresses_scan])


class EPMBase(BaseModel):
    """Abstract base class for Elasto-Plastic Models.

    This class provides common infrastructure for EPM variants:
    - Common parameters (L, dt, mu, sigma_c_mean, sigma_c_std, etc.)
    - Threshold initialization (_init_thresholds)
    - Parameter extraction (_get_param_dict)
    - Protocol runner templates (flow_curve, startup, relaxation, creep, oscillation)

    Subclasses must implement:
    - _init_stress(key): Initialize stress field with appropriate shape
    - _epm_step(...): Call appropriate kernel for their EPM variant

    Parameters:
        L (int): Lattice size (LxL for 2D). Default 64.
        dt (float): Time step for numerical integration. Default 0.01.
        mu (float): Shear modulus. Default 1.0.
        tau_pl (float): Plastic relaxation timescale. Default 1.0.
        sigma_c_mean (float): Mean yield threshold. Default 1.0.
        sigma_c_std (float): Yield threshold standard deviation (disorder). Default 0.1.
        n_bayesian_steps (int): Number of time steps for Bayesian inference.
            Reduced from simulation default (1000) to speed up JIT compilation.
            Default 200.
    """

    def __init__(
        self,
        L: int = 64,
        dt: float = 0.01,
        mu: float = 1.0,
        tau_pl: float = 1.0,
        sigma_c_mean: float = 1.0,
        sigma_c_std: float = 0.1,
        n_fluid: float = 1.0,
        n_bayesian_steps: int = 200,
        fluidity_form: str = "overstress",
    ):
        """Initialize EPM base with common parameters.

        The `fluidity_form` argument selects the constitutive law for the plastic
        strain rate. It is a *static configuration*, not a fitted parameter:

        - ``"linear"`` — classical Bingham EPM (``plastic_rate ~ sigma``). High-rate
          asymptote is ``sigma ~ gamma_dot * tau_pl`` with no yield-stress baseline.
          Use only for genuinely Newtonian-at-high-rate materials.
        - ``"power"`` — power-law fluidity from soft-glassy rheology
          (``plastic_rate ~ |sigma/sigma_c_mean|^n_fluid * sigma_c_mean``).
          Shear-thinning at high rates but no additive yield-stress baseline.
        - ``"overstress"`` — Herschel-Bulkley overstress law
          (``plastic_rate ~ (|sigma| - sigma_c_mean)_+^n_fluid``). Produces the full
          HB form ``sigma = sigma_y + K * gamma_dot^n_HB`` with ``sigma_y = sigma_c_mean``
          and ``n_HB = 1/n_fluid``. **Default and recommended** for yield-stress fluids
          (emulsions, gels, foams, pastes).
        """
        super().__init__()

        # Configuration (Static)
        self.L = L
        self.dt = dt
        self.n_bayesian_steps = n_bayesian_steps
        self._precompiled = False

        # Validate and store the constitutive-law selector.
        if fluidity_form not in ("linear", "power", "overstress"):
            raise ValueError(
                f"fluidity_form must be 'linear', 'power', or 'overstress'; "
                f"got {fluidity_form!r}."
            )
        self.fluidity_form = fluidity_form

        # Parameters (Optimizable) - use inherited self.parameters from BaseModel
        self.parameters.add(
            "mu", mu, bounds=(0.1, 1e9), units="Pa", description="Shear modulus"
        )
        self.parameters.add(
            "tau_pl",
            tau_pl,
            bounds=(0.01, 100.0),
            units="s",
            description="Plastic relaxation timescale",
        )
        self.parameters.add(
            "sigma_c_mean",
            sigma_c_mean,
            bounds=(0.1, 1e6),
            units="Pa",
            description="Mean yield threshold",
        )
        self.parameters.add(
            "sigma_c_std",
            sigma_c_std,
            bounds=(0.0, 100.0),
            units="Pa",
            description="Yield threshold standard deviation (disorder)",
        )
        # n_fluid: exponent of the plastic-flow power law.
        # plastic_strain_rate = activation * sign(s) * |s/sigma_c_mean|^n_fluid * sigma_c_mean / tau_pl
        # At n_fluid=1 this is the classical linear EPM (Bingham-like high-rate asymptote).
        # At n_fluid>1 the high-rate asymptote becomes shear-thinning with exponent 1/n_fluid
        # (e.g. n_fluid=2 -> sigma ~ gamma_dot^0.5, Herschel-Bulkley). This is a power-law
        # fluidity constitutive law, well-established in soft glassy and amorphous plasticity.
        self.parameters.add(
            "n_fluid",
            n_fluid,
            bounds=(0.5, 5.0),
            units="dimensionless",
            description="Power-law fluidity exponent (1=Bingham, 2=HB with n=0.5)",
        )
        self.parameters.add(
            "smoothing_width",
            0.1,
            bounds=(0.01, 100.0),
            units="Pa",
            description="Smooth yielding transition width",
        )

    def _init_thresholds(self, key: jax.Array) -> jax.Array:
        """Initialize yield thresholds from Gaussian distribution.

        Args:
            key: PRNG key for random number generation.

        Returns:
            Array of shape (L, L) with Gaussian-distributed yield thresholds.
        """
        mean = self.parameters.get_value("sigma_c_mean")
        std = self.parameters.get_value("sigma_c_std")
        thresholds = mean + std * jax.random.normal(key, (self.L, self.L))
        # Ensure positive thresholds
        thresholds = jnp.maximum(thresholds, 1e-4)
        return thresholds

    def _get_param_dict(self) -> dict[str, float]:
        """Extract parameters as dictionary for kernel calls.

        Returns:
            Dictionary with all EPM parameters (mu, tau_pl, sigma_c_mean, etc.).
        """
        mu = self.parameters.get_value("mu")
        tau_pl = self.parameters.get_value("tau_pl")
        sigma_c_mean = self.parameters.get_value("sigma_c_mean")
        sigma_c_std = self.parameters.get_value("sigma_c_std")
        n_fluid = self.parameters.get_value("n_fluid")
        smoothing_width = self.parameters.get_value("smoothing_width")
        if mu is None:
            raise ValueError("Parameter 'mu' must be set before use")
        if tau_pl is None:
            raise ValueError("Parameter 'tau_pl' must be set before use")
        if sigma_c_mean is None:
            raise ValueError("Parameter 'sigma_c_mean' must be set before use")
        if sigma_c_std is None:
            raise ValueError("Parameter 'sigma_c_std' must be set before use")
        if n_fluid is None:
            raise ValueError("Parameter 'n_fluid' must be set before use")
        if smoothing_width is None:
            raise ValueError("Parameter 'smoothing_width' must be set before use")
        return {
            "mu": mu,
            "tau_pl": tau_pl,
            "sigma_c_mean": sigma_c_mean,
            "sigma_c_std": sigma_c_std,
            "n_fluid": n_fluid,
            "smoothing_width": smoothing_width,
        }

    @abstractmethod
    def _init_stress(self, key: jax.Array) -> jax.Array:
        """Initialize stress field (subclass-specific shape).

        Args:
            key: PRNG key for random number generation.

        Returns:
            Stress array with shape appropriate for EPM variant:
            - Scalar EPM: (L, L)
            - Tensorial EPM: (L, L, 3) for (σ_xx, σ_xy, σ_yy)
        """
        pass

    @abstractmethod
    def _epm_step(
        self,
        state: tuple,
        propagator_q: jax.Array,
        shear_rate: float,
        dt: float,
        params: dict,
        smooth: bool,
    ) -> tuple:
        """Perform one EPM time step (subclass-specific kernel).

        Args:
            state: Current state tuple (stress, thresholds, strain, key).
            propagator_q: Precomputed Fourier-space propagator.
            shear_rate: Imposed macroscopic shear rate.
            dt: Time step size.
            params: Dictionary of model parameters.
            smooth: Whether to use smooth yielding (True) or hard threshold (False).

        Returns:
            Updated state tuple.
        """
        pass

    def _init_state(
        self, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, float, jax.Array]:
        """Initialize full simulation state.

        Args:
            key: PRNG key for random number generation.

        Returns:
            Tuple (stress, thresholds, strain, key) where:
            - stress: Initialized stress field (shape from _init_stress)
            - thresholds: Yield thresholds (L, L)
            - strain: Accumulated macroscopic strain (scalar 0.0)
            - key: Updated PRNG key
        """
        k1, k2 = jax.random.split(key)

        # Subclass determines stress shape
        stress = self._init_stress(k1)

        # Common threshold initialization
        thresholds = self._init_thresholds(k2)

        strain = 0.0

        return (stress, thresholds, strain, k2)

    # --- Protocol Runner Templates ---
    # These methods call _epm_step in loops for different test protocols

    def _run_flow_curve(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Steady state flow curve: Stress vs Shear Rate.

        Args:
            data: RheoData with x=shear_rates.
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=shear_rates, y=steady_stress.
        """
        shear_rates = data.x
        sigma_c_mean = params["sigma_c_mean"]
        tau_pl = params["tau_pl"]
        n_fluid = params.get("n_fluid", 1.0)

        def scan_fn(gdot):
            # Run simulation for sufficient steps to reach steady state
            n_steps = 1000
            state = self._init_state(key)
            stress0, thresholds, strain, k = state

            # Warm-start at the analytical steady state for the configured
            # fluidity form. The same formula works for both the "power" and
            # "overstress" forms (at steady state they agree at leading order):
            #   sigma = sigma_c_mean + sigma_c_mean * (|gdot| * tau_pl / sigma_c_mean)^(1/n_fluid)
            # For scalar EPM the field has shape (L, L); for tensorial EPM it
            # is (3, L, L) with index 2 == sigma_xy.
            gdot_abs = jnp.abs(gdot)
            inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
            warm_excess = sigma_c_mean * (gdot_abs * tau_pl * inv_scm) ** (1.0 / n_fluid)
            sigma_warm = jnp.sign(gdot) * (sigma_c_mean + warm_excess)
            if stress0.ndim == 2:
                stress0 = stress0 + sigma_warm
            else:
                stress0 = stress0.at[2].add(sigma_warm)
            state = (stress0, thresholds, strain, k)

            def body(carrier, _):
                curr_state = carrier
                new_state = self._epm_step(
                    curr_state, propagator_q, gdot, self.dt, params, smooth
                )
                # Extract stress mean (works for both scalar and tensorial)
                return new_state, jnp.mean(new_state[0])

            _, history = jax.lax.scan(body, state, None, length=n_steps)

            # Average last 50% for steady state
            steady_stress = jnp.mean(history[n_steps // 2 :])
            return steady_stress

        # Vectorize over shear rates
        stresses = jax.vmap(scan_fn)(shear_rates)
        return RheoData(x=shear_rates, y=stresses, initial_test_mode="flow_curve")

    def _run_startup(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Start-up shear: Stress(t) at constant rate.

        Args:
            data: RheoData with x=time, metadata['gamma_dot'].
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=time, y=stress.
        """
        time = data.x
        if time is None:
            raise ValueError("data.x (time array) must not be None")

        # Calculate dt from data if possible
        dt = self.dt
        if len(time) > 1:
            dt = float(time[1] - time[0])

        # Constant shear rate from metadata
        gdot = data.metadata.get("gamma_dot", 0.1)

        # Scan for N-1 steps
        n_steps = max(0, len(time) - 1)
        state = self._init_state(key)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(
                curr_state, propagator_q, gdot, dt, params, smooth
            )
            return new_state, jnp.mean(new_state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, None, length=n_steps)
            # Prepend initial stress
            initial_stress = jnp.mean(state[0])
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([jnp.mean(state[0])])

        return RheoData(x=time, y=stresses, initial_test_mode="startup")

    def _run_relaxation(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Stress relaxation: G(t) after step strain.

        Args:
            data: RheoData with x=time, metadata['gamma'].
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=time, y=modulus.
        """
        time = data.x
        if time is None:
            raise ValueError("data.x (time array) must not be None")

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = float(time[1] - time[0])

        # Step strain magnitude from metadata
        strain_step = data.metadata.get("gamma", 0.1)

        state = self._init_state(key)
        stress, thresh, strain, k = state

        # Apply Step Strain (Elastic Load)
        mu = params["mu"]
        stress = stress + mu * strain_step
        state = (stress, thresh, strain + strain_step, k)

        # Initial G(0)
        g_0 = jnp.mean(stress) / strain_step

        # Relax (gdot = 0) for N-1 steps
        n_steps = max(0, len(time) - 1)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(
                curr_state, propagator_q, 0.0, dt, params, smooth
            )
            # Return G(t) = Stress / gamma_0
            return new_state, jnp.mean(new_state[0]) / strain_step

        if n_steps > 0:
            _, moduli_scan = jax.lax.scan(body, state, None, length=n_steps)
            moduli = jnp.concatenate([jnp.array([g_0]), moduli_scan])
        else:
            moduli = jnp.array([g_0])

        return RheoData(x=time, y=moduli, initial_test_mode="relaxation")

    def _run_creep(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Creep: Strain(t) at constant stress using Adaptive P-Controller.

        The controller is substepped at ``self.dt`` between adjacent data
        points so the explicit-Euler integration stays stable at coarse data
        cadences and the controller has enough iterations to drive the
        lattice stress to the target. Target stress is taken from
        ``data.metadata['stress']`` if present; otherwise falls back to
        ``mean(data.y)`` for the historical ``y=constant`` call pattern.

        Args:
            data: RheoData with x=time; target from metadata['stress'] or y.
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=time, y=strain.
        """
        time = data.x
        if time is None:
            raise ValueError("data.x (time array) must not be None")

        # Outer cadence from data; substep the controller at self.dt inside.
        dt_data = float(time[1] - time[0]) if len(time) > 1 else self.dt
        dt_sub = min(self.dt, dt_data)
        n_sub = max(1, int(round(dt_data / dt_sub))) if dt_sub > 0 else 1

        # Target stress: metadata is canonical (predict-time shape uses
        # y=dummy_zeros + metadata['stress']). Fall back to mean(y) for the
        # legacy test pattern that passes y=full_like(t, target_stress).
        target_stress = data.metadata.get("stress") if data.metadata else None
        if target_stress is None:
            if data.y is not None and data.y.size > 0:
                y_mean = float(jnp.mean(data.y))
                target_stress = y_mean if abs(y_mean) > 1e-12 else 1.0
            else:
                target_stress = 1.0
        target_stress = float(target_stress)

        # Controller Params
        Kp_base = 0.01
        alpha = 10.0

        state = self._init_state(key)
        # Augmented state: (EPM_State, current_gdot)
        aug_state = (state, 0.0)

        # Initial strain (0.0)
        initial_strain = state[2]

        n_steps = max(0, len(time) - 1)

        def sub_body(carrier, _):
            curr_epm, gdot = carrier
            stress_grid = curr_epm[0]
            curr_stress = jnp.mean(stress_grid)

            # Adaptive Control
            error = target_stress - curr_stress
            rel_error = jnp.abs(error) / (jnp.abs(target_stress) + 1e-6)
            Kp = Kp_base * (1.0 + alpha * rel_error)

            gdot_new = gdot + Kp * error
            gdot_new = jnp.maximum(gdot_new, 0.0)

            # Step EPM at the stable substep
            new_epm = self._epm_step(
                curr_epm, propagator_q, gdot_new, dt_sub, params, smooth
            )
            return (new_epm, gdot_new), None

        def outer_body(carrier, _):
            carrier_next, _ = jax.lax.scan(sub_body, carrier, None, length=n_sub)
            return carrier_next, carrier_next[0][2]  # sampled strain

        if n_steps > 0:
            _, strains_scan = jax.lax.scan(
                outer_body, aug_state, None, length=n_steps
            )
            strains = jnp.concatenate([jnp.array([initial_strain]), strains_scan])
        else:
            strains = jnp.array([initial_strain])

        return RheoData(x=time, y=strains, initial_test_mode="creep")

    def _run_oscillation(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """SAOS/LAOS: Stress(t) for sinusoidal strain.

        Args:
            data: RheoData with x=time, metadata['gamma0', 'omega'].
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=time, y=stress.
        """
        time = data.x
        if time is None:
            raise ValueError("data.x (time array) must not be None")

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = float(time[1] - time[0])

        # Params
        gamma0 = data.metadata.get("gamma0", 1.0)
        omega = data.metadata.get("omega", 1.0)

        state = self._init_state(key)

        # Initial stress
        initial_stress = jnp.mean(state[0])

        # Run for N-1 steps
        n_steps = max(0, len(time) - 1)
        scan_time = time[:-1] if n_steps > 0 else jnp.array([])

        def body(carrier, t):
            curr_state = carrier
            # Time varying shear rate at current time t
            gdot = gamma0 * omega * jnp.cos(omega * t)

            new_state = self._epm_step(
                curr_state, propagator_q, gdot, dt, params, smooth
            )
            return new_state, jnp.mean(new_state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, scan_time, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return RheoData(x=time, y=stresses, initial_test_mode="oscillation")

    def _fit(self, X, y, **kwargs):
        """Fit EPM parameters to data using NLSQ with smooth yielding.

        This method uses GPU-accelerated NLSQ optimization with smooth yielding
        approximation to fit EPM parameters. The smooth approximation replaces
        the hard yield threshold with a tanh transition, enabling gradient-based
        optimization.

        Args:
            X: Input data (shear rates, time, or frequency depending on mode)
            y: Target data (stress, modulus, or strain depending on mode)
            **kwargs: Additional fitting options including:
                test_mode (str): Protocol type ('flow_curve', 'startup',
                    'relaxation', 'creep', 'oscillation'). Required.
                seed (int): Random seed for reproducibility (default: 42)
                gamma_dot (float): Shear rate for startup mode (default: 0.1)
                gamma (float): Step strain for relaxation mode (default: 0.1)
                stress (float): Target stress for creep mode (default: 1.0)
                gamma0 (float): Strain amplitude for oscillation (default: 0.01)
                omega (float): Angular frequency for oscillation (default: 1.0)
                max_iter (int): Maximum NLSQ iterations (default: 500)
                use_log_residuals (bool): Use log-space residuals (default: True)

        Returns:
            self for method chaining
        """
        from rheojax.logging import get_logger, log_fit
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        logger = get_logger(__name__)

        # Extract and cache test mode
        test_mode = kwargs.get("test_mode")
        if test_mode is None:
            if hasattr(self, "_test_mode") and self._test_mode is not None:
                test_mode = self._test_mode
            else:
                raise ValueError("test_mode must be specified for EPM fitting")

        # Cache metadata for model_function use
        self._test_mode = test_mode
        self._cached_seed = kwargs.get("seed", 42)
        self._cached_gamma_dot = kwargs.get("gamma_dot", 0.1)
        self._cached_gamma = kwargs.get("gamma", 0.1)
        self._cached_stress = kwargs.get("stress", 1.0)
        self._cached_gamma0 = kwargs.get("gamma0", 0.01)
        self._cached_omega = kwargs.get("omega", 1.0)

        # Creep substep cache: the P-controller must run at dt ≤ self.dt for
        # stable integration. We compute the data cadence here (outside any
        # JIT trace) so _model_creep_jit can derive dt_sub and n_sub as
        # Python scalars.
        if test_mode == "creep":
            import numpy as _np

            _X_np = _np.asarray(X)
            if _X_np.ndim >= 1 and _X_np.shape[0] > 1:
                self._creep_dt_data = float(_X_np[1] - _X_np[0])
            else:
                self._creep_dt_data = self.dt

        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger,
            model=self.__class__.__name__,
            data_shape=data_shape,
            test_mode=test_mode,
        ) as ctx:
            # Convert to JAX arrays
            X_jax = jnp.asarray(X, dtype=jnp.float64)
            y_jax = jnp.asarray(y, dtype=jnp.float64)

            # Create model function wrapper for NLSQ
            def model_fn(x_data, params):
                return self.model_function(x_data, params, test_mode=test_mode)

            # Create least squares objective
            objective = create_least_squares_objective(
                model_fn,
                X_jax,
                y_jax,
                use_log_residuals=kwargs.get("use_log_residuals", True),
            )

            # Run NLSQ optimization
            result = nlsq_optimize(
                objective,
                self.parameters,
                max_iter=kwargs.get("max_iter", 500),
                ftol=kwargs.get("ftol", 1e-6),
                xtol=kwargs.get("xtol", 1e-6),
            )

            if not result.success:
                logger.warning(
                    f"{self.__class__.__name__} fit warning: {result.message}"
                )

            ctx["success"] = result.success
            ctx["cost"] = float(result.cost) if result.cost is not None else None
            ctx["n_iter"] = result.nit

            self.fitted_ = True

        return self

    # --- Bayesian / Model Function Interface ---

    def precompile(
        self,
        test_mode: str = "relaxation",
        X=None,
        y=None,
        *,
        n_points: int = 5,
        verbose: bool = True,
    ) -> float:
        """Pre-compile JIT kernels for faster Bayesian inference.

        Triggers JAX JIT compilation with dummy data so the first Bayesian
        inference call doesn't incur compilation overhead.

        Args:
            test_mode: Accepted for parent compatibility (unused).
            X: Accepted for parent compatibility (unused).
            y: Accepted for parent compatibility (unused).
            n_points: Number of data points for dummy compilation (default 5).
            verbose: Whether to log compilation progress (default True).

        Returns:
            Compilation time in seconds.

        Example:
            >>> model = LatticeEPM(L=16)
            >>> compile_time = model.precompile()  # Triggers JIT
            >>> # Now Bayesian inference will be faster
            >>> result = model.fit_bayesian(x, y, test_mode='flow_curve')
        """
        from rheojax.logging import get_logger

        logger = get_logger(__name__)

        if verbose:
            logger.info(
                "Precompiling EPM kernels",
                L=self.L,
                n_bayesian_steps=self.n_bayesian_steps,
            )

        start_time = time_module.perf_counter()

        # Get propagator
        if not hasattr(self, "_propagator_q_norm"):
            raise NotImplementedError(
                "Subclass must define _propagator_q_norm. "
                "Use LatticeEPM or TensorialEPM instead of EPMBase directly."
            )

        # Dummy data for compilation
        seed = 42
        key = jax.random.PRNGKey(seed)
        shear_rates = jnp.logspace(-1, 1, n_points)
        params_array = jnp.array(
            [
                self.parameters.get_value("mu"),
                self.parameters.get_value("tau_pl"),
                self.parameters.get_value("sigma_c_mean"),
                self.parameters.get_value("sigma_c_std"),
                self.parameters.get_value("n_fluid"),
                self.parameters.get_value("smoothing_width"),
            ]
        )

        # Scale propagator
        propagator_q = self._propagator_q_norm * params_array[0]

        # Compile flow curve (most expensive)
        _ = _jit_flow_curve_batch(
            shear_rates,
            key,
            propagator_q,
            params_array,
            self.dt,
            self.n_bayesian_steps,
            self.L,
            n_points,
            fluidity_form=self.fluidity_form,
        )

        # Block until compilation is done
        jax.block_until_ready(_)

        elapsed = time_module.perf_counter() - start_time
        self._precompiled = True

        if verbose:
            logger.info(
                "EPM kernels precompiled",
                compile_time_s=f"{elapsed:.2f}",
                L=self.L,
                n_steps=self.n_bayesian_steps,
            )

        return elapsed

    def _is_scalar_epm(self) -> bool:
        """Check if this is a scalar (not tensorial) EPM.

        Returns True for LatticeEPM (scalar stress field), False for TensorialEPM.
        This determines whether JIT-optimized scalar kernels can be used.
        """
        # Default to True (scalar) - TensorialEPM will override this
        return True

    def model_function(self, X, params, test_mode=None, **protocol_kwargs):
        """Compute EPM predictions for BayesianMixin integration.

        This method provides a pure-function interface for Bayesian inference,
        allowing NumPyro to sample from the parameter space. The implementation
        uses JIT-compiled kernels for efficient computation.

        Args:
            X: Input array (shear rates, time, frequency depending on mode)
            params: Tuple or array of parameter values in self.parameters order
            test_mode: Protocol mode ('flow_curve', 'startup', 'relaxation',
                      'creep', 'oscillation'). If None, uses cached test_mode.
            **protocol_kwargs: Protocol-specific parameters (gamma_dot, etc.)

        Returns:
            JAX array of predictions (stress, modulus, or strain depending on mode)
        """
        # Resolve test mode
        mode = (
            test_mode
            if test_mode is not None
            else getattr(self, "_test_mode", "flow_curve")
        )

        # Ensure JAX array (no numpy conversion for traceability)
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Convert params to array if needed
        params_array = jnp.asarray(params, dtype=jnp.float64)

        # F-001/F-003 fix: Resolve protocol kwargs with fallback to cached values.
        # This ensures fit_bayesian(gamma_dot=X) uses X, not the stale default.
        resolved_kwargs = {
            "gamma_dot": protocol_kwargs.get(
                "gamma_dot", getattr(self, "_cached_gamma_dot", 0.1)
            ),
            "gamma": protocol_kwargs.get("gamma", getattr(self, "_cached_gamma", 0.1)),
            "stress": protocol_kwargs.get(
                "stress", getattr(self, "_cached_stress", 1.0)
            ),
            "gamma0": protocol_kwargs.get(
                "gamma0", getattr(self, "_cached_gamma0", 0.01)
            ),
            "omega": protocol_kwargs.get("omega", getattr(self, "_cached_omega", 1.0)),
        }
        seed = protocol_kwargs.get("seed", getattr(self, "_cached_seed", 42))
        key = jax.random.PRNGKey(seed)

        # Get scaled propagator (subclass must have _propagator_q_norm)
        if not hasattr(self, "_propagator_q_norm"):
            raise NotImplementedError(
                "Subclass must define _propagator_q_norm. "
                "Use LatticeEPM or TensorialEPM instead of EPMBase directly."
            )
        # Scale by mu (first parameter)
        propagator_q = self._propagator_q_norm * params_array[0]

        # Use JIT-compiled scalar kernels for LatticeEPM (scalar stress)
        # TensorialEPM (tensorial stress) uses the general model functions
        if self._is_scalar_epm():
            return self._model_function_scalar(
                X_jax, key, propagator_q, params_array, mode, resolved_kwargs
            )
        else:
            # Convert params array back to dict for general model functions
            param_names = list(self.parameters.keys())
            p_values = dict(zip(param_names, params, strict=True))
            return self._model_function_general(
                X_jax, key, propagator_q, p_values, mode, resolved_kwargs
            )

    def _model_function_scalar(
        self,
        X_jax: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        mode: str,
        resolved_kwargs: dict,
    ) -> jax.Array:
        """Model function using JIT-compiled scalar kernels (for LatticeEPM)."""
        if mode in ["flow_curve", "rotation", "steady_shear"]:
            n_rates = int(X_jax.shape[0])
            return _jit_flow_curve_batch(
                X_jax,
                key,
                propagator_q,
                params_array,
                self.dt,
                self.n_bayesian_steps,
                self.L,
                n_rates,
                fluidity_form=self.fluidity_form,
            )
        elif mode == "startup":
            return self._model_startup_jit(
                X_jax, key, propagator_q, params_array, resolved_kwargs["gamma_dot"]
            )
        elif mode == "relaxation":
            return self._model_relaxation_jit(
                X_jax, key, propagator_q, params_array, resolved_kwargs["gamma"]
            )
        elif mode == "creep":
            return self._model_creep_jit(
                X_jax, key, propagator_q, params_array, resolved_kwargs["stress"]
            )
        elif mode in ["oscillation", "saos"]:
            return self._model_oscillation_jit(
                X_jax,
                key,
                propagator_q,
                params_array,
                resolved_kwargs["gamma0"],
                resolved_kwargs["omega"],
            )
        else:
            raise ValueError(f"Unknown test mode: {mode}")

    def _model_function_general(
        self,
        X_jax: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        p_values: dict,
        mode: str,
        resolved_kwargs: dict,
    ) -> jax.Array:
        """Model function using general (non-JIT) methods (for TensorialEPM)."""
        if mode in ["flow_curve", "rotation", "steady_shear"]:
            return self._model_flow_curve(X_jax, key, propagator_q, p_values)
        elif mode == "startup":
            return self._model_startup(
                X_jax, key, propagator_q, p_values, resolved_kwargs["gamma_dot"]
            )
        elif mode == "relaxation":
            return self._model_relaxation(
                X_jax, key, propagator_q, p_values, resolved_kwargs["gamma"]
            )
        elif mode == "creep":
            return self._model_creep(
                X_jax, key, propagator_q, p_values, resolved_kwargs["stress"]
            )
        elif mode in ["oscillation", "saos"]:
            return self._model_oscillation(
                X_jax,
                key,
                propagator_q,
                p_values,
                resolved_kwargs["gamma0"],
                resolved_kwargs["omega"],
            )
        else:
            raise ValueError(f"Unknown test mode: {mode}")

    # --- JIT-friendly time protocol wrappers ---

    def _model_startup_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        gamma_dot: float,
    ) -> jax.Array:
        """JIT-friendly startup simulation."""
        n_steps = max(0, int(time.shape[0]) - 1)
        dt = self.dt
        if n_steps > 0:
            # Use JAX-compatible array difference (avoids float() on traced arrays)
            dt = time[1] - time[0]

        return self._run_startup_kernel(
            time,
            key,
            propagator_q,
            params_array,
            gamma_dot,
            dt,
            n_steps,
            self.L,
            fluidity_form=self.fluidity_form,
        )

    def _model_relaxation_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        strain_step: float,
    ) -> jax.Array:
        """JIT-friendly relaxation simulation.

        Threads ``self.fluidity_form`` through the kernel so the fit path
        honours the same constitutive law as the Python ``_run_relaxation``
        predict path (which routes through ``epm_step``).
        """
        n_steps = max(0, int(time.shape[0]) - 1)
        dt = self.dt
        if n_steps > 0:
            # Use JAX-compatible array difference (avoids float() on traced arrays)
            dt = time[1] - time[0]

        return self._run_relaxation_kernel(
            time,
            key,
            propagator_q,
            params_array,
            strain_step,
            dt,
            n_steps,
            self.L,
            getattr(self, "fluidity_form", "overstress"),
        )

    def _model_creep_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        target_stress: float,
    ) -> jax.Array:
        """JIT-friendly creep simulation with controller substep.

        The creep P-controller is substepped at ``self.dt`` between adjacent
        data points. This keeps the explicit-Euler step stable regardless of
        data cadence and gives the controller enough iterations to drive the
        lattice stress to the target even on coarse grids (e.g. dt_data=0.5 s
        from a 20-point creep curve). See
        tests/models/epm/test_lattice_epm.py::test_lattice_epm_creep_coarse_dt_matches_fine_dt.

        ``dt_data`` is taken from ``self._creep_dt_data``, cached in ``_fit``
        before any JIT tracing, so slicing a traced ``time`` array here is
        unnecessary and we avoid ConcretizationTypeError.
        """
        n_steps = max(0, int(time.shape[0]) - 1)
        dt_data = float(getattr(self, "_creep_dt_data", self.dt))
        dt_sub = min(self.dt, dt_data) if dt_data > 0 else self.dt
        n_sub = max(1, int(round(dt_data / dt_sub))) if dt_sub > 0 else 1

        return self._run_creep_kernel(
            time,
            key,
            propagator_q,
            params_array,
            target_stress,
            dt_sub,
            n_steps,
            self.L,
            n_sub,
            getattr(self, "fluidity_form", "overstress"),
        )

    def _model_oscillation_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        gamma0: float,
        omega: float,
    ) -> jax.Array:
        """JIT-friendly oscillation simulation."""
        n_steps = max(0, int(time.shape[0]) - 1)
        dt = self.dt
        if n_steps > 0:
            # Use JAX-compatible array difference (avoids float() on traced arrays)
            dt = time[1] - time[0]

        return self._run_oscillation_kernel(
            time,
            key,
            propagator_q,
            params_array,
            gamma0,
            omega,
            dt,
            n_steps,
            self.L,
            fluidity_form=self.fluidity_form,
        )

    # --- Kernel dispatch methods (call JIT-compiled functions) ---

    def _run_startup_kernel(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        gamma_dot: float,
        dt: float,
        n_steps: int,
        L: int,
        fluidity_form: str = "overstress",
    ) -> jax.Array:
        """Dispatch to JIT-compiled startup kernel."""
        return _jit_startup_kernel(
            time,
            key,
            propagator_q,
            params_array,
            gamma_dot,
            dt,
            n_steps,
            L,
            fluidity_form=fluidity_form,
        )

    def _run_relaxation_kernel(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        strain_step: float,
        dt: float,
        n_steps: int,
        L: int,
        fluidity_form: str = "overstress",
    ) -> jax.Array:
        """Dispatch to JIT-compiled relaxation kernel."""
        return _jit_relaxation_kernel(
            time,
            key,
            propagator_q,
            params_array,
            strain_step,
            dt,
            n_steps,
            L,
            fluidity_form=fluidity_form,
        )

    def _run_creep_kernel(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        target_stress: float,
        dt_sub: float,
        n_steps: int,
        L: int,
        n_sub: int,
        fluidity_form: str = "overstress",
    ) -> jax.Array:
        """Dispatch to JIT-compiled creep kernel (substepped controller)."""
        return _jit_creep_kernel(
            time,
            key,
            propagator_q,
            params_array,
            target_stress,
            dt_sub,
            n_steps,
            L,
            n_sub,
            fluidity_form=fluidity_form,
        )

    def _run_oscillation_kernel(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        gamma0: float,
        omega: float,
        dt: float,
        n_steps: int,
        L: int,
        fluidity_form: str = "overstress",
    ) -> jax.Array:
        """Dispatch to JIT-compiled oscillation kernel."""
        return _jit_oscillation_kernel(
            time,
            key,
            propagator_q,
            params_array,
            gamma0,
            omega,
            dt,
            n_steps,
            L,
            fluidity_form=fluidity_form,
        )

    # --- JAX-Pure Model Functions for Bayesian Inference ---

    def _model_flow_curve(
        self,
        shear_rates: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
    ) -> jax.Array:
        """JAX-pure flow curve simulation (no RheoData, no numpy)."""
        n_steps = 1000
        dt = self.dt
        sigma_c_mean = params["sigma_c_mean"]
        tau_pl = params["tau_pl"]
        n_fluid = params.get("n_fluid", 1.0)
        tau_pl_shear = params.get("tau_pl_shear", tau_pl)

        def scan_fn(gdot):
            state = self._init_state(key)
            stress0, thresholds, strain, k = state

            # Warm-start at the analytical overstress steady state. Scalar
            # EPM has stress shape (L, L) and uses the scalar warm-start;
            # tensorial EPM has stress shape (3, L, L) with index [2] = σ_xy,
            # and the pure-shear steady state picks up a factor of √3 from
            # von Mises and a factor of 2 from the dσ/dt = 2μ·ε̇ convention:
            #
            #   scalar    :  σ_warm = σ_c_mean + σ_c_mean·(|γ̇|·τ_pl/σ_c_mean)^(1/n_fluid)
            #   tensorial :  σ_xy_warm = σ_c_mean/√3
            #              + (1/√3)·(√3/2)^(1/n_fluid)·σ_c_mean^((n_fluid-1)/n_fluid)
            #              · (|γ̇|·τ_pl_shear)^(1/n_fluid)
            #
            # The scalar warm-start is exact for the overstress form there;
            # the tensorial one is the analytical Bingham/HB solution with
            # the von Mises + factor-of-2 corrections.
            # Safety clamps for NLSQ-exploration regions where parameters
            # can transiently go near zero or negative. Without these guards,
            # the analytical warm-start can produce NaN/inf for certain
            # parameter combinations, which poisons the whole flow curve.
            scm_safe = jnp.maximum(sigma_c_mean, 1e-6)
            n_safe = jnp.maximum(n_fluid, 1e-3)
            tau_safe = jnp.maximum(tau_pl, 1e-6)
            tau_shear_safe = jnp.maximum(tau_pl_shear, 1e-6)

            gdot_abs = jnp.abs(gdot)
            inv_scm = 1.0 / scm_safe
            inv_n = 1.0 / n_safe
            if stress0.ndim == 2:
                warm_excess = scm_safe * (gdot_abs * tau_safe * inv_scm) ** inv_n
                sigma_warm_raw = jnp.sign(gdot) * (scm_safe + warm_excess)
                sigma_warm = jnp.where(
                    jnp.isfinite(sigma_warm_raw), sigma_warm_raw, jnp.sign(gdot) * scm_safe
                )
                stress0 = stress0 + sigma_warm
            else:
                sqrt3 = jnp.sqrt(3.0)
                warm_excess = (
                    (1.0 / sqrt3)
                    * (sqrt3 / 2.0) ** inv_n
                    * scm_safe ** ((n_safe - 1.0) * inv_n)
                    * (gdot_abs * tau_shear_safe) ** inv_n
                )
                sigma_xy_warm_raw = jnp.sign(gdot) * (scm_safe / sqrt3 + warm_excess)
                sigma_xy_warm = jnp.where(
                    jnp.isfinite(sigma_xy_warm_raw),
                    sigma_xy_warm_raw,
                    jnp.sign(gdot) * scm_safe / sqrt3,
                )
                stress0 = stress0.at[2].add(sigma_xy_warm)
            state = (stress0, thresholds, strain, k)

            def body(carrier, _):
                curr_state = carrier
                new_state = self._epm_step(
                    curr_state, propagator_q, gdot, dt, params, smooth=True
                )
                # Extract the mean shear stress. For scalar (L, L), jnp.mean
                # over the entire field is σ. For tensorial (3, L, L), we
                # need the mean of index [2] = σ_xy only, NOT the average
                # over all three stress components (which would mix σ_xx,
                # σ_yy, σ_xy).
                stress_arr = new_state[0]
                if stress_arr.ndim == 2:
                    sigma_obs = jnp.mean(stress_arr)
                else:
                    sigma_obs = jnp.mean(stress_arr[2])
                return new_state, sigma_obs

            _, history = jax.lax.scan(body, state, None, length=n_steps)
            steady_stress = jnp.mean(history[n_steps // 2 :])
            # NaN safety net — during NLSQ finite-differencing some
            # parameter combinations can poison the scan with NaN; replace
            # with the analytical plateau so the loss is finite and the
            # optimiser can recover on the next iteration.
            steady_stress = jnp.where(
                jnp.isfinite(steady_stress),
                steady_stress,
                scm_safe / jnp.sqrt(3.0) if stress0.ndim == 3 else scm_safe,
            )
            return steady_stress

        return jax.vmap(scan_fn)(shear_rates)

    @staticmethod
    def _mean_shear_stress(stress_field: jax.Array) -> jax.Array:
        """Extract the mean shear stress from a stress field.

        For scalar EPM (shape (L, L)) the mean of the entire field is the
        shear stress. For tensorial EPM (shape (3, L, L)) we need the mean
        of index [2] (σ_xy) — NOT the mean over all three components, which
        would mix σ_xx, σ_yy, σ_xy and give a physically meaningless number.
        """
        if stress_field.ndim == 2:
            return jnp.mean(stress_field)
        return jnp.mean(stress_field[2])

    def _model_startup(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        gamma_dot: float,
    ) -> jax.Array:
        """JAX-pure startup simulation."""
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

        n_steps = jnp.maximum(0, len(time) - 1)
        state = self._init_state(key)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(
                curr_state, propagator_q, gamma_dot, dt, params, smooth=True
            )
            return new_state, self._mean_shear_stress(new_state[0])

        initial_stress = self._mean_shear_stress(state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, None, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return stresses

    def _model_relaxation(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        strain_step: float,
    ) -> jax.Array:
        """JAX-pure relaxation simulation."""
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

        state = self._init_state(key)
        stress, thresh, strain, k = state

        # Apply step strain — for scalar add uniformly; for tensorial add
        # only to the σ_xy component (index [2]).
        mu = params["mu"]
        if stress.ndim == 2:
            stress = stress + mu * strain_step
        else:
            stress = stress.at[2].add(mu * strain_step)
        state = (stress, thresh, strain + strain_step, k)

        g_0 = self._mean_shear_stress(stress) / strain_step
        n_steps = jnp.maximum(0, len(time) - 1)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(
                curr_state, propagator_q, 0.0, dt, params, smooth=True
            )
            return new_state, self._mean_shear_stress(new_state[0]) / strain_step

        if n_steps > 0:
            _, moduli_scan = jax.lax.scan(body, state, None, length=n_steps)
            moduli = jnp.concatenate([jnp.array([g_0]), moduli_scan])
        else:
            moduli = jnp.array([g_0])

        return moduli

    def _model_creep(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        target_stress: float,
    ) -> jax.Array:
        """JAX-pure creep simulation with P-controller."""
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

        Kp_base = 0.01
        alpha = 10.0

        state = self._init_state(key)
        aug_state = (state, 0.0)
        initial_strain = state[2]
        n_steps = jnp.maximum(0, len(time) - 1)

        def body(carrier, _):
            curr_epm, gdot = carrier
            curr_stress = self._mean_shear_stress(curr_epm[0])

            error = target_stress - curr_stress
            rel_error = jnp.abs(error) / (jnp.abs(target_stress) + 1e-6)
            Kp = Kp_base * (1.0 + alpha * rel_error)

            gdot_new = gdot + Kp * error
            gdot_new = jnp.maximum(gdot_new, 0.0)

            new_epm = self._epm_step(
                curr_epm, propagator_q, gdot_new, dt, params, smooth=True
            )
            return (new_epm, gdot_new), new_epm[2]

        if n_steps > 0:
            _, strains_scan = jax.lax.scan(body, aug_state, None, length=n_steps)
            strains = jnp.concatenate([jnp.array([initial_strain]), strains_scan])
        else:
            strains = jnp.array([initial_strain])

        return strains

    def _model_oscillation(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        gamma0: float,
        omega: float,
    ) -> jax.Array:
        """JAX-pure oscillation simulation."""
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

        state = self._init_state(key)
        initial_stress = self._mean_shear_stress(state[0])
        n_steps = jnp.maximum(0, len(time) - 1)
        scan_time = time[:-1] if n_steps > 0 else jnp.array([])

        def body(carrier, t):
            curr_state = carrier
            gdot = gamma0 * omega * jnp.cos(omega * t)
            new_state = self._epm_step(
                curr_state, propagator_q, gdot, dt, params, smooth=True
            )
            return new_state, self._mean_shear_stress(new_state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, scan_time, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return stresses
