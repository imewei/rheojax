"""Kernels for Elasto-Plastic Models (EPM).

This module implements the core physics kernels for scalar EPM simulations using JAX.
It includes the FFT-based elastic propagator for stress redistribution, logic
for plastic events (dual-mode: hard/smooth), and the full time-stepping kernel.
"""

from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


@partial(jax.jit, static_argnames=("L_x", "L_y"))
def make_propagator_q(L_x: int, L_y: int, shear_modulus: float = 1.0) -> jax.Array:
    """Create the quadrupolar Eshelby propagator in Fourier space.

    G(q) = -4 * mu * (qx * qy)^2 / (q^2)^2 for q != 0
    G(0) = 0

    Args:
        L_x: Lattice size in x.
        L_y: Lattice size in y.
        shear_modulus: Shear modulus mu.

    Returns:
        2D array of the propagator in Fourier space (L_x, L_y // 2 + 1).
    """
    qx = jnp.fft.fftfreq(L_x) * 2 * jnp.pi
    qy = jnp.fft.rfftfreq(L_y) * 2 * jnp.pi

    # Create meshgrid of wave vectors (indexing='ij' for matrix coords)
    QX, QY = jnp.meshgrid(qx, qy, indexing="ij")
    Q2 = QX**2 + QY**2

    # Handle q=0 singularity safely
    valid_mask = Q2 > 0
    safe_Q2 = jnp.where(valid_mask, Q2, 1.0)

    # Standard scalar EPM propagator (quadrupolar)
    propagator_q = jnp.where(
        valid_mask, -4.0 * shear_modulus * (QX**2 * QY**2) / (safe_Q2**2), 0.0
    )

    # Explicitly enforce zero mean redistribution at q=0
    propagator_q = propagator_q.at[0, 0].set(0.0)

    return propagator_q


@jax.jit
def solve_elastic_propagator(
    plastic_strain_rate: jax.Array, propagator_q: jax.Array
) -> jax.Array:
    """Solve for the elastic stress redistribution rate using FFT.

    Calculates sigma_dot_el = G * epsilon_dot_pl.

    Args:
        plastic_strain_rate: 2D array of plastic strain rate field (L, L).
        propagator_q: Precomputed propagator in Fourier space (L, L // 2 + 1).

    Returns:
        2D array of elastic stress redistribution rate.
    """
    # Perform convolution in Fourier space using Real-FFT
    plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
    stress_q = propagator_q * plastic_strain_q

    # Transform back to real space
    # We pass the shape `s` to ensure correct reconstruction if dimensions are odd
    stress_redistribution = jnp.fft.irfft2(stress_q, s=plastic_strain_rate.shape)

    return stress_redistribution


@partial(jax.jit, static_argnames=("smooth",))
def compute_plastic_strain_rate(
    stress: jax.Array,
    yield_thresholds: jax.Array,
    fluidity: float = 1.0,
    smooth: bool = False,
    smoothing_width: float = 0.1,
) -> jax.Array:
    r"""Compute the local plastic strain rate.

    Supports two modes:

    1. Hard Mode: gamma_dot_p = Gamma * sigma * Theta(\|sigma\| - sigma_c)
    2. Smooth Mode: gamma_dot_p = Gamma * sigma * 0.5 * (1 + tanh((\|sigma\| - sigma_c)/w))

    Args:
        stress: Local stress field.
        yield_thresholds: Local yield thresholds.
        fluidity: Inverse plastic timescale ($1/\tau_{pl}$).
        smooth: Whether to use the differentiable smooth approximation.
        smoothing_width: Width parameter $w$ for smoothing.

    Returns:
        Plastic strain rate field.
    """
    stress_mag = jnp.abs(stress)

    if smooth:
        # Differentiable approximation (Hyperbolic tangent)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - yield_thresholds) / smoothing_width)
        )
    else:
        # Hard threshold
        activation = (stress_mag > yield_thresholds).astype(stress.dtype)

    return activation * stress * fluidity


@jax.jit
def update_yield_thresholds(
    key: jax.Array,
    active_mask: jax.Array,
    current_thresholds: jax.Array,
    mean: float = 1.0,
    std: float = 0.1,
) -> jax.Array:
    """Renew yield thresholds for active sites.

    Args:
        key: PRNG Key.
        active_mask: Boolean mask of sites that yielded.
        current_thresholds: Current thresholds.
        mean: Mean of Gaussian distribution.
        std: Std dev of Gaussian distribution.

    Returns:
        Updated yield thresholds.
    """
    # Generate random thresholds for the entire grid
    shape = current_thresholds.shape
    random_thresholds = mean + std * jax.random.normal(key, shape)

    # Ensure positive thresholds
    random_thresholds = jnp.maximum(random_thresholds, 1e-4)

    # Only update active sites
    return jnp.where(active_mask, random_thresholds, current_thresholds)


@partial(jax.jit, static_argnames=("smooth",))
def epm_step(
    state: tuple[jax.Array, jax.Array, float, jax.Array],
    propagator_q: jax.Array,
    shear_rate: float,
    dt: float,
    params: dict,
    smooth: bool = False,
) -> tuple[jax.Array, jax.Array, float, jax.Array]:
    """Perform one full EPM time step.

    Dynamics:
    sigma_dot = mu * gamma_dot - mu * gamma_dot_p + G * gamma_dot_p

    Args:
        state: Tuple (stress, yield_thresholds, accumulated_strain, key).
        propagator_q: Precomputed propagator.
        shear_rate: Macroscopic imposed shear rate gamma_dot.
        dt: Time step size.
        params: Dictionary of model parameters (mu, tau_pl, sigma_c_mean, etc.).
        smooth: Use smooth yielding (for inference) vs hard yielding (for simulation).

    Returns:
        Updated state tuple.
    """
    stress, thresholds, strain, key = state

    mu = params.get("mu", 1.0)
    tau_pl = params.get("tau_pl", 1.0)
    fluidity = 1.0 / tau_pl

    # 1. Compute Plastic Strain Rate
    # Note: If smooth=True, this gives a continuous field.
    # If smooth=False, it's sparse (only yielded sites).
    plastic_strain_rate = compute_plastic_strain_rate(
        stress,
        thresholds,
        fluidity=fluidity,
        smooth=smooth,
        smoothing_width=params.get("smoothing_width", 0.1),
    )

    # 2. Compute Stress Rates
    # Elastic Loading: mu * gdot
    loading_rate = mu * shear_rate

    # Plastic Relaxation: -mu * gdot_p
    relaxation_rate = -mu * plastic_strain_rate

    # Redistribution: G * gdot_p
    # Note: The propagator G already includes the factor 'mu' if derived from stress balance,
    # or strictly G describes stress-from-strain.
    # In standard form (Nicolas 2018): dSigma/dt = mu*gdot - mu*gdot_p + G_stress_from_strain * gdot_p
    # Our propagator factory generates G_stress_from_strain (includes -4*mu factor).
    redistribution_rate = solve_elastic_propagator(plastic_strain_rate, propagator_q)

    total_stress_rate = loading_rate + relaxation_rate + redistribution_rate

    # 3. Update State (Euler Integration)
    new_stress = stress + total_stress_rate * dt
    new_strain = strain + shear_rate * dt

    # 4. Structural Renewal (Only in Hard Mode usually, or stochastic smooth)
    # For fully differentiable inference (Smooth Mode), we often fix the disorder
    # or treat it as parameters. For forward simulation (Hard Mode), we renew.
    # Here we renew if NOT smooth, or if explicitly desired.
    # We'll stick to the standard: Renewal happens on yielding.

    if not smooth:
        # Identify sites that yielded in this step (or are currently yielding)
        # Using the stress from start of step determines activation
        active_mask = jnp.abs(stress) > thresholds

        # Split key for renewal
        key, subkey = jax.random.split(key)

        # Only renew if they yielded.
        # Note: In continuous time, renewal is often a Poisson process or instantaneous reset.
        # Standard EPM: Yield -> Stress drops -> If stress drops below threshold, it heals?
        # Or does it heal immediately?
        # Common Protocol: Threshold is renewed when the site yields.
        new_thresholds = update_yield_thresholds(
            subkey,
            active_mask,
            thresholds,
            mean=params.get("sigma_c_mean", 1.0),
            std=params.get("sigma_c_std", 0.1),
        )
    else:
        # In smooth mode, we typically keep the landscape fixed or evolve it continuously.
        # For NLSQ fitting of static parameters, we keep thresholds fixed.
        new_thresholds = thresholds

    return (new_stress, new_thresholds, new_strain, key)
