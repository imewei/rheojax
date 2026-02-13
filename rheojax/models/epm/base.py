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


@partial(jax.jit, static_argnums=(5, 6))
def _jit_flow_curve_single(
    gdot: float,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    dt: float,
    n_steps: int,
    L: int,
) -> jax.Array:
    """JIT-compiled flow curve for a single shear rate.

    Args:
        gdot: Shear rate
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        dt: Time step
        n_steps: Number of simulation steps
        L: Lattice size

    Returns:
        Steady-state stress
    """
    # Unpack params
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    smoothing_width = params_array[4]

    fluidity = 1.0 / tau_pl

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Smooth plastic strain rate (differentiable)
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )
        plastic_strain_rate = activation * stress_curr * fluidity

        # Stress evolution
        loading_rate = mu * gdot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        new_strain = strain_curr + gdot * dt

        return (new_stress, thresholds_curr, new_strain, key_curr), jnp.mean(new_stress)

    _, history = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    # Average second half for steady state
    steady_stress = jnp.mean(history[n_steps // 2:])
    return steady_stress


@partial(jax.jit, static_argnums=(5, 6, 7))
def _jit_flow_curve_batch(
    shear_rates: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    dt: float,
    n_steps: int,
    L: int,
    n_rates: int,
) -> jax.Array:
    """JIT-compiled flow curve for batch of shear rates.

    Args:
        shear_rates: Array of shear rates
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        dt: Time step
        n_steps: Number of simulation steps
        L: Lattice size
        n_rates: Number of shear rates (static for JIT)

    Returns:
        Array of steady-state stresses
    """
    # Use different keys for each shear rate
    keys = jax.random.split(key, n_rates)

    def single_rate(gdot_key):
        gdot, k = gdot_key
        return _jit_flow_curve_single(gdot, k, propagator_q, params_array, dt, n_steps, L)

    return jax.vmap(single_rate)((shear_rates, keys))


@partial(jax.jit, static_argnums=(6, 7))
def _jit_startup_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    gamma_dot: float,
    dt: float,
    n_steps: int,
    L: int,
) -> jax.Array:
    """JIT-compiled startup shear simulation.

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        gamma_dot: Applied shear rate
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size

    Returns:
        Array of stress over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    smoothing_width = params_array[4]

    fluidity = 1.0 / tau_pl

    # Initialize state
    k1, k2 = jax.random.split(key)
    stress = jnp.zeros((L, L))
    thresholds = sigma_c_mean + sigma_c_std * jax.random.normal(k2, (L, L))
    thresholds = jnp.maximum(thresholds, 1e-4)
    strain = 0.0

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr = carrier

        # Smooth plastic strain rate
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )
        plastic_strain_rate = activation * stress_curr * fluidity

        # Stress evolution
        loading_rate = mu * gamma_dot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        new_strain = strain_curr + gamma_dot * dt

        return (new_stress, thresholds_curr, new_strain, key_curr), jnp.mean(new_stress)

    initial_stress = jnp.mean(stress)
    _, stresses_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([initial_stress]), stresses_scan])


@partial(jax.jit, static_argnums=(6, 7))
def _jit_relaxation_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    strain_step: float,
    dt: float,
    n_steps: int,
    L: int,
) -> jax.Array:
    """JIT-compiled stress relaxation simulation.

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        strain_step: Applied step strain
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size

    Returns:
        Array of modulus G(t) over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    smoothing_width = params_array[4]

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

        # Smooth plastic strain rate
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )
        plastic_strain_rate = activation * stress_curr * fluidity

        # Stress evolution (no loading - relaxation only)
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update (no loading)
        new_stress = stress_curr + (relaxation_rate + redistribution_rate) * dt

        return (new_stress, thresholds_curr, strain_curr, key_curr), jnp.mean(new_stress) / strain_step

    _, moduli_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([g_0]), moduli_scan])


@partial(jax.jit, static_argnums=(6, 7))
def _jit_creep_kernel(
    time: jax.Array,
    key: jax.Array,
    propagator_q: jax.Array,
    params_array: jax.Array,
    target_stress: float,
    dt: float,
    n_steps: int,
    L: int,
) -> jax.Array:
    """JIT-compiled creep simulation with P-controller.

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        target_stress: Target stress for creep
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size

    Returns:
        Array of strain over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    smoothing_width = params_array[4]

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

    # Augmented state: (stress, thresholds, strain, key, gdot)
    initial_strain = strain

    def body_fn(carrier, _):
        stress_curr, thresholds_curr, strain_curr, key_curr, gdot = carrier

        # P-controller: adjust shear rate to maintain target stress
        curr_stress = jnp.mean(stress_curr)
        error = target_stress - curr_stress
        rel_error = jnp.abs(error) / (jnp.abs(target_stress) + 1e-6)
        Kp = Kp_base * (1.0 + alpha * rel_error)
        gdot_new = jnp.maximum(gdot + Kp * error, 0.0)

        # Smooth plastic strain rate
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )
        plastic_strain_rate = activation * stress_curr * fluidity

        # Stress evolution
        loading_rate = mu * gdot_new
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
        new_strain = strain_curr + gdot_new * dt

        return (new_stress, thresholds_curr, new_strain, key_curr, gdot_new), new_strain

    _, strains_scan = jax.lax.scan(
        body_fn, (stress, thresholds, strain, k2, 0.0), None, length=n_steps
    )

    return jnp.concatenate([jnp.array([initial_strain]), strains_scan])


@partial(jax.jit, static_argnums=(7, 8))
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
) -> jax.Array:
    """JIT-compiled oscillatory shear simulation.

    Args:
        time: Time array
        key: PRNG key
        propagator_q: Propagator in Fourier space
        params_array: [mu, tau_pl, sigma_c_mean, sigma_c_std, smoothing_width]
        gamma0: Strain amplitude
        omega: Angular frequency
        dt: Time step
        n_steps: Number of time points minus 1
        L: Lattice size

    Returns:
        Array of stress over time
    """
    mu = params_array[0]
    tau_pl = params_array[1]
    sigma_c_mean = params_array[2]
    sigma_c_std = params_array[3]
    smoothing_width = params_array[4]

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

        # Smooth plastic strain rate
        stress_mag = jnp.abs(stress_curr)
        activation = 0.5 * (
            1.0 + jnp.tanh((stress_mag - thresholds_curr) / smoothing_width)
        )
        plastic_strain_rate = activation * stress_curr * fluidity

        # Stress evolution
        loading_rate = mu * gdot
        relaxation_rate = -mu * plastic_strain_rate

        # FFT-based redistribution
        plastic_strain_q = jnp.fft.rfft2(plastic_strain_rate)
        stress_q = propagator_q * plastic_strain_q
        redistribution_rate = jnp.fft.irfft2(stress_q, s=(L, L))

        # Update
        new_stress = stress_curr + (loading_rate + relaxation_rate + redistribution_rate) * dt
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
        n_bayesian_steps: int = 200,
    ):
        """Initialize EPM base with common parameters."""
        super().__init__()

        # Configuration (Static)
        self.L = L
        self.dt = dt
        self.n_bayesian_steps = n_bayesian_steps
        self._precompiled = False

        # Parameters (Optimizable) - use inherited self.parameters from BaseModel
        self.parameters.add(
            "mu", mu, bounds=(0.1, 10000.0), units="Pa", description="Shear modulus"
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
            bounds=(0.1, 1000.0),
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
        smoothing_width = self.parameters.get_value("smoothing_width")
        assert mu is not None
        assert tau_pl is not None
        assert sigma_c_mean is not None
        assert sigma_c_std is not None
        assert smoothing_width is not None
        return {
            "mu": mu,
            "tau_pl": tau_pl,
            "sigma_c_mean": sigma_c_mean,
            "sigma_c_std": sigma_c_std,
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

        def scan_fn(gdot):
            # Run simulation for sufficient steps to reach steady state
            n_steps = 1000
            state = self._init_state(key)

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
        assert time is not None

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
        assert time is not None

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

        Args:
            data: RheoData with x=time, y=target_stress or metadata['stress'].
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=time, y=strain.
        """
        time = data.x
        assert time is not None

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = float(time[1] - time[0])

        # Target stress from metadata or mean of y
        if data.y is not None:
            target_stress = jnp.mean(data.y)
        else:
            target_stress = data.metadata.get("stress", 1.0)

        # Controller Params
        Kp_base = 0.01
        alpha = 10.0

        state = self._init_state(key)
        # Augmented state: (EPM_State, current_gdot)
        aug_state = (state, 0.0)

        # Initial strain (0.0)
        initial_strain = state[2]

        n_steps = max(0, len(time) - 1)

        def body(carrier, _):
            curr_epm, gdot = carrier
            stress_grid = curr_epm[0]
            curr_stress = jnp.mean(stress_grid)

            # Adaptive Control
            error = target_stress - curr_stress
            # Gain scheduling: Boost gain if error is large relative to target
            rel_error = jnp.abs(error) / (jnp.abs(target_stress) + 1e-6)
            Kp = Kp_base * (1.0 + alpha * rel_error)

            # Update shear rate (P-control on rate)
            gdot_new = gdot + Kp * error
            # Prevent negative shear rate
            gdot_new = jnp.maximum(gdot_new, 0.0)

            # Step EPM
            new_epm = self._epm_step(
                curr_epm, propagator_q, gdot_new, dt, params, smooth
            )

            # Return Strain
            return (new_epm, gdot_new), new_epm[2]

        if n_steps > 0:
            _, strains_scan = jax.lax.scan(body, aug_state, None, length=n_steps)
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
        assert time is not None

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
            if hasattr(self, "_test_mode") and self._test_mode:
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

    def precompile(self, n_points: int = 5, verbose: bool = True) -> float:
        """Pre-compile JIT kernels for faster Bayesian inference.

        Triggers JAX JIT compilation with dummy data so the first Bayesian
        inference call doesn't incur compilation overhead.

        Args:
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
        params_array = jnp.array([
            self.parameters.get_value("mu"),
            self.parameters.get_value("tau_pl"),
            self.parameters.get_value("sigma_c_mean"),
            self.parameters.get_value("sigma_c_std"),
            self.parameters.get_value("smoothing_width"),
        ])

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
        mode = test_mode or getattr(self, "_test_mode", "flow_curve")

        # Ensure JAX array (no numpy conversion for traceability)
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Convert params to array if needed
        params_array = jnp.asarray(params, dtype=jnp.float64)

        # Get cached seed for reproducibility during MCMC sampling
        seed = getattr(self, "_cached_seed", 42)
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
                X_jax, key, propagator_q, params_array, mode
            )
        else:
            # Convert params array back to dict for general model functions
            param_names = list(self.parameters.keys())
            p_values = dict(zip(param_names, params, strict=True))
            return self._model_function_general(
                X_jax, key, propagator_q, p_values, mode
            )

    def _model_function_scalar(
        self,
        X_jax: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        mode: str,
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
            )
        elif mode == "startup":
            gamma_dot = getattr(self, "_cached_gamma_dot", 0.1)
            return self._model_startup_jit(X_jax, key, propagator_q, params_array, gamma_dot)
        elif mode == "relaxation":
            gamma = getattr(self, "_cached_gamma", 0.1)
            return self._model_relaxation_jit(X_jax, key, propagator_q, params_array, gamma)
        elif mode == "creep":
            stress = getattr(self, "_cached_stress", 1.0)
            return self._model_creep_jit(X_jax, key, propagator_q, params_array, stress)
        elif mode in ["oscillation", "saos"]:
            gamma0 = getattr(self, "_cached_gamma0", 0.01)
            omega = getattr(self, "_cached_omega", 1.0)
            return self._model_oscillation_jit(
                X_jax, key, propagator_q, params_array, gamma0, omega
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
    ) -> jax.Array:
        """Model function using general (non-JIT) methods (for TensorialEPM)."""
        if mode in ["flow_curve", "rotation", "steady_shear"]:
            return self._model_flow_curve(X_jax, key, propagator_q, p_values)
        elif mode == "startup":
            gamma_dot = getattr(self, "_cached_gamma_dot", 0.1)
            return self._model_startup(X_jax, key, propagator_q, p_values, gamma_dot)
        elif mode == "relaxation":
            gamma = getattr(self, "_cached_gamma", 0.1)
            return self._model_relaxation(X_jax, key, propagator_q, p_values, gamma)
        elif mode == "creep":
            stress = getattr(self, "_cached_stress", 1.0)
            return self._model_creep(X_jax, key, propagator_q, p_values, stress)
        elif mode in ["oscillation", "saos"]:
            gamma0 = getattr(self, "_cached_gamma0", 0.01)
            omega = getattr(self, "_cached_omega", 1.0)
            return self._model_oscillation(
                X_jax, key, propagator_q, p_values, gamma0, omega
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
            time, key, propagator_q, params_array, gamma_dot, dt, n_steps, self.L
        )

    def _model_relaxation_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        strain_step: float,
    ) -> jax.Array:
        """JIT-friendly relaxation simulation."""
        n_steps = max(0, int(time.shape[0]) - 1)
        dt = self.dt
        if n_steps > 0:
            # Use JAX-compatible array difference (avoids float() on traced arrays)
            dt = time[1] - time[0]

        return self._run_relaxation_kernel(
            time, key, propagator_q, params_array, strain_step, dt, n_steps, self.L
        )

    def _model_creep_jit(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        target_stress: float,
    ) -> jax.Array:
        """JIT-friendly creep simulation."""
        n_steps = max(0, int(time.shape[0]) - 1)
        dt = self.dt
        if n_steps > 0:
            # Use JAX-compatible array difference (avoids float() on traced arrays)
            dt = time[1] - time[0]

        return self._run_creep_kernel(
            time, key, propagator_q, params_array, target_stress, dt, n_steps, self.L
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
            time, key, propagator_q, params_array, gamma0, omega, dt, n_steps, self.L
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
    ) -> jax.Array:
        """Dispatch to JIT-compiled startup kernel."""
        return _jit_startup_kernel(
            time, key, propagator_q, params_array, gamma_dot, dt, n_steps, L
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
    ) -> jax.Array:
        """Dispatch to JIT-compiled relaxation kernel."""
        return _jit_relaxation_kernel(
            time, key, propagator_q, params_array, strain_step, dt, n_steps, L
        )

    def _run_creep_kernel(
        self,
        time: jax.Array,
        key: jax.Array,
        propagator_q: jax.Array,
        params_array: jax.Array,
        target_stress: float,
        dt: float,
        n_steps: int,
        L: int,
    ) -> jax.Array:
        """Dispatch to JIT-compiled creep kernel."""
        return _jit_creep_kernel(
            time, key, propagator_q, params_array, target_stress, dt, n_steps, L
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
    ) -> jax.Array:
        """Dispatch to JIT-compiled oscillation kernel."""
        return _jit_oscillation_kernel(
            time, key, propagator_q, params_array, gamma0, omega, dt, n_steps, L
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

        def scan_fn(gdot):
            state = self._init_state(key)

            def body(carrier, _):
                curr_state = carrier
                new_state = self._epm_step(
                    curr_state, propagator_q, gdot, dt, params, smooth=True
                )
                return new_state, jnp.mean(new_state[0])

            _, history = jax.lax.scan(body, state, None, length=n_steps)
            steady_stress = jnp.mean(history[n_steps // 2 :])
            return steady_stress

        return jax.vmap(scan_fn)(shear_rates)

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
            return new_state, jnp.mean(new_state[0])

        initial_stress = jnp.mean(state[0])

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

        # Apply step strain
        mu = params["mu"]
        stress = stress + mu * strain_step
        state = (stress, thresh, strain + strain_step, k)

        g_0 = jnp.mean(stress) / strain_step
        n_steps = jnp.maximum(0, len(time) - 1)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(
                curr_state, propagator_q, 0.0, dt, params, smooth=True
            )
            return new_state, jnp.mean(new_state[0]) / strain_step

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
            stress_grid = curr_epm[0]
            curr_stress = jnp.mean(stress_grid)

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
        initial_stress = jnp.mean(state[0])
        n_steps = jnp.maximum(0, len(time) - 1)
        scan_time = time[:-1] if n_steps > 0 else jnp.array([])

        def body(carrier, t):
            curr_state = carrier
            gdot = gamma0 * omega * jnp.cos(omega * t)
            new_state = self._epm_step(
                curr_state, propagator_q, gdot, dt, params, smooth=True
            )
            return new_state, jnp.mean(new_state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, scan_time, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return stresses
