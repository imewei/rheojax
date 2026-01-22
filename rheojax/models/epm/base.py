"""Base class for Elasto-Plastic Models (EPM).

This module provides the abstract base class for all EPM variants (scalar lattice,
tensorial, etc.), extracting common parameters, initialization logic, and protocol
runner templates.
"""

from abc import abstractmethod
from typing import Dict, Tuple, Optional

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


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
    """

    def __init__(
        self,
        L: int = 64,
        dt: float = 0.01,
        mu: float = 1.0,
        tau_pl: float = 1.0,
        sigma_c_mean: float = 1.0,
        sigma_c_std: float = 0.1,
    ):
        """Initialize EPM base with common parameters."""
        super().__init__()

        # Configuration (Static)
        self.L = L
        self.dt = dt

        # Parameters (Optimizable) - use inherited self.parameters from BaseModel
        self.parameters.add(
            "mu", mu, bounds=(0.1, 100.0),
            units="Pa", description="Shear modulus"
        )
        self.parameters.add(
            "tau_pl", tau_pl, bounds=(0.01, 100.0),
            units="s", description="Plastic relaxation timescale"
        )
        self.parameters.add(
            "sigma_c_mean", sigma_c_mean, bounds=(0.1, 10.0),
            units="Pa", description="Mean yield threshold"
        )
        self.parameters.add(
            "sigma_c_std", sigma_c_std, bounds=(0.0, 1.0),
            units="Pa", description="Yield threshold standard deviation (disorder)"
        )
        self.parameters.add(
            "smoothing_width", 0.1, bounds=(0.01, 1.0),
            units="Pa", description="Smooth yielding transition width"
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

    def _get_param_dict(self) -> Dict[str, float]:
        """Extract parameters as dictionary for kernel calls.

        Returns:
            Dictionary with all EPM parameters (mu, tau_pl, sigma_c_mean, etc.).
        """
        return {
            "mu": self.parameters.get_value("mu"),
            "tau_pl": self.parameters.get_value("tau_pl"),
            "sigma_c_mean": self.parameters.get_value("sigma_c_mean"),
            "sigma_c_std": self.parameters.get_value("sigma_c_std"),
            "smoothing_width": self.parameters.get_value("smoothing_width"),
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
        state: Tuple,
        propagator_q: jax.Array,
        shear_rate: float,
        dt: float,
        params: dict,
        smooth: bool,
    ) -> Tuple:
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

    def _init_state(self, key: jax.Array) -> Tuple[jax.Array, jax.Array, float, jax.Array]:
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
            steady_stress = jnp.mean(history[n_steps // 2:])
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

        # Calculate dt from data if possible
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

        # Constant shear rate from metadata
        gdot = data.metadata.get("gamma_dot", 0.1)

        # Scan for N-1 steps
        n_steps = max(0, len(time) - 1)
        state = self._init_state(key)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(curr_state, propagator_q, gdot, dt, params, smooth)
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

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

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
            new_state = self._epm_step(curr_state, propagator_q, 0.0, dt, params, smooth)
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

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

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
            (curr_epm, gdot) = carrier
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
            new_epm = self._epm_step(curr_epm, propagator_q, gdot_new, dt, params, smooth)

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

        # Calculate dt from data
        dt = self.dt
        if len(time) > 1:
            dt = time[1] - time[0]

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

            new_state = self._epm_step(curr_state, propagator_q, gdot, dt, params, smooth)
            return new_state, jnp.mean(new_state[0])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, scan_time, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return RheoData(x=time, y=stresses, initial_test_mode="oscillation")

    def _fit(self, X, y, **kwargs):
        """Fit model parameters to data (NLSQ).

        EPM fitting is complex and requires smooth yielding approximation.
        Base implementation does nothing (pass-through).
        """
        pass
