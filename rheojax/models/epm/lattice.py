"""Lattice-based Elasto-Plastic Model (EPM) implementation."""

from functools import partial
from typing import Dict, Tuple, Optional, Any

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import Parameter, ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.epm_kernels import (
    make_propagator_q,
    epm_step,
    compute_plastic_strain_rate
)

jax, jnp = safe_import_jax()


@ModelRegistry.register("lattice_epm")
class LatticeEPM(BaseModel):
    """2D Lattice Elasto-Plastic Model (EPM).

    A mesoscopic model for amorphous solids (glasses, gels) that explicitly resolves
    spatial heterogeneity, plastic avalanches, and stress redistribution.

    Physics:
        - Lattice of elastoplastic blocks.
        - Elastic loading (affine).
        - Local yielding when stress > threshold.
        - Long-range stress redistribution via quadrupolar Eshelby propagator.
        - Structural renewal (disorder).

    Parameters:
        mu (float): Shear modulus. Default 1.0.
        tau_pl (float): Plastic relaxation timescale. Default 1.0.
        sigma_c_mean (float): Mean yield threshold. Default 1.0.
        sigma_c_std (float): Disorder strength (std dev of thresholds). Default 0.1.
        smoothing_width (float): Width for smooth yielding approx (inference only). Default 0.1.

    Configuration:
        L (int): Lattice size (LxL). Default 64.
        dt (float): Time step. Default 0.01.
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
        """Initialize the Lattice EPM."""
        super().__init__()

        # Configuration (Static)
        self.L = L
        self.dt = dt

        # Parameters (Optimizable)
        self.params = ParameterSet()
        self.params.add("mu", mu, bounds=(0.1, 100.0))
        self.params.add("tau_pl", tau_pl, bounds=(0.01, 100.0))
        self.params.add("sigma_c_mean", sigma_c_mean, bounds=(0.1, 10.0))
        self.params.add("sigma_c_std", sigma_c_std, bounds=(0.0, 1.0))
        self.params.add("smoothing_width", 0.1, bounds=(0.01, 1.0))

        # Precompute Propagator (Cached)
        self._propagator_q_norm = make_propagator_q(L, L, shear_modulus=1.0)

    def _init_state(self, key: jax.Array) -> Tuple[jax.Array, jax.Array, float, jax.Array]:
        """Initialize simulation state (Stress, Thresholds, Strain)."""
        k1, k2 = jax.random.split(key)

        # Initial Stress: Start relaxed (zero) or maybe aged? Standard is zero.
        stress = jnp.zeros((self.L, self.L))

        # Initial Thresholds: Gaussian
        mean = self.params.get_value("sigma_c_mean")
        std = self.params.get_value("sigma_c_std")
        thresholds = mean + std * jax.random.normal(k1, (self.L, self.L))
        thresholds = jnp.maximum(thresholds, 1e-4)

        strain = 0.0

        return (stress, thresholds, strain, k2)

    def _fit(self, X, y, **kwargs):
        """Fit model parameters to data (NLSQ)."""
        # EPM fitting is complex. We use the standard pipeline but ensure
        # smooth=True is passed to _predict implicitly via test_mode config or similar.
        # For now, base implementation.
        pass

    def _predict(self, rheo_data: RheoData, **kwargs) -> RheoData:
        """Simulate the model for the given protocol.

        Args:
            rheo_data: Input data defining the protocol (t, gamma_dot, stress, etc.).
            kwargs:
                test_mode (str): 'flow_curve', 'startup', 'relaxation', 'creep', 'oscillation'.
                smooth (bool): Use smooth yielding (default False for simulation, True for fitting).
                seed (int): Random seed (default 0).

        Returns:
            RheoData with simulation results (stress or strain).
        """
        test_mode = kwargs.get("test_mode", rheo_data.test_mode)
        smooth = kwargs.get("smooth", False)
        seed = kwargs.get("seed", 0)
        key = jax.random.PRNGKey(seed)

        # Extract Parameters
        # Scale propagator by current mu
        mu = self.params.get_value("mu")
        propagator_q = self._propagator_q_norm * mu

        # Safe parameter extraction for kernels
        param_dict = {
            "mu": mu,
            "tau_pl": self.params.get_value("tau_pl"),
            "sigma_c_mean": self.params.get_value("sigma_c_mean"),
            "sigma_c_std": self.params.get_value("sigma_c_std"),
            "smoothing_width": self.params.get_value("smoothing_width"),
        }

        if test_mode == "flow_curve":
            return self._run_flow_curve(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "startup":
            return self._run_startup(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "relaxation":
            return self._run_relaxation(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "creep":
            return self._run_creep(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "oscillation":
            return self._run_oscillation(rheo_data, key, propagator_q, param_dict, smooth)
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

    # --- Protocol Runners ---

    def _run_flow_curve(self, data: RheoData, key: jax.Array, propagator_q: jax.Array, params: dict, smooth: bool):
        """Steady state flow curve: Stress vs Shear Rate."""
        # Use data.x as shear rates
        shear_rates = data.x

        def scan_fn(gdot):
            # Run simulation for a fixed number of steps
            # To ensure steady state, we choose a sufficient number
            n_steps = 1000

            state = self._init_state(key)

            def body(carrier, _):
                curr_state = carrier
                new_state = epm_step(curr_state, propagator_q, gdot, self.dt, params, smooth)
                # Return stress average
                return new_state, jnp.mean(new_state[0])

            _, history = jax.lax.scan(body, state, None, length=n_steps)

            # Average last 50% for steady state
            steady_stress = jnp.mean(history[n_steps // 2:])
            return steady_stress

        # Vectorize over shear rates
        stresses = jax.vmap(scan_fn)(shear_rates)
        return RheoData(x=shear_rates, y=stresses, initial_test_mode="flow_curve")

    def _run_startup(self, data: RheoData, key: jax.Array, propagator_q: jax.Array, params: dict, smooth: bool):
        """Start-up shear: Stress(t) at constant rate."""
        time = data.x

        # Assume constant shear rate provided in metadata
        gdot = data.metadata.get("gamma_dot", 0.1)

        n_steps = len(time)
        state = self._init_state(key)

        def body(carrier, _):
            curr_state = carrier
            new_state = epm_step(curr_state, propagator_q, gdot, self.dt, params, smooth)
            return new_state, jnp.mean(new_state[0])

        _, stresses = jax.lax.scan(body, state, None, length=n_steps)
        return RheoData(x=time, y=stresses, initial_test_mode="startup")

    def _run_relaxation(self, data: RheoData, key: jax.Array, propagator_q: jax.Array, params: dict, smooth: bool):
        """Stress relaxation: G(t) after step strain."""
        time = data.x
        n_steps = len(time)

        # Step strain magnitude from metadata
        strain_step = data.metadata.get("gamma", 0.1)

        state = self._init_state(key)
        stress, thresh, strain, k = state

        # Apply Step Strain (Elastic Load)
        mu = params["mu"]
        stress = stress + mu * strain_step
        state = (stress, thresh, strain + strain_step, k)

        # Relax (gdot = 0)
        def body(carrier, _):
            curr_state = carrier
            new_state = epm_step(curr_state, propagator_q, 0.0, self.dt, params, smooth)
            # Return G(t) = Stress / gamma_0
            return new_state, jnp.mean(new_state[0]) / strain_step

        _, moduli = jax.lax.scan(body, state, None, length=n_steps)
        return RheoData(x=time, y=moduli, initial_test_mode="relaxation")

    def _run_creep(self, data: RheoData, key: jax.Array, propagator_q: jax.Array, params: dict, smooth: bool):
        """Creep: Strain(t) at constant stress using Adaptive P-Controller."""
        time = data.x
        n_steps = len(time)

        # Target stress from metadata or mean of y (if y is stress input)
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
            # dot_gamma_new = dot_gamma_old + Kp * error
            gdot_new = gdot + Kp * error
            # Prevent negative shear rate if physics dictates (usually creep is monotonic)
            gdot_new = jnp.maximum(gdot_new, 0.0)

            # Step EPM
            new_epm = epm_step(curr_epm, propagator_q, gdot_new, self.dt, params, smooth)

            # Return Strain
            return (new_epm, gdot_new), new_epm[2]

        _, strains = jax.lax.scan(body, aug_state, None, length=n_steps)
        return RheoData(x=time, y=strains, initial_test_mode="creep")

    def _run_oscillation(self, data: RheoData, key: jax.Array, propagator_q: jax.Array, params: dict, smooth: bool):
        """SAOS/LAOS: Stress(t) for sinusoidal strain."""
        time = data.x
        n_steps = len(time)

        # Params
        gamma0 = data.metadata.get("gamma0", 1.0)
        omega = data.metadata.get("omega", 1.0)

        state = self._init_state(key)

        def body(carrier, t):
            curr_state = carrier
            # Time varying shear rate
            gdot = gamma0 * omega * jnp.cos(omega * t)

            new_state = epm_step(curr_state, propagator_q, gdot, self.dt, params, smooth)
            return new_state, jnp.mean(new_state[0])

        _, stresses = jax.lax.scan(body, state, time, length=n_steps)
        return RheoData(x=time, y=stresses, initial_test_mode="oscillation")
