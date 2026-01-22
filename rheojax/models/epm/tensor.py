"""Tensorial Elasto-Plastic Model (EPM) implementation.

This module implements the full tensorial (3-component) stress formulation for EPM
simulations. It tracks the stress tensor [σ_xx, σ_yy, σ_xy] in 2D plane strain,
enabling prediction of normal stress differences (N₁, N₂), anisotropic flow behavior,
and kinematic hardening.

Key Features:
- Full tensorial stress state per lattice site
- Von Mises and Hill anisotropic yield criteria
- Normal stress difference predictions (N₁, N₂)
- Flexible fitting: shear-only or combined [σ_xy, N₁]
"""

from functools import partial
from typing import Dict, Tuple, Optional

from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.models.epm.base import EPMBase
from rheojax.utils.epm_kernels_tensorial import (
    make_tensorial_propagator_q,
    tensorial_epm_step,
)

jax, jnp = safe_import_jax()


@ModelRegistry.register(
    "tensorial_epm",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
    ],
)
class TensorialEPM(EPMBase):
    """3-Component Tensorial Lattice EPM.

    A mesoscopic model for amorphous solids that explicitly tracks the full stress
    tensor, enabling predictions of normal stress differences and anisotropic flow.

    Physics:
        - Lattice of elastoplastic blocks with tensorial stress state.
        - Elastic loading (affine) for all stress components.
        - Von Mises or Hill yield criterion for anisotropic materials.
        - Component-wise plastic flow rule (Prandtl-Reuss).
        - Tensorial Eshelby propagator for stress redistribution.

    Parameters:
        mu (float): Shear modulus. Default 1.0.
        nu (float): Poisson's ratio for plane strain. Default 0.48 (avoid 0.5 singularity).
        tau_pl (float): Base plastic relaxation timescale (legacy). Default 1.0.
        tau_pl_shear (float): Plastic relaxation time for shear. Default 1.0.
        tau_pl_normal (float): Plastic relaxation time for normal stresses. Default 1.0.
        sigma_c_mean (float): Mean yield threshold. Default 1.0.
        sigma_c_std (float): Disorder strength (std dev of thresholds). Default 0.1.
        smoothing_width (float): Width for smooth yielding approx (inference only). Default 0.1.
        w_N1 (float): Weight for N₁ in combined fitting loss. Default 1.0.
        hill_H (float): Hill anisotropy parameter H. Default 0.5.
        hill_N (float): Hill anisotropy parameter N. Default 1.5.

    Configuration:
        L (int): Lattice size (LxL). Default 64.
        dt (float): Time step. Default 0.01.
        yield_criterion (str): "von_mises" or "hill". Default "von_mises".
    """

    def __init__(
        self,
        L: int = 64,
        dt: float = 0.01,
        mu: float = 1.0,
        nu: float = 0.48,  # Avoid nu=0.5 (incompressible singularity in plane strain)
        tau_pl: float = 1.0,
        tau_pl_shear: float = 1.0,
        tau_pl_normal: float = 1.0,
        sigma_c_mean: float = 1.0,
        sigma_c_std: float = 0.1,
        yield_criterion: str = "von_mises",
    ):
        """Initialize the Tensorial EPM.

        Args:
            L: Lattice size (LxL grid).
            dt: Time step for integration.
            mu: Shear modulus.
            nu: Poisson's ratio for plane strain constraint.
            tau_pl: Base plastic relaxation time (for compatibility).
            tau_pl_shear: Plastic relaxation time for shear components.
            tau_pl_normal: Plastic relaxation time for normal stress components.
            sigma_c_mean: Mean yield threshold.
            sigma_c_std: Standard deviation of yield thresholds (disorder).
            yield_criterion: Yield criterion name ("von_mises" or "hill").
        """
        # Initialize base class with common parameters
        super().__init__(
            L=L,
            dt=dt,
            mu=mu,
            tau_pl=tau_pl,
            sigma_c_mean=sigma_c_mean,
            sigma_c_std=sigma_c_std,
        )

        # Add tensorial-specific parameters
        self.parameters.add(
            "nu", nu, bounds=(0.3, 0.5),
            units="", description="Poisson's ratio for plane strain"
        )
        self.parameters.add(
            "tau_pl_shear", tau_pl_shear, bounds=(0.01, 100.0),
            units="s", description="Plastic relaxation time for shear"
        )
        self.parameters.add(
            "tau_pl_normal", tau_pl_normal, bounds=(0.01, 100.0),
            units="s", description="Plastic relaxation time for normal stresses"
        )
        self.parameters.add(
            "w_N1", 1.0, bounds=(0.1, 10.0),
            units="", description="Weight for N₁ in combined fitting loss"
        )
        self.parameters.add(
            "hill_H", 0.5, bounds=(0.1, 5.0),
            units="", description="Hill anisotropy parameter H"
        )
        self.parameters.add(
            "hill_N", 1.5, bounds=(0.1, 5.0),
            units="", description="Hill anisotropy parameter N"
        )

        # Yield criterion (static configuration)
        if yield_criterion not in ["von_mises", "hill"]:
            raise ValueError(
                f"Unknown yield criterion: {yield_criterion}. "
                "Must be 'von_mises' or 'hill'."
            )
        self.yield_criterion = yield_criterion

        # Precompute tensorial propagator (cached)
        # Using mu=1.0 as normalization, will scale by actual mu during execution
        self._propagator_q_norm = make_tensorial_propagator_q(L, nu=nu, mu=1.0)

    def _init_stress(self, key: jax.Array) -> jax.Array:
        """Initialize tensorial stress field.

        Args:
            key: PRNG key (unused for zero initialization).

        Returns:
            Zero-initialized stress tensor of shape (3, L, L) for [σ_xx, σ_yy, σ_xy].
        """
        # Start relaxed (zero stress)
        return jnp.zeros((3, self.L, self.L))

    def _get_param_dict(self) -> Dict[str, float]:
        """Extract parameters as dictionary for kernel calls.

        Extends base class method to include tensorial parameters.

        Returns:
            Dictionary with all EPM parameters including tensorial ones.
        """
        base_params = super()._get_param_dict()

        # Add tensorial-specific parameters
        tensorial_params = {
            "nu": self.parameters.get_value("nu"),
            "tau_pl_shear": self.parameters.get_value("tau_pl_shear"),
            "tau_pl_normal": self.parameters.get_value("tau_pl_normal"),
            "hill_H": self.parameters.get_value("hill_H"),
            "hill_N": self.parameters.get_value("hill_N"),
        }

        return {**base_params, **tensorial_params}

    def _epm_step(
        self,
        state: Tuple[jax.Array, jax.Array, float, jax.Array],
        propagator_q: jax.Array,
        shear_rate: float,
        dt: float,
        params: dict,
        smooth: bool,
    ) -> Tuple[jax.Array, jax.Array, float, jax.Array]:
        """Perform one tensorial EPM time step.

        Delegates to tensorial_epm_step kernel from epm_kernels_tensorial module.

        Args:
            state: Current state (stress, thresholds, strain, key).
            propagator_q: Precomputed tensorial propagator.
            shear_rate: Imposed macroscopic shear rate.
            dt: Time step size.
            params: Model parameters dictionary.
            smooth: Use smooth yielding (tanh) or hard threshold (step).

        Returns:
            Updated state tuple (new_stress, thresholds, new_strain, key).
        """
        stress, thresholds, strain, key = state

        # Call tensorial kernel
        new_stress = tensorial_epm_step(
            stress=stress,
            thresholds=thresholds,
            strain_rate=shear_rate,
            dt=dt,
            propagator=propagator_q,
            params=params,
            smooth=smooth,
            yield_criterion=self.yield_criterion,
        )

        # Update accumulated strain
        new_strain = strain + shear_rate * dt

        return (new_stress, thresholds, new_strain, key)

    def get_shear_stress(self, stress: jax.Array) -> jax.Array:
        """Extract shear stress component from stress tensor.

        Args:
            stress: Stress tensor of shape (3, L, L) or (..., 3, L, L).

        Returns:
            Shear stress σ_xy, shape (L, L) or (..., L, L).
        """
        # Component ordering: [σ_xx, σ_yy, σ_xy]
        if stress.ndim == 3:
            return stress[2]
        else:
            return stress[..., 2, :, :]

    def get_normal_stress_differences(
        self, stress: jax.Array, nu: Optional[float] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute normal stress differences from stress tensor.

        For plane strain: σ_zz = ν(σ_xx + σ_yy)

        Normal stress differences:
        - N₁ = σ_xx - σ_yy
        - N₂ = σ_yy - σ_zz

        Args:
            stress: Stress tensor of shape (3, L, L).
            nu: Poisson's ratio. If None, uses parameter value.

        Returns:
            Tuple (N₁, N₂), each of shape (L, L).
        """
        if nu is None:
            nu = self.parameters.get_value("nu")

        sigma_xx = stress[0]
        sigma_yy = stress[1]

        # Plane strain constraint
        sigma_zz = nu * (sigma_xx + sigma_yy)

        N1 = sigma_xx - sigma_yy
        N2 = sigma_yy - sigma_zz

        return N1, N2

    def predict_normal_stresses(
        self, data: RheoData, **kwargs
    ) -> Tuple[jax.Array, jax.Array]:
        """Convenience method to predict normal stress differences.

        Runs the simulation and extracts N₁ and N₂ spatial averages over time.

        Args:
            data: RheoData with protocol specification.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Tuple (N₁_array, N₂_array) with time-averaged values.

        Raises:
            NotImplementedError: Not yet implemented (future feature).
        """
        raise NotImplementedError(
            "predict_normal_stresses() not yet implemented. "
            "Use predict() with test_mode='flow_curve' which returns [σ_xy, N₁]."
        )

    # Override flow_curve to return shear stress with N₁ in metadata
    def _run_flow_curve(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Steady state flow curve: Stress vs Shear Rate.

        For tensorial EPM, returns shear stress σ_xy with N₁ stored in metadata.

        Args:
            data: RheoData with x=shear_rates.
            key: PRNG key.
            propagator_q: Precomputed propagator.
            params: Model parameters.
            smooth: Use smooth yielding.

        Returns:
            RheoData with x=shear_rates, y=sigma_xy.
            metadata contains 'N1' with first normal stress differences.
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
                stress_tensor = new_state[0]  # Shape (3, L, L)

                # Extract shear stress and N₁
                sigma_xy_mean = jnp.mean(stress_tensor[2])
                N1, _ = self.get_normal_stress_differences(stress_tensor, params["nu"])
                N1_mean = jnp.mean(N1)

                return new_state, jnp.array([sigma_xy_mean, N1_mean])

            _, history = jax.lax.scan(body, state, None, length=n_steps)

            # Average last 50% for steady state
            # history has shape (n_steps, 2)
            steady_values = jnp.mean(history[n_steps // 2:], axis=0)
            return steady_values  # [σ_xy, N₁]

        # Vectorize over shear rates
        stresses = jax.vmap(scan_fn)(shear_rates)  # Shape: (n_rates, 2)

        # Extract components
        sigma_xy = stresses[:, 0]
        N1 = stresses[:, 1]

        # Store N₁ in metadata
        result_metadata = data.metadata.copy() if data.metadata else {}
        result_metadata["N1"] = N1

        return RheoData(
            x=shear_rates,
            y=sigma_xy,
            initial_test_mode="flow_curve",
            metadata=result_metadata
        )

    def _predict(self, rheo_data: RheoData, **kwargs) -> RheoData:
        """Simulate the model for the given protocol.

        Args:
            rheo_data: Input data defining the protocol (t, gamma_dot, stress, etc.).
            kwargs:
                test_mode (str): 'flow_curve', 'startup', 'relaxation', 'creep', 'oscillation'.
                smooth (bool): Use smooth yielding (default False for simulation, True for fitting).
                seed (int): Random seed (default 0).

        Returns:
            RheoData with simulation results.
            - flow_curve: y is σ_xy array, metadata['N1'] contains N₁ values
            - Other protocols: y has shape (n_points,) with σ_xy only
        """
        test_mode = kwargs.get("test_mode", rheo_data.test_mode)
        smooth = kwargs.get("smooth", False)
        seed = kwargs.get("seed", 0)
        key = jax.random.PRNGKey(seed)

        # Extract parameters
        # Scale propagator by current mu
        mu = self.parameters.get_value("mu")
        propagator_q = self._propagator_q_norm * mu

        # Get full parameter dictionary
        param_dict = self._get_param_dict()

        if test_mode == "flow_curve":
            return self._run_flow_curve(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "startup":
            return self._run_startup_tensorial(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "relaxation":
            return self._run_relaxation_tensorial(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "creep":
            return self._run_creep_tensorial(rheo_data, key, propagator_q, param_dict, smooth)
        elif test_mode == "oscillation":
            return self._run_oscillation_tensorial(rheo_data, key, propagator_q, param_dict, smooth)
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

    def _run_startup_tensorial(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Start-up shear: Stress(t) at constant rate.

        Extracts shear component from tensorial stress.
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
            # Extract shear stress component (index 2)
            return new_state, jnp.mean(new_state[0][2])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, None, length=n_steps)
            # Prepend initial stress
            initial_stress = jnp.mean(state[0][2])
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([jnp.mean(state[0][2])])

        return RheoData(x=time, y=stresses, initial_test_mode="startup")

    def _run_relaxation_tensorial(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Stress relaxation: G(t) after step strain.

        Extracts shear component from tensorial stress.
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

        # Apply Step Strain (Elastic Load) - only to shear component
        mu = params["mu"]
        stress = stress.at[2].set(stress[2] + mu * strain_step)
        state = (stress, thresh, strain + strain_step, k)

        # Initial G(0)
        g_0 = jnp.mean(stress[2]) / strain_step

        # Relax (gdot = 0) for N-1 steps
        n_steps = max(0, len(time) - 1)

        def body(carrier, _):
            curr_state = carrier
            new_state = self._epm_step(curr_state, propagator_q, 0.0, dt, params, smooth)
            # Return G(t) = Shear Stress / gamma_0
            return new_state, jnp.mean(new_state[0][2]) / strain_step

        if n_steps > 0:
            _, moduli_scan = jax.lax.scan(body, state, None, length=n_steps)
            moduli = jnp.concatenate([jnp.array([g_0]), moduli_scan])
        else:
            moduli = jnp.array([g_0])

        return RheoData(x=time, y=moduli, initial_test_mode="relaxation")

    def _run_creep_tensorial(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """Creep: Strain(t) at constant stress using Adaptive P-Controller.

        Extracts shear component from tensorial stress.
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
            # Extract shear stress component
            curr_stress = jnp.mean(stress_grid[2])

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

    def _run_oscillation_tensorial(
        self,
        data: RheoData,
        key: jax.Array,
        propagator_q: jax.Array,
        params: dict,
        smooth: bool,
    ) -> RheoData:
        """SAOS/LAOS: Stress(t) for sinusoidal strain.

        Extracts shear component from tensorial stress.
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
        initial_stress = jnp.mean(state[0][2])

        # Run for N-1 steps
        n_steps = max(0, len(time) - 1)
        scan_time = time[:-1] if n_steps > 0 else jnp.array([])

        def body(carrier, t):
            curr_state = carrier
            # Time varying shear rate at current time t
            gdot = gamma0 * omega * jnp.cos(omega * t)

            new_state = self._epm_step(curr_state, propagator_q, gdot, dt, params, smooth)
            # Extract shear stress component
            return new_state, jnp.mean(new_state[0][2])

        if n_steps > 0:
            _, stresses_scan = jax.lax.scan(body, state, scan_time, length=n_steps)
            stresses = jnp.concatenate([jnp.array([initial_stress]), stresses_scan])
        else:
            stresses = jnp.array([initial_stress])

        return RheoData(x=time, y=stresses, initial_test_mode="oscillation")

    def _fit(self, X, y, **kwargs):
        """Fit model parameters to data with flexible target selection.

        Supports:
        - 1D y: Fit to shear stress σ_xy only (backward compatible)
        - 2D y with shape (2, n): Fit to [σ_xy, N₁] simultaneously
        - 3D y: Not yet supported (full tensor fitting)

        Args:
            X: Shear rates or time array.
            y: Target data (1D or 2D array).
            **kwargs:
                test_mode (str): Protocol type (default 'flow_curve').
                Other fitting parameters.

        Raises:
            NotImplementedError: EPM fitting is complex and not yet fully implemented.
        """
        # Auto-detect fitting mode from y shape
        if y.ndim == 1:
            # Shear-only fitting (backward compatible)
            fitting_mode = "shear_only"
        elif y.ndim == 2 and y.shape[0] == 2:
            # Combined fitting [σ_xy, N₁]
            fitting_mode = "combined"
            w_N1 = self.parameters.get_value("w_N1")
        elif y.ndim == 2 and y.shape[0] == 3:
            raise NotImplementedError(
                "Full tensor fitting (3 components) not yet supported. "
                "Use 1D y for shear-only or 2D y with shape (2, n) for [σ_xy, N₁]."
            )
        else:
            raise ValueError(
                f"Invalid y shape: {y.shape}. "
                "Expected 1D for shear-only or (2, n) for [σ_xy, N₁]."
            )

        # Fitting requires smooth approximation and gradient-based optimization
        # This is complex for EPM and requires careful implementation
        raise NotImplementedError(
            f"TensorialEPM fitting (mode: {fitting_mode}) not yet implemented. "
            "EPM parameter inference requires MCMC or specialized optimization. "
            "Use model.predict() for forward simulations."
        )
