"""Base class for Fluidity models.

Provides shared infrastructure for:
1. Parameter initialization for Local and Non-Local variants
2. Common parameter definitions for yield-stress fluid behavior
3. Shared methods for ODE/PDE integration with Diffrax
"""

from __future__ import annotations

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


class FluidityBase(BaseModel):
    """Base class for Fluidity models for yield-stress fluids.

    Implements shared parameter management and trajectory storage
    for Local (0D) and Non-Local (1D) variants.

    The Fluidity model describes yield-stress fluids through a scalar
    fluidity field f(t) or f(y,t) that evolves via:
    - Aging (structural build-up) toward equilibrium fluidity f_eq
    - Rejuvenation (flow-induced breakdown) toward high-shear fluidity f_inf

    Attributes:
        parameters: ParameterSet containing model constants
    """

    def __init__(self):
        """Initialize Fluidity Base Model."""
        super().__init__()
        self._setup_parameters()

        # Internal storage for trajectories (for plotting/debugging)
        self._trajectory: dict[str, np.ndarray] | None = None

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    def _setup_parameters(self):
        """Initialize ParameterSet with shared parameters."""
        self.parameters = ParameterSet()

        # --- Elastic/Viscoelastic Parameters ---

        # G: Elastic modulus (Pa)
        self.parameters.add(
            name="G",
            value=1e6,
            bounds=(1e3, 1e9),
            units="Pa",
            description="Elastic modulus",
        )

        # --- Yield Stress / Flow Curve Parameters (Herschel-Bulkley) ---

        # tau_y: Yield stress (Pa)
        self.parameters.add(
            name="tau_y",
            value=10.0,
            bounds=(1e-3, 1e6),
            units="Pa",
            description="Yield stress",
        )

        # K: Flow consistency (Pa·s^n)
        self.parameters.add(
            name="K",
            value=10.0,
            bounds=(1e-3, 1e6),
            units="Pa·s^n",
            description="Flow consistency (HB K parameter)",
        )

        # n_flow: Flow exponent (dimensionless)
        self.parameters.add(
            name="n_flow",
            value=0.5,
            bounds=(0.1, 2.0),
            units="dimensionless",
            description="Flow exponent (HB n parameter)",
        )

        # --- Fluidity Dynamics Parameters ---

        # f_eq: Equilibrium (low-shear) fluidity (1/(Pa·s))
        self.parameters.add(
            name="f_eq",
            value=1e-6,
            bounds=(1e-12, 1e-3),
            units="1/(Pa·s)",
            description="Equilibrium fluidity (aging limit)",
        )

        # f_inf: High-shear fluidity (1/(Pa·s))
        self.parameters.add(
            name="f_inf",
            value=1e-3,
            bounds=(1e-6, 1.0),
            units="1/(Pa·s)",
            description="High-shear fluidity (rejuvenation limit)",
        )

        # theta: Relaxation time / aging timescale (s)
        self.parameters.add(
            name="theta",
            value=10.0,
            bounds=(0.1, 1e4),
            units="s",
            description="Structural relaxation time (aging timescale)",
        )

        # a: Rejuvenation amplitude (dimensionless)
        self.parameters.add(
            name="a",
            value=1.0,
            bounds=(0.0, 100.0),
            units="dimensionless",
            description="Rejuvenation amplitude",
        )

        # n_rejuv: Rejuvenation exponent (dimensionless)
        self.parameters.add(
            name="n_rejuv",
            value=1.0,
            bounds=(0.0, 2.0),
            units="dimensionless",
            description="Rejuvenation exponent",
        )

    def _init_hb_from_data(
        self, gamma_dot: np.ndarray, stress: np.ndarray
    ) -> None:
        """Seed HB parameters (tau_y, K, n_flow) from flow-curve data.

        Estimates a Herschel-Bulkley fit ``σ = τ_y + K·γ̇^n`` directly
        from the data so the NLSQ optimizer starts close to the
        solution. Without this, the generic defaults are typically
        orders of magnitude away from real measurements and the
        optimizer terminates after a single step on the xtol criterion.

        - tau_y is taken as 90 % of the smallest measured stress
          (the low-rate plateau).
        - n_flow is estimated from the high-rate slope of
          log(σ - τ_y) vs log(γ̇).
        - K is back-solved at the highest reliable shear rate.

        All seeds are clipped to the parameter bounds.
        """
        gamma = np.abs(np.asarray(gamma_dot, dtype=float))
        sig = np.asarray(stress, dtype=float)

        if gamma.size == 0 or sig.size == 0:
            return

        # Sort by shear rate so the slope estimate is well-defined
        order = np.argsort(gamma)
        gamma_s = gamma[order]
        sig_s = sig[order]

        # Yield stress: low-rate plateau, slightly below the minimum
        # measured stress (so the residual at γ̇→0 is non-zero).
        sig_min = float(np.min(sig_s[sig_s > 0])) if np.any(sig_s > 0) else 1.0
        tau_y_seed = max(sig_min * 0.9, 1e-3)

        # Use the upper half of the (sorted) data to estimate the
        # high-rate slope without contamination from the plateau.
        n_pts = len(gamma_s)
        hi = max(n_pts // 2, 2)
        gamma_hi = gamma_s[-hi:]
        sig_hi = sig_s[-hi:]
        excess = sig_hi - tau_y_seed
        mask = (gamma_hi > 0) & (excess > 0)
        if np.count_nonzero(mask) >= 2:
            log_g = np.log(gamma_hi[mask])
            log_e = np.log(excess[mask])
            slope, intercept = np.polyfit(log_g, log_e, 1)
            n_seed = float(np.clip(slope, 0.1, 1.5))
            K_seed = float(np.exp(intercept))
        else:
            n_seed = 0.5
            K_seed = max((sig_s[-1] - tau_y_seed), 1.0) / max(
                gamma_s[-1] ** n_seed, 1e-6
            )

        # Clip seeds to bounds before applying so set_value() does not
        # raise on edge cases (e.g. data well below the lower bound).
        def _clipped(name: str, value: float) -> float:
            param = self.parameters[name]
            lo, hi_b = param.bounds if param.bounds else (None, None)
            lo_v = lo if lo is not None else -np.inf
            hi_v = hi_b if hi_b is not None else np.inf
            return float(np.clip(value, lo_v, hi_v))

        self.parameters.set_value("tau_y", _clipped("tau_y", tau_y_seed))
        self.parameters.set_value("K", _clipped("K", max(K_seed, 1e-3)))
        self.parameters.set_value("n_flow", _clipped("n_flow", n_seed))

    def get_initial_fluidity(self) -> float:
        """Get initial fluidity value (equilibrium).

        Returns:
            Initial fluidity f_0 = f_eq
        """
        f_eq = self.parameters.get_value("f_eq")
        return f_eq if f_eq is not None else 1e-6

    def get_parameter_dict(self) -> dict:
        """Get all parameters as a dictionary.

        Returns:
            Dictionary of parameter name -> value
        """
        return {k: self.parameters.get_value(k) for k in self.parameters.keys()}

    def _get_base_ode_args(self, params: dict | None = None) -> dict:
        """Build base args dictionary for ODE integration.

        Args:
            params: Optional parameter dictionary. If None, uses stored values.

        Returns:
            Dictionary with all parameters needed for ODE RHS.
        """
        if params is None:
            params = self.get_parameter_dict()

        return {
            "G": params["G"],
            "tau_y": params["tau_y"],
            "K": params["K"],
            "n_flow": params["n_flow"],
            "f_eq": params["f_eq"],
            "f_inf": params["f_inf"],
            "theta": params["theta"],
            "a": params["a"],
            "n_rejuv": params["n_rejuv"],
        }
