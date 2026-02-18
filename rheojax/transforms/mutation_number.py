"""Mutation number analysis for relaxation data.

This module calculates the mutation number (Δ) from relaxation modulus data
to quantify the degree of time-dependence in viscoelastic materials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.inventory import TransformType
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from jax import Array


IntegrationMethod = Literal["trapz", "simpson", "cumulative"]


@TransformRegistry.register("mutation_number", type=TransformType.ANALYSIS)
class MutationNumber(BaseTransform):
    """Calculate mutation number from relaxation modulus data.

    The mutation number (Δ) is a dimensionless parameter that quantifies
    how quickly a viscoelastic material relaxes. It is defined as:

        Δ = ∫[0 to ∞] G(t) dt / (G(0) × τ_avg)

    where:
    - G(t) is the relaxation modulus
    - G(0) is the initial modulus
    - τ_avg is the average relaxation time

    The mutation number ranges from:
    - Δ → 0: Elastic solid (no relaxation)
    - Δ → 1: Viscous fluid (complete relaxation)

    For practical calculations with finite time data:
        Δ = ∫G(t)dt / (G(0) × τ_avg)
        where τ_avg = ∫G(t)dt / G(0)

    This simplifies to:
        Δ = [∫G(t)dt]² / [G(0) × ∫t×G(t)dt]

    Parameters
    ----------
    integration_method : IntegrationMethod, default='trapz'
        Numerical integration method: 'trapz', 'simpson', or 'cumulative'
    extrapolate : bool, default=False
        Whether to extrapolate to infinite time
    extrapolation_model : str, default='exponential'
        Model for extrapolation: 'exponential', 'powerlaw'

    Examples
    --------
    >>> from rheojax.core.data import RheoData
    >>> from rheojax.transforms.mutation_number import MutationNumber
    >>>
    >>> # Create relaxation data
    >>> t = jnp.linspace(0, 100, 1000)
    >>> G_t = 1000 * jnp.exp(-t/10.0)  # Exponential relaxation
    >>> data = RheoData(x=t, y=G_t, domain='time',
    ...                 metadata={'test_mode': 'relaxation'})
    >>>
    >>> # Calculate mutation number
    >>> mutation = MutationNumber(integration_method='trapz')
    >>> delta = mutation.calculate(data)
    >>> print(f"Mutation number: {delta:.4f}")
    """

    def __init__(
        self,
        integration_method: IntegrationMethod = "trapz",
        extrapolate: bool = False,
        extrapolation_model: str = "exponential",
    ):
        """Initialize Mutation Number transform.

        Parameters
        ----------
        integration_method : IntegrationMethod
            Numerical integration method
        extrapolate : bool
            Whether to extrapolate to infinite time
        extrapolation_model : str
            Model for extrapolation
        """
        super().__init__()
        self.integration_method = integration_method
        self.extrapolate = extrapolate
        self.extrapolation_model = extrapolation_model

    def _integrate(self, x, y) -> float:
        """Perform numerical integration.

        Parameters
        ----------
        x : jnp.ndarray
            Independent variable
        y : jnp.ndarray
            Dependent variable to integrate

        Returns
        -------
        float
            Integral value
        """
        logger.debug(
            "Performing numerical integration",
            method=self.integration_method,
            n_points=len(x),
        )

        if self.integration_method == "trapz":
            # Use trapezoidal rule
            from jax.scipy.integrate import trapezoid

            return float(trapezoid(y, x))
        elif self.integration_method == "simpson":
            # Simpson's rule requires odd number of points
            # Use scipy simpson
            from scipy.integrate import simpson

            x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
            y_np = np.array(y) if isinstance(y, jnp.ndarray) else y
            return float(simpson(y_np, x_np))
        elif self.integration_method == "cumulative":
            # Use trapezoid for cumulative integral
            from jax.scipy.integrate import trapezoid

            return float(trapezoid(y, x))
        else:
            logger.error(  # type: ignore[unreachable]
                "Unknown integration method",
                method=self.integration_method,
            )
            raise ValueError(f"Unknown integration method: {self.integration_method}")

    def _extrapolate_tail(self, t, G_t) -> float:
        """Extrapolate tail contribution to infinite time.

        Parameters
        ----------
        t : jnp.ndarray
            Time data
        G_t : jnp.ndarray
            Relaxation modulus data

        Returns
        -------
        float
            Extrapolated tail integral contribution
        """
        # Use last few points to fit extrapolation model
        n_fit = min(10, len(t) // 4)
        t_fit = t[-n_fit:]
        G_fit = G_t[-n_fit:]

        if self.extrapolation_model == "exponential":
            # Fit G(t) = A * exp(-t/tau)
            # ln(G) = ln(A) - t/tau
            log_G = jnp.log(jnp.maximum(G_fit, 1e-10))  # Clamp positive before log

            # Linear regression
            t_mean = jnp.mean(t_fit)
            log_G_mean = jnp.mean(log_G)

            denom = jnp.sum((t_fit - t_mean) ** 2)
            slope = jnp.where(
                denom > 1e-30,
                jnp.sum((t_fit - t_mean) * (log_G - log_G_mean)) / denom,
                0.0,
            )
            intercept = log_G_mean - slope * t_mean

            # Extract parameters (guard slope near zero)
            tau = -1.0 / jnp.where(jnp.abs(slope) > 1e-20, slope, -1e-20)
            A = jnp.exp(intercept)

            # Integrate from t_max to infinity
            t_max = t[-1]
            tail_integral = A * tau * jnp.exp(-t_max / tau)

            return float(tail_integral)

        elif self.extrapolation_model == "powerlaw":
            # Fit G(t) = A * t^(-n)
            # ln(G) = ln(A) - n*ln(t)
            log_t = jnp.log(t_fit)
            log_G = jnp.log(jnp.maximum(G_fit, 1e-10))

            # Linear regression
            log_t_mean = jnp.mean(log_t)
            log_G_mean = jnp.mean(log_G)

            denom = jnp.sum((log_t - log_t_mean) ** 2)
            n = jnp.where(
                denom > 1e-30,
                -jnp.sum((log_t - log_t_mean) * (log_G - log_G_mean)) / denom,
                0.0,
            )
            A = jnp.exp(log_G_mean + n * log_t_mean)

            # Integrate from t_max to infinity
            t_max = t[-1]
            if n > 1:
                # Converges: ∫[t_max to ∞] A*t^(-n) dt = A/(n-1) * t_max^(1-n)
                tail_integral = A / (n - 1) * jnp.power(t_max, 1 - n)
            else:
                # Diverges, use finite cutoff
                tail_integral = 0.0

            return float(tail_integral)

        else:
            raise ValueError(f"Unknown extrapolation model: {self.extrapolation_model}")

    def _validate_and_prepare_data(self, rheo_data: RheoData) -> tuple:
        """Validate relaxation data and prepare arrays.

        Parameters
        ----------
        rheo_data : RheoData
            Relaxation modulus data

        Returns
        -------
        tuple
            (t, G_t, G_0) - time array, modulus array, initial modulus

        Raises
        ------
        ValueError
            If data is not valid relaxation data
        """
        logger.debug("Validating and preparing relaxation data")

        mode = detect_test_mode(rheo_data)
        if mode != TestMode.RELAXATION:
            logger.error(
                "Invalid test mode for mutation number",
                expected="RELAXATION",
                got=str(mode),
            )
            raise ValueError(
                f"Mutation number requires RELAXATION data, got {mode}. "
                f"Ensure data is monotonically decreasing in time domain."
            )

        t = rheo_data.x
        G_t = rheo_data.y

        logger.debug(
            "Data extracted",
            data_points=len(t),
            time_range=(float(t[0]), float(t[-1])),
        )

        # Convert to JAX arrays
        if not isinstance(t, jnp.ndarray):
            t = jnp.array(t)
        if not isinstance(G_t, jnp.ndarray):
            G_t = jnp.array(G_t)

        # Handle complex data
        if jnp.iscomplexobj(G_t):
            logger.debug("Converting complex modulus to real part")
            G_t = jnp.real(G_t)

        G_0 = G_t[0]
        if G_0 <= 0:
            logger.error("Initial modulus is not positive", G_0=float(G_0))
            raise ValueError("Initial modulus G(0) must be positive")

        logger.debug("Initial modulus extracted", G_0=float(G_0))

        return t, G_t, G_0

    def _estimate_equilibrium_modulus(self, G_t: Array) -> float:
        """Estimate equilibrium modulus from tail of data.

        Parameters
        ----------
        G_t : jnp.ndarray
            Relaxation modulus data

        Returns
        -------
        float
            Estimated equilibrium modulus G_eq
        """
        n_tail = max(10, len(G_t) // 10)
        return float(jnp.mean(G_t[-n_tail:]))

    def _should_extrapolate(self, G_relax: Array, G_0_relax: float) -> bool:
        """Check if extrapolation should be applied.

        Parameters
        ----------
        G_relax : jnp.ndarray
            Relaxing modulus component
        G_0_relax : float
            Initial relaxing modulus

        Returns
        -------
        bool
            Whether extrapolation conditions are met
        """
        if not self.extrapolate:
            return False

        n_check = min(10, len(G_relax) // 4)
        tail_values = G_relax[-n_check:]
        tail_mean = float(jnp.mean(tail_values))
        tail_positive = float(jnp.min(tail_values)) > 0

        # Only extrapolate if tail is positive and > 1% of initial relaxing modulus
        return tail_positive and tail_mean > 0.01 * G_0_relax

    def _extrapolate_tG_integral(self, t: Array, G_relax: Array) -> float:
        """Extrapolate the ∫t×G_relax(t)dt integral contribution.

        Parameters
        ----------
        t : jnp.ndarray
            Time data
        G_relax : jnp.ndarray
            Relaxing modulus component

        Returns
        -------
        float
            Extrapolated tail contribution (0 if extrapolation fails)
        """
        t_max = t[-1]
        G_relax_tail = G_relax[-1]

        n_fit = min(10, len(t) // 4)
        if len(t) < n_fit or G_relax_tail <= 0:
            return 0.0

        # Fit exponential decay to estimate tau
        t_fit = t[-n_fit:]
        G_relax_fit = G_relax[-n_fit:]
        log_G = jnp.log(G_relax_fit + 1e-10)

        t_mean = jnp.mean(t_fit)
        log_G_mean = jnp.mean(log_G)
        slope = jnp.sum((t_fit - t_mean) * (log_G - log_G_mean)) / (
            jnp.sum((t_fit - t_mean) ** 2) + 1e-10
        )
        safe_slope = slope if abs(float(slope)) > 1e-20 else -1e-20
        tau = -1.0 / safe_slope

        # Reasonable tau bounds
        if tau <= 0 or tau >= 1e6:
            return 0.0

        # Compute tail integral: ∫[t_max to ∞] t×A×exp(-t/tau)dt
        A = G_relax_tail * jnp.exp(t_max / tau)
        tail_tG = A * tau * (tau + t_max) * jnp.exp(-t_max / tau)

        if jnp.isnan(tail_tG) or jnp.isinf(tail_tG) or tail_tG <= 0:
            return 0.0

        return float(tail_tG)

    def _apply_extrapolation(
        self,
        t: Array,
        G_relax: Array,
        G_0_relax: float,
        integral_G: float,
        integral_tG: float,
    ) -> tuple[float, float]:
        """Apply tail extrapolation to integrals.

        Parameters
        ----------
        t : jnp.ndarray
            Time data
        G_relax : jnp.ndarray
            Relaxing modulus component
        G_0_relax : float
            Initial relaxing modulus
        integral_G : float
            Current ∫G_relax dt integral
        integral_tG : float
            Current ∫t×G_relax dt integral

        Returns
        -------
        tuple
            Updated (integral_G, integral_tG) with extrapolation
        """
        if integral_tG <= 0 or not self._should_extrapolate(G_relax, G_0_relax):
            logger.debug("Extrapolation not applied (conditions not met)")
            return integral_G, integral_tG

        try:
            logger.debug(
                "Applying tail extrapolation",
                extrapolation_model=self.extrapolation_model,
            )
            tail_G = self._extrapolate_tail(t, G_relax)
            if not jnp.isnan(tail_G) and not jnp.isinf(tail_G) and tail_G > 0:
                integral_G += tail_G
                integral_tG += self._extrapolate_tG_integral(t, G_relax)
                logger.debug(
                    "Extrapolation applied successfully",
                    tail_G=tail_G,
                    updated_integral_G=integral_G,
                    updated_integral_tG=integral_tG,
                )
        except (ValueError, RuntimeWarning, ZeroDivisionError) as e:
            logger.debug(
                "Extrapolation failed, continuing without it",
                error=str(e),
            )

        return integral_G, integral_tG

    def _compute_mutation_number(
        self,
        G_0: float,
        G_eq: float,
        G_0_relax: float,
        integral_G: float,
        integral_tG: float,
    ) -> float:
        """Compute mutation number from integrals.

        Parameters
        ----------
        G_0 : float
            Initial modulus
        G_eq : float
            Equilibrium modulus
        G_0_relax : float
            Initial relaxing modulus
        integral_G : float
            ∫G_relax dt integral
        integral_tG : float
            ∫t×G_relax dt integral

        Returns
        -------
        float
            Mutation number clamped to [0, 1]
        """
        if G_0_relax <= 0 or integral_tG <= 0 or G_0 <= 0:
            # Fallback estimate: Δ ≈ 1 - G_eq/G_0 (guard G_0 = 0)
            delta = (1.0 - G_eq / G_0) if G_0 > 0 else 0.0
        else:
            # Δ = [∫G_relax(t)dt]² / [G_0_relax × ∫t×G_relax(t)dt]
            delta = (integral_G**2) / (G_0_relax * integral_tG)

        return float(jnp.clip(delta, 0.0, 1.0))

    def calculate(self, rheo_data: RheoData) -> float:
        """Calculate mutation number from relaxation data.

        Parameters
        ----------
        rheo_data : RheoData
            Relaxation modulus data

        Returns
        -------
        float
            Mutation number Δ

        Raises
        ------
        ValueError
            If data is not relaxation mode
        """
        logger.info(
            "Starting mutation number calculation",
            integration_method=self.integration_method,
            extrapolate=self.extrapolate,
        )

        # Phase 1: Validate and prepare data
        logger.debug("Phase 1: Validating and preparing data")
        t, G_t, G_0 = self._validate_and_prepare_data(rheo_data)

        # Phase 2: Estimate equilibrium modulus
        logger.debug("Phase 2: Estimating equilibrium modulus")
        G_eq = self._estimate_equilibrium_modulus(G_t)
        logger.debug("Equilibrium modulus estimated", G_eq=G_eq)

        # Phase 3: Calculate relaxing component
        logger.debug("Phase 3: Calculating relaxing component")
        G_relax = G_t - G_eq
        G_0_relax = G_0 - G_eq
        logger.debug(
            "Relaxing component calculated",
            G_0_relax=float(G_0_relax),
        )

        # Check for pure elastic solid (no relaxation)
        if G_0_relax <= 0:
            logger.info(
                "Pure elastic solid detected (no relaxation)",
                mutation_number=0.0,
            )
            return 0.0

        # Phase 4: Compute base integrals
        logger.debug("Phase 4: Computing base integrals")
        integral_G = self._integrate(t, G_relax)
        integral_tG = self._integrate(t, t * G_relax)
        logger.debug(
            "Base integrals computed",
            integral_G=integral_G,
            integral_tG=integral_tG,
        )

        # Phase 5: Apply extrapolation if requested
        logger.debug("Phase 5: Applying extrapolation if requested")
        integral_G, integral_tG = self._apply_extrapolation(
            t, G_relax, G_0_relax, integral_G, integral_tG
        )

        # Phase 6: Compute final mutation number
        logger.debug("Phase 6: Computing final mutation number")
        delta = self._compute_mutation_number(
            G_0, G_eq, G_0_relax, integral_G, integral_tG
        )

        logger.info(
            "Mutation number calculation completed",
            mutation_number=delta,
            G_0=float(G_0),
            G_eq=float(G_eq),
        )

        return delta

    def _transform(self, data: RheoData) -> RheoData:
        """Transform relaxation data to mutation number.

        This returns a scalar RheoData with single mutation number value.

        Parameters
        ----------
        data : RheoData
            Relaxation modulus data

        Returns
        -------
        RheoData
            Scalar data containing mutation number
        """
        logger.info(
            "Starting mutation number transform",
            integration_method=self.integration_method,
            extrapolate=self.extrapolate,
        )

        # Calculate mutation number
        delta = self.calculate(data)

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update(
            {
                "transform": "mutation_number",
                "mutation_number": delta,
                "integration_method": self.integration_method,
                "extrapolated": self.extrapolate,
            }
        )

        logger.info(
            "Mutation number transform completed",
            mutation_number=delta,
        )

        # Return scalar RheoData
        return RheoData(
            x=jnp.array([0.0]),
            y=jnp.array([delta]),
            x_units=None,
            y_units="dimensionless",
            domain="scalar",
            metadata=new_metadata,
            validate=False,
        )

    def get_relaxation_time(self, rheo_data: RheoData) -> float:
        """Calculate average relaxation time from relaxation data.

        Parameters
        ----------
        rheo_data : RheoData
            Relaxation modulus data

        Returns
        -------
        float
            Average relaxation time τ_avg
        """
        # Get time and modulus data
        t = rheo_data.x
        G_t = rheo_data.y

        # Convert to JAX arrays
        if not isinstance(t, jnp.ndarray):
            t = jnp.array(t)
        if not isinstance(G_t, jnp.ndarray):
            G_t = jnp.array(G_t)

        # Handle complex data
        if jnp.iscomplexobj(G_t):
            G_t = jnp.real(G_t)

        # Get initial modulus
        G_0 = G_t[0]

        # Calculate ∫G(t)dt
        integral_G = self._integrate(t, G_t)

        # Average relaxation time
        tau_avg = integral_G / G_0

        return float(tau_avg)

    def get_equilibrium_modulus(self, rheo_data: RheoData) -> float:
        """Estimate equilibrium modulus from long-time behavior.

        Parameters
        ----------
        rheo_data : RheoData
            Relaxation modulus data

        Returns
        -------
        float
            Equilibrium modulus G_eq (0 for viscous fluids)
        """
        G_t = rheo_data.y

        # Convert to array
        if not isinstance(G_t, jnp.ndarray):
            G_t = jnp.array(G_t)

        # Handle complex
        if jnp.iscomplexobj(G_t):
            G_t = jnp.real(G_t)

        # Estimate as average of last 10% of data
        n_tail = max(10, len(G_t) // 10)
        G_eq = float(jnp.mean(G_t[-n_tail:]))

        return G_eq


__all__ = ["MutationNumber"]
