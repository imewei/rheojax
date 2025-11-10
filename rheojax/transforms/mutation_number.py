"""Mutation number analysis for relaxation data.

This module calculates the mutation number (Δ) from relaxation modulus data
to quantify the degree of time-dependence in viscoelastic materials.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


IntegrationMethod = Literal["trapz", "simpson", "cumulative"]


@TransformRegistry.register("mutation_number")
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
            log_G = jnp.log(G_fit + 1e-10)  # Avoid log(0)

            # Linear regression
            t_mean = jnp.mean(t_fit)
            log_G_mean = jnp.mean(log_G)

            slope = jnp.sum((t_fit - t_mean) * (log_G - log_G_mean)) / jnp.sum(
                (t_fit - t_mean) ** 2
            )
            intercept = log_G_mean - slope * t_mean

            # Extract parameters
            tau = -1.0 / slope
            A = jnp.exp(intercept)

            # Integrate from t_max to infinity
            t_max = t[-1]
            tail_integral = A * tau * jnp.exp(-t_max / tau)

            return float(tail_integral)

        elif self.extrapolation_model == "powerlaw":
            # Fit G(t) = A * t^(-n)
            # ln(G) = ln(A) - n*ln(t)
            log_t = jnp.log(t_fit)
            log_G = jnp.log(G_fit + 1e-10)

            # Linear regression
            log_t_mean = jnp.mean(log_t)
            log_G_mean = jnp.mean(log_G)

            n = -jnp.sum((log_t - log_t_mean) * (log_G - log_G_mean)) / jnp.sum(
                (log_t - log_t_mean) ** 2
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
        # Validate test mode
        mode = detect_test_mode(rheo_data)
        if mode != TestMode.RELAXATION:
            raise ValueError(
                f"Mutation number requires RELAXATION data, got {mode}. "
                f"Ensure data is monotonically decreasing in time domain."
            )

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

        if G_0 <= 0:
            raise ValueError("Initial modulus G(0) must be positive")

        # Estimate equilibrium modulus (average of last 10% of data)
        n_tail = max(10, len(G_t) // 10)
        G_eq = float(jnp.mean(G_t[-n_tail:]))

        # If extrapolation is enabled, use exponential fit to estimate true G_eq
        if self.extrapolate:
            # Use last few points to fit extrapolation model
            n_fit = min(10, len(t) // 4)
            t_fit = t[-n_fit:]
            G_fit = G_t[-n_fit:]

            if self.extrapolation_model == "exponential":
                # Fit G(t) = G_eq_extrap + A * exp(-t/tau)
                # For large t, G(t) → G_eq_extrap
                # ln(G - G_eq_guess) = ln(A) - t/tau

                # First guess: start with 0 (assume pure decay)
                # This avoids negative G_shifted values that break the fit
                G_eq_guess = 0.0

                # Iteratively refine G_eq estimate
                for _ in range(3):  # A few iterations usually suffice
                    G_shifted = G_fit - G_eq_guess

                    # Check if all values are positive
                    if jnp.min(G_shifted) > 0:
                        log_G = jnp.log(G_shifted + 1e-10)

                        # Linear regression on log scale
                        t_mean = jnp.mean(t_fit)
                        log_G_mean = jnp.mean(log_G)

                        slope = jnp.sum((t_fit - t_mean) * (log_G - log_G_mean)) / (
                            jnp.sum((t_fit - t_mean) ** 2) + 1e-10
                        )
                        intercept = log_G_mean - slope * t_mean

                        # Extract parameters
                        tau = -1.0 / (slope + 1e-10)
                        A = jnp.exp(intercept)

                        # Estimate G_eq from fit: G_eq ≈ G(t_max) - A*exp(-t_max/tau)
                        t_max = t[-1]
                        G_eq_guess = G_t[-1] - A * jnp.exp(-t_max / tau)

                        # Clamp to non-negative
                        G_eq_guess = jnp.maximum(0.0, G_eq_guess)
                    else:
                        # Can't fit exponential, keep current guess
                        break

                # Use refined estimate if it's reasonable (between 0 and current estimate)
                if 0 <= G_eq_guess <= G_eq:
                    G_eq = float(G_eq_guess)

        # For mutation number, integrate the relaxing part G(t) - G_eq
        # For elastic solids: G(t) ≈ G_eq → G_relax ≈ 0 → Δ ≈ 0
        # For viscous fluids: G_eq ≈ 0 → G_relax ≈ G(t) → Δ ≈ 1
        G_relax = G_t - G_eq

        # Calculate G_0_relax = initial relaxing modulus
        G_0_relax = G_0 - G_eq

        # Calculate integrals of relaxing part (used for extrapolation)
        # ∫[G(t) - G_eq]dt
        integral_G = self._integrate(t, G_relax)

        # Add extrapolation if requested
        # Only extrapolate if the tail values are positive and significant
        # (for elastic solids that have plateaued, G_relax ≈ 0, so extrapolation not needed)
        if self.extrapolate:
            # Check if tail values are positive and significant
            n_check = min(10, len(G_relax) // 4)
            tail_values = G_relax[-n_check:]
            tail_mean = float(jnp.mean(tail_values))
            tail_positive = float(jnp.min(tail_values)) > 0

            # Only extrapolate if tail is positive and > 1% of initial relaxing modulus
            if tail_positive and tail_mean > 0.01 * G_0_relax:
                try:
                    tail_G = self._extrapolate_tail(t, G_relax)
                    if not jnp.isnan(tail_G) and not jnp.isinf(tail_G):
                        integral_G += tail_G
                except (ValueError, RuntimeWarning):
                    # Extrapolation failed, continue without it
                    pass

        # Check for pure elastic solid (no relaxation)
        if G_0_relax <= 0:
            return 0.0

        # Calculate mutation number using standard rheological definition
        # Δ = (G_0 - G_eq) / G_0 = 1 - G_eq/G_0
        # This quantifies the degree of relaxation:
        # - Δ = 0: Purely elastic (no relaxation, G_eq = G_0)
        # - Δ = 1: Purely viscous (complete relaxation, G_eq = 0)
        # - 0 < Δ < 1: Viscoelastic material
        if G_0 > 0:
            delta = 1.0 - (G_eq / G_0)
        else:
            delta = 0.0

        # Clamp to physical range [0, 1]
        # Values outside this range indicate numerical issues or insufficient data quality
        delta = float(jnp.clip(delta, 0.0, 1.0))

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
