"""SPP Yield Stress model for LAOS-based yield stress extraction.

This module implements the SPP (Sequence of Physical Processes) yield stress
model that extracts physically meaningful yield parameters from Large Amplitude
Oscillatory Shear (LAOS) data. The model combines amplitude-sweep SPP analysis
with power-law scaling to characterize yield stress fluids.

The model parameterizes:
- G_cage: Apparent cage modulus (elastic response)
- σ_sy: Static yield stress (stress at strain reversal)
- σ_dy: Dynamic yield stress (stress at rate reversal)
- η_inf: Infinite-shear viscosity
- n: Flow power-law index

For Bayesian inference, the model uses physically-motivated priors:
- LogNormal for scale parameters (G_cage, stress scales, viscosity)
- Beta for exponents bounded in [0, 1] or [0, 2]
- HalfCauchy for noise scale

References
----------
- S.A. Rogers et al., J. Rheol. 56(1), 2012
- S.A. Rogers, Rheol. Acta 56, 2017
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpyro.distributions as dist

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from jax import Array

    from rheojax.core.data import RheoData


@ModelRegistry.register("spp_yield_stress")
class SPPYieldStress(BaseModel):
    """SPP-based yield stress model for LAOS analysis.

    This model extracts yield stress parameters from amplitude-sweep LAOS data
    using the SPP framework. It parameterizes the nonlinear response in terms
    of physically meaningful quantities: cage modulus, static/dynamic yield
    stresses, and post-yield flow parameters.

    The model supports both OSCILLATION mode (LAOS amplitude sweeps) and
    ROTATION mode (steady shear flow curves) for comprehensive yield stress
    characterization.

    Parameters
    ----------
    G_cage : float
        Apparent cage modulus (Pa), elastic stiffness of the cage structure
    sigma_sy_scale : float
        Static yield stress scale factor (Pa)
    sigma_sy_exp : float
        Static yield stress exponent for amplitude scaling
    sigma_dy_scale : float
        Dynamic yield stress scale factor (Pa)
    sigma_dy_exp : float
        Dynamic yield stress exponent for amplitude scaling
    eta_inf : float
        Infinite-shear viscosity (Pa·s)
    n_power_law : float
        Power-law flow index (dimensionless)
    noise : float
        Observation noise scale (Pa), for Bayesian inference

    Constitutive Equations
    ----------------------
    For OSCILLATION (LAOS at amplitude γ_0):
        σ_sy(γ_0) = sigma_sy_scale * γ_0^sigma_sy_exp
        σ_dy(γ_0) = sigma_dy_scale * γ_0^sigma_dy_exp
        σ_max(γ_0) = G_cage * γ_0 + η_inf * ω * γ_0

    For ROTATION (steady shear at rate γ̇):
        σ(γ̇) = σ_dy + η_inf * γ̇^n_power_law  (Herschel-Bulkley-like)

    Test Modes
    ----------
    - OSCILLATION: Amplitude sweep analysis (γ_0 as independent variable)
    - ROTATION: Steady shear flow curve (γ̇ as independent variable)

    Examples
    --------
    Fit to amplitude sweep data:

    >>> from rheojax.models import SPPYieldStress
    >>> from rheojax.transforms import SPPDecomposer
    >>>
    >>> # Amplitude sweep data (multiple γ_0 values)
    >>> gamma_0_values = jnp.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    >>> sigma_sy_data = jnp.array([1.0, 10.0, 40.0, 80.0, 140.0, 280.0])
    >>>
    >>> model = SPPYieldStress()
    >>> model.fit(gamma_0_values, sigma_sy_data, test_mode='oscillation')
    >>> print(model)

    Bayesian inference:

    >>> result = model.fit_bayesian(gamma_0_values, sigma_sy_data,
    ...                             test_mode='oscillation')
    >>> print(result.summary)
    """

    def __init__(self):
        """Initialize SPP yield stress model."""
        super().__init__()
        self.parameters = ParameterSet()

        # Cage modulus (elastic stiffness)
        self.parameters.add(
            name="G_cage",
            value=1000.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Apparent cage modulus",
        )

        # Static yield stress scaling
        self.parameters.add(
            name="sigma_sy_scale",
            value=100.0,
            bounds=(1e-6, 1e9),
            units="Pa",
            description="Static yield stress scale",
        )
        self.parameters.add(
            name="sigma_sy_exp",
            value=1.0,
            bounds=(0.0, 2.0),
            units="dimensionless",
            description="Static yield stress exponent",
        )

        # Dynamic yield stress scaling
        self.parameters.add(
            name="sigma_dy_scale",
            value=50.0,
            bounds=(1e-6, 1e9),
            units="Pa",
            description="Dynamic yield stress scale",
        )
        self.parameters.add(
            name="sigma_dy_exp",
            value=0.5,
            bounds=(0.0, 2.0),
            units="dimensionless",
            description="Dynamic yield stress exponent",
        )

        # Flow parameters
        self.parameters.add(
            name="eta_inf",
            value=1.0,
            bounds=(1e-9, 1e6),
            units="Pa·s",
            description="Infinite-shear viscosity",
        )
        self.parameters.add(
            name="n_power_law",
            value=0.5,
            bounds=(0.01, 2.0),
            units="dimensionless",
            description="Flow power-law index",
        )

        # Noise scale for Bayesian inference
        self.parameters.add(
            name="noise",
            value=1.0,
            bounds=(1e-10, 1e6),
            units="Pa",
            description="Observation noise scale",
        )

        # Internal state
        self._test_mode = TestMode.OSCILLATION
        self._omega = 1.0  # Default angular frequency
        self._amplitude_data: dict = {}  # Store per-amplitude SPP results
        self._yield_type = "static"

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> SPPYieldStress:
        """Fit SPP yield stress model to data.

        The fitting strategy depends on the test mode:
        - OSCILLATION: Sequential per-amplitude SPP extraction, then power-law fit
        - ROTATION: Direct Herschel-Bulkley-like fit

        Parameters
        ----------
        X : np.ndarray
            Independent variable:
            - OSCILLATION: strain amplitudes γ_0
            - ROTATION: shear rates γ̇
        y : np.ndarray
            Dependent variable:
            - OSCILLATION: measured yield stress (σ_sy or σ_dy)
            - ROTATION: shear stress σ
        **kwargs : dict
            Additional options:
            - test_mode: 'oscillation' or 'rotation'
            - omega: angular frequency (rad/s) for oscillation mode
            - yield_type: 'static' or 'dynamic' (which yield stress to fit)

        Returns
        -------
        SPPYieldStress
            Fitted model (self)
        """
        from rheojax.core.data import RheoData

        # Handle RheoData input
        if isinstance(X, RheoData):
            rheo_data = X
            X_array = np.asarray(rheo_data.x, dtype=np.float64)
            y_array = np.asarray(rheo_data.y, dtype=np.float64)
            test_mode = detect_test_mode(rheo_data)
            self._omega = rheo_data.metadata.get("omega", 1.0)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            y_array = np.asarray(y, dtype=np.float64)
            test_mode = kwargs.get("test_mode", TestMode.OSCILLATION)
            if isinstance(test_mode, str):
                test_mode = TestMode(test_mode.lower())
            self._omega = kwargs.get("omega", 1.0)

        self._test_mode = test_mode
        yield_type = kwargs.get("yield_type", "static")
        self._yield_type = yield_type

        if test_mode == TestMode.OSCILLATION:
            self._fit_oscillation(X_array, y_array, yield_type)
        elif test_mode == TestMode.ROTATION:
            self._fit_rotation(X_array, y_array)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

        self.fitted_ = True
        return self

    def _fit_oscillation(
        self, gamma_0_array: np.ndarray, sigma_array: np.ndarray, yield_type: str
    ):
        """Fit to amplitude sweep data (oscillation mode).

        Uses power-law scaling: σ(γ_0) = scale * γ_0^exp

        Parameters
        ----------
        gamma_0_array : np.ndarray
            Strain amplitudes
        sigma_array : np.ndarray
            Yield stress values
        yield_type : str
            'static' or 'dynamic'
        """
        # Log-log linear regression for power-law fit
        # log(σ) = log(scale) + exp * log(γ_0)
        valid_mask = (gamma_0_array > 0) & (sigma_array > 0)
        log_gamma = np.log(gamma_0_array[valid_mask])
        log_sigma = np.log(sigma_array[valid_mask])

        if len(log_gamma) < 2:
            raise ValueError("Need at least 2 valid data points for fitting")

        # Linear regression
        coeffs = np.polyfit(log_gamma, log_sigma, 1)
        exponent = float(coeffs[0])
        scale = float(np.exp(coeffs[1]))

        # Update parameters based on yield type
        if yield_type == "static":
            self.parameters.set_value("sigma_sy_scale", scale)
            self.parameters.set_value("sigma_sy_exp", np.clip(exponent, 0.0, 2.0))
        else:  # dynamic
            self.parameters.set_value("sigma_dy_scale", scale)
            self.parameters.set_value("sigma_dy_exp", np.clip(exponent, 0.0, 2.0))

        # Estimate cage modulus from low-amplitude linear regime
        # G_cage ≈ σ_sy / γ_0 at small γ_0
        low_amp_mask = gamma_0_array < 0.1 * np.max(gamma_0_array)
        if np.any(low_amp_mask):
            G_cage_est = np.mean(
                sigma_array[low_amp_mask] / gamma_0_array[low_amp_mask]
            )
            self.parameters.set_value("G_cage", np.clip(G_cage_est, 1e-3, 1e9))

    def _fit_rotation(self, gamma_dot_array: np.ndarray, sigma_array: np.ndarray):
        """Fit to steady shear flow curve (rotation mode).

        Uses Herschel-Bulkley-like model: σ = σ_dy + η_inf * γ̇^n

        Parameters
        ----------
        gamma_dot_array : np.ndarray
            Shear rates
        sigma_array : np.ndarray
            Shear stress values
        """
        # Sort by shear rate
        sort_idx = np.argsort(gamma_dot_array)
        gamma_dot_sorted = gamma_dot_array[sort_idx]
        sigma_sorted = sigma_array[sort_idx]

        # Estimate yield stress from low-rate extrapolation
        low_rate_mask = gamma_dot_sorted < 0.1 * np.max(gamma_dot_sorted)
        if np.any(low_rate_mask):
            sigma_dy_est = np.min(sigma_sorted[low_rate_mask])
        else:
            sigma_dy_est = np.min(sigma_sorted)

        sigma_dy_est = max(sigma_dy_est, 0.0)

        # Fit power-law to post-yield region
        sigma_corrected = sigma_sorted - sigma_dy_est
        sigma_corrected = np.maximum(sigma_corrected, 1e-10)

        # High-rate region for power-law fit
        high_rate_mask = gamma_dot_sorted > 0.1 * np.max(gamma_dot_sorted)
        if np.sum(high_rate_mask) >= 2:
            log_rate = np.log(gamma_dot_sorted[high_rate_mask])
            log_sigma_corr = np.log(sigma_corrected[high_rate_mask])
            coeffs = np.polyfit(log_rate, log_sigma_corr, 1)
            n_est = float(coeffs[0])
            eta_est = float(np.exp(coeffs[1]))
        else:
            # Fallback to simple estimation
            n_est = 1.0
            eta_est = np.mean(sigma_corrected / np.maximum(gamma_dot_sorted, 1e-10))

        # Update parameters
        self.parameters.set_value("sigma_dy_scale", np.clip(sigma_dy_est, 1e-6, 1e9))
        self.parameters.set_value("sigma_dy_exp", 0.0)  # Not amplitude-dependent
        self.parameters.set_value("n_power_law", np.clip(n_est, 0.01, 2.0))
        self.parameters.set_value("eta_inf", np.clip(eta_est, 1e-9, 1e6))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress using fitted parameters.

        Parameters
        ----------
        X : np.ndarray
            Independent variable (γ_0 for oscillation, γ̇ for rotation)

        Returns
        -------
        np.ndarray
            Predicted stress
        """
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Get parameters
        params = jnp.array(
            [
                self.parameters.get_value("G_cage"),
                self.parameters.get_value("sigma_sy_scale"),
                self.parameters.get_value("sigma_sy_exp"),
                self.parameters.get_value("sigma_dy_scale"),
                self.parameters.get_value("sigma_dy_exp"),
                self.parameters.get_value("eta_inf"),
                self.parameters.get_value("n_power_law"),
                self.parameters.get_value("noise"),
            ]
        )

        predictions = self.model_function(X_jax, params, self._test_mode)
        return np.array(predictions)

    def model_function(
        self,
        X: Array,
        params: Array,
        test_mode: TestMode | None = None,
    ) -> Array:
        """Model function for predictions and Bayesian inference.

        This is the NumPyro-facing model used by BayesianMixin.
        It maps inputs and parameter vectors to stress predictions.

        Parameters
        ----------
        X : Array
            Independent variable
            - OSCILLATION: strain amplitudes (gamma_0)
            - ROTATION: shear rates (gamma_dot)
        params : Array
            [G_cage, sigma_sy_scale, sigma_sy_exp, sigma_dy_scale,
             sigma_dy_exp, eta_inf, n_power_law, noise]
        test_mode : TestMode, optional
            Mode selector; defaults to the model's stored test_mode.

        Returns
        -------
        Array
            Predicted stress values.
        """
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", TestMode.OSCILLATION)

        # Extract parameters
        G_cage = params[0]
        sigma_sy_scale = params[1]
        sigma_sy_exp = params[2]
        sigma_dy_scale = params[3]
        sigma_dy_exp = params[4]
        eta_inf = params[5]
        n_power_law = params[6]
        # noise = params[7]  # Used only in likelihood

        if test_mode == TestMode.OSCILLATION:
            yield_type = getattr(self, "_yield_type", "static")
            if yield_type == "dynamic":
                return self._predict_dynamic_yield(
                    X, sigma_dy_scale, sigma_dy_exp
                )
            return self._predict_oscillation(
                X, G_cage, sigma_sy_scale, sigma_sy_exp, eta_inf
            )
        elif test_mode == TestMode.ROTATION:
            return self._predict_rotation(X, sigma_dy_scale, eta_inf, n_power_law)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    @staticmethod
    @jax.jit
    def _predict_oscillation(
        gamma_0: Array,
        G_cage: float,
        sigma_sy_scale: float,
        sigma_sy_exp: float,
        eta_inf: float,
    ) -> Array:
        """Predict static yield stress for amplitude sweep.

        σ_sy(γ_0) = sigma_sy_scale * γ_0^sigma_sy_exp

        Parameters
        ----------
        gamma_0 : Array
            Strain amplitudes
        G_cage : float
            Cage modulus (not directly used, but available)
        sigma_sy_scale : float
            Yield stress scale
        sigma_sy_exp : float
            Yield stress exponent
        eta_inf : float
            Infinite-shear viscosity (not directly used)

        Returns
        -------
        Array
            Predicted static yield stress
        """
        # Power-law scaling
        sigma_sy = sigma_sy_scale * jnp.power(jnp.abs(gamma_0), sigma_sy_exp)
        return sigma_sy

    @staticmethod
    @jax.jit
    def _predict_dynamic_yield(
        gamma_0: Array,
        sigma_dy_scale: float,
        sigma_dy_exp: float,
    ) -> Array:
        """Predict dynamic yield stress for amplitude sweep.

        σ_dy(γ_0) = sigma_dy_scale * γ_0^sigma_dy_exp
        """
        return sigma_dy_scale * jnp.power(jnp.abs(gamma_0), sigma_dy_exp)

    @staticmethod
    @jax.jit
    def _predict_rotation(
        gamma_dot: Array,
        sigma_dy: float,
        eta_inf: float,
        n_power_law: float,
    ) -> Array:
        """Predict stress for steady shear flow.

        σ(γ̇) = σ_dy + η_inf * |γ̇|^n

        Parameters
        ----------
        gamma_dot : Array
            Shear rates
        sigma_dy : float
            Dynamic yield stress
        eta_inf : float
            Infinite-shear viscosity
        n_power_law : float
            Power-law index

        Returns
        -------
        Array
            Predicted shear stress
        """
        # Herschel-Bulkley-like formulation
        abs_rate = jnp.abs(gamma_dot)
        sigma = sigma_dy + eta_inf * jnp.power(abs_rate, n_power_law)
        return sigma

    def bayesian_prior_factory(
        self, name: str, lower: float | None, upper: float | None
    ) -> dist.Distribution | None:
        """Create NumPyro-friendly priors for model parameters.

        Uses physically-motivated prior distributions:
        - LogNormal for scale parameters (positive, often log-distributed)
        - Beta for bounded exponents
        - HalfCauchy for noise scale

        Parameters
        ----------
        name : str
            Parameter name
        lower : float
            Lower bound
        upper : float
            Upper bound

        Returns
        -------
        dist.Distribution or None
            NumPyro distribution, or None to use default uniform
        """
        # Scale parameters: LogNormal priors
        if name == "G_cage":
            # Cage modulus typically 100-10000 Pa for soft materials
            return dist.LogNormal(loc=jnp.log(1000.0), scale=2.0)

        elif name == "sigma_sy_scale":
            # Static yield stress scale
            return dist.LogNormal(loc=jnp.log(100.0), scale=2.0)

        elif name == "sigma_dy_scale":
            # Dynamic yield stress scale (typically < σ_sy)
            return dist.LogNormal(loc=jnp.log(50.0), scale=2.0)

        elif name == "eta_inf":
            # Viscosity: wide log-normal prior
            return dist.LogNormal(loc=jnp.log(1.0), scale=3.0)

        # Exponent parameters: Beta priors on [0, 2]
        elif name == "sigma_sy_exp":
            # Exponent bounded [0,1]; time-domain yield often linear/sublinear
            return dist.TransformedDistribution(
                dist.Beta(2.0, 2.0),
                dist.transforms.AffineTransform(loc=0.0, scale=1.0),
            )

        elif name == "sigma_dy_exp":
            # Often sublinear scaling (exp < 1) bounded [0,1]
            return dist.TransformedDistribution(
                dist.Beta(2.0, 3.0),  # Skewed toward lower values
                dist.transforms.AffineTransform(loc=0.0, scale=1.0),
            )

        elif name == "n_power_law":
            # Power-law index: often 0.3-0.8 for yield stress fluids
            return dist.TransformedDistribution(
                dist.Beta(2.0, 2.0),
                dist.transforms.AffineTransform(loc=0.01, scale=0.99),
            )

        elif name == "noise":
            # Noise: HalfCauchy for heavy-tailed robustness
            return dist.HalfCauchy(scale=10.0)

        # Default: use uniform prior
        return None

    def predict_amplitude_sweep(
        self,
        gamma_0_array: np.ndarray,
        omega: float = 1.0,
        yield_type: str = "static",
    ) -> dict:
        """Predict full amplitude sweep response.

        Computes both static and dynamic yield stresses across amplitudes.

        Parameters
        ----------
        gamma_0_array : np.ndarray
            Strain amplitudes to evaluate
        omega : float
            Angular frequency (rad/s)
        yield_type : str
            'static', 'dynamic', or 'both'

        Returns
        -------
        dict
            Dictionary with:
            - gamma_0: strain amplitudes
            - sigma_sy: static yield stresses (if requested)
            - sigma_dy: dynamic yield stresses (if requested)
        """
        gamma_0_jax = jnp.asarray(gamma_0_array, dtype=jnp.float64)

        self.parameters.get_value("G_cage")
        sigma_sy_scale = self.parameters.get_value("sigma_sy_scale")
        sigma_sy_exp = self.parameters.get_value("sigma_sy_exp")
        sigma_dy_scale = self.parameters.get_value("sigma_dy_scale")
        sigma_dy_exp = self.parameters.get_value("sigma_dy_exp")
        self.parameters.get_value("eta_inf")

        result = {"gamma_0": gamma_0_array}

        if yield_type in ("static", "both"):
            sigma_sy = sigma_sy_scale * jnp.power(gamma_0_jax, sigma_sy_exp)
            result["sigma_sy"] = np.array(sigma_sy)

        if yield_type in ("dynamic", "both"):
            sigma_dy = sigma_dy_scale * jnp.power(gamma_0_jax, sigma_dy_exp)
            result["sigma_dy"] = np.array(sigma_dy)

        return result

    def predict_flow_curve(
        self,
        gamma_dot_array: np.ndarray,
    ) -> dict:
        """Predict steady shear flow curve.

        Parameters
        ----------
        gamma_dot_array : np.ndarray
            Shear rates to evaluate

        Returns
        -------
        dict
            Dictionary with:
            - gamma_dot: shear rates
            - sigma: shear stress
            - eta_app: apparent viscosity
        """
        gamma_dot_jax = jnp.asarray(gamma_dot_array, dtype=jnp.float64)

        sigma_dy = self.parameters.get_value("sigma_dy_scale")
        eta_inf = self.parameters.get_value("eta_inf")
        n = self.parameters.get_value("n_power_law")

        # Stress
        abs_rate = jnp.abs(gamma_dot_jax)
        sigma = sigma_dy + eta_inf * jnp.power(abs_rate, n)

        # Apparent viscosity
        eta_app = sigma / jnp.maximum(abs_rate, 1e-10)

        return {
            "gamma_dot": gamma_dot_array,
            "sigma": np.array(sigma),
            "eta_app": np.array(eta_app),
        }

    def predict_rheo(
        self,
        rheo_data: RheoData,
        test_mode: TestMode | None = None,
    ) -> RheoData:
        """Predict rheological response for RheoData.

        Parameters
        ----------
        rheo_data : RheoData
            Input rheological data
        test_mode : TestMode, optional
            Test mode override

        Returns
        -------
        RheoData
            Predicted response
        """
        from rheojax.core.data import RheoData

        if test_mode is None:
            test_mode = detect_test_mode(rheo_data)

        X = rheo_data.x
        predictions = self._predict(X)

        return RheoData(
            x=np.array(X),
            y=predictions,
            x_units=rheo_data.x_units,
            y_units="Pa",
            domain=rheo_data.domain,
            metadata={
                "model": "SPPYieldStress",
                "test_mode": (
                    test_mode.value if isinstance(test_mode, TestMode) else test_mode
                ),
                **self.get_params(),
            },
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        G_cage = self.parameters.get_value("G_cage")
        sigma_sy = self.parameters.get_value("sigma_sy_scale")
        sigma_dy = self.parameters.get_value("sigma_dy_scale")
        n = self.parameters.get_value("n_power_law")
        return (
            f"SPPYieldStress(G_cage={G_cage:.2e}, "
            f"σ_sy={sigma_sy:.2e}, σ_dy={sigma_dy:.2e}, n={n:.2f})"
        )


__all__ = ["SPPYieldStress"]
