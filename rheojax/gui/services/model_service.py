"""
Model Service
============

Service for model fitting, prediction, and parameter management.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.registry import Registry
from rheojax.utils.compatibility import check_model_compatibility

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """Result from model fitting.

    Attributes
    ----------
    model_name : str
        Name of the fitted model
    parameters : dict
        Fitted parameter values
    residuals : np.ndarray
        Residuals from fit
    chi_squared : float
        Chi-squared goodness of fit
    success : bool
        Whether fit was successful
    message : str
        Status message
    x_fit : np.ndarray
        X values for fitted curve
    y_fit : np.ndarray
        Y values for fitted curve
    metadata : dict
        Additional metadata (convergence info, etc.)
    """

    model_name: str
    parameters: dict[str, float]
    residuals: np.ndarray
    chi_squared: float
    success: bool
    message: str
    x_fit: np.ndarray
    y_fit: np.ndarray
    metadata: dict[str, Any]


class ModelService:
    """Service for rheological model operations.

    Features:
        - Model fitting with NLSQ optimization
        - Model comparison and selection
        - Parameter management and validation
        - Compatibility checking
        - Smart initialization

    Example
    -------
    >>> service = ModelService()
    >>> models = service.get_available_models()
    >>> result = service.fit('maxwell', data, {}, test_mode='relaxation')
    """

    def __init__(self) -> None:
        """Initialize model service."""
        self._registry = Registry.get_instance()
        self._model_cache = {}

    def get_available_models(self) -> dict[str, list[str]]:
        """Get models grouped by category from ModelRegistry.

        Returns
        -------
        dict
            Models grouped by category:
            - 'classical': Maxwell, Zener, SpringPot
            - 'fractional_maxwell': FractionalMaxwellGel, etc.
            - 'fractional_zener': FractionalZenerSolidLiquid, etc.
            - 'fractional_advanced': FractionalBurgersModel, etc.
            - 'flow': PowerLaw, Carreau, HerschelBulkley, etc.
            - 'multi_mode': GeneralizedMaxwell
            - 'sgr': SGRConventional, SGRGeneric
        """
        all_models = self._registry.get_all_models()

        # Categorize models
        categories = {
            "classical": [],
            "fractional_maxwell": [],
            "fractional_zener": [],
            "fractional_advanced": [],
            "flow": [],
            "multi_mode": [],
            "sgr": [],
            "spp_laos": [],
            "other": [],
        }

        classical = ["maxwell", "zener", "springpot"]
        fractional_maxwell = [
            "fractional_maxwell_gel",
            "fractional_maxwell_liquid",
            "fractional_maxwell_model",
            "fractional_kelvin_voigt",
        ]
        fractional_zener = [
            "fractional_zener_sl",
            "fractional_zener_ss",
            "fractional_zener_ll",
            "fractional_kv_zener",
        ]
        fractional_advanced = [
            "fractional_burgers",
            "fractional_poynting_thomson",
            "fractional_jeffreys",
        ]
        flow = [
            "power_law",
            "carreau",
            "carreau_yasuda",
            "cross",
            "herschel_bulkley",
            "bingham",
        ]
        multi_mode = ["generalized_maxwell"]
        sgr = ["sgr_conventional", "sgr_generic"]
        spp_laos = ["spp_yield_stress"]

        for model_name in all_models:
            if model_name in classical:
                categories["classical"].append(model_name)
            elif model_name in fractional_maxwell:
                categories["fractional_maxwell"].append(model_name)
            elif model_name in fractional_zener:
                categories["fractional_zener"].append(model_name)
            elif model_name in fractional_advanced:
                categories["fractional_advanced"].append(model_name)
            elif model_name in flow:
                categories["flow"].append(model_name)
            elif model_name in multi_mode:
                categories["multi_mode"].append(model_name)
            elif model_name in sgr:
                categories["sgr"].append(model_name)
            elif model_name in spp_laos:
                categories["spp_laos"].append(model_name)
            else:
                categories["other"].append(model_name)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def get_parameter_defaults(self, model_name: str) -> dict[str, "ParameterState"]:
        """Return ParameterState mapping for a model using registry defaults."""

        from rheojax.gui.state.store import ParameterState

        model = self._registry.create_instance(model_name, plugin_type="model")
        defaults: dict[str, ParameterState] = {}
        for name, param in model.parameters.items():
            bounds = getattr(param, "bounds", (float("-inf"), float("inf")))
            defaults[name] = ParameterState(
                name=name,
                value=float(getattr(param, "value", 0.0)),
                min_bound=float(bounds[0]),
                max_bound=float(bounds[1]),
                fixed=False,
                unit=getattr(param, "unit", ""),
                description=getattr(param, "description", ""),
            )
        return defaults

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get model parameters, bounds, and description.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        dict
            Model information with keys:
            - name: str
            - description: str
            - parameters: dict[str, dict] with 'default', 'bounds', 'units'
            - supported_test_modes: list[str]
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Get parameter info
            params_info = {}
            for param_name, param in model.parameters.items():
                params_info[param_name] = {
                    "default": param.value,
                    "bounds": param.bounds,
                    "units": getattr(param, "units", None),
                    "description": getattr(param, "description", ""),
                }

            # Get docstring
            description = model.__class__.__doc__ or "No description available"

            # Determine supported test modes based on model type
            supported_modes = ["relaxation", "creep", "oscillation"]
            if any(
                flow in model_name.lower()
                for flow in ["power_law", "carreau", "herschel", "bingham", "cross"]
            ):
                supported_modes = ["flow"]
            elif "spp" in model_name.lower():
                supported_modes = ["oscillation", "rotation"]

            return {
                "name": model_name,
                "description": description.strip(),
                "parameters": params_info,
                "supported_test_modes": supported_modes,
            }

        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {
                "name": model_name,
                "description": "Error loading model info",
                "parameters": {},
                "supported_test_modes": [],
                "error": str(e),
            }

    def check_compatibility(
        self, model_name: str, data: RheoData, test_mode: str | None = None
    ) -> dict[str, Any]:
        """Check model-data compatibility.

        Parameters
        ----------
        model_name : str
            Name of the model
        data : RheoData
            Rheological data
        test_mode : str, optional
            Test mode (relaxation, creep, oscillation, flow)

        Returns
        -------
        dict
            Compatibility report with keys:
            - compatible: bool
            - decay_type: str
            - material_type: str
            - warnings: list[str]
            - recommendations: list[str]
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Extract x, y
            x = np.asarray(data.x)
            y = np.asarray(data.y)

            # Determine test mode
            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "unknown")

            # Check compatibility
            compat_kwargs = {}
            if test_mode == "relaxation":
                compat_kwargs = {"t": x, "G_t": y, "test_mode": "relaxation"}
            elif test_mode == "oscillation":
                compat_kwargs = {"omega": x, "G_star": y, "test_mode": "oscillation"}
            elif test_mode == "flow":
                compat_kwargs = {
                    "shear_rate": x,
                    "viscosity": y,
                    "test_mode": "flow",
                }

            result = check_model_compatibility(model=model, **compat_kwargs)

            return {
                "compatible": result.get("compatible", True),
                "decay_type": result.get("decay_type", "unknown"),
                "material_type": result.get("material_type", "unknown"),
                "warnings": result.get("warnings", []),
                "recommendations": result.get("recommendations", []),
                "details": result,
            }

        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return {
                "compatible": True,  # Default to compatible if check fails
                "decay_type": "unknown",
                "material_type": "unknown",
                "warnings": [f"Compatibility check failed: {e}"],
                "recommendations": [],
            }

    def get_smart_init(
        self, model_name: str, data: RheoData, test_mode: str | None = None
    ) -> dict[str, float]:
        """Get smart initialization values.

        Parameters
        ----------
        model_name : str
            Name of the model
        data : RheoData
            Rheological data
        test_mode : str, optional
            Test mode

        Returns
        -------
        dict
            Initial parameter values
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Check if model has smart initialization
            if hasattr(model, "smart_initial_guess"):
                x = np.asarray(data.x)
                y = np.asarray(data.y)

                # Determine test mode
                if test_mode is None:
                    test_mode = data.metadata.get("test_mode", "oscillation")

                # Get smart initialization
                init_params = model.smart_initial_guess(x, y, test_mode=test_mode)
                return init_params
            else:
                # Return default values
                return {name: param.value for name, param in model.parameters.items()}

        except Exception as e:
            logger.warning(f"Smart initialization failed, using defaults: {e}")
            # Return default values on failure
            try:
                model = self._registry.create_instance(model_name, plugin_type="model")
                return {name: param.value for name, param in model.parameters.items()}
            except Exception:
                return {}

    def fit(
        self,
        model_name: str,
        data: RheoData,
        params: dict[str, Any],
        test_mode: str | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
        **fit_kwargs: Any,
    ) -> FitResult:
        """Run NLSQ fitting with progress updates.

        Parameters
        ----------
        model_name : str
            Name of the model
        data : RheoData
            Rheological data
        params : dict
            Parameter configuration (initial values, bounds, etc.)
        test_mode : str, optional
            Test mode (relaxation, creep, oscillation, flow)
        progress_callback : Callable[[int, float], None], optional
            Callback function for progress updates: callback(iteration, loss)
        **fit_kwargs
            Additional fitting options (max_iter, ftol, xtol, etc.)

        Returns
        -------
        FitResult
            Fitting result with parameters, residuals, and goodness of fit
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Set initial parameter values if provided
            if params:
                for name, value in params.items():
                    if name in model.parameters:
                        if isinstance(value, dict):
                            # Handle dict with 'value', 'bounds', etc.
                            model.parameters[name].value = value.get(
                                "value", value.get("default", model.parameters[name].value)
                            )
                            if "bounds" in value:
                                model.parameters[name].bounds = value["bounds"]
                        else:
                            # Handle direct value
                            model.parameters[name].value = value

            # Extract data
            x = np.asarray(data.x)
            y = np.asarray(data.y)

            # Determine test mode
            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "oscillation")

            # Fit model
            logger.info(f"Fitting {model_name} model with test_mode={test_mode}")

            # Add test_mode to fit_kwargs
            fit_kwargs["test_mode"] = test_mode

            # Fit (this uses NLSQ by default)
            model.fit(x, y, **fit_kwargs)

            # Get fitted values
            y_pred = model.predict(x)

            # Calculate residuals
            if np.iscomplexobj(y):
                # For complex data, use magnitude of residuals
                residuals = np.abs(y - y_pred)
                y_real = np.abs(y)
            else:
                residuals = y - y_pred
                y_real = y

            # Calculate chi-squared
            chi_squared = float(np.sum(residuals**2))

            # Calculate R-squared (coefficient of determination)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_real - np.mean(y_real))**2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Calculate MPE (Mean Percentage Error)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_errors = np.abs(residuals / y_real) * 100
                pct_errors = pct_errors[np.isfinite(pct_errors)]
                mpe = float(np.mean(pct_errors)) if len(pct_errors) > 0 else 0.0

            # Get fitted parameters
            fitted_params = {name: param.value for name, param in model.parameters.items()}

            # Get metadata with R² and MPE included
            metadata = {
                "test_mode": test_mode,
                "n_iterations": getattr(model, "_n_iterations", None),
                "convergence": getattr(model, "_convergence", None),
                "r_squared": r_squared,
                "mpe": mpe,
            }

            # Add NLSQ result if available
            if hasattr(model, "_nlsq_result") and model._nlsq_result:
                metadata["nlsq_result"] = {
                    "success": model._nlsq_result.converged,
                    "nfev": model._nlsq_result.nfev,
                    "njev": model._nlsq_result.njev,
                }

            return FitResult(
                model_name=model_name,
                parameters=fitted_params,
                residuals=residuals,
                chi_squared=chi_squared,
                success=True,
                message="Fit successful",
                x_fit=x,
                y_fit=y_pred,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Fitting failed for {model_name}: {e}")
            return FitResult(
                model_name=model_name,
                parameters={},
                residuals=np.array([]),
                chi_squared=np.inf,
                success=False,
                message=f"Fit failed: {e}",
                x_fit=np.array([]),
                y_fit=np.array([]),
                metadata={"error": str(e)},
            )

    def predict(
        self, model_name: str, parameters: dict[str, float], x_values: np.ndarray
    ) -> np.ndarray:
        """Generate model predictions.

        Parameters
        ----------
        model_name : str
            Name of the model
        parameters : dict
            Fitted parameter values
        x_values : np.ndarray
            X values for prediction

        Returns
        -------
        np.ndarray
            Predicted y values
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Set parameters
            for name, value in parameters.items():
                if name in model.parameters:
                    model.parameters[name].value = value

            # Mark as fitted
            model.fitted_ = True

            # Predict
            return model.predict(x_values)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e

    def get_model_equation(self, model_name: str) -> str:
        """Get LaTeX equation for model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        str
            LaTeX equation string
        """
        try:
            model = self._registry.create_instance(model_name, plugin_type="model")
            if hasattr(model, "equation"):
                return model.equation
            return "Equation not available"
        except Exception:
            return "Error loading equation"

    def validate_parameters(
        self, model_name: str, parameters: dict[str, float]
    ) -> list[str]:
        """Validate parameter values against bounds.

        Parameters
        ----------
        model_name : str
            Name of the model
        parameters : dict
            Parameter values to validate

        Returns
        -------
        list[str]
            List of validation warnings
        """
        warnings = []

        try:
            model = self._registry.create_instance(model_name, plugin_type="model")

            for name, value in parameters.items():
                if name not in model.parameters:
                    warnings.append(f"Unknown parameter: {name}")
                    continue

                param = model.parameters[name]
                lower, upper = param.bounds

                if value < lower:
                    warnings.append(
                        f"{name}={value} is below lower bound {lower}"
                    )
                if value > upper:
                    warnings.append(
                        f"{name}={value} is above upper bound {upper}"
                    )

        except Exception as e:
            warnings.append(f"Validation failed: {e}")

        return warnings
