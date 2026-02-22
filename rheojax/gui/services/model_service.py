"""
Model Service
============

Service for model fitting, prediction, and parameter management.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.registry import Registry
from rheojax.logging import get_logger
from rheojax.utils.compatibility import check_model_compatibility

if TYPE_CHECKING:
    from rheojax.gui.state.store import ParameterState

logger = get_logger(__name__)


def infer_model_kwargs(model_name: str, param_names: list[str]) -> dict[str, Any]:
    """Infer model constructor kwargs from parameter names.

    For GeneralizedMaxwell: count G_i/E_i parameters to determine n_modes
    after element minimization may have reduced the mode count.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "generalized_maxwell")
    param_names : list[str]
        Parameter names from NLSQ warm-start or posterior samples

    Returns
    -------
    dict
        Model initialization kwargs (e.g., {"n_modes": 2})
    """
    import re

    model_kwargs: dict[str, Any] = {}

    if "maxwell" in model_name.lower() and "generalized" in model_name.lower():
        # Match G_1, G_2, ... or E_1, E_2, ... (tensile DMTA)
        g_pattern = re.compile(r"^[GE]_(\d+)$")
        g_indices = [
            int(m.group(1))
            for name in param_names
            if (m := g_pattern.match(name))
        ]
        if g_indices:
            n_modes = max(g_indices)
            model_kwargs["n_modes"] = n_modes
            logger.debug(
                "Inferred n_modes from parameter names",
                model=model_name,
                n_modes=n_modes,
            )

    if "stz" in model_name.lower():
        # Infer variant from parameter names:
        # minimal: no tau_beta, no m_inf
        # standard: has tau_beta, no m_inf
        # full: has tau_beta and m_inf
        has_tau_beta = "tau_beta" in param_names
        has_m_inf = "m_inf" in param_names
        if has_m_inf:
            model_kwargs["variant"] = "full"
        elif has_tau_beta:
            model_kwargs["variant"] = "standard"
        else:
            model_kwargs["variant"] = "minimal"
        logger.debug(
            "Inferred STZ variant from parameter names",
            model=model_name,
            variant=model_kwargs["variant"],
        )

    return model_kwargs


def normalize_model_name(model_name: str) -> str:
    """Normalize user-entered model identifiers to registry slugs.

    Accepts case-insensitive aliases (e.g., "GMM" -> "generalized_maxwell") and
    trims whitespace so editable combos can safely map typed text.
    """
    logger.debug("Normalizing model name", input_name=model_name)

    key = model_name.strip()
    alias_map = {
        "gmm": "generalized_maxwell",
        "generalized maxwell": "generalized_maxwell",
        # Fluidity
        "fluidity local": "fluidity_local",
        "fluidity nonlocal": "fluidity_nonlocal",
        # EPM
        "lattice epm": "lattice_epm",
        "tensorial epm": "tensorial_epm",
        # IKH
        "ikh": "mikh",
        "mlikh": "ml_ikh",
        # HL
        "hl": "hebraud_lequeux",
        "hebraud lequeux": "hebraud_lequeux",
        # ITT-MCT
        "itt-mct": "itt_mct_schematic",
        "itt mct": "itt_mct_schematic",
        "mct": "itt_mct_schematic",
        "itt_mct": "itt_mct_schematic",
        "itt-mct schematic": "itt_mct_schematic",
        "itt-mct isotropic": "itt_mct_isotropic",
        "ism": "itt_mct_isotropic",
        # DMT
        "dmt": "dmt_local",
        "dmt local": "dmt_local",
        "dmt nonlocal": "dmt_nonlocal",
        "de souza mendes": "dmt_local",
        # Fluidity-Saramito
        "saramito": "fluidity_saramito_local",
        "fluidity saramito": "fluidity_saramito_local",
        "fluidity-saramito": "fluidity_saramito_local",
        "fluidity saramito local": "fluidity_saramito_local",
        "fluidity saramito nonlocal": "fluidity_saramito_nonlocal",
        "evp": "fluidity_saramito_local",
        # VLB
        "vlb": "vlb_local",
        "vlb local": "vlb_local",
        "vlb multi": "vlb_multi_network",
        "vlb multi network": "vlb_multi_network",
        "vlb multi-network": "vlb_multi_network",
        "vlb variant": "vlb_variant",
        "vlb nonlocal": "vlb_nonlocal",
        "vlb bell": "vlb_variant",
        "vlb fene": "vlb_variant",
        # HVM
        "hvm": "hvm_local",
        "hvm local": "hvm_local",
        "vitrimer": "hvm_local",
        "hybrid vitrimer": "hvm_local",
        # Giesekus
        "giesekus_single_mode": "giesekus_single",
        "giesekus_multi_mode": "giesekus_multi",
        "giesekus single mode": "giesekus_single",
        "giesekus multi mode": "giesekus_multi",
        # HVNM
        "hvnm": "hvnm_local",
        "hvnm local": "hvnm_local",
        "vitrimer nanocomposite": "hvnm_local",
        "hybrid vitrimer nanocomposite": "hvnm_local",
        # SPP
        "spp": "spp_yield_stress",
        "spp yield stress": "spp_yield_stress",
        # STZ
        "stz": "stz_conventional",
        "stz conventional": "stz_conventional",
        "shear transformation zone": "stz_conventional",
        # TNT
        "tnt": "tnt_single_mode",
        "tnt single": "tnt_single_mode",
        "tnt single mode": "tnt_single_mode",
        "tnt cates": "tnt_cates",
        "tnt loop bridge": "tnt_loop_bridge",
        "tnt loop-bridge": "tnt_loop_bridge",
        "tnt multi species": "tnt_multi_species",
        "tnt multi-species": "tnt_multi_species",
        "tnt sticky rouse": "tnt_sticky_rouse",
        "tnt sticky-rouse": "tnt_sticky_rouse",
        "sticky rouse": "tnt_sticky_rouse",
    }

    if key in alias_map.values():
        logger.debug("Model name already normalized", model_name=key)
        return key

    result = alias_map.get(key.lower(), key)
    logger.debug("Model name normalized", input_name=model_name, output_name=result)
    return result


def _is_placeholder_model(model_name: str | None) -> bool:
    """Return True when the incoming name is a UI placeholder or empty."""

    if model_name is None:
        return True

    key = model_name.strip().lower()
    return key == "" or key.startswith("select model")


# Import canonical FitResult from store (single source of truth)
from rheojax.gui.state.store import FitResult


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
        logger.debug("Initializing ModelService")
        self._registry = Registry.get_instance()
        logger.debug("ModelService initialized", registry_available=True)

    def _normalize_model_name(self, model_name: str) -> str:
        """Map friendly aliases to registered model slugs."""
        return normalize_model_name(model_name)

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
            - 'spp_laos': SPPYieldStress
            - 'stz': STZConventional
            - 'fluidity': FluidityLocal, FluidityNonlocal
            - 'fluidity_saramito': FluiditySaramitoLocal, FluiditySaramitoNonlocal
            - 'epm': LatticeEPM, TensorialEPM
            - 'ikh': MIKH, MLIKH
            - 'hl': HebraudLequeux
            - 'itt_mct': ITTMCTSchematic, ITTMCTIsotropic
            - 'dmt': DMTLocal, DMTNonlocal
            - 'giesekus': GiesekusSingleMode, GiesekusMultiMode
            - 'tnt': TNTSingleMode, TNTLoopBridge, TNTStickyRouse, TNTCates, TNTMultiSpecies
            - 'vlb': VLBLocal, VLBMultiNetwork, VLBVariant, VLBNonlocal
            - 'hvm': HVMLocal
            - 'hvnm': HVNMLocal
            - 'fikh': FIKH, FMLIKH
        """
        logger.debug("Getting available models from registry")
        all_models = self._registry.get_all_models()
        logger.debug("Retrieved models from registry", n_models=len(all_models))

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
            "stz": [],
            "fluidity": [],
            "fluidity_saramito": [],
            "epm": [],
            "ikh": [],
            "hl": [],
            "itt_mct": [],
            "dmt": [],
            "giesekus": [],
            "tnt": [],
            "vlb": [],
            "hvm": [],
            "hvnm": [],
            "fikh": [],
            "other": [],
        }

        # known prefixes or exact matches for categorization
        # In a future update, we should add 'category' to registry metadata
        for model_name in all_models:
            name_lower = model_name.lower()
            if name_lower in ["maxwell", "zener", "springpot"]:
                categories["classical"].append(model_name)
            elif (
                "fractional_maxwell" in name_lower
                or name_lower == "fractional_kelvin_voigt"
            ):
                categories["fractional_maxwell"].append(model_name)
            elif (
                "fractional_zener" in name_lower or "fractional_kv_zener" in name_lower
            ):
                categories["fractional_zener"].append(model_name)
            elif "fractional" in name_lower and (
                "burgers" in name_lower
                or "jeffreys" in name_lower
                or "poynting" in name_lower
            ):
                categories["fractional_advanced"].append(model_name)
            elif name_lower in [
                "power_law",
                "carreau",
                "carreau_yasuda",
                "cross",
                "herschel_bulkley",
                "bingham",
            ]:
                categories["flow"].append(model_name)
            elif name_lower == "generalized_maxwell":
                categories["multi_mode"].append(model_name)
            elif name_lower.startswith("sgr"):
                categories["sgr"].append(model_name)
            elif name_lower.startswith("spp"):
                categories["spp_laos"].append(model_name)
            elif name_lower.startswith("stz"):
                categories["stz"].append(model_name)
            elif name_lower.startswith("itt_mct"):
                categories["itt_mct"].append(model_name)
            elif name_lower.startswith("dmt"):
                categories["dmt"].append(model_name)
            # Check fluidity_saramito before fluidity (more specific first)
            elif name_lower.startswith("fluidity_saramito"):
                categories["fluidity_saramito"].append(model_name)
            elif name_lower.startswith("fluidity"):
                categories["fluidity"].append(model_name)
            elif name_lower.endswith("_epm") or name_lower.startswith("epm"):
                categories["epm"].append(model_name)
            elif name_lower in ["mikh", "ml_ikh"]:
                categories["ikh"].append(model_name)
            elif name_lower == "hebraud_lequeux":
                categories["hl"].append(model_name)
            elif name_lower.startswith("giesekus"):
                categories["giesekus"].append(model_name)
            elif name_lower.startswith("tnt"):
                categories["tnt"].append(model_name)
            elif name_lower.startswith("vlb"):
                categories["vlb"].append(model_name)
            # Check hvnm before hvm (more specific prefix first)
            elif name_lower.startswith("hvnm"):
                categories["hvnm"].append(model_name)
            elif name_lower.startswith("hvm"):
                categories["hvm"].append(model_name)
            elif name_lower in ["fikh", "fmlikh"]:
                categories["fikh"].append(model_name)
            else:
                categories["other"].append(model_name)

        # Remove empty categories
        result = {k: v for k, v in categories.items() if v}
        logger.debug("Models categorized", n_categories=len(result))
        return result

    def get_parameter_defaults(self, model_name: str) -> dict[str, ParameterState]:
        """Return ParameterState mapping for a model using registry defaults."""
        logger.debug("Getting parameter defaults", model=model_name)

        if _is_placeholder_model(model_name):
            logger.debug("Placeholder model, returning empty defaults")
            return {}

        model_name = self._normalize_model_name(model_name)

        from rheojax.gui.state.store import ParameterState

        try:
            model = self._registry.create_instance(model_name, plugin_type="model")
        except Exception as exc:
            logger.warning(
                "Could not load parameter defaults",
                model=model_name,
                error=str(exc),
            )
            return {}
        defaults: dict[str, ParameterState] = {}
        for name, param in model.parameters.items():
            bounds = getattr(param, "bounds", (float("-inf"), float("inf")))
            defaults[name] = ParameterState(
                name=name,
                value=float(getattr(param, "value", 0.0)),
                min_bound=float(bounds[0]),
                max_bound=float(bounds[1]),
                fixed=False,
                unit=getattr(param, "units", ""),
                description=getattr(param, "description", ""),
            )
        logger.debug(
            "Parameter defaults retrieved",
            model=model_name,
            n_parameters=len(defaults),
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
        logger.debug("Getting model info", model=model_name)

        try:
            if _is_placeholder_model(model_name):
                logger.debug("Placeholder model, returning empty info")
                return {
                    "name": model_name or "",
                    "description": "No model selected",
                    "parameters": {},
                    "supported_test_modes": [],
                }

            model_name = self._normalize_model_name(model_name)

            # Get registry info for protocols
            from rheojax.core.registry import ModelRegistry

            reg_info = ModelRegistry.get_info(model_name)
            supported_modes = []

            if reg_info and reg_info.protocols:
                # Use Protocol.value directly to match fit_page combo labels
                supported_modes = [p.value for p in reg_info.protocols]
            else:
                # Fallback to heuristics (should not happen for migrated models)
                supported_modes = ["relaxation", "creep", "oscillation"]
                if any(
                    flow in model_name.lower()
                    for flow in ["power_law", "carreau", "herschel", "bingham", "cross"]
                ):
                    supported_modes = ["flow"]
                elif "spp" in model_name.lower():
                    supported_modes = ["oscillation", "rotation"]
                elif "sgr" in model_name.lower():
                    supported_modes = ["oscillation", "relaxation"]

            # Create model instance for parameters
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

            logger.debug(
                "Model info retrieved",
                model=model_name,
                n_parameters=len(params_info),
                supported_modes=supported_modes,
            )

            return {
                "name": model_name,
                "description": description.strip(),
                "parameters": params_info,
                "supported_test_modes": supported_modes,
            }

        except Exception as e:
            logger.error(
                "Failed to get model info",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
            return {
                "name": model_name,
                "description": "Error loading model info",
                "parameters": {},
                "supported_test_modes": [],
                "error": str(e),
            }

    def get_supported_deformation_modes(self, model_name: str) -> list[str]:
        """Return list of supported deformation mode strings for a model.

        Returns empty list for shear-only models (no DMTA support).
        """
        if _is_placeholder_model(model_name):
            return []
        try:
            from rheojax.core.registry import ModelRegistry

            model_name = self._normalize_model_name(model_name)
            reg_info = ModelRegistry.get_info(model_name)
            if reg_info and reg_info.deformation_modes:
                return [dm.value for dm in reg_info.deformation_modes]
        except Exception:
            logger.debug(
                "Could not get deformation modes",
                model=model_name,
                exc_info=True,
            )
        return []

    def supports_fitting(self, model_name: str) -> bool:
        """Check if a model supports NLSQ fitting (not just prediction).

        Some models (e.g., TensorialEPM) can only predict, not fit.
        Returns False if the model's _fit() raises NotImplementedError.
        """
        if _is_placeholder_model(model_name):
            return False
        try:
            model_name = self._normalize_model_name(model_name)
            model = self._registry.create_instance(model_name, plugin_type="model")
            # Check if _fit is explicitly overridden to raise NotImplementedError
            import inspect

            source = inspect.getsource(model._fit)
            return "NotImplementedError" not in source
        except Exception:
            return True  # Assume fittable if we can't check

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
        logger.debug(
            "Checking model-data compatibility",
            model=model_name,
            test_mode=test_mode,
            n_points=len(data.x) if data is not None and data.x is not None else 0,
        )

        try:
            if _is_placeholder_model(model_name):
                logger.debug("Placeholder model, returning incompatible")
                return {
                    "compatible": False,
                    "decay_type": "unknown",
                    "material_type": "unknown",
                    "warnings": ["No model selected"],
                    "recommendations": [],
                }

            model_name = self._normalize_model_name(model_name)
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

            compatible = result.get("compatible", True)
            logger.info(
                "Compatibility check complete",
                model=model_name,
                test_mode=test_mode,
                compatible=compatible,
                decay_type=result.get("decay_type", "unknown"),
                material_type=result.get("material_type", "unknown"),
            )

            return {
                "compatible": compatible,
                "decay_type": result.get("decay_type", "unknown"),
                "material_type": result.get("material_type", "unknown"),
                "warnings": result.get("warnings", []),
                "recommendations": result.get("recommendations", []),
                "details": result,
            }

        except Exception as e:
            logger.error(
                "Compatibility check failed",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
            return {
                "compatible": False,
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
        logger.debug(
            "Getting smart initialization",
            model=model_name,
            test_mode=test_mode,
            n_points=len(data.x) if data.x is not None else 0,
        )

        try:
            model_name = self._normalize_model_name(model_name)
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
                logger.info(
                    "Smart initialization computed",
                    model=model_name,
                    test_mode=test_mode,
                    n_params=len(init_params),
                )
                return init_params
            else:
                # Return default values
                defaults = {
                    name: param.value for name, param in model.parameters.items()
                }
                logger.debug(
                    "Using default initialization (no smart_initial_guess)",
                    model=model_name,
                    n_params=len(defaults),
                )
                return defaults

        except Exception as e:
            logger.warning(
                "Smart initialization failed, using defaults",
                model=model_name,
                error=str(e),
            )
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
        n_points = len(data.x) if data.x is not None else 0
        logger.info(
            "Starting model fit",
            model=model_name,
            n_points=n_points,
            test_mode=test_mode,
        )
        logger.debug(
            "Fit parameters",
            model=model_name,
            initial_params=params,
            fit_kwargs=fit_kwargs,
        )

        try:
            model_name = self._normalize_model_name(model_name)
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")
            logger.debug("Model instance created", model=model_name)

            # Set initial parameter values if provided
            if params:
                for name, value in params.items():
                    if name in model.parameters:
                        if isinstance(value, dict):
                            # Handle dict with 'value', 'bounds', 'fixed'
                            param_val = value.get(
                                "value",
                                value.get("default", model.parameters[name].value),
                            )
                            # Use set_value() to enforce bounds validation (F-007 fix)
                            model.parameters.set_value(name, param_val)
                            if value.get("fixed"):
                                # Lock parameter by setting bounds to (value, value)
                                model.parameters[name].bounds = (param_val, param_val)
                            elif "bounds" in value:
                                model.parameters[name].bounds = value["bounds"]
                        else:
                            # Handle direct value — use set_value() for validation
                            model.parameters.set_value(name, value)
                logger.debug("Initial parameters set", n_params=len(params))

            # Extract data
            x = np.asarray(data.x)
            y = np.asarray(data.y)

            # Determine test mode
            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "oscillation")

            # Validate test mode is supported by this model
            model_info = self.get_model_info(model_name)
            supported_modes = model_info.get("supported_test_modes", [])
            if supported_modes and test_mode not in supported_modes:
                error_msg = (
                    f"Model '{model_name}' does not support test_mode='{test_mode}'. "
                    f"Supported modes: {supported_modes}"
                )
                logger.error(
                    "Unsupported test mode",
                    model=model_name,
                    test_mode=test_mode,
                    supported_modes=supported_modes,
                )
                raise ValueError(error_msg)

            # Fit model
            logger.debug(
                "Calling model.fit",
                model=model_name,
                test_mode=test_mode,
                n_points=len(x),
            )

            # Add test_mode to fit_kwargs
            fit_kwargs["test_mode"] = test_mode

            # F-002: Extract protocol-specific kwargs from data metadata
            # Models like FIKH/FMLIKH need strain, gamma_dot, sigma_applied, etc.
            _PROTOCOL_KWARGS = (
                "strain", "gamma_dot", "sigma_applied", "sigma_0",
                "gamma_0", "omega_laos", "omega", "T_init", "T", "n_cycles",
                "t_wait", "return_components", "return_full",
                # HL-specific kwargs (non-standard names)
                "gdot", "stress_target", "gamma0",
                # ITT-MCT-specific kwargs
                "gamma_pre", "use_diffrax", "t_max", "n_harmonics",
            )
            for key in _PROTOCOL_KWARGS:
                if key not in fit_kwargs and key in data.metadata:
                    fit_kwargs[key] = data.metadata[key]

            # Inject progress callback if provided and not already set
            if progress_callback and "callback" not in fit_kwargs:
                fit_kwargs["callback"] = progress_callback

            # Translate GUI option names to BaseModel.fit() parameter names
            _GUI_TO_FIT_KEYS = {
                "algorithm": "method",
                "multistart": "use_multi_start",
                "num_starts": "n_starts",
            }
            for gui_key, fit_key in _GUI_TO_FIT_KEYS.items():
                if gui_key in fit_kwargs and fit_key not in fit_kwargs:
                    fit_kwargs[fit_key] = fit_kwargs.pop(gui_key)
                elif gui_key in fit_kwargs:
                    fit_kwargs.pop(gui_key)

            # Remove GUI-only keys that BaseModel.fit() does not accept
            for gui_only in ("use_bounds", "verbose"):
                fit_kwargs.pop(gui_only, None)

            # Fit (this uses NLSQ by default)
            model.fit(x, y, **fit_kwargs)
            logger.debug("Model fit completed", model=model_name)

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

            # Try to get metrics from OptimizationResult (NLSQ 0.6.0 compatibility)
            nlsq_result = (
                model.get_nlsq_result() if hasattr(model, "get_nlsq_result") else None
            )

            if nlsq_result is not None and nlsq_result.r_squared is not None:
                # Use OptimizationResult properties
                r_squared = nlsq_result.r_squared
                chi_squared = float(np.sum(residuals**2))
                rmse = nlsq_result.rmse
                mae = nlsq_result.mae
                aic = nlsq_result.aic
                bic = nlsq_result.bic
                adj_r_squared = nlsq_result.adj_r_squared

                # Calculate MPE from residuals
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_errors = np.abs(residuals / y_real) * 100
                    pct_errors = pct_errors[np.isfinite(pct_errors)]
                    mpe = float(np.mean(pct_errors)) if len(pct_errors) > 0 else 0.0
            else:
                # Fallback: Calculate metrics manually
                chi_squared = float(np.sum(residuals**2))

                # Calculate R-squared (coefficient of determination)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Calculate adjusted R-squared
                n = len(y)
                p = len(model.parameters)
                if n > p + 1 and ss_tot > 0:
                    adj_r_squared = 1.0 - ((1 - r_squared) * (n - 1) / (n - p - 1))
                else:
                    adj_r_squared = r_squared

                # Calculate RMSE and MAE
                rmse = float(np.sqrt(np.mean(residuals**2)))
                mae = float(np.mean(np.abs(residuals)))

                # Calculate AIC and BIC
                if n > 0:
                    mse = ss_res / n
                    if mse > 0:
                        log_likelihood = -n / 2 * (np.log(2 * np.pi * mse) + 1)
                        aic = float(2 * p - 2 * log_likelihood)
                        bic = float(p * np.log(n) - 2 * log_likelihood)
                    else:
                        aic = None
                        bic = None
                else:
                    aic = None
                    bic = None

                # Calculate MPE (Mean Percentage Error)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_errors = np.abs(residuals / y_real) * 100
                    pct_errors = pct_errors[np.isfinite(pct_errors)]
                    mpe = float(np.mean(pct_errors)) if len(pct_errors) > 0 else 0.0

            # Get fitted parameters
            fitted_params = {
                name: param.value for name, param in model.parameters.items()
            }

            # Get metadata with all metrics included
            metadata = {
                "test_mode": test_mode,
                "n_iterations": getattr(model, "_n_iterations", None),
                "convergence": getattr(model, "_convergence", None),
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "rmse": rmse,
                "mae": mae,
                "aic": aic,
                "bic": bic,
                "mpe": mpe,
            }

            # F-HL-005 fix: Capture fitted model state for stateful models
            # (HL, DMT, ITT-MCT, etc.) so BayesianService can restore it.
            _fitted_model_state = {}
            for attr in (
                "_last_fit_kwargs",
                "_fit_data_metadata",
                "_use_forward_mode_ad",
                # HVNM/HVM/VLB/STZ protocol state (model_function reads these
                # instead of _last_fit_kwargs for startup/creep/LAOS context)
                "_gamma_dot_applied",
                "_sigma_applied",
                "_sigma_0",
                "_gamma_0",
                "_omega_laos",
                # IKH/ML-IKH protocol state for model_function
                "_fit_gamma_dot",
                "_fit_sigma_applied",
                "_fit_sigma_0",
                # SGR startup protocol state for model_function
                "_startup_gamma_dot",
                # GMM protocol state for startup/LAOS model_function
                "_laos_omega",
                "_laos_gamma_0",
                "_n_modes",
                # SPP protocol state for model_function
                "_yield_type",
                "_omega",
                # ITT-MCT Prony decomposition state
                "_prony_amplitudes",
                "_prony_times",
                "_memory_form",
                "_use_lorentzian",
                "n_prony_modes",
            ):
                val = getattr(model, attr, None)
                if val is not None:
                    _fitted_model_state[attr] = val
            if _fitted_model_state:
                metadata["fitted_model_state"] = _fitted_model_state

            # Extract pcov (parameter covariance) from NLSQ result for uncertainty bands
            pcov = None
            if nlsq_result is not None:
                pcov = nlsq_result.pcov
                metadata["nlsq_result"] = {
                    "success": nlsq_result.success,
                    "nfev": nlsq_result.nfev,
                    "njev": nlsq_result.njev,
                }
            elif hasattr(model, "_nlsq_result") and model._nlsq_result:
                pcov = getattr(model._nlsq_result, "pcov", None)
                metadata["nlsq_result"] = {
                    "success": getattr(model._nlsq_result, "success", False),
                    "nfev": model._nlsq_result.nfev,
                    "njev": model._nlsq_result.njev,
                }

            # Propagate convergence status from NLSQ result (F-013 fix)
            fit_success = True
            fit_message = "Fit successful"
            if nlsq_result is not None:
                fit_success = getattr(nlsq_result, "success", True)
                fit_message = getattr(nlsq_result, "message", fit_message)
            elif hasattr(model, "_nlsq_result") and model._nlsq_result:
                fit_success = getattr(model._nlsq_result, "success", True)
                fit_message = getattr(model._nlsq_result, "message", fit_message)

            logger.info(
                "Model fit complete",
                model=model_name,
                success=fit_success,
                r_squared=r_squared,
                chi_squared=chi_squared,
                rmse=rmse,
                n_params=len(fitted_params),
            )

            return FitResult(
                model_name=model_name,
                parameters=fitted_params,
                chi_squared=chi_squared,
                success=fit_success,
                message=fit_message,
                timestamp=datetime.now(),
                r_squared=float(r_squared) if r_squared is not None else 0.0,
                mpe=float(mpe) if mpe is not None else 0.0,
                residuals=residuals,
                x_fit=x,
                y_fit=y_pred,
                pcov=pcov,
                rmse=float(rmse) if rmse is not None else None,
                mae=float(mae) if mae is not None else None,
                aic=float(aic) if aic is not None else None,
                bic=float(bic) if bic is not None else None,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                "Fit failed",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
            return FitResult(
                model_name=model_name,
                parameters={},
                chi_squared=np.inf,
                success=False,
                message=f"Fit failed: {e}",
                timestamp=datetime.now(),
                residuals=np.array([]),
                x_fit=np.array([]),
                y_fit=np.array([]),
                metadata={"error": str(e)},
            )

    def predict(
        self,
        model_name: str,
        parameters: dict[str, float],
        x_values: np.ndarray,
        test_mode: str | None = None,
        model_kwargs: dict | None = None,
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
        test_mode : str, optional
            Optional test mode for prediction (relaxation/oscillation/creep/etc.)
        model_kwargs : dict, optional
            Model initialization kwargs (e.g., n_modes for GeneralizedMaxwell)

        Returns
        -------
        np.ndarray
            Predicted y values
        """
        logger.debug(
            "Starting prediction",
            model=model_name,
            n_points=len(x_values),
            test_mode=test_mode,
        )

        try:
            # Create model instance with optional configuration
            model_kwargs = model_kwargs or {}
            model = self._registry.create_instance(
                self._normalize_model_name(model_name),
                plugin_type="model",
                **model_kwargs,
            )

            # Set parameters
            for name, value in parameters.items():
                if name in model.parameters:
                    model.parameters[name].value = value

            # Mark as fitted
            model.fitted_ = True

            # F-HL-006 fix: Set _test_mode on fresh model instance so that
            # stateful models (HL, DMT, etc.) whose _predict() checks
            # self._test_mode don't raise ValueError.
            if test_mode and hasattr(model, "_test_mode"):
                model._test_mode = test_mode

            # Transfer fitted model state if provided (protocol kwargs, grid settings)
            fitted_state = model_kwargs.pop("fitted_model_state", None) if model_kwargs else None
            if fitted_state and isinstance(fitted_state, dict):
                for attr in (
                    "_last_fit_kwargs",
                    "_fit_data_metadata",
                    # HVNM/HVM/VLB/STZ protocol state for model_function
                    "_gamma_dot_applied",
                    "_sigma_applied",
                    "_sigma_0",
                    "_gamma_0",
                    "_omega_laos",
                    # IKH/ML-IKH protocol state for model_function
                    "_fit_gamma_dot",
                    "_fit_sigma_applied",
                    "_fit_sigma_0",
                    # SGR startup protocol state for model_function
                    "_startup_gamma_dot",
                    # SPP protocol state for model_function
                    "_yield_type",
                    "_omega",
                    # ITT-MCT Prony decomposition state
                    "_prony_amplitudes",
                    "_prony_times",
                    "_memory_form",
                    "_use_lorentzian",
                    "n_prony_modes",
                ):
                    if attr in fitted_state:
                        setattr(model, attr, fitted_state[attr])

            # Predict
            if test_mode is None:
                result = model.predict(x_values)
            else:
                try:
                    result = model.predict(x_values, test_mode=test_mode)
                except TypeError:
                    # Backward compatibility for model implementations without test_mode.
                    result = model.predict(x_values)

            logger.debug(
                "Prediction complete",
                model=model_name,
                n_points=len(result),
            )
            return result

        except Exception as e:
            logger.error(
                "Prediction failed",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
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
        logger.debug("Getting model equation", model=model_name)

        try:
            model = self._registry.create_instance(model_name, plugin_type="model")
            if hasattr(model, "equation"):
                logger.debug("Equation retrieved", model=model_name)
                return model.equation
            logger.debug("No equation available", model=model_name)
            return "Equation not available"
        except Exception as e:
            logger.error(
                "Failed to get model equation",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
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
        logger.debug(
            "Validating parameters",
            model=model_name,
            n_params=len(parameters),
        )

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
                    warnings.append(f"{name}={value} is below lower bound {lower}")
                if value > upper:
                    warnings.append(f"{name}={value} is above upper bound {upper}")

            if warnings:
                logger.info(
                    "Parameter validation warnings",
                    model=model_name,
                    n_warnings=len(warnings),
                    warnings=warnings,
                )
            else:
                logger.debug("All parameters valid", model=model_name)

        except Exception as e:
            logger.error(
                "Parameter validation failed",
                model=model_name,
                error=str(e),
                exc_info=True,
            )
            warnings.append(f"Validation failed: {e}")

        return warnings
