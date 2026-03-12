"""
Transform Service
================

Service for applying rheological transforms (mastercurve, FFT, SRFS, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.logging import get_logger
from rheojax.transforms import (
    SRFS,
    CoxMerz,
    FFTAnalysis,
    LVEEnvelope,
    Mastercurve,
    MutationNumber,
    OWChirp,
    PronyConversion,
    SmoothDerivative,
    SpectrumInversion,
    SPPDecomposer,
)

logger = get_logger(__name__)


class TransformService:
    """Service for rheological transform operations.

    Transforms:
        - Mastercurve (TTS with auto shift factors)
        - FFT (oscillation <-> time domain)
        - SRFS (strain-rate frequency superposition)
        - OWChirp (optimal windowed chirp)
        - Numerical derivatives
        - Mutation number analysis

    Example
    -------
    >>> service = TransformService()
    >>> transforms = service.get_available_transforms()
    >>> result = service.apply_transform('mastercurve', data, params)
    """

    def __init__(self) -> None:
        """Initialize transform service."""
        logger.debug("Initializing TransformService")
        self._transforms = {
            "mastercurve": Mastercurve,
            "fft": FFTAnalysis,
            "srfs": SRFS,
            "owchirp": OWChirp,
            "derivative": SmoothDerivative,
            "mutation_number": MutationNumber,
            "spp": SPPDecomposer,
            "cox_merz": CoxMerz,
            "lve_envelope": LVEEnvelope,
            "prony_conversion": PronyConversion,
            "spectrum_inversion": SpectrumInversion,
        }
        logger.debug(
            "TransformService initialized",
            available_transforms=list(self._transforms.keys()),
        )

    def get_available_transforms(self) -> list[str]:
        """Get list of available transforms.

        Returns
        -------
        list[str]
            Transform names
        """
        logger.debug("Getting available transforms")
        return list(self._transforms.keys())

    def get_transform_params(self, name: str) -> dict[str, Any]:
        """Get configurable parameters for transform.

        Parameters
        ----------
        name : str
            Transform name

        Returns
        -------
        dict
            Parameter specifications with defaults and descriptions
        """
        logger.debug("Getting transform parameters", transform=name)
        params_map = {
            "mastercurve": {
                "reference_temp": {
                    "type": "float",
                    "default": 25.0,
                    "label": "Reference Temperature",
                    "range": (-100, 300),
                    "description": "Reference temperature (C)",
                },
                "auto_shift": {
                    "type": "bool",
                    "default": True,
                    "label": "Auto Shift",
                    "description": "Automatically calculate shift factors",
                },
                "shift_method": {
                    "type": "choice",
                    "choices": ["wlf", "arrhenius", "manual"],
                    "default": "wlf",
                    "label": "Shift Method",
                    "description": "Method for calculating shift factors",
                },
            },
            "fft": {
                "direction": {
                    "type": "choice",
                    "choices": ["forward", "inverse"],
                    "default": "forward",
                    "label": "Direction",
                    "description": "Transform direction",
                },
                "window": {
                    "type": "choice",
                    "choices": [
                        "hann",
                        "hamming",
                        "blackman",
                        "bartlett",
                        "rectangular",
                    ],
                    "default": "hann",
                    "label": "Window Function",
                    "description": "Window function (rectangular = none)",
                },
                "detrend": {
                    "type": "bool",
                    "default": True,
                    "label": "Detrend",
                    "description": "Remove linear trend before FFT",
                },
                "return_psd": {
                    "type": "bool",
                    "default": False,
                    "label": "Return PSD",
                    "description": "Return power spectral density instead of magnitude",
                },
                "normalize": {
                    "type": "bool",
                    "default": True,
                    "label": "Normalize",
                    "description": "Normalize FFT output",
                },
            },
            "srfs": {
                "reference_gamma_dot": {
                    "type": "float",
                    "default": 1.0,
                    "label": "Reference Shear Rate",
                    "range": (0.001, 1000),
                    "description": "Reference shear rate (1/s)",
                },
                "auto_shift": {
                    "type": "bool",
                    "default": True,
                    "label": "Auto Shift",
                    "description": "Auto-calculate shift factors",
                },
            },
            "owchirp": {
                "min_frequency": {
                    "type": "float",
                    "default": 0.01,
                    "label": "Min Frequency",
                    "range": (0.0001, 1e6),
                    "description": "Minimum frequency (rad/s)",
                },
                "max_frequency": {
                    "type": "float",
                    "default": 100.0,
                    "label": "Max Frequency",
                    "range": (0.0001, 1e6),
                    "description": "Maximum frequency (rad/s)",
                },
                "n_frequencies": {
                    "type": "int",
                    "default": 100,
                    "label": "Number of Frequencies",
                    "range": (4, 5000),
                    "description": "Number of frequency points",
                },
                "wavelet_width": {
                    "type": "float",
                    "default": 5.0,
                    "label": "Wavelet Width",
                    "range": (1.0, 20.0),
                    "description": "Width parameter for wavelet",
                },
                "extract_harmonics": {
                    "type": "bool",
                    "default": True,
                    "label": "Extract Harmonics",
                    "description": "Extract harmonic components",
                },
                "max_harmonic": {
                    "type": "int",
                    "default": 7,
                    "label": "Max Harmonic",
                    "range": (1, 99),
                    "description": "Maximum harmonic to extract",
                },
            },
            "derivative": {
                "order": {
                    "type": "int",
                    "default": 1,
                    "label": "Derivative Order",
                    "range": (1, 4),
                    "description": "Derivative order",
                },
                "window_length": {
                    "type": "int",
                    "default": 11,
                    "label": "Window Length",
                    "range": (3, 201),
                    "description": "Savitzky-Golay window length (must be odd)",
                },
                "poly_order": {
                    "type": "int",
                    "default": 3,
                    "label": "Polynomial Order",
                    "range": (1, 10),
                    "description": "Polynomial order for Savitzky-Golay (must be < window_length)",
                },
                "method": {
                    "type": "choice",
                    "choices": ["savgol", "finite_diff", "spline", "total_variation"],
                    "default": "savgol",
                    "label": "Method",
                    "description": "Differentiation method",
                },
                "mode": {
                    "type": "choice",
                    "choices": ["mirror", "nearest", "constant", "wrap"],
                    "default": "mirror",
                    "label": "Padding Mode",
                    "description": "Boundary extension mode for Savitzky-Golay filter",
                },
                "smooth_before": {
                    "type": "bool",
                    "default": False,
                    "label": "Smooth Before",
                    "description": "Apply smoothing before differentiation",
                },
                "smooth_after": {
                    "type": "bool",
                    "default": False,
                    "label": "Smooth After",
                    "description": "Apply smoothing after differentiation",
                },
                "validate_window": {
                    "type": "bool",
                    "default": True,
                    "label": "Validate Window",
                    "description": "Force odd window length",
                },
            },
            "mutation_number": {
                "integration_method": {
                    "type": "choice",
                    "choices": ["trapz", "simpson", "romberg"],
                    "default": "trapz",
                    "label": "Integration Method",
                    "description": "Numerical integration method",
                },
                "extrapolate": {
                    "type": "bool",
                    "default": False,
                    "label": "Extrapolate",
                    "description": "Extrapolate data outside measured range",
                },
                "extrapolation_model": {
                    "type": "choice",
                    "choices": ["exponential", "power_law", "linear"],
                    "default": "exponential",
                    "label": "Extrapolation Model",
                    "description": "Model for data extrapolation",
                },
            },
            "spp": {
                "omega": {
                    "type": "float",
                    "default": 1.0,
                    "label": "Angular Frequency",
                    "range": (0.001, 1000),
                    "description": "Angular frequency (rad/s)",
                },
                "gamma_0": {
                    "type": "float",
                    "default": 1.0,
                    "label": "Strain Amplitude",
                    "range": (0.0001, 100),
                    "description": "Strain amplitude (dimensionless)",
                },
                "n_harmonics": {
                    "type": "int",
                    "default": 39,
                    "label": "Number of Harmonics",
                    "range": (1, 99),
                    "description": "Number of harmonics to extract",
                },
                "yield_tolerance": {
                    "type": "float",
                    "default": 0.02,
                    "label": "Yield Tolerance",
                    "range": (0.0001, 1.0),
                    "description": "Tolerance for yield point detection",
                },
                "start_cycle": {
                    "type": "int",
                    "default": 0,
                    "label": "Start Cycle",
                    "range": (0, 100),
                    "description": "First cycle to analyze (skip transients)",
                },
                "end_cycle": {
                    "type": "int",
                    "default": 0,
                    "label": "End Cycle",
                    "range": (0, 1000),
                    "description": "Last cycle to analyze (0 = use all cycles)",
                },
                "use_numerical_method": {
                    "type": "bool",
                    "default": False,
                    "label": "Numerical Method",
                    "description": "Use numerical differentiation (MATLAB-compatible)",
                },
            },
            "cox_merz": {
                "tolerance": {
                    "type": "float",
                    "default": 0.10,
                    "label": "Pass Tolerance",
                    "range": (0.001, 1.0),
                    "description": "Maximum mean relative deviation for the rule to pass",
                },
                "n_points": {
                    "type": "int",
                    "default": 50,
                    "label": "Interpolation Points",
                    "range": (10, 500),
                    "description": "Number of points on the common rate grid",
                },
            },
            "lve_envelope": {
                "shear_rate": {
                    "type": "float",
                    "default": 1.0,
                    "label": "Shear Rate",
                    "range": (1e-6, 1e6),
                    "description": "Applied shear rate gamma_dot_0 (1/s)",
                },
                "G_e": {
                    "type": "float",
                    "default": 0.0,
                    "label": "Equilibrium Modulus",
                    "range": (0.0, 1e12),
                    "description": "Equilibrium modulus G_e (Pa, 0 = no equilibrium term)",
                },
            },
            "prony_conversion": {
                "n_modes": {
                    "type": "int",
                    "default": 0,
                    "label": "Number of Modes",
                    "range": (0, 100),
                    "description": "Prony modes (0 = auto-select)",
                },
                "direction": {
                    "type": "choice",
                    "choices": ["time_to_freq", "freq_to_time"],
                    "default": "time_to_freq",
                    "label": "Direction",
                    "description": "Conversion direction",
                },
            },
            "spectrum_inversion": {
                "method": {
                    "type": "choice",
                    "choices": ["tikhonov", "max_entropy"],
                    "default": "tikhonov",
                    "label": "Method",
                    "description": "Regularization method",
                },
                "n_tau": {
                    "type": "int",
                    "default": 100,
                    "label": "Tau Grid Points",
                    "range": (2, 1000),
                    "description": "Number of relaxation time points",
                },
                "source": {
                    "type": "choice",
                    "choices": ["oscillation", "relaxation"],
                    "default": "oscillation",
                    "label": "Data Source",
                    "description": "Input data type",
                },
                "G_e": {
                    "type": "float",
                    "default": 0.0,
                    "label": "Equilibrium Modulus",
                    "range": (0.0, 1e12),
                    "description": "Equilibrium modulus G_e (Pa)",
                },
                "regularization": {
                    "type": "float",
                    "default": 0.0,
                    "label": "Regularization (lambda)",
                    "range": (0.0, 1e4),
                    "description": "Manual regularization parameter (0 = auto-select via GCV)",
                },
            },
        }

        return params_map.get(name, {})

    def get_transform_metadata(self) -> list[dict[str, Any]]:
        """Return display metadata for all transforms.

        Returns
        -------
        list[dict]
            Each dict has: key, name, description, requires_multiple
        """
        return [
            {
                "key": "fft",
                "name": "FFT",
                "description": "Fast Fourier Transform for frequency analysis",
                "requires_multiple": False,
            },
            {
                "key": "mastercurve",
                "name": "Mastercurve",
                "description": "Time-temperature superposition",
                "requires_multiple": True,
            },
            {
                "key": "srfs",
                "name": "SRFS",
                "description": "Strain-rate frequency superposition",
                "requires_multiple": True,
            },
            {
                "key": "mutation_number",
                "name": "Mutation Number",
                "description": "Calculate mutation number",
                "requires_multiple": False,
            },
            {
                "key": "owchirp",
                "name": "OW Chirp",
                "description": "Optimally-windowed chirp analysis",
                "requires_multiple": False,
            },
            {
                "key": "spp",
                "name": "SPP Analysis",
                "description": "LAOS yield stress and cage modulus extraction",
                "requires_multiple": False,
            },
            {
                "key": "derivative",
                "name": "Derivatives",
                "description": "Calculate numerical derivatives",
                "requires_multiple": False,
            },
            {
                "key": "cox_merz",
                "name": "Cox-Merz Rule",
                "description": "Validate Cox-Merz rule (|eta*| vs eta)",
                "requires_multiple": True,
            },
            {
                "key": "lve_envelope",
                "name": "LVE Envelope",
                "description": "Linear viscoelastic startup stress envelope",
                "requires_multiple": False,
            },
            {
                "key": "prony_conversion",
                "name": "Prony Conversion",
                "description": "Time to/from frequency domain via Prony series",
                "requires_multiple": False,
            },
            {
                "key": "spectrum_inversion",
                "name": "Spectrum Inversion",
                "description": "Recover relaxation spectrum H(tau) from G(t) or G*(omega)",
                "requires_multiple": False,
            },
        ]

    def apply_transform(
        self,
        name: str,
        data: RheoData | list[RheoData],
        params: dict[str, Any],
    ) -> RheoData | tuple[RheoData, dict]:
        """Apply transform and return new data.

        Parameters
        ----------
        name : str
            Transform name
        data : RheoData or list[RheoData]
            Input data (single or multiple datasets)
        params : dict
            Transform parameters

        Returns
        -------
        RheoData or tuple
            Transformed data, optionally with additional results
        """
        logger.debug("Entering apply_transform", transform=name, params=params)
        logger.info("Starting transform", transform=name)
        try:
            if name not in self._transforms:
                logger.error("Unknown transform requested", transform=name)
                raise ValueError(f"Unknown transform: {name}")

            # Validate input before processing (T-007)
            warnings = self.validate_transform_input(name, data)
            if warnings:
                for w in warnings:
                    logger.warning("Transform input warning", transform=name, warning=w)

            def _with_provenance(result_data: RheoData) -> RheoData:
                # G-017 fix: Build provenance on a copy of metadata and return
                # a new RheoData rather than mutating result_data.metadata in-
                # place.  In-place mutation breaks callers that retain a
                # reference to the pre-provenance object (e.g., undo stacks,
                # caches).
                prov_entry = {
                    "transform": name,
                    "params": {
                        k: v
                        for k, v in params.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                }
                new_meta = dict(result_data.metadata)
                history = list(new_meta.get("provenance", []))
                history.append(prov_entry)
                new_meta["provenance"] = history
                # Track transform chain for reproducibility (T-013)
                applied = list(new_meta.get("transforms_applied", []))
                applied.append(name)
                new_meta["transforms_applied"] = applied
                new_meta["last_transform"] = name
                # Return a new RheoData with the updated metadata dict;
                # do NOT assign back to result_data.metadata.
                return RheoData(
                    x=result_data.x,
                    y=result_data.y,
                    x_units=result_data.x_units,
                    y_units=result_data.y_units,
                    domain=result_data.domain,
                    initial_test_mode=new_meta.get("test_mode"),
                    metadata=new_meta,
                    validate=False,
                )

            if name == "mastercurve":
                # Mastercurve requires multiple datasets
                if not isinstance(data, list):
                    logger.error(
                        "Mastercurve requires list of datasets",
                        transform=name,
                        received_type=type(data).__name__,
                    )
                    raise ValueError("Mastercurve requires list of datasets")

                reference_temp = params.get("reference_temp", 25.0)
                auto_shift = params.get("auto_shift", True)

                logger.debug(
                    "Applying mastercurve",
                    reference_temp=reference_temp,
                    auto_shift=auto_shift,
                    n_datasets=len(data),
                )
                mc = Mastercurve(reference_temp=reference_temp, auto_shift=auto_shift)
                mastercurve_data, shift_factors = mc.transform(data)

                logger.info(
                    "Transform complete",
                    transform=name,
                    reference_temp=reference_temp,
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(mastercurve_data), {
                    "shift_factors": shift_factors
                }

            elif name == "fft":
                # FFT transform
                direction = params.get("direction", "forward")
                # Map 'rectangular' to 'none' for window parameter
                window = params.get("window", "hann")
                if window == "rectangular":
                    window = "none"
                detrend = params.get("detrend", True)
                return_psd = params.get("return_psd", False)
                normalize = params.get("normalize", True)

                logger.debug(
                    "Applying FFT transform",
                    direction=direction,
                    window=window,
                    detrend=detrend,
                    return_psd=return_psd,
                    normalize=normalize,
                )
                fft = FFTAnalysis(
                    window=window,
                    detrend=detrend,
                    return_psd=return_psd,
                    normalize=normalize,
                )

                if direction == "forward":
                    result = fft.transform(data)
                else:
                    result = fft.inverse_transform(data)

                logger.info("Transform complete", transform=name, direction=direction)
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result)

            elif name == "srfs":
                # SRFS transform
                if not isinstance(data, list):
                    logger.error(
                        "SRFS requires list of datasets",
                        transform=name,
                        received_type=type(data).__name__,
                    )
                    raise ValueError("SRFS requires list of datasets")

                reference_gamma_dot = params.get("reference_gamma_dot", 1.0)
                auto_shift = params.get("auto_shift", True)

                logger.debug(
                    "Applying SRFS transform",
                    reference_gamma_dot=reference_gamma_dot,
                    auto_shift=auto_shift,
                    n_datasets=len(data),
                )
                srfs = SRFS(
                    reference_gamma_dot=reference_gamma_dot, auto_shift=auto_shift
                )
                master_curve, shift_factors = srfs.transform(data)

                logger.info(
                    "Transform complete",
                    transform=name,
                    reference_gamma_dot=reference_gamma_dot,
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(master_curve), {"shift_factors": shift_factors}

            elif name == "owchirp":
                # OWChirp transform
                # Note: OWChirp uses n_frequencies, frequency_range, wavelet_width, etc.
                min_freq = params.get("min_frequency", 0.01)
                max_freq = params.get("max_frequency", 100.0)
                n_frequencies = params.get("n_frequencies", 100)
                wavelet_width = params.get("wavelet_width", 5.0)
                extract_harmonics = params.get("extract_harmonics", True)
                max_harmonic = params.get("max_harmonic", 7)

                logger.debug(
                    "Applying OWChirp transform",
                    min_frequency=min_freq,
                    max_frequency=max_freq,
                    n_frequencies=n_frequencies,
                    wavelet_width=wavelet_width,
                    extract_harmonics=extract_harmonics,
                    max_harmonic=max_harmonic,
                )
                owchirp = OWChirp(
                    n_frequencies=n_frequencies,
                    frequency_range=(min_freq, max_freq),
                    wavelet_width=wavelet_width,
                    extract_harmonics=extract_harmonics,
                    max_harmonic=max_harmonic,
                )
                result = owchirp.transform(data)

                logger.info("Transform complete", transform=name)
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result)

            elif name == "derivative":
                # Smooth derivative
                # Map 'order' from Page to 'deriv' for SmoothDerivative
                deriv_order = int(params.get("order", 1))
                window_length = int(params.get("window_length", 11))
                # Map 'poly_order' from Page to 'polyorder' for SmoothDerivative
                polyorder = int(params.get("poly_order", params.get("polyorder", 3)))
                method = params.get("method", "savgol")
                smooth_before = params.get("smooth_before", False)
                smooth_after = params.get("smooth_after", False)

                logger.debug(
                    "Applying derivative transform",
                    order=deriv_order,
                    window_length=window_length,
                    polyorder=polyorder,
                    method=method,
                    smooth_before=smooth_before,
                    smooth_after=smooth_after,
                )
                deriv = SmoothDerivative(
                    method=method,
                    window_length=window_length,
                    polyorder=polyorder,
                    deriv=deriv_order,
                    smooth_before=smooth_before,
                    smooth_after=smooth_after,
                )
                result = deriv.transform(data)

                logger.info(
                    "Transform complete", transform=name, derivative_order=deriv_order
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result)

            elif name == "mutation_number":
                # Mutation number
                # Note: MutationNumber doesn't take reference_frequency as __init__ param
                # It uses integration_method, extrapolate, extrapolation_model
                integration_method = params.get("integration_method", "trapz")
                extrapolate = params.get("extrapolate", False)
                extrapolation_model = params.get("extrapolation_model", "exponential")

                logger.debug(
                    "Applying mutation number transform",
                    integration_method=integration_method,
                    extrapolate=extrapolate,
                    extrapolation_model=extrapolation_model,
                )
                mn = MutationNumber(
                    integration_method=integration_method,
                    extrapolate=extrapolate,
                    extrapolation_model=extrapolation_model,
                )
                result = mn.transform(data)

                logger.info("Transform complete", transform=name)
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result)

            elif name == "spp":
                # SPP decomposition for LAOS yield stress extraction
                omega = params.get("omega", 1.0)
                gamma_0 = params.get("gamma_0", 1.0)
                n_harmonics = int(params.get("n_harmonics", 39))
                yield_tolerance = params.get("yield_tolerance", 0.02)
                start_cycle = int(params.get("start_cycle", 0))
                # Page defaults to 0, but 0 means "use all cycles" so convert to None
                end_cycle_raw = params.get("end_cycle", None)
                end_cycle = (
                    None if end_cycle_raw in (None, 0, 0.0) else int(end_cycle_raw)
                )
                use_numerical_method = params.get("use_numerical_method", False)

                logger.debug(
                    "Applying SPP transform",
                    omega=omega,
                    gamma_0=gamma_0,
                    n_harmonics=n_harmonics,
                    yield_tolerance=yield_tolerance,
                    start_cycle=start_cycle,
                    end_cycle=end_cycle,
                    use_numerical_method=use_numerical_method,
                )
                spp = SPPDecomposer(
                    omega=omega,
                    gamma_0=gamma_0,
                    n_harmonics=n_harmonics,
                    yield_tolerance=yield_tolerance,
                    start_cycle=start_cycle,
                    end_cycle=end_cycle,
                    use_numerical_method=use_numerical_method,
                )
                result = spp.transform(data)
                spp_results = spp.get_results()

                logger.info(
                    "Transform complete",
                    transform=name,
                    sigma_sy=spp_results["sigma_sy"],
                    sigma_dy=spp_results["sigma_dy"],
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result), {"spp_results": spp_results}

            elif name == "cox_merz":
                # Cox-Merz rule validation — requires list of 2 datasets
                if not isinstance(data, list) or len(data) != 2:
                    logger.error(
                        "CoxMerz requires exactly 2 datasets",
                        transform=name,
                        received_type=type(data).__name__,
                    )
                    raise ValueError(
                        "Cox-Merz requires exactly 2 datasets: [oscillation, flow_curve]"
                    )

                tolerance = float(params.get("tolerance", 0.10))
                n_points = int(params.get("n_points", 50))

                logger.debug(
                    "Applying Cox-Merz transform",
                    tolerance=tolerance,
                    n_points=n_points,
                )
                cox_merz = CoxMerz(tolerance=tolerance, n_points=n_points)
                result, extra = cox_merz.transform(data)

                logger.info(
                    "Transform complete",
                    transform=name,
                    passes=extra.get("cox_merz_result", {}).passes
                    if hasattr(extra.get("cox_merz_result"), "passes")
                    else None,
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result), extra

            elif name == "lve_envelope":
                # LVE startup stress envelope — uses Prony params from data.metadata
                shear_rate = float(params.get("shear_rate", 1.0))
                G_e = float(params.get("G_e", 0.0))

                logger.debug(
                    "Applying LVE envelope transform",
                    shear_rate=shear_rate,
                    G_e=G_e,
                )
                lve = LVEEnvelope(shear_rate=shear_rate, G_e=G_e)
                result, extra = lve.transform(data)

                logger.info("Transform complete", transform=name)
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result), extra

            elif name == "prony_conversion":
                # Prony series time ↔ frequency conversion
                n_modes_raw = int(params.get("n_modes", 0))
                n_modes = n_modes_raw if n_modes_raw > 0 else None
                direction = params.get("direction", "time_to_freq")

                logger.debug(
                    "Applying Prony conversion transform",
                    n_modes=n_modes,
                    direction=direction,
                )
                prony = PronyConversion(n_modes=n_modes, direction=direction)
                result, extra = prony.transform(data)

                logger.info(
                    "Transform complete", transform=name, direction=direction
                )
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result), extra

            elif name == "spectrum_inversion":
                # Relaxation spectrum H(τ) recovery
                method = params.get("method", "tikhonov")
                n_tau = int(params.get("n_tau", 100))
                source = params.get("source", "oscillation")
                G_e = float(params.get("G_e", 0.0))
                regularization_raw = float(params.get("regularization", 0.0))
                regularization = regularization_raw if regularization_raw > 0.0 else None

                logger.debug(
                    "Applying spectrum inversion transform",
                    method=method,
                    n_tau=n_tau,
                    source=source,
                    G_e=G_e,
                    regularization=regularization,
                )
                spectrum = SpectrumInversion(
                    method=method,
                    n_tau=n_tau,
                    source=source,
                    G_e=G_e,
                    regularization=regularization,
                )
                result, extra = spectrum.transform(data)

                logger.info("Transform complete", transform=name, method=method)
                logger.debug("Exiting apply_transform", transform=name)
                return _with_provenance(result), extra

            else:
                logger.error("Transform not implemented", transform=name)
                raise ValueError(f"Transform {name} not implemented")

        except (ValueError, TypeError):
            # GUI-IO-024: Re-raise user-facing validation errors as-is so
            # callers can distinguish them from unexpected failures and
            # surface meaningful messages in the UI without wrapping.
            logger.error(
                "Transform failed (validation error)",
                transform=name,
                exc_info=True,
            )
            raise
        except MemoryError:
            # Re-raise OOM errors as-is so callers can handle them
            # separately (e.g. reduce dataset size, free caches).
            logger.error(
                "Transform failed (out of memory)",
                transform=name,
                exc_info=True,
            )
            raise
        except Exception as e:
            # Only unknown/unexpected errors are wrapped as RuntimeError.
            logger.error(
                "Transform failed",
                transform=name,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Transform failed: {e}") from e

    def preview_transform(
        self,
        name: str,
        data: RheoData | list[RheoData],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute transform and return plot data for preview.

        Parameters
        ----------
        name : str
            Transform name
        data : RheoData or list[RheoData]
            Input data
        params : dict
            Transform parameters

        Returns
        -------
        dict
            On success: x_before, y_before, x_after, y_after (numpy arrays)
            On failure: error (str)
        """
        logger.debug("Computing preview", transform=name, params=params)

        # Guard: multi-dataset transforms (mastercurve, srfs) cannot be
        # previewed with a single dataset.  Return a silent no-data result
        # instead of calling apply_transform which logs ERROR-level messages
        # for this expected condition and confuses the user.
        _MULTI_DATASET = {"mastercurve", "srfs", "cox_merz"}
        if name in _MULTI_DATASET and not isinstance(data, list):
            logger.debug(
                "Preview skipped: multi-dataset transform requires list input",
                transform=name,
            )
            return {"error": f"{name} preview requires multiple datasets"}

        try:
            # Capture before data
            if isinstance(data, list):
                # Multi-dataset: use first dataset as "before" representative
                x_before = np.asarray(data[0].x)
                y_before = np.asarray(data[0].y)
            else:
                x_before = np.asarray(data.x)
                y_before = np.asarray(data.y)

            # Compute actual transform
            result = self.apply_transform(name, data, params)

            # Unpack result (some transforms return (RheoData, metadata_dict))
            if isinstance(result, tuple):
                result_data = result[0]
            else:
                result_data = result

            x_after = np.asarray(result_data.x)
            y_after = np.asarray(result_data.y)

            return {
                "x_before": x_before,
                "y_before": y_before,
                "x_after": x_after,
                "y_after": y_after,
            }
        except Exception as e:
            logger.warning("Preview failed", transform=name, error=str(e))
            return {"error": str(e)}

    def validate_transform_input(
        self,
        name: str,
        data: RheoData | list[RheoData],
    ) -> list[str]:
        """Validate data for transform.

        Parameters
        ----------
        name : str
            Transform name
        data : RheoData or list[RheoData]
            Input data

        Returns
        -------
        list[str]
            Validation warnings
        """
        logger.debug("Entering validate_transform_input", transform=name)
        warnings = []

        # Check if transform requires multiple datasets
        multi_dataset_transforms = ["mastercurve", "srfs", "cox_merz"]
        if name in multi_dataset_transforms and not isinstance(data, list):
            warnings.append(f"{name} requires multiple datasets")

        # Check data characteristics
        if isinstance(data, list):
            if len(data) < 2:
                warnings.append(f"{name} requires at least 2 datasets")

            # Check for consistent test modes
            test_modes = [d.metadata.get("test_mode") for d in data]
            if len(set(test_modes)) > 1:
                warnings.append("Datasets have different test modes")

        else:
            # Single dataset checks
            if len(data.x) < 10:
                warnings.append("Insufficient data points for transform")

            # Check for monotonicity (required for some transforms)
            if name in ["derivative", "mutation_number"]:
                if not np.all(np.diff(data.x) > 0):
                    warnings.append("Data must be monotonically increasing")

            # SPP requires time-domain LAOS data
            if name == "spp":
                if data.domain != "time":
                    warnings.append("SPP requires time-domain stress waveform data")
                if len(data.x) < 100:
                    warnings.append(
                        "SPP typically requires at least 100 points per cycle"
                    )
                test_mode = data.metadata.get("test_mode", "")
                if test_mode and test_mode != "oscillation":
                    warnings.append(
                        f"SPP is for oscillatory (LAOS) data, got {test_mode}"
                    )

        if warnings:
            logger.debug(
                "Validation warnings found",
                transform=name,
                warning_count=len(warnings),
            )
        logger.debug("Exiting validate_transform_input", transform=name)
        return warnings
