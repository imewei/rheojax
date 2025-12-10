"""
Transform Service
================

Service for applying rheological transforms (mastercurve, FFT, SRFS, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.transforms import (
    SRFS,
    FFTAnalysis,
    Mastercurve,
    MutationNumber,
    OWChirp,
    SmoothDerivative,
    SPPDecomposer,
)

logger = logging.getLogger(__name__)


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
        self._transforms = {
            "mastercurve": Mastercurve,
            "fft": FFTAnalysis,
            "srfs": SRFS,
            "owchirp": OWChirp,
            "derivative": SmoothDerivative,
            "mutation_number": MutationNumber,
            "spp": SPPDecomposer,
        }

    def get_available_transforms(self) -> list[str]:
        """Get list of available transforms.

        Returns
        -------
        list[str]
            Transform names
        """
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
        params_map = {
            "mastercurve": {
                "reference_temp": {
                    "type": "float",
                    "default": 25.0,
                    "description": "Reference temperature (C)",
                },
                "auto_shift": {
                    "type": "bool",
                    "default": True,
                    "description": "Automatically calculate shift factors",
                },
            },
            "fft": {
                "direction": {
                    "type": "choice",
                    "choices": ["forward", "inverse"],
                    "default": "forward",
                    "description": "Transform direction",
                },
                "window": {
                    "type": "choice",
                    "choices": ["hann", "hamming", "blackman", "bartlett", "rectangular"],
                    "default": "hann",
                    "description": "Window function (rectangular = none)",
                },
                "detrend": {
                    "type": "bool",
                    "default": True,
                    "description": "Remove linear trend before FFT",
                },
                "return_psd": {
                    "type": "bool",
                    "default": False,
                    "description": "Return power spectral density instead of magnitude",
                },
                "normalize": {
                    "type": "bool",
                    "default": True,
                    "description": "Normalize FFT output",
                },
            },
            "srfs": {
                "reference_gamma_dot": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Reference shear rate (1/s)",
                },
                "auto_shift": {
                    "type": "bool",
                    "default": True,
                    "description": "Auto-calculate shift factors",
                },
            },
            "owchirp": {
                "min_frequency": {
                    "type": "float",
                    "default": 0.01,
                    "description": "Minimum frequency (rad/s)",
                },
                "max_frequency": {
                    "type": "float",
                    "default": 100.0,
                    "description": "Maximum frequency (rad/s)",
                },
                "n_frequencies": {
                    "type": "int",
                    "default": 100,
                    "description": "Number of frequency points",
                },
                "wavelet_width": {
                    "type": "float",
                    "default": 5.0,
                    "description": "Width parameter for wavelet",
                },
                "extract_harmonics": {
                    "type": "bool",
                    "default": True,
                    "description": "Extract harmonic components",
                },
                "max_harmonic": {
                    "type": "int",
                    "default": 7,
                    "description": "Maximum harmonic to extract",
                },
            },
            "derivative": {
                "order": {
                    "type": "int",
                    "default": 1,
                    "description": "Derivative order",
                },
                "window_length": {
                    "type": "int",
                    "default": 11,
                    "description": "Savitzky-Golay window length (must be odd)",
                },
                "poly_order": {
                    "type": "int",
                    "default": 3,
                    "description": "Polynomial order for Savitzky-Golay (must be < window_length)",
                },
                "method": {
                    "type": "choice",
                    "choices": ["savgol", "finite_diff", "spline", "total_variation"],
                    "default": "savgol",
                    "description": "Differentiation method",
                },
                "smooth_before": {
                    "type": "bool",
                    "default": False,
                    "description": "Apply smoothing before differentiation",
                },
                "smooth_after": {
                    "type": "bool",
                    "default": False,
                    "description": "Apply smoothing after differentiation",
                },
                "validate_window": {
                    "type": "bool",
                    "default": True,
                    "description": "Force odd window length",
                },
            },
            "mutation_number": {
                "integration_method": {
                    "type": "choice",
                    "choices": ["trapz", "simpson", "romberg"],
                    "default": "trapz",
                    "description": "Numerical integration method",
                },
                "extrapolate": {
                    "type": "bool",
                    "default": False,
                    "description": "Extrapolate data outside measured range",
                },
                "extrapolation_model": {
                    "type": "choice",
                    "choices": ["exponential", "power_law", "linear"],
                    "default": "exponential",
                    "description": "Model for data extrapolation",
                },
            },
            "spp": {
                "omega": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Angular frequency (rad/s)",
                },
                "gamma_0": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Strain amplitude (dimensionless)",
                },
                "n_harmonics": {
                    "type": "int",
                    "default": 39,
                    "description": "Number of harmonics to extract",
                },
                "yield_tolerance": {
                    "type": "float",
                    "default": 0.02,
                    "description": "Tolerance for yield point detection",
                },
                "start_cycle": {
                    "type": "int",
                    "default": 0,
                    "description": "First cycle to analyze (skip transients)",
                },
                "use_numerical_method": {
                    "type": "bool",
                    "default": False,
                    "description": "Use numerical differentiation (MATLAB-compatible)",
                },
            },
        }

        return params_map.get(name, {})

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
        try:
            if name not in self._transforms:
                raise ValueError(f"Unknown transform: {name}")

            def _with_provenance(result_data: RheoData) -> RheoData:
                # Attach a simple provenance entry
                prov_entry = {
                    "transform": name,
                    "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
                }
                history = list(result_data.metadata.get("provenance", []))
                history.append(prov_entry)
                result_data.metadata["provenance"] = history
                return result_data

            if name == "mastercurve":
                # Mastercurve requires multiple datasets
                if not isinstance(data, list):
                    raise ValueError("Mastercurve requires list of datasets")

                reference_temp = params.get("reference_temp", 25.0)
                auto_shift = params.get("auto_shift", True)

                mc = Mastercurve(reference_temp=reference_temp, auto_shift=auto_shift)
                mastercurve_data, shift_factors = mc.transform(data)

                logger.info(f"Applied mastercurve at T_ref={reference_temp}°C")
                return _with_provenance(mastercurve_data), {"shift_factors": shift_factors}

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

                logger.info(f"Applied FFT transform ({direction})")
                return _with_provenance(result)

            elif name == "srfs":
                # SRFS transform
                if not isinstance(data, list):
                    raise ValueError("SRFS requires list of datasets")

                reference_gamma_dot = params.get("reference_gamma_dot", 1.0)
                auto_shift = params.get("auto_shift", True)

                srfs = SRFS(reference_gamma_dot=reference_gamma_dot, auto_shift=auto_shift)
                master_curve, shift_factors = srfs.transform(data)

                logger.info(f"Applied SRFS at γ̇_ref={reference_gamma_dot} 1/s")
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

                owchirp = OWChirp(
                    n_frequencies=n_frequencies,
                    frequency_range=(min_freq, max_freq),
                    wavelet_width=wavelet_width,
                    extract_harmonics=extract_harmonics,
                    max_harmonic=max_harmonic,
                )
                result = owchirp.transform(data)

                logger.info("Applied OWChirp transform")
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

                deriv = SmoothDerivative(
                    method=method,
                    window_length=window_length,
                    polyorder=polyorder,
                    deriv=deriv_order,
                    smooth_before=smooth_before,
                    smooth_after=smooth_after,
                )
                result = deriv.transform(data)

                logger.info(f"Applied {deriv_order}-order derivative")
                return _with_provenance(result)

            elif name == "mutation_number":
                # Mutation number
                # Note: MutationNumber doesn't take reference_frequency as __init__ param
                # It uses integration_method, extrapolate, extrapolation_model
                integration_method = params.get("integration_method", "trapz")
                extrapolate = params.get("extrapolate", False)
                extrapolation_model = params.get("extrapolation_model", "exponential")

                mn = MutationNumber(
                    integration_method=integration_method,
                    extrapolate=extrapolate,
                    extrapolation_model=extrapolation_model,
                )
                result = mn.transform(data)

                logger.info("Calculated mutation number")
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
                end_cycle = None if end_cycle_raw in (None, 0, 0.0) else int(end_cycle_raw)
                use_numerical_method = params.get("use_numerical_method", False)

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
                    f"SPP analysis: σ_sy={spp_results['sigma_sy']:.2f} Pa, "
                    f"σ_dy={spp_results['sigma_dy']:.2f} Pa"
                )
                return _with_provenance(result), {"spp_results": spp_results}

            else:
                raise ValueError(f"Transform {name} not implemented")

        except Exception as e:
            logger.error(f"Transform {name} failed: {e}")
            raise RuntimeError(f"Transform failed: {e}") from e

    def preview_transform(
        self,
        name: str,
        data: RheoData | list[RheoData],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Preview transform result without applying.

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
            Preview information (statistics, expected output shape, etc.)
        """
        try:
            preview = {
                "transform": name,
                "input_type": "multiple" if isinstance(data, list) else "single",
                "params": params,
            }

            if isinstance(data, list):
                preview["n_datasets"] = len(data)
                preview["total_points"] = sum(len(d.x) for d in data)
            else:
                preview["n_points"] = len(data.x)
                preview["x_range"] = (float(np.min(data.x)), float(np.max(data.x)))

            # Add transform-specific preview info
            if name == "mastercurve":
                if isinstance(data, list):
                    temps = [d.metadata.get("temperature") for d in data]
                    preview["temperatures"] = temps

            elif name == "fft":
                if not isinstance(data, list):
                    preview["expected_output_domain"] = (
                        "frequency" if data.domain == "time" else "time"
                    )

            return preview

        except Exception as e:
            logger.error(f"Preview failed: {e}")
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
        warnings = []

        # Check if transform requires multiple datasets
        multi_dataset_transforms = ["mastercurve", "srfs"]
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

        return warnings
