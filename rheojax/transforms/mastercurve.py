"""Time-Temperature Superposition (TTS) mastercurve generation.

This module implements time-temperature superposition for creating mastercurves
from multi-temperature rheological data using WLF or Arrhenius shift factors,
or automatic shift factor calculation via power-law intersection (pyvisco algorithm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import TransformRegistry
from rheojax.logging import get_logger, log_transform
from rheojax.utils.optimization import create_least_squares_objective, nlsq_optimize

# Module logger
logger = get_logger(__name__)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    import jax.numpy as jnp_typing
else:  # pragma: no cover - typing fallback
    jnp_typing = np

type JaxArray = jnp_typing.ndarray
type ScalarOrArray = float | JaxArray


ShiftMethod = Literal["wlf", "arrhenius", "manual"]


@TransformRegistry.register("mastercurve")
class Mastercurve(BaseTransform):
    """Time-Temperature Superposition (TTS) mastercurve generation.

    This transform applies time-temperature superposition to create mastercurves
    from multi-temperature rheological data. Supports both WLF and Arrhenius
    shift factor models for horizontal shifting, with optional vertical shifting,
    or automatic shift factor calculation via power-law intersection method.

    The WLF equation is:
        log(a_T) = -C1 * (T - T_ref) / (C2 + (T - T_ref))

    The Arrhenius equation is:
        log(a_T) = (E_a / R) * (1/T - 1/T_ref)

    Automatic shift factors use power-law intersection (pyvisco algorithm):
        Fits each curve to y = a*x^b + e, then computes shift from intersection

    Parameters
    ----------
    reference_temp : float, default=298.15
        Reference temperature in Kelvin
    method : ShiftMethod, default='wlf'
        Shift factor method: 'wlf', 'arrhenius', or 'manual'
    C1 : float, default=17.44
        WLF parameter C1 (universal value for polymers)
    C2 : float, default=51.6
        WLF parameter C2 in Kelvin (universal value)
    E_a : float, optional
        Activation energy for Arrhenius (J/mol)
    vertical_shift : bool, default=False
        Whether to apply vertical shifting (for modulus scaling)
    optimize_shifts : bool, default=True
        Whether to optimize shift factors to minimize overlap error
    auto_shift : bool, default=False
        Whether to use automatic shift factor calculation via power-law intersection.
        If True, overrides manual WLF/Arrhenius calculations.

    Examples
    --------
    >>> from rheojax.core.data import RheoData
    >>> from rheojax.transforms.mastercurve import Mastercurve
    >>>
    >>> # Create multi-temperature frequency sweep data
    >>> # (In practice, this would come from experimental measurements)
    >>> temps = [273, 298, 323]  # K
    >>> freq = jnp.logspace(-2, 2, 50)
    >>> datasets = []
    >>> for T in temps:
    ...     G_prime = some_modulus_function(freq, T)
    ...     data = RheoData(x=freq, y=G_prime, domain='frequency',
    ...                     metadata={'temperature': T})
    ...     datasets.append(data)
    >>>
    >>> # Create mastercurve at reference temperature (two equivalent APIs)
    >>> mc = Mastercurve(reference_temp=298.15, method='wlf')
    >>>
    >>> # Option 1: Using create_mastercurve (explicit)
    >>> mastercurve = mc.create_mastercurve(datasets)
    >>>
    >>> # Option 2: Using transform with list (returns shift factors too)
    >>> mastercurve, shift_factors = mc.transform(datasets)
    >>> print(shift_factors)  # {273.0: 42.5, 298.15: 1.0, 323.0: 0.024}
    >>>
    >>> # Option 3: Automatic shift factor calculation
    >>> mc_auto = Mastercurve(reference_temp=298.15, auto_shift=True)
    >>> mastercurve_auto, shifts_auto = mc_auto.transform(datasets)
    """

    def __init__(
        self,
        reference_temp: float = 298.15,
        method: ShiftMethod = "wlf",
        C1: float = 17.44,
        C2: float = 51.6,
        E_a: float | None = None,
        vertical_shift: bool = False,
        optimize_shifts: bool = True,
        auto_shift: bool = False,
    ):
        """Initialize Mastercurve transform.

        Parameters
        ----------
        reference_temp : float
            Reference temperature in Kelvin
        method : ShiftMethod
            Shift factor method
        C1 : float
            WLF parameter C1
        C2 : float
            WLF parameter C2 (Kelvin)
        E_a : float, optional
            Activation energy for Arrhenius (J/mol)
        vertical_shift : bool
            Apply vertical shifting
        optimize_shifts : bool
            Optimize shift factors
        auto_shift : bool
            Use automatic power-law intersection for shift calculation
        """
        super().__init__()
        self.T_ref = reference_temp
        self.method = method
        self.C1 = C1
        self.C2 = C2
        self.E_a = E_a
        self.vertical_shift = vertical_shift
        self.optimize_shifts = optimize_shifts
        self._auto_shift = auto_shift

        # Store computed shift factors
        self.shift_factors_: dict[float, float] | None = None
        self.vertical_shifts_: dict[float, float] | None = None
        self._auto_shift_factors: np.ndarray | None = None

    def _calculate_wlf_shift(
        self, T: ScalarOrArray, T_ref: float, C1: float, C2: float
    ) -> ScalarOrArray:
        """Calculate WLF shift factor.

        Parameters
        ----------
        T : float or jnp.ndarray
            Temperature(s) in Kelvin
        T_ref : float
            Reference temperature in Kelvin
        C1 : float
            WLF parameter C1
        C2 : float
            WLF parameter C2 (Kelvin)

        Returns
        -------
        float or jnp.ndarray
            Shift factor a_T
        """
        # WLF equation: log(a_T) = -C1(T-T_ref)/(C2+(T-T_ref))
        log_aT = -C1 * (T - T_ref) / (C2 + (T - T_ref))
        return jnp.power(10.0, log_aT)

    def _calculate_arrhenius_shift(
        self, T: ScalarOrArray, T_ref: float, E_a: float
    ) -> ScalarOrArray:
        """Calculate Arrhenius shift factor.

        Parameters
        ----------
        T : float or jnp.ndarray
            Temperature(s) in Kelvin
        T_ref : float
            Reference temperature in Kelvin
        E_a : float
            Activation energy (J/mol)

        Returns
        -------
        float or jnp.ndarray
            Shift factor a_T
        """
        R = 8.314  # Gas constant (J/(mol·K))

        # Arrhenius: log(a_T) = (E_a/R) * (1/T - 1/T_ref)
        log_aT = (E_a / R) * (1.0 / T - 1.0 / T_ref)
        return jnp.exp(log_aT)

    def _fit_power_law(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit power-law model: y = a*x^b + e using NLSQ.

        Parameters
        ----------
        x : ndarray
            X data (frequency or time)
        y : ndarray
            Y data (modulus or other response)

        Returns
        -------
        popt : ndarray
            Optimal parameters [a, b, e]
        perr : ndarray
            Parameter uncertainties (standard errors)
        """
        logger.debug(
            "Fitting power-law model",
            n_points=len(x),
            x_range=(float(x.min()), float(x.max())),
        )

        # Create parameter set with reasonable bounds
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(1e-10, 1e10))
        params.add("b", value=-0.5, bounds=(-5.0, 5.0))
        params.add("e", value=0.0, bounds=(-1e10, 1e10))

        # Define model function for power-law
        def power_law_model(x_data: np.ndarray, param_values: np.ndarray) -> np.ndarray:
            """Power-law model: y = a*x^b + e."""
            a, b, e = param_values
            return a * jnp.power(x_data, b) + e

        # Create least-squares objective
        objective = create_least_squares_objective(power_law_model, x, y)

        # Optimize using NLSQ
        result = nlsq_optimize(
            objective, params, use_jax=True, max_iter=1000, ftol=1e-8, xtol=1e-8
        )

        # Extract parameter uncertainties from Jacobian
        if result.jac is not None:
            # Estimate covariance from Jacobian: Cov ≈ (J^T J)^-1
            try:
                jtj = result.jac.T @ result.jac
                cov = np.linalg.inv(jtj)
                perr = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                # Singular matrix, use large uncertainties
                logger.debug("Jacobian singular, using large uncertainties")
                perr = np.full(3, 1e6)
        else:
            perr = np.full(3, 1e6)

        logger.debug(
            "Power-law fit completed",
            a=float(result.x[0]),
            b=float(result.x[1]),
            e=float(result.x[2]),
        )

        return result.x, perr

    def _detect_outliers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        popt_full: np.ndarray,
        perr_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect and remove outliers (first point) if it improves fit.

        Following pyvisco algorithm: try removing first point, keep removal
        if exponent error improves.

        Parameters
        ----------
        x : ndarray
            X data
        y : ndarray
            Y data
        popt_full : ndarray
            Parameters from full fit [a, b, e]
        perr_full : ndarray
            Uncertainties from full fit

        Returns
        -------
        x_clean : ndarray
            Cleaned x data
        y_clean : ndarray
            Cleaned y data
        popt_clean : ndarray
            Parameters from cleaned fit
        perr_clean : ndarray
            Uncertainties from cleaned fit
        """
        # Try fit without first point
        x_no_first = x[1:]
        y_no_first = y[1:]

        popt_no_first, perr_no_first = self._fit_power_law(x_no_first, y_no_first)

        # Compare exponent (b) uncertainty
        if perr_no_first[1] < perr_full[1]:
            # Removing first point improves fit
            return x_no_first, y_no_first, popt_no_first, perr_no_first
        else:
            # Keep all points
            return x, y, popt_full, perr_full

    def _compute_pairwise_shift(
        self,
        curve_top: np.ndarray,
        curve_bot: np.ndarray,
        popt_top: np.ndarray,
        popt_bot: np.ndarray,
    ) -> float:
        """Compute shift factor between two adjacent curves via intersection.

        Uses power-law intersection method: sample points in overlap/gap region,
        compute inverse power-law to find x-shift, average log(aT).

        Parameters
        ----------
        curve_top : ndarray
            Top curve data [x, y] with shape (N, 2)
        curve_bot : ndarray
            Bottom curve data [x, y] with shape (M, 2)
        popt_top : ndarray
            Power-law parameters [a, b, e] for top curve
        popt_bot : ndarray
            Power-law parameters [a, b, e] for bottom curve

        Returns
        -------
        log_aT : float
            Log10 shift factor
        """
        # Extract x and y ranges
        x_top, y_top = curve_top[:, 0], curve_top[:, 1]
        x_bot, y_bot = curve_bot[:, 0], curve_bot[:, 1]

        # Determine overlap or gap
        y_min_top, y_max_top = y_top.min(), y_top.max()
        y_min_bot, y_max_bot = y_bot.min(), y_bot.max()

        # Find y-range for intersection sampling
        if y_min_top < y_max_bot and y_max_bot < y_max_top:
            # Overlap case
            y_sample_min = max(y_min_top, y_min_bot)
            y_sample_max = min(y_max_top, y_max_bot)
        else:
            # Gap case: sample in gap region
            y_sample_min = min(y_max_bot, y_max_top)
            y_sample_max = max(y_min_bot, y_min_top)

        # Sample 10 points in y-range
        y_samples = np.linspace(y_sample_min, y_sample_max, 10)

        # Compute x from inverse power-law: x = ((y - e) / a)^(1/b)
        a_top, b_top, e_top = popt_top
        a_bot, b_bot, e_bot = popt_bot

        # Inverse power-law for top curve
        x_top_inv = np.power((y_samples - e_top) / a_top, 1.0 / b_top)

        # Inverse power-law for bottom curve
        x_bot_inv = np.power((y_samples - e_bot) / a_bot, 1.0 / b_bot)

        # Compute log shift factors and average
        log_shift_factors = np.log10(x_top_inv / x_bot_inv)

        # Handle infinities and NaNs
        valid = np.isfinite(log_shift_factors)
        if np.sum(valid) == 0:
            # Fallback: use geometric mean of x-ranges
            return float(np.log10(np.mean(x_top) / np.mean(x_bot)))

        log_aT = float(np.mean(log_shift_factors[valid]))

        return log_aT

    def _compute_auto_shift_factors(
        self, datasets: list[RheoData], ref_temp_idx: int
    ) -> np.ndarray:
        """Compute automatic shift factors via sequential pairwise power-law intersection.

        Follows pyvisco algorithm:
        1. Fit power-law to each temperature curve
        2. Detect and remove outliers
        3. Compute pairwise shifts via intersection
        4. Accumulate shifts sequentially from reference temperature

        Parameters
        ----------
        datasets : list of RheoData
            Multi-temperature datasets (must be sorted by temperature)
        ref_temp_idx : int
            Index of reference temperature in datasets

        Returns
        -------
        log_aT_array : ndarray
            Cumulative log10 shift factors for all temperatures
        """
        n_temps = len(datasets)
        logger.debug(
            "Computing automatic shift factors",
            n_temperatures=n_temps,
            ref_temp_idx=ref_temp_idx,
        )

        log_aT_array = np.zeros(n_temps)

        # Fit power-law to each curve with outlier detection
        power_law_params = []
        curves = []

        for i, data in enumerate(datasets):
            x = np.asarray(data.x, dtype=np.float64)
            y = np.asarray(data.y, dtype=np.float64)

            logger.debug(
                "Fitting power-law for temperature curve",
                curve_index=i,
                n_points=len(x),
            )

            # Fit power-law
            popt, perr = self._fit_power_law(x, y)

            # Detect outliers
            x_clean, y_clean, popt_clean, perr_clean = self._detect_outliers(
                x, y, popt, perr
            )

            power_law_params.append(popt_clean)
            curves.append(np.column_stack([x_clean, y_clean]))

        # Sequential cumulative shifting below reference temperature
        for i in range(ref_temp_idx - 1, -1, -1):
            # Shift from i+1 to i (going down in temperature)
            log_shift = self._compute_pairwise_shift(
                curves[i + 1], curves[i], power_law_params[i + 1], power_law_params[i]
            )
            log_aT_array[i] = log_aT_array[i + 1] + log_shift
            logger.debug(
                "Pairwise shift computed",
                from_idx=i + 1,
                to_idx=i,
                log_shift=float(log_shift),
            )

        # Sequential cumulative shifting above reference temperature
        for i in range(ref_temp_idx + 1, n_temps):
            # Shift from i-1 to i (going up in temperature)
            log_shift = self._compute_pairwise_shift(
                curves[i - 1], curves[i], power_law_params[i - 1], power_law_params[i]
            )
            log_aT_array[i] = log_aT_array[i - 1] + log_shift
            logger.debug(
                "Pairwise shift computed",
                from_idx=i - 1,
                to_idx=i,
                log_shift=float(log_shift),
            )

        # Store for later retrieval
        self._auto_shift_factors = log_aT_array

        logger.debug(
            "Auto shift factors computed",
            shift_factors=log_aT_array.tolist(),
        )

        return log_aT_array

    def get_auto_shift_factors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get automatic shift factors as arrays for plotting.

        Returns
        -------
        temperatures : ndarray or None
            Array of temperatures in Kelvin, or None if not computed
        log_aT : ndarray or None
            Array of log10 shift factors, or None if not computed
        """
        if self._auto_shift_factors is None:
            return None

        if self.shift_factors_ is None:
            return None

        temps = np.array(sorted(self.shift_factors_.keys()))
        return temps, self._auto_shift_factors

    def get_shift_factor(self, T: float) -> float:
        """Get shift factor for a given temperature.

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            Horizontal shift factor a_T
        """
        if self.method == "wlf":
            return float(self._calculate_wlf_shift(T, self.T_ref, self.C1, self.C2))
        elif self.method == "arrhenius":
            if self.E_a is None:
                raise ValueError("E_a must be provided for Arrhenius method")
            return float(self._calculate_arrhenius_shift(T, self.T_ref, self.E_a))
        elif self.method == "manual":
            if self.shift_factors_ is None:
                raise ValueError("Manual shift factors not set")
            return self.shift_factors_.get(T, 1.0)
        else:
            raise ValueError(f"Unknown shift method: {self.method}")

    def set_manual_shifts(self, shift_factors: dict[float, float]):
        """Set manual shift factors for each temperature.

        Parameters
        ----------
        shift_factors : dict
            Dictionary mapping temperature (K) to shift factor
        """
        self.method = "manual"
        self.shift_factors_ = shift_factors

    def get_wlf_parameters(self) -> dict[str, float]:
        """Get WLF parameters.

        Returns
        -------
        dict
            Dictionary with keys 'C1', 'C2', and 'T_ref' (reference temperature)

        Raises
        ------
        ValueError
            If method is not 'wlf'
        """
        if self.method != "wlf":
            raise ValueError(f"WLF parameters not available for method '{self.method}'")

        return {
            "C1": self.C1,
            "C2": self.C2,
            "T_ref": self.T_ref,
        }

    def get_arrhenius_parameters(self) -> dict[str, float]:
        """Get Arrhenius parameters.

        Returns
        -------
        dict
            Dictionary with keys 'E_a' (activation energy) and 'T_ref' (reference temperature)

        Raises
        ------
        ValueError
            If method is not 'arrhenius' or E_a is not set
        """
        if self.method != "arrhenius":
            raise ValueError(
                f"Arrhenius parameters not available for method '{self.method}'"
            )

        if self.E_a is None:
            raise ValueError("E_a (activation energy) not set")

        return {
            "E_a": self.E_a,
            "T_ref": self.T_ref,
        }

    def get_shift_factors_array(
        self, temperatures: list[float] | np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get shift factors as arrays for plotting and analysis.

        Parameters
        ----------
        temperatures : list or ndarray, optional
            Temperatures in Kelvin. If None, uses temperatures from the last
            mastercurve creation (stored in ``shift_factors_``).

        Returns
        -------
        temperatures : ndarray
            Array of temperatures in Kelvin (sorted)
        shift_factors : ndarray
            Array of shift factors corresponding to temperatures

        Raises
        ------
        ValueError
            If temperatures is None and no shift factors have been computed

        Examples
        --------
        >>> mc = Mastercurve(reference_temp=298.15, method='wlf')
        >>> temps, shifts = mc.get_shift_factors_array([273.15, 298.15, 323.15])
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(temps - 273.15, np.log10(shifts))
        """
        if temperatures is None:
            # Use stored shift factors from last mastercurve creation
            if self.shift_factors_ is None:
                raise ValueError(
                    "No shift factors available. Either provide temperatures or "
                    "create a mastercurve first."
                )

            # Extract from stored shift factors
            temps_array = np.array(sorted(self.shift_factors_.keys()))
            shifts_array = np.array([self.shift_factors_[T] for T in temps_array])

        else:
            # Calculate shift factors for provided temperatures
            temps_array = np.array(temperatures)

            # Sort by temperature
            sort_idx = np.argsort(temps_array)
            temps_array = temps_array[sort_idx]

            # Calculate shift factors
            shifts_array = np.array(
                [self.get_shift_factor(float(T)) for T in temps_array]
            )

        return temps_array, shifts_array

    def _transform_single(self, data: RheoData) -> RheoData:
        """Apply horizontal shift to single-temperature data.

        Parameters
        ----------
        data : RheoData
            Single-temperature data to shift

        Returns
        -------
        RheoData
            Shifted data

        Raises
        ------
        ValueError
            If temperature metadata is missing
        """
        # Get temperature from metadata
        if "temperature" not in data.metadata:
            raise ValueError("Temperature must be in metadata for mastercurve shifting")

        T = data.metadata["temperature"]

        # Get shift factor
        a_T = self.get_shift_factor(T)

        # Apply horizontal shift (frequency or time shift)
        x_shifted = data.x * a_T  # type: ignore[operator]

        # Apply vertical shift if requested
        y_shifted = data.y
        if self.vertical_shift:
            # For temperature-dependent modulus: G(T) ~ rho(T) * T
            # Vertical shift factor: b_T = rho(T) * T / (rho(T_ref) * T_ref)
            # Simplified: b_T = T / T_ref
            b_T = T / self.T_ref
            y_shifted = y_shifted * b_T

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update(
            {
                "transform": "mastercurve",
                "reference_temperature": self.T_ref,
                "shift_method": self.method,
                "horizontal_shift": float(a_T),
                "vertical_shift": float(T / self.T_ref) if self.vertical_shift else 1.0,
            }
        )

        return RheoData(
            x=x_shifted,
            y=y_shifted,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata=new_metadata,
            validate=False,
        )

    def _transform(
        self, data: RheoData | list[RheoData]
    ) -> RheoData | tuple[RheoData, dict[float, float]]:
        """Apply horizontal shift to single-temperature data or create mastercurve.

        Parameters
        ----------
        data : RheoData or list of RheoData
            Single-temperature data to shift, or list of datasets for mastercurve

        Returns
        -------
        RheoData or tuple of (RheoData, dict)
            If data is a single RheoData: returns shifted data
            If data is a list: returns (mastercurve, ``shift_factors``)

        Raises
        ------
        ValueError
            If temperature metadata is missing
        """
        # Determine input shape for logging
        if isinstance(data, list):
            input_shape = (len(data),)
        else:
            input_shape = (len(data.x),) if hasattr(data.x, "__len__") else (1,)

        with log_transform(
            logger,
            "mastercurve",
            input_shape=input_shape,
            method=self.method,
            reference_temp=self.T_ref,
            auto_shift=self._auto_shift,
        ) as ctx:
            # Handle list of datasets (create mastercurve)
            if isinstance(data, list):
                result = self.create_mastercurve(data, return_shifts=True)
                if isinstance(result, tuple):
                    mastercurve, shift_factors = result
                    ctx["output_shape"] = (len(mastercurve.x),)
                    ctx["n_temperatures"] = len(shift_factors)
                return result  # type: ignore[return-value]

            # Handle single dataset
            result = self._transform_single(data)
            ctx["output_shape"] = (len(result.x),)
            return result

    def create_mastercurve(
        self, datasets: list[RheoData], merge: bool = True, return_shifts: bool = False
    ) -> RheoData | list[RheoData] | tuple[RheoData, dict[float, float]]:
        """Create mastercurve from multiple temperature datasets.

        Parameters
        ----------
        datasets : list of RheoData
            List of datasets at different temperatures
        merge : bool, default=True
            If True, merge all shifted data into single RheoData.
            If False, return list of shifted datasets.
        return_shifts : bool, default=False
            If True, return tuple of (mastercurve, shift_factors).
            Only valid when merge=True.

        Returns
        -------
        RheoData or list of RheoData or tuple
            If merge=True and return_shifts=False: RheoData
            If merge=False: list of RheoData
            If merge=True and return_shifts=True: (RheoData, dict of shift factors)

        Raises
        ------
        ValueError
            If datasets don't have temperature metadata or if return_shifts=True with merge=False
        """
        from rheojax.core.data import RheoData

        logger.debug(
            "Creating mastercurve",
            n_datasets=len(datasets),
            merge=merge,
            return_shifts=return_shifts,
        )

        if return_shifts and not merge:
            logger.error(
                "Invalid configuration: return_shifts=True requires merge=True"
            )
            raise ValueError("return_shifts=True requires merge=True")

        # Extract temperatures and sort datasets
        temperatures = []
        for data in datasets:
            if "temperature" not in data.metadata:
                raise ValueError("All datasets must have 'temperature' in metadata")
            temperatures.append(data.metadata["temperature"])

        # Sort by temperature
        temp_indices = np.argsort(temperatures)
        datasets = [datasets[i] for i in temp_indices]
        temperatures = [temperatures[i] for i in temp_indices]

        # Find reference temperature index
        ref_temp_idx = np.argmin(np.abs(np.array(temperatures) - self.T_ref))

        # Compute shift factors
        if self._auto_shift:
            # Use automatic shift factor calculation
            log_aT_array = self._compute_auto_shift_factors(datasets, ref_temp_idx)
            shift_factors = {
                T: 10.0**log_aT
                for T, log_aT in zip(temperatures, log_aT_array, strict=False)
            }
        else:
            # Use manual WLF/Arrhenius/manual method
            shift_factors = {}
            for T in temperatures:
                a_T = self.get_shift_factor(T)
                shift_factors[T] = float(a_T)

        # Shift all datasets
        shifted_datasets = []

        for data, T in zip(datasets, temperatures, strict=False):
            # Get shift factor
            a_T = shift_factors[T]

            # Apply horizontal shift
            x_shifted = data.x * a_T

            # Apply vertical shift if requested
            y_shifted = data.y
            if self.vertical_shift:
                b_T = T / self.T_ref
                y_shifted = y_shifted * b_T

            # Create metadata
            new_metadata = data.metadata.copy()
            new_metadata.update(
                {
                    "transform": "mastercurve",
                    "reference_temperature": self.T_ref,
                    "shift_method": "auto" if self._auto_shift else self.method,
                    "horizontal_shift": float(a_T),
                    "vertical_shift": (
                        float(T / self.T_ref) if self.vertical_shift else 1.0
                    ),
                }
            )

            shifted = RheoData(
                x=x_shifted,
                y=y_shifted,
                x_units=data.x_units,
                y_units=data.y_units,
                domain=data.domain,
                metadata=new_metadata,
                validate=False,
            )
            shifted_datasets.append(shifted)

        # If not merging, return list
        if not merge:
            return shifted_datasets

        # Merge all shifted data
        all_x = []
        all_y = []
        all_temps = []

        for data, T in zip(shifted_datasets, temperatures, strict=False):
            x_data = data.x if isinstance(data.x, np.ndarray) else np.array(data.x)
            y_data = data.y if isinstance(data.y, np.ndarray) else np.array(data.y)

            all_x.append(x_data)
            all_y.append(y_data)
            all_temps.extend([T] * len(x_data))

        # Concatenate
        merged_x = np.concatenate(all_x)
        merged_y = np.concatenate(all_y)
        merged_temps = np.array(all_temps)

        # Sort by x-axis
        sort_idx = np.argsort(merged_x)
        merged_x = merged_x[sort_idx]
        merged_y = merged_y[sort_idx]
        merged_temps = merged_temps[sort_idx]

        # Create merged metadata
        merged_metadata = {
            "transform": "mastercurve",
            "reference_temperature": self.T_ref,
            "shift_method": "auto" if self._auto_shift else self.method,
            "temperatures": temperatures,
            "n_datasets": len(datasets),
            "source_temperatures": merged_temps,
            "shift_factors": shift_factors,
        }

        mastercurve = RheoData(
            x=merged_x,
            y=merged_y,
            x_units=datasets[0].x_units if datasets else None,
            y_units=datasets[0].y_units if datasets else None,
            domain=datasets[0].domain if datasets else "frequency",
            metadata=merged_metadata,
            validate=False,
        )

        # Store shift factors for later retrieval
        self.shift_factors_ = shift_factors

        if return_shifts:
            return mastercurve, shift_factors
        return mastercurve

    def compute_overlap_error(self, datasets: list[RheoData]) -> float:
        """Compute overlap error for multi-temperature data.

        This metric quantifies how well the datasets collapse onto a
        mastercurve. Lower values indicate better superposition.

        Parameters
        ----------
        datasets : list of RheoData
            List of datasets at different temperatures

        Returns
        -------
        float
            Overlap error (normalized RMSE in overlap regions)
        """
        # Create shifted datasets
        shifted_datasets = self.create_mastercurve(datasets, merge=False)

        if not isinstance(shifted_datasets, list):
            shifted_datasets = [shifted_datasets]  # type: ignore[list-item]

        # Find overlapping regions and compute RMSE
        total_error = 0.0
        n_overlaps = 0

        for i in range(len(shifted_datasets)):
            for j in range(i + 1, len(shifted_datasets)):
                data_i = shifted_datasets[i]
                data_j = shifted_datasets[j]

                # Find overlap region
                x_i = (
                    data_i.x if isinstance(data_i.x, np.ndarray) else np.array(data_i.x)
                )
                x_j = (
                    data_j.x if isinstance(data_j.x, np.ndarray) else np.array(data_j.x)
                )

                x_min = max(x_i.min(), x_j.min())
                x_max = min(x_i.max(), x_j.max())

                if x_max <= x_min:
                    continue  # No overlap

                # Interpolate both datasets to common x-axis in overlap region
                x_common = np.linspace(x_min, x_max, 50)

                y_i_interp = np.interp(x_common, x_i, data_i.y)
                y_j_interp = np.interp(x_common, x_j, data_j.y)

                # Compute RMSE
                error = np.sqrt(np.mean((y_i_interp - y_j_interp) ** 2))
                total_error += error
                n_overlaps += 1

        if n_overlaps == 0:
            return float("inf")

        return total_error / n_overlaps

    def optimize_wlf_parameters(
        self,
        datasets: list[RheoData],
        initial_C1: float = 17.44,
        initial_C2: float = 51.6,
    ) -> tuple[float, float]:
        """Optimize WLF parameters to minimize overlap error.

        Parameters
        ----------
        datasets : list of RheoData
            Multi-temperature datasets
        initial_C1 : float
            Initial guess for C1
        initial_C2 : float
            Initial guess for C2

        Returns
        -------
        C1_opt : float
            Optimized C1 parameter
        C2_opt : float
            Optimized C2 parameter
        """
        from scipy.optimize import minimize

        def objective(params):
            """Objective function: overlap error."""
            C1, C2 = params
            self.C1 = C1
            self.C2 = C2
            return self.compute_overlap_error(datasets)

        # Optimize
        result = minimize(
            objective,
            x0=[initial_C1, initial_C2],
            method="Nelder-Mead",
            bounds=[(5, 50), (20, 200)],
        )

        C1_opt, C2_opt = result.x
        self.C1 = C1_opt
        self.C2 = C2_opt

        return C1_opt, C2_opt


__all__ = ["Mastercurve"]
