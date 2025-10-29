"""Time-Temperature Superposition (TTS) mastercurve generation.

This module implements time-temperature superposition for creating mastercurves
from multi-temperature rheological data using WLF or Arrhenius shift factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np

from rheo.core.base import BaseTransform
from rheo.core.registry import TransformRegistry

if TYPE_CHECKING:
    from rheo.core.data import RheoData


ShiftMethod = Literal["wlf", "arrhenius", "manual"]


@TransformRegistry.register("mastercurve")
class Mastercurve(BaseTransform):
    """Time-Temperature Superposition (TTS) mastercurve generation.

    This transform applies time-temperature superposition to create mastercurves
    from multi-temperature rheological data. Supports both WLF and Arrhenius
    shift factor models for horizontal shifting, with optional vertical shifting.

    The WLF equation is:
        log(a_T) = -C1 * (T - T_ref) / (C2 + (T - T_ref))

    The Arrhenius equation is:
        log(a_T) = (E_a / R) * (1/T - 1/T_ref)

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

    Examples
    --------
    >>> from rheo.core.data import RheoData
    >>> from rheo.transforms.mastercurve import Mastercurve
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
    >>> # Create mastercurve at reference temperature
    >>> mc = Mastercurve(reference_temp=298.15, method='wlf')
    >>> mastercurve = mc.create_mastercurve(datasets)
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
        """
        super().__init__()
        self.T_ref = reference_temp
        self.method = method
        self.C1 = C1
        self.C2 = C2
        self.E_a = E_a
        self.vertical_shift = vertical_shift
        self.optimize_shifts = optimize_shifts

        # Store computed shift factors
        self.shift_factors_: dict[float, float] | None = None
        self.vertical_shifts_: dict[float, float] | None = None

    def _calculate_wlf_shift(
        self, T: float | jnp.ndarray, T_ref: float, C1: float, C2: float
    ) -> float | jnp.ndarray:
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
        self, T: float | jnp.ndarray, T_ref: float, E_a: float
    ) -> float | jnp.ndarray:
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

    def _transform(self, data: RheoData) -> RheoData:
        """Apply horizontal shift to single-temperature data.

        This method is for single datasets. For creating mastercurves from
        multiple temperatures, use create_mastercurve() instead.

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
        from rheo.core.data import RheoData

        # Get temperature from metadata
        if "temperature" not in data.metadata:
            raise ValueError("Temperature must be in metadata for mastercurve shifting")

        T = data.metadata["temperature"]

        # Get shift factor
        a_T = self.get_shift_factor(T)

        # Apply horizontal shift (frequency or time shift)
        x_shifted = data.x * a_T

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

    def create_mastercurve(
        self, datasets: list[RheoData], merge: bool = True
    ) -> RheoData | list[RheoData]:
        """Create mastercurve from multiple temperature datasets.

        Parameters
        ----------
        datasets : list of RheoData
            List of datasets at different temperatures
        merge : bool, default=True
            If True, merge all shifted data into single RheoData.
            If False, return list of shifted datasets.

        Returns
        -------
        RheoData or list of RheoData
            Mastercurve data (merged or list)

        Raises
        ------
        ValueError
            If datasets don't have temperature metadata
        """
        from rheo.core.data import RheoData

        # Shift all datasets
        shifted_datasets = []
        temperatures = []

        for data in datasets:
            if "temperature" not in data.metadata:
                raise ValueError("All datasets must have 'temperature' in metadata")

            temperatures.append(data.metadata["temperature"])
            shifted = self.transform(data)
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
            "shift_method": self.method,
            "temperatures": temperatures,
            "n_datasets": len(datasets),
            "source_temperatures": merged_temps,
        }

        return RheoData(
            x=merged_x,
            y=merged_y,
            x_units=datasets[0].x_units if datasets else None,
            y_units=datasets[0].y_units if datasets else None,
            domain=datasets[0].domain if datasets else "frequency",
            metadata=merged_metadata,
            validate=False,
        )

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
            shifted_datasets = [shifted_datasets]

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
