"""Strain-Rate Frequency Superposition (SRFS) transform.

This module implements SRFS for collapsing flow curves at different shear rates
onto a master curve, analogous to time-temperature superposition (TTS) but based
on shear rate rather than temperature.

SRFS is particularly useful for soft glassy materials where the SGR model predicts
a power-law relationship between shift factor and shear rate:
    a(gamma_dot) ~ (gamma_dot)^m
where m = (2 - x) depends on the noise temperature x.

Thixotropy kinetics and shear banding detection are also implemented for
complete characterization of complex flow behavior in soft glassy materials.

Physical Background:
    - SRFS exploits the fact that flow curves at different reference shear rates
      can be collapsed via horizontal shifting
    - For SGR materials, the shift factor has power-law form determined by x
    - Thixotropy arises from microstructure build-up (at rest) and breakdown (under shear)
    - Shear banding occurs when the constitutive curve becomes non-monotonic

References:
    - P. Sollich, Rheological constitutive equation for a model of soft glassy
      materials, Physical Review E, 1998, 58(1), 738-759
    - M. Wyss et al., Strain-rate frequency superposition: A rheological probe
      of structural relaxation in soft materials, Physical Review Letters, 2007
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    import jax.numpy as jnp_typing
else:  # pragma: no cover - typing fallback
    jnp_typing = np

type JaxArray = jnp_typing.ndarray
type ScalarOrArray = float | JaxArray


@TransformRegistry.register("srfs")
class SRFS(BaseTransform):
    """Strain-Rate Frequency Superposition (SRFS) transform.

    SRFS collapses flow curves measured at different shear rates onto a master
    curve by applying horizontal shift factors. This is analogous to time-temperature
    superposition (TTS) but uses shear rate rather than temperature.

    For SGR (Soft Glassy Rheology) materials, the shift factor follows:
        a(gamma_dot) = (gamma_dot / gamma_dot_ref)^m
    where m = (2 - x) and x is the noise temperature.

    Parameters
    ----------
    reference_gamma_dot : float, default=1.0
        Reference shear rate for the master curve (1/s)
    auto_shift : bool, default=False
        If True, automatically compute optimal shift factors from data overlap

    Attributes
    ----------
    reference_gamma_dot : float
        Reference shear rate
    shift_factors_ : dict[float, float] or None
        Computed shift factors after transform

    Examples
    --------
    >>> from rheojax.transforms.srfs import SRFS
    >>> from rheojax.core.data import RheoData
    >>>
    >>> # Create flow curve datasets at different reference shear rates
    >>> datasets = [
    ...     RheoData(x=gamma_dots_1, y=eta_1, metadata={'reference_gamma_dot': 0.1}),
    ...     RheoData(x=gamma_dots_2, y=eta_2, metadata={'reference_gamma_dot': 1.0}),
    ...     RheoData(x=gamma_dots_3, y=eta_3, metadata={'reference_gamma_dot': 10.0}),
    ... ]
    >>>
    >>> # Create SRFS transform
    >>> srfs = SRFS(reference_gamma_dot=1.0)
    >>>
    >>> # Apply SRFS shift (requires SGR parameters)
    >>> mastercurve, shift_factors = srfs.transform(datasets, x=1.5, tau0=1e-3)

    Notes
    -----
    - Shift factors depend on SGR noise temperature x
    - For x < 1 (glass), shift behavior changes near yield stress
    - For x >= 2 (Newtonian), shift factor approaches 1
    """

    def __init__(
        self,
        reference_gamma_dot: float = 1.0,
        auto_shift: bool = False,
    ):
        """Initialize SRFS transform.

        Parameters
        ----------
        reference_gamma_dot : float
            Reference shear rate for the master curve
        auto_shift : bool
            Whether to automatically compute optimal shift factors
        """
        super().__init__()
        self.reference_gamma_dot = reference_gamma_dot
        self._auto_shift = auto_shift
        self.shift_factors_: dict[float, float] | None = None

    def compute_shift_factor(
        self,
        gamma_dot: float,
        x: float,
        tau0: float,
    ) -> float:
        """Compute SRFS shift factor from SGR theory.

        For SGR materials, the shift factor follows a power-law:
            a(gamma_dot) = (gamma_dot / gamma_dot_ref)^m
        where m = (2 - x) for the power-law fluid regime (1 < x < 2).

        Parameters
        ----------
        gamma_dot : float
            Shear rate to compute shift for (1/s)
        x : float
            SGR noise temperature (dimensionless)
        tau0 : float
            SGR attempt time (s)

        Returns
        -------
        float
            Shift factor a(gamma_dot)

        Notes
        -----
        - For x = 1.5, exponent m = 0.5
        - For x = 2 (Newtonian), m = 0, shift factor = 1
        - For x < 1 (glass), behavior near yield stress is different
        """
        # Compute shift exponent from SGR theory
        # In power-law regime: a ~ gamma_dot^(2-x)
        # This comes from the scaling of viscosity eta ~ gamma_dot^(x-2)
        # and the requirement that shifted curves collapse

        # Exponent for shift factor
        m = 2.0 - x

        # Handle special cases
        if abs(gamma_dot - self.reference_gamma_dot) < 1e-12:
            return 1.0

        # Compute shift factor
        # a(gamma_dot) = (gamma_dot * tau0)^m / (gamma_dot_ref * tau0)^m
        #              = (gamma_dot / gamma_dot_ref)^m
        ratio = gamma_dot / self.reference_gamma_dot
        a_gamma_dot = ratio ** m

        return float(a_gamma_dot)

    def _transform_single(
        self,
        data: RheoData,
        x: float,
        tau0: float,
    ) -> RheoData:
        """Apply SRFS shift to a single dataset.

        Parameters
        ----------
        data : RheoData
            Single flow curve dataset
        x : float
            SGR noise temperature
        tau0 : float
            SGR attempt time

        Returns
        -------
        RheoData
            Shifted dataset
        """
        # Get reference shear rate from metadata
        if "reference_gamma_dot" not in data.metadata:
            raise ValueError(
                "reference_gamma_dot must be in metadata for SRFS shifting"
            )

        gamma_dot_ref = data.metadata["reference_gamma_dot"]

        # Compute shift factor
        a_gamma_dot = self.compute_shift_factor(gamma_dot_ref, x, tau0)

        # Apply horizontal shift to shear rate axis
        x_shifted = data.x * a_gamma_dot

        # Create shifted dataset
        new_metadata = data.metadata.copy()
        new_metadata.update({
            "transform": "srfs",
            "reference_gamma_dot_master": self.reference_gamma_dot,
            "shift_factor": float(a_gamma_dot),
            "sgr_x": x,
            "sgr_tau0": tau0,
        })

        return RheoData(
            x=x_shifted,
            y=data.y,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata=new_metadata,
            validate=False,
        )

    def _transform(
        self,
        data: RheoData | list[RheoData],
        x: float | None = None,
        tau0: float | None = None,
        return_shifts: bool = False,
    ) -> RheoData | tuple[RheoData, dict[float, float]]:
        """Apply SRFS transformation.

        Parameters
        ----------
        data : RheoData or list of RheoData
            Single dataset or list of datasets to transform
        x : float, optional
            SGR noise temperature (required if not using auto_shift)
        tau0 : float, optional
            SGR attempt time (required if not using auto_shift)
        return_shifts : bool, default=False
            If True, return shift factors dict along with mastercurve

        Returns
        -------
        RheoData or tuple
            If data is single RheoData: shifted dataset
            If data is list and return_shifts=True: (mastercurve, shift_factors)
            If data is list and return_shifts=False: mastercurve
        """
        # Handle single dataset
        if isinstance(data, RheoData):
            if x is None or tau0 is None:
                raise ValueError("x and tau0 are required for SRFS transformation")
            return self._transform_single(data, x, tau0)

        # Handle list of datasets
        if x is None or tau0 is None:
            raise ValueError("x and tau0 are required for SRFS transformation")

        return self.create_mastercurve(data, x, tau0, return_shifts=return_shifts)

    def transform(
        self,
        data: RheoData | list[RheoData],
        x: float | None = None,
        tau0: float | None = None,
        return_shifts: bool = False,
    ) -> RheoData | tuple[RheoData, dict[float, float]]:
        """Apply SRFS transformation (public interface).

        Parameters
        ----------
        data : RheoData or list of RheoData
            Single dataset or list of datasets to transform
        x : float, optional
            SGR noise temperature
        tau0 : float, optional
            SGR attempt time
        return_shifts : bool, default=False
            If True, return shift factors dict along with mastercurve

        Returns
        -------
        RheoData or tuple
            Transformed data, optionally with shift factors
        """
        return self._transform(data, x=x, tau0=tau0, return_shifts=return_shifts)

    def create_mastercurve(
        self,
        datasets: list[RheoData],
        x: float,
        tau0: float,
        merge: bool = True,
        return_shifts: bool = False,
    ) -> RheoData | list[RheoData] | tuple[RheoData, dict[float, float]]:
        """Create SRFS master curve from multiple flow curve datasets.

        Parameters
        ----------
        datasets : list of RheoData
            Flow curves at different reference shear rates
        x : float
            SGR noise temperature
        tau0 : float
            SGR attempt time
        merge : bool, default=True
            If True, merge all shifted data into single RheoData
        return_shifts : bool, default=False
            If True, return shift factors dict with mastercurve

        Returns
        -------
        RheoData or list or tuple
            Master curve or list of shifted datasets, optionally with shifts
        """
        # Extract reference shear rates and sort
        ref_gamma_dots = []
        for data in datasets:
            if "reference_gamma_dot" not in data.metadata:
                raise ValueError(
                    "All datasets must have 'reference_gamma_dot' in metadata"
                )
            ref_gamma_dots.append(data.metadata["reference_gamma_dot"])

        # Sort by reference shear rate
        sorted_indices = np.argsort(ref_gamma_dots)
        datasets = [datasets[i] for i in sorted_indices]
        ref_gamma_dots = [ref_gamma_dots[i] for i in sorted_indices]

        # Compute shift factors
        shift_factors = {}
        for gamma_dot_ref in ref_gamma_dots:
            a_gamma_dot = self.compute_shift_factor(gamma_dot_ref, x, tau0)
            shift_factors[gamma_dot_ref] = a_gamma_dot

        # Apply shifts
        shifted_datasets = []
        for data, _gamma_dot_ref in zip(datasets, ref_gamma_dots, strict=False):
            shifted = self._transform_single(data, x, tau0)
            shifted_datasets.append(shifted)

        # Store shift factors
        self.shift_factors_ = shift_factors

        if not merge:
            return shifted_datasets

        # Merge all shifted data
        all_x = []
        all_y = []
        all_refs = []

        for data, ref in zip(shifted_datasets, ref_gamma_dots, strict=False):
            x_data = np.asarray(data.x)
            y_data = np.asarray(data.y)

            all_x.append(x_data)
            all_y.append(y_data)
            all_refs.extend([ref] * len(x_data))

        # Concatenate and sort
        merged_x = np.concatenate(all_x)
        merged_y = np.concatenate(all_y)
        merged_refs = np.array(all_refs)

        sort_idx = np.argsort(merged_x)
        merged_x = merged_x[sort_idx]
        merged_y = merged_y[sort_idx]
        merged_refs = merged_refs[sort_idx]

        # Create mastercurve
        mastercurve_metadata = {
            "transform": "srfs",
            "reference_gamma_dot": self.reference_gamma_dot,
            "source_gamma_dots": ref_gamma_dots,
            "n_datasets": len(datasets),
            "source_refs": merged_refs,
            "shift_factors": shift_factors,
            "sgr_x": x,
            "sgr_tau0": tau0,
        }

        mastercurve = RheoData(
            x=merged_x,
            y=merged_y,
            x_units=datasets[0].x_units if datasets else None,
            y_units=datasets[0].y_units if datasets else None,
            domain=datasets[0].domain if datasets else "shear_rate",
            metadata=mastercurve_metadata,
            validate=False,
        )

        if return_shifts:
            return mastercurve, shift_factors
        return mastercurve

    def get_shift_factors_array(
        self,
        gamma_dots: list[float] | np.ndarray | None = None,
        x: float | None = None,
        tau0: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get shift factors as arrays for plotting.

        Parameters
        ----------
        gamma_dots : list or ndarray, optional
            Shear rates to compute shifts for. If None, uses stored values.
        x : float, optional
            SGR noise temperature (required if computing new shifts)
        tau0 : float, optional
            SGR attempt time (required if computing new shifts)

        Returns
        -------
        gamma_dots : ndarray
            Array of shear rates (sorted)
        shift_factors : ndarray
            Array of corresponding shift factors
        """
        if gamma_dots is None:
            if self.shift_factors_ is None:
                raise ValueError(
                    "No shift factors available. Either provide gamma_dots or "
                    "create a mastercurve first."
                )
            gamma_dots_arr = np.array(sorted(self.shift_factors_.keys()))
            shifts_arr = np.array([self.shift_factors_[gd] for gd in gamma_dots_arr])
        else:
            if x is None or tau0 is None:
                raise ValueError("x and tau0 required to compute shift factors")
            gamma_dots_arr = np.array(gamma_dots)
            sort_idx = np.argsort(gamma_dots_arr)
            gamma_dots_arr = gamma_dots_arr[sort_idx]
            shifts_arr = np.array([
                self.compute_shift_factor(float(gd), x, tau0) for gd in gamma_dots_arr
            ])

        return gamma_dots_arr, shifts_arr


# ============================================================================
# Shear Banding Detection Functions
# ============================================================================


def detect_shear_banding(
    gamma_dot: np.ndarray,
    sigma: np.ndarray,
    warn: bool = False,
    threshold: float = -0.01,
) -> tuple[bool, dict | None]:
    """Detect shear banding from non-monotonic constitutive curve.

    Shear banding occurs when the derivative d(sigma)/d(gamma_dot) < 0,
    indicating a region of mechanical instability where the material
    splits into bands with different local shear rates.

    Parameters
    ----------
    gamma_dot : ndarray
        Shear rate array (1/s)
    sigma : ndarray
        Stress array (Pa)
    warn : bool, default=False
        If True, issue a warning when shear banding is detected
    threshold : float, default=-0.01
        Threshold for detecting negative slope (allows for numerical noise)

    Returns
    -------
    is_banding : bool
        True if shear banding is detected
    banding_info : dict or None
        Information about the banding region if detected:
        - 'gamma_dot_low': Lower shear rate of banding region
        - 'gamma_dot_high': Upper shear rate of banding region
        - 'sigma_range': Stress range in banding region
        - 'negative_slope_fraction': Fraction of curve with negative slope

    Examples
    --------
    >>> gamma_dot = np.logspace(-2, 2, 100)
    >>> sigma = gamma_dot ** 0.5  # Monotonic power-law
    >>> is_banding, info = detect_shear_banding(gamma_dot, sigma)
    >>> print(is_banding)  # False

    >>> # Non-monotonic curve
    >>> sigma_nm = sigma * (1 - 0.3 * np.exp(-((gamma_dot - 1)**2) / 0.1))
    >>> is_banding, info = detect_shear_banding(gamma_dot, sigma_nm)
    >>> print(is_banding)  # True
    """
    # Sort by shear rate
    sort_idx = np.argsort(gamma_dot)
    gamma_dot = gamma_dot[sort_idx]
    sigma = sigma[sort_idx]

    # Compute derivative d(sigma)/d(gamma_dot) using finite differences
    d_sigma = np.diff(sigma)
    d_gamma_dot = np.diff(gamma_dot)

    # Avoid division by zero
    d_gamma_dot = np.maximum(d_gamma_dot, 1e-20)

    derivative = d_sigma / d_gamma_dot

    # Detect regions with negative slope
    negative_slope_mask = derivative < threshold

    # Check if any negative slope regions exist
    is_banding = np.any(negative_slope_mask)

    if not is_banding:
        return False, None

    # Find the banding region bounds
    negative_indices = np.where(negative_slope_mask)[0]

    if len(negative_indices) == 0:
        return False, None

    # Get bounds of non-monotonic region
    first_neg_idx = negative_indices[0]
    last_neg_idx = negative_indices[-1]

    gamma_dot_low = gamma_dot[first_neg_idx]
    gamma_dot_high = gamma_dot[min(last_neg_idx + 1, len(gamma_dot) - 1)]

    # Get stress range in banding region
    sigma_low = sigma[first_neg_idx]
    sigma_high = sigma[min(last_neg_idx + 1, len(sigma) - 1)]

    # Compute fraction of curve with negative slope
    neg_fraction = np.sum(negative_slope_mask) / len(derivative)

    banding_info = {
        "gamma_dot_low": float(gamma_dot_low),
        "gamma_dot_high": float(gamma_dot_high),
        "sigma_low": float(sigma_low),
        "sigma_high": float(sigma_high),
        "sigma_range": (float(min(sigma_low, sigma_high)), float(max(sigma_low, sigma_high))),
        "negative_slope_fraction": float(neg_fraction),
    }

    if warn:
        warnings.warn(
            f"Shear banding detected in flow curve. "
            f"Non-monotonic region: gamma_dot = [{gamma_dot_low:.3g}, {gamma_dot_high:.3g}] 1/s. "
            f"This may indicate mechanical instability.",
            UserWarning,
            stacklevel=2,
        )

    return True, banding_info


def compute_shear_band_coexistence(
    gamma_dot: np.ndarray,
    sigma: np.ndarray,
    gamma_dot_applied: float,
) -> dict | None:
    """Compute shear band coexistence using lever rule.

    When shear banding occurs, the material splits into bands with different
    local shear rates (gamma_dot_low and gamma_dot_high) that coexist at
    a common stress plateau. The fraction of each band is determined by
    the lever rule from the applied average shear rate.

    Parameters
    ----------
    gamma_dot : ndarray
        Shear rate array (1/s)
    sigma : ndarray
        Stress array (Pa)
    gamma_dot_applied : float
        Applied (average) shear rate (1/s)

    Returns
    -------
    coexistence : dict or None
        Coexistence information if banding detected:
        - 'gamma_dot_low': Shear rate in low-shear band
        - 'gamma_dot_high': Shear rate in high-shear band
        - 'fraction_low': Volume fraction of low-shear band
        - 'fraction_high': Volume fraction of high-shear band
        - 'stress_plateau': Common stress in banding regime
        Returns None if no banding or applied rate outside banding region.

    Notes
    -----
    The lever rule states:
        gamma_dot_applied = f_low * gamma_dot_low + f_high * gamma_dot_high
    where f_low + f_high = 1.

    The stress plateau is found by equal area construction (Maxwell rule)
    or by finding the stress at which both bands coexist stably.
    """
    # First detect if banding exists
    is_banding, banding_info = detect_shear_banding(gamma_dot, sigma)

    if not is_banding or banding_info is None:
        return None

    # Get banding region bounds
    gamma_dot_low_bound = banding_info["gamma_dot_low"]
    gamma_dot_high_bound = banding_info["gamma_dot_high"]

    # Check if applied shear rate is in banding region
    if gamma_dot_applied < gamma_dot_low_bound or gamma_dot_applied > gamma_dot_high_bound:
        return None

    # Find stress plateau using simplified approach
    # (In practice, would use equal area Maxwell construction)

    # Sort data
    sort_idx = np.argsort(gamma_dot)
    gamma_dot_sorted = gamma_dot[sort_idx]
    sigma_sorted = sigma[sort_idx]

    # Find indices bounding the banding region
    low_idx = np.searchsorted(gamma_dot_sorted, gamma_dot_low_bound)
    high_idx = np.searchsorted(gamma_dot_sorted, gamma_dot_high_bound)

    # Estimate stress plateau as average in banding region
    stress_plateau = np.mean(sigma_sorted[low_idx:high_idx+1])

    # Find coexisting shear rates at stress plateau
    # These are the intersections of horizontal line at stress_plateau
    # with the constitutive curve (on the stable branches)

    # Left branch (before banding onset)
    left_mask = gamma_dot_sorted < gamma_dot_low_bound
    if np.any(left_mask):
        gamma_dot_left = gamma_dot_sorted[left_mask]
        sigma_left = sigma_sorted[left_mask]
        # Interpolate to find gamma_dot at stress_plateau
        if len(gamma_dot_left) > 1:
            gamma_dot_low = np.interp(stress_plateau, sigma_left, gamma_dot_left)
        else:
            gamma_dot_low = gamma_dot_low_bound
    else:
        gamma_dot_low = gamma_dot_low_bound

    # Right branch (after banding ends)
    right_mask = gamma_dot_sorted > gamma_dot_high_bound
    if np.any(right_mask):
        gamma_dot_right = gamma_dot_sorted[right_mask]
        sigma_right = sigma_sorted[right_mask]
        # Interpolate
        if len(gamma_dot_right) > 1:
            gamma_dot_high = np.interp(stress_plateau, sigma_right, gamma_dot_right)
        else:
            gamma_dot_high = gamma_dot_high_bound
    else:
        gamma_dot_high = gamma_dot_high_bound

    # Lever rule for band fractions
    # gamma_dot_applied = f_low * gamma_dot_low + (1 - f_low) * gamma_dot_high
    # f_low = (gamma_dot_high - gamma_dot_applied) / (gamma_dot_high - gamma_dot_low)

    delta_gamma = gamma_dot_high - gamma_dot_low
    if abs(delta_gamma) < 1e-12:
        return None

    f_low = (gamma_dot_high - gamma_dot_applied) / delta_gamma
    f_high = 1.0 - f_low

    # Clamp fractions to [0, 1]
    f_low = np.clip(f_low, 0, 1)
    f_high = np.clip(f_high, 0, 1)

    return {
        "gamma_dot_low": float(gamma_dot_low),
        "gamma_dot_high": float(gamma_dot_high),
        "fraction_low": float(f_low),
        "fraction_high": float(f_high),
        "stress_plateau": float(stress_plateau),
    }


# ============================================================================
# Thixotropy Kinetics Functions
# ============================================================================


@jax.jit
def thixotropy_lambda_derivative(
    lambda_val: float,
    gamma_dot: float,
    k_build: float,
    k_break: float,
) -> float:
    """Compute time derivative of structural parameter lambda.

    The structural parameter lambda represents the state of internal
    microstructure, with lambda = 1 being fully built and lambda = 0
    being fully broken.

    Evolution equation:
        d(lambda)/dt = k_build * (1 - lambda) - k_break * gamma_dot * lambda

    Parameters
    ----------
    lambda_val : float
        Current structural parameter value [0, 1]
    gamma_dot : float
        Current shear rate (1/s)
    k_build : float
        Structure build-up rate (1/s)
    k_break : float
        Structure breakdown rate (dimensionless)

    Returns
    -------
    float
        Time derivative d(lambda)/dt
    """
    # Build-up term: drives lambda toward 1 at rest
    build_up = k_build * (1.0 - lambda_val)

    # Breakdown term: shear destroys structure
    breakdown = k_break * gamma_dot * lambda_val

    return build_up - breakdown


def evolve_thixotropy_lambda(
    t: np.ndarray,
    gamma_dot: np.ndarray,
    lambda_initial: float,
    k_build: float,
    k_break: float,
) -> np.ndarray:
    """Evolve structural parameter lambda(t) for given shear history.

    Integrates the thixotropy kinetics equation:
        d(lambda)/dt = k_build * (1 - lambda) - k_break * gamma_dot * lambda

    Parameters
    ----------
    t : ndarray
        Time array (s)
    gamma_dot : ndarray
        Shear rate array (1/s), same shape as t
    lambda_initial : float
        Initial structural parameter [0, 1]
    k_build : float
        Structure build-up rate (1/s)
    k_break : float
        Structure breakdown rate (dimensionless)

    Returns
    -------
    lambda_t : ndarray
        Structural parameter evolution, same shape as t
    """
    if t.shape != gamma_dot.shape:
        raise ValueError(
            f"Time and shear rate arrays must have same shape: "
            f"t.shape={t.shape}, gamma_dot.shape={gamma_dot.shape}"
        )

    # Use simple Euler integration for stability
    dt = np.diff(t)
    dt = np.concatenate([[0], dt])  # Prepend 0 for first step

    lambda_t = np.zeros_like(t)
    lambda_t[0] = lambda_initial

    for i in range(1, len(t)):
        dlambda_dt = thixotropy_lambda_derivative(
            lambda_t[i-1], gamma_dot[i], k_build, k_break
        )
        lambda_t[i] = lambda_t[i-1] + dlambda_dt * dt[i]
        # Clamp to [0, 1]
        lambda_t[i] = np.clip(lambda_t[i], 0.0, 1.0)

    return lambda_t


def compute_thixotropic_stress(
    t: np.ndarray,
    gamma_dot: np.ndarray,
    lambda_t: np.ndarray,
    G0: float,
    tau0: float,
    x: float,
    n_struct: float = 2.0,
) -> np.ndarray:
    """Compute stress response with thixotropic modulus.

    The effective modulus is coupled to the structural parameter:
        G_eff(t) = G0 * lambda(t)^n_struct

    Parameters
    ----------
    t : ndarray
        Time array (s)
    gamma_dot : ndarray
        Shear rate array (1/s)
    lambda_t : ndarray
        Structural parameter array [0, 1]
    G0 : float
        Base modulus scale (Pa)
    tau0 : float
        Attempt time (s)
    x : float
        SGR noise temperature
    n_struct : float, default=2.0
        Structural coupling exponent

    Returns
    -------
    sigma : ndarray
        Stress response (Pa)
    """
    # Effective modulus from structure
    G_eff = G0 * np.power(lambda_t, n_struct)

    # Viscosity from power-law (SGR-like)
    gamma_dot_safe = np.maximum(np.abs(gamma_dot), 1e-12)
    eta_factor = np.power(gamma_dot_safe * tau0, x - 2.0)

    # Stress = G_eff * gamma_dot * tau0 * eta_factor
    sigma = G_eff * gamma_dot * tau0 * eta_factor

    return sigma


__all__ = [
    "SRFS",
    "detect_shear_banding",
    "compute_shear_band_coexistence",
    "thixotropy_lambda_derivative",
    "evolve_thixotropy_lambda",
    "compute_thixotropic_stress",
]
