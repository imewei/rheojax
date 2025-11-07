"""Smart parameter initialization for fractional models in oscillation mode.

This module provides data-driven initialization strategies to improve optimization
convergence for fractional viscoelastic models when fitting frequency-domain data.

The initialization extracts features from the complex modulus G*(ω) such as:
- Low-frequency plateau (equilibrium modulus)
- High-frequency plateau (total modulus)
- Transition frequency (characteristic relaxation time)
- Slope in transition region (fractional order)

These features provide much better starting points than arbitrary default values,
helping the optimizer avoid local minima in the non-convex landscape created by
Mittag-Leffler functions.

References
----------
- Issue #9: Fractional models fail to optimize in oscillation mode due to local minima
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def extract_frequency_features(omega: np.ndarray, G_star: np.ndarray) -> dict:
    """Extract features from frequency-domain complex modulus data.

    Analyzes frequency sweep data to identify characteristic features like
    low/high frequency plateaus, transition frequency, and fractional order.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array (rad/s)
    G_star : np.ndarray
        Complex modulus G* = G' + iG" (complex array or 2D [G', G"] format)

    Returns
    -------
    dict
        Dictionary with extracted features:
        - low_plateau : float
            Low-frequency |G*| plateau value (Pa)
        - high_plateau : float
            High-frequency |G*| plateau value (Pa)
        - omega_mid : float
            Transition frequency where slope is steepest (rad/s)
        - alpha_estimate : float
            Fractional order estimated from slope
        - valid : bool
            True if features extracted successfully

    Notes
    -----
    Uses Savitzky-Golay filtering to reduce noise before feature extraction.
    Requires at least 1.5 decades of frequency range for reliable results.
    """
    # Convert to magnitude
    if np.iscomplexobj(G_star):
        G_mag = np.abs(G_star)
    else:  # 2D [G', G"] format
        if G_star.ndim == 2 and G_star.shape[1] == 2:
            G_mag = np.sqrt(G_star[:, 0] ** 2 + G_star[:, 1] ** 2)
        else:
            G_mag = np.abs(G_star)  # Fall back to abs for 1D arrays

    # Smooth to reduce noise (window=5, poly=2)
    if len(G_mag) >= 5:
        G_mag_smooth = savgol_filter(G_mag, window_length=5, polyorder=2)
    else:
        G_mag_smooth = G_mag.copy()

    # Low-frequency plateau: average lowest 10%
    n_low = max(1, len(G_mag) // 10)
    low_plateau = np.mean(np.sort(G_mag_smooth)[:n_low])

    # High-frequency plateau: average highest 10%
    n_high = max(1, len(G_mag) // 10)
    high_plateau = np.mean(np.sort(G_mag_smooth)[-n_high:])

    # Find transition frequency (steepest slope in log-log)
    log_omega = np.log10(omega + 1e-12)
    log_G = np.log10(G_mag_smooth + 1e-12)
    d_log_G = np.gradient(log_G, log_omega)
    idx_mid = np.argmax(np.abs(d_log_G))
    omega_mid = omega[idx_mid]

    # Estimate alpha from slope at transition
    alpha_estimate = d_log_G[idx_mid]
    alpha_estimate = np.clip(alpha_estimate, 0.01, 0.99)

    # Check validity
    freq_range = np.log10((omega.max() + 1e-12) / (omega.min() + 1e-12))
    plateau_ratio = high_plateau / (low_plateau + 1e-12)
    valid = freq_range > 1.5 and plateau_ratio > 1.1

    return {
        "low_plateau": float(low_plateau),
        "high_plateau": float(high_plateau),
        "omega_mid": float(omega_mid),
        "alpha_estimate": float(alpha_estimate),
        "valid": bool(valid),
    }


def initialize_fractional_zener_ss(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalZenerSolidSolid from oscillation data.

    Model equation:
        G*(ω) = Ge + Gm / (1 + (iωτ_α)^(-α))

    Extraction strategy:
        - Ge: equilibrium modulus from low-frequency plateau
        - Gm: Maxwell arm modulus from plateau difference
        - tau_alpha: relaxation time from transition frequency
        - alpha: fractional order from slope or default to 0.5

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array (rad/s)
    G_star : np.ndarray
        Complex modulus (complex or 2D array)
    param_set : ParameterSet
        ParameterSet object to update with initial values

    Returns
    -------
    bool
        True if initialization succeeded, False if fell back to defaults

    Examples
    --------
    >>> from rheojax.models import FractionalZenerSolidSolid
    >>> model = FractionalZenerSolidSolid()
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = ...  # complex modulus data
    >>> success = initialize_fractional_zener_ss(omega, G_star, model.parameters)
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False  # Fall back to defaults

    epsilon = 1e-12

    # Ge: equilibrium modulus from low-frequency plateau
    Ge_init = max(features["low_plateau"], epsilon)

    # Gm: Maxwell arm modulus from plateau difference
    Gm_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

    # tau_alpha: relaxation time from transition frequency
    tau_alpha_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: fractional order from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Ge_bounds = param_set._parameters["Ge"].bounds
    Gm_bounds = param_set._parameters["Gm"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_alpha_bounds = param_set._parameters["tau_alpha"].bounds

    Ge_init = np.clip(Ge_init, Ge_bounds[0], Ge_bounds[1])
    Gm_init = np.clip(Gm_init, Gm_bounds[0], Gm_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_alpha_init = np.clip(tau_alpha_init, tau_alpha_bounds[0], tau_alpha_bounds[1])

    # Update parameters
    param_set.set_value("Ge", Ge_init)
    param_set.set_value("Gm", Gm_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau_alpha", tau_alpha_init)

    return True


__all__ = ["extract_frequency_features", "initialize_fractional_zener_ss"]
