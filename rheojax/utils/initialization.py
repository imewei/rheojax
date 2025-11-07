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


def initialize_fractional_maxwell_liquid(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalMaxwellLiquid from oscillation data.

    Model equation:
        G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)

    Extraction strategy:
        - Gm: Maxwell modulus from high-frequency plateau
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
    >>> from rheojax.models import FractionalMaxwellLiquid
    >>> model = FractionalMaxwellLiquid()
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = ...  # complex modulus data
    >>> success = initialize_fractional_maxwell_liquid(omega, G_star, model.parameters)
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False  # Fall back to defaults

    epsilon = 1e-12

    # Gm: Maxwell modulus from high-frequency plateau
    Gm_init = max(features["high_plateau"], epsilon)

    # tau_alpha: relaxation time from transition frequency
    tau_alpha_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: fractional order from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Gm_bounds = param_set._parameters["Gm"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_alpha_bounds = param_set._parameters["tau_alpha"].bounds

    Gm_init = np.clip(Gm_init, Gm_bounds[0], Gm_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_alpha_init = np.clip(tau_alpha_init, tau_alpha_bounds[0], tau_alpha_bounds[1])

    # Update parameters
    param_set.set_value("Gm", Gm_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau_alpha", tau_alpha_init)

    return True


def initialize_fractional_maxwell_gel(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalMaxwellGel from oscillation data.

    Model equation:
        G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))
        where τ = η / c_α^(1/(1-α))

    Extraction strategy:
        - c_alpha: estimated from high-frequency plateau
        - alpha: fractional order from slope or default to 0.5
        - eta: estimated from transition frequency and approximate tau

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
    >>> from rheojax.models import FractionalMaxwellGel
    >>> model = FractionalMaxwellGel()
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = ...  # complex modulus data
    >>> success = initialize_fractional_maxwell_gel(omega, G_star, model.parameters)
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False  # Fall back to defaults

    epsilon = 1e-12

    # alpha: fractional order from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # c_alpha: approximate from high-frequency behavior
    # At high frequency, G* ~ c_α τ^(α-1), but we'll use high plateau as first estimate
    c_alpha_init = max(features["high_plateau"], epsilon)

    # eta: estimate from transition frequency
    # tau ~ 1/omega_mid, and tau = eta / c_alpha^(1/(1-alpha))
    # So eta ~ tau * c_alpha^(1/(1-alpha))
    tau_est = 1.0 / (features["omega_mid"] + epsilon)
    eta_init = tau_est * (c_alpha_init ** (1.0 / (1.0 - alpha_init + epsilon)))

    # Clip to parameter bounds
    c_alpha_bounds = param_set._parameters["c_alpha"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    eta_bounds = param_set._parameters["eta"].bounds

    c_alpha_init = np.clip(c_alpha_init, c_alpha_bounds[0], c_alpha_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    eta_init = np.clip(eta_init, eta_bounds[0], eta_bounds[1])

    # Update parameters
    param_set.set_value("c_alpha", c_alpha_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("eta", eta_init)

    return True


def initialize_fractional_zener_ll(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalZenerLiquidLiquid from oscillation data.

    Model equation:
        G*(ω) = c_1 * (iω)^α / (1 + (iωτ)^β) + c_2 * (iω)^γ

    Extraction strategy (simplified for 6-parameter model):
        - c1, c2: Split high-frequency plateau between both terms
        - alpha, beta, gamma: Use slope estimate or defaults
        - tau: From transition frequency

    Note: This is a simplified initialization for a highly complex model.
    The optimizer will refine these starting values.

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False  # Fall back to defaults

    epsilon = 1e-12

    # Split the high-frequency modulus between c1 and c2
    total_modulus = max(features["high_plateau"], epsilon)
    c1_init = total_modulus * 0.6  # Allocate more to first term
    c2_init = total_modulus * 0.4

    # Use slope estimate for alpha, defaults for beta and gamma
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    beta_init = 0.5  # Default for second fractional order
    gamma_init = 0.5  # Default for third fractional order

    # tau from transition frequency
    tau_init = 1.0 / (features["omega_mid"] + epsilon)

    # Clip to parameter bounds
    c1_bounds = param_set._parameters["c1"].bounds
    c2_bounds = param_set._parameters["c2"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    beta_bounds = param_set._parameters["beta"].bounds
    gamma_bounds = param_set._parameters["gamma"].bounds
    tau_bounds = param_set._parameters["tau"].bounds

    c1_init = np.clip(c1_init, c1_bounds[0], c1_bounds[1])
    c2_init = np.clip(c2_init, c2_bounds[0], c2_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    beta_init = np.clip(beta_init, beta_bounds[0], beta_bounds[1])
    gamma_init = np.clip(gamma_init, gamma_bounds[0], gamma_bounds[1])
    tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

    # Update parameters
    param_set.set_value("c1", c1_init)
    param_set.set_value("c2", c2_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("beta", beta_init)
    param_set.set_value("gamma", gamma_init)
    param_set.set_value("tau", tau_init)

    return True


def initialize_fractional_zener_sl(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalZenerSolidLiquid from oscillation data.

    Model equation:
        G*(ω) = G_e + c_α * (iω)^α / (1 + (iωτ)^(1-α))

    Extraction strategy:
        - Ge: equilibrium modulus from low-frequency plateau
        - c_alpha: from plateau difference (high - low)
        - alpha: fractional order from slope or default to 0.5
        - tau: relaxation time from transition frequency

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
    >>> from rheojax.models import FractionalZenerSolidLiquid
    >>> model = FractionalZenerSolidLiquid()
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = ...  # complex modulus data
    >>> success = initialize_fractional_zener_sl(omega, G_star, model.parameters)
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False  # Fall back to defaults

    epsilon = 1e-12

    # Ge: equilibrium modulus from low-frequency plateau
    Ge_init = max(features["low_plateau"], epsilon)

    # c_alpha: from plateau difference (high - low)
    c_alpha_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

    # tau: relaxation time from transition frequency
    tau_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: fractional order from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Ge_bounds = param_set._parameters["Ge"].bounds
    c_alpha_bounds = param_set._parameters["c_alpha"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_bounds = param_set._parameters["tau"].bounds

    Ge_init = np.clip(Ge_init, Ge_bounds[0], Ge_bounds[1])
    c_alpha_init = np.clip(c_alpha_init, c_alpha_bounds[0], c_alpha_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

    # Update parameters
    param_set.set_value("Ge", Ge_init)
    param_set.set_value("c_alpha", c_alpha_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau", tau_init)

    return True


def initialize_fractional_kelvin_voigt(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalKelvinVoigt from oscillation data.

    Model equation:
        G*(ω) = G_e + c_α (iω)^α

    Extraction strategy:
        - Ge: equilibrium modulus from low-frequency plateau
        - c_alpha: from plateau difference (high - low)
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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # Ge: equilibrium modulus from low-frequency plateau
    Ge_init = max(features["low_plateau"], epsilon)

    # c_alpha: from plateau difference (high - low)
    c_alpha_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

    # alpha: fractional order from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Ge_bounds = param_set._parameters["Ge"].bounds
    c_alpha_bounds = param_set._parameters["c_alpha"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds

    Ge_init = np.clip(Ge_init, Ge_bounds[0], Ge_bounds[1])
    c_alpha_init = np.clip(c_alpha_init, c_alpha_bounds[0], c_alpha_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])

    # Update parameters
    param_set.set_value("Ge", Ge_init)
    param_set.set_value("c_alpha", c_alpha_init)
    param_set.set_value("alpha", alpha_init)

    return True


def initialize_fractional_maxwell_model(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalMaxwellModel from oscillation data.

    Model equation:
        G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

    Extraction strategy:
        - c1: from high-frequency plateau
        - alpha, beta: from slope or defaults to 0.5
        - tau: from transition frequency

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # c1: from high-frequency plateau
    c1_init = max(features["high_plateau"], epsilon)

    # alpha: from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # beta: default to 0.5 (optimizer will refine)
    beta_init = 0.5

    # tau: from transition frequency
    tau_init = 1.0 / (features["omega_mid"] + epsilon)

    # Clip to parameter bounds
    c1_bounds = param_set._parameters["c1"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    beta_bounds = param_set._parameters["beta"].bounds
    tau_bounds = param_set._parameters["tau"].bounds

    c1_init = np.clip(c1_init, c1_bounds[0], c1_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    beta_init = np.clip(beta_init, beta_bounds[0], beta_bounds[1])
    tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

    # Update parameters
    param_set.set_value("c1", c1_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("beta", beta_init)
    param_set.set_value("tau", tau_init)

    return True


def initialize_fractional_kv_zener(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalKelvinVoigtZener from oscillation data.

    Model equation (in compliance):
        J*(ω) = 1/G_e + (1/G_k) / (1 + (iωτ)^α)
        G*(ω) = 1 / J*(ω)

    Extraction strategy:
        - Ge: from high-frequency limit (1/J_min)
        - Gk: from difference in compliances
        - tau: from transition frequency
        - alpha: from slope or default to 0.5

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # Ge: from high-frequency plateau (series spring)
    Ge_init = max(features["high_plateau"], epsilon)

    # Gk: from modulus difference
    Gk_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

    # tau: from transition frequency
    tau_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Ge_bounds = param_set._parameters["Ge"].bounds
    Gk_bounds = param_set._parameters["Gk"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_bounds = param_set._parameters["tau"].bounds

    Ge_init = np.clip(Ge_init, Ge_bounds[0], Ge_bounds[1])
    Gk_init = np.clip(Gk_init, Gk_bounds[0], Gk_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

    # Update parameters
    param_set.set_value("Ge", Ge_init)
    param_set.set_value("Gk", Gk_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau", tau_init)

    return True


def initialize_fractional_poynting_thomson(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalPoyntingThomson from oscillation data.

    Model equation (in compliance):
        J*(ω) = 1/G_e + (1/G_k) / (1 + (iωτ)^α)
        G*(ω) = 1 / J*(ω)

    Extraction strategy:
        - Ge: from high-frequency limit
        - Gk: from modulus difference
        - tau: from transition frequency
        - alpha: from slope or default to 0.5

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # Ge: instantaneous modulus from high-frequency plateau
    Ge_init = max(features["high_plateau"], epsilon)

    # Gk: retarded modulus from difference
    Gk_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

    # tau: from transition frequency
    tau_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Ge_bounds = param_set._parameters["Ge"].bounds
    Gk_bounds = param_set._parameters["Gk"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_bounds = param_set._parameters["tau"].bounds

    Ge_init = np.clip(Ge_init, Ge_bounds[0], Ge_bounds[1])
    Gk_init = np.clip(Gk_init, Gk_bounds[0], Gk_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

    # Update parameters
    param_set.set_value("Ge", Ge_init)
    param_set.set_value("Gk", Gk_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau", tau_init)

    return True


def initialize_fractional_jeffreys(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalJeffreys from oscillation data.

    Model equation:
        G*(ω) = η_1(iω) · [1 + (iωτ_2)^α] / [1 + (iωτ_1)^α]
        where τ_2 = (η_2/η_1) · τ_1

    Extraction strategy (simplified):
        - eta1, eta2: from high-frequency slope and plateau
        - tau1: from transition frequency
        - alpha: from slope or default to 0.5

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # eta1: from high-frequency behavior (G* ~ eta1 * omega at high freq)
    omega_high = omega[-1] if len(omega) > 0 else 1.0
    eta1_init = max(features["high_plateau"] / (omega_high + epsilon), epsilon)

    # eta2: assume ratio around 0.5
    eta2_init = eta1_init * 0.5

    # tau1: from transition frequency
    tau1_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    eta1_bounds = param_set._parameters["eta1"].bounds
    eta2_bounds = param_set._parameters["eta2"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau1_bounds = param_set._parameters["tau1"].bounds

    eta1_init = np.clip(eta1_init, eta1_bounds[0], eta1_bounds[1])
    eta2_init = np.clip(eta2_init, eta2_bounds[0], eta2_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau1_init = np.clip(tau1_init, tau1_bounds[0], tau1_bounds[1])

    # Update parameters
    param_set.set_value("eta1", eta1_init)
    param_set.set_value("eta2", eta2_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau1", tau1_init)

    return True


def initialize_fractional_burgers(
    omega: np.ndarray, G_star: np.ndarray, param_set
) -> bool:
    """Smart initialization for FractionalBurgers from oscillation data.

    Model equation (in compliance):
        J*(ω) = J_g + (iω)^(-α) / (η_1 Γ(1-α)) + J_k / (1 + (iωτ_k)^α)
        G*(ω) = 1 / J*(ω)

    Extraction strategy (simplified for 5-parameter model):
        - Jg: from 1/high_plateau (glassy compliance)
        - Jk: from compliance difference
        - eta1: from low-frequency flow behavior
        - tau_k: from transition frequency
        - alpha: from slope or default to 0.5

    Note: This is a simplified initialization for a complex model.
    The optimizer will refine these starting values.

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
    """
    features = extract_frequency_features(omega, G_star)

    if not features["valid"]:
        return False

    epsilon = 1e-12

    # Jg: glassy compliance from 1/high_plateau
    Jg_init = 1.0 / max(features["high_plateau"], epsilon)

    # Jk: Kelvin compliance from compliance difference
    Jk_init = (1.0 / max(features["low_plateau"], epsilon)) - Jg_init
    Jk_init = max(Jk_init, epsilon)

    # eta1: viscosity from low-frequency behavior
    omega_low = omega[0] if len(omega) > 0 else 1e-2
    eta1_init = max(features["low_plateau"] / (omega_low + epsilon), epsilon)

    # tau_k: from transition frequency
    tau_k_init = 1.0 / (features["omega_mid"] + epsilon)

    # alpha: from slope or default to 0.5
    if 0.01 <= features["alpha_estimate"] <= 0.99:
        alpha_init = features["alpha_estimate"]
    else:
        alpha_init = 0.5

    # Clip to parameter bounds
    Jg_bounds = param_set._parameters["Jg"].bounds
    eta1_bounds = param_set._parameters["eta1"].bounds
    Jk_bounds = param_set._parameters["Jk"].bounds
    alpha_bounds = param_set._parameters["alpha"].bounds
    tau_k_bounds = param_set._parameters["tau_k"].bounds

    Jg_init = np.clip(Jg_init, Jg_bounds[0], Jg_bounds[1])
    eta1_init = np.clip(eta1_init, eta1_bounds[0], eta1_bounds[1])
    Jk_init = np.clip(Jk_init, Jk_bounds[0], Jk_bounds[1])
    alpha_init = np.clip(alpha_init, alpha_bounds[0], alpha_bounds[1])
    tau_k_init = np.clip(tau_k_init, tau_k_bounds[0], tau_k_bounds[1])

    # Update parameters
    param_set.set_value("Jg", Jg_init)
    param_set.set_value("eta1", eta1_init)
    param_set.set_value("Jk", Jk_init)
    param_set.set_value("alpha", alpha_init)
    param_set.set_value("tau_k", tau_k_init)

    return True


__all__ = [
    "extract_frequency_features",
    "initialize_fractional_zener_ss",
    "initialize_fractional_maxwell_liquid",
    "initialize_fractional_maxwell_gel",
    "initialize_fractional_zener_ll",
    "initialize_fractional_zener_sl",
    "initialize_fractional_kelvin_voigt",
    "initialize_fractional_maxwell_model",
    "initialize_fractional_kv_zener",
    "initialize_fractional_poynting_thomson",
    "initialize_fractional_jeffreys",
    "initialize_fractional_burgers",
]
