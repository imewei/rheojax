"""Base initializer abstract class for fractional model initialization.

This module provides the template method pattern for smart parameter initialization
of fractional viscoelastic models from oscillation data. All concrete initializers
inherit from BaseInitializer and implement model-specific feature extraction logic.

The template method enforces a consistent algorithm across all 11 fractional models:
1. Validate input data (common validation)
2. Estimate equilibrium modulus (model-specific)
3. Estimate glassy modulus (model-specific)
4. Estimate fractional order (model-specific)
5. Estimate characteristic time (model-specific)
6. Set parameters in ParameterSet (model-specific)

This pattern eliminates 70-90% code duplication while maintaining exact behavior
of the original initialization functions.

Phase 2: Constants extracted to constants.py module for improved maintainability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rheojax.logging import get_logger
from rheojax.utils.initialization.constants import FEATURE_CONFIG, PARAM_BOUNDS

logger = get_logger(__name__)


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
    logger.debug(
        "Extracting frequency features",
        omega_len=len(omega),
        G_star_len=len(G_star),
    )

    # Handle empty arrays defensively
    if len(omega) == 0 or len(G_star) == 0:
        from rheojax.utils.initialization.constants import DEFAULT_PARAMS

        logger.debug(
            "Empty array detected, returning defaults",
            omega_len=len(omega),
            G_star_len=len(G_star),
        )
        return {
            "low_plateau": DEFAULT_PARAMS.modulus,
            "high_plateau": DEFAULT_PARAMS.modulus,
            "omega_mid": 1.0 / DEFAULT_PARAMS.tau,
            "alpha_estimate": DEFAULT_PARAMS.alpha,
            "valid": False,
        }

    # Convert to magnitude
    if np.iscomplexobj(G_star):
        G_mag = np.abs(G_star)
    else:  # 2D [G', G"] format
        if G_star.ndim == 2 and G_star.shape[1] == 2:
            G_mag = np.sqrt(G_star[:, 0] ** 2 + G_star[:, 1] ** 2)
        else:
            G_mag = np.abs(G_star)  # Fall back to abs for 1D arrays

    # Smooth to reduce noise using configurable parameters
    if len(G_mag) >= FEATURE_CONFIG.savgol_window:
        from scipy.signal import savgol_filter

        G_mag_smooth = savgol_filter(
            G_mag,
            window_length=FEATURE_CONFIG.savgol_window,
            polyorder=FEATURE_CONFIG.savgol_poly,
        )
    else:
        G_mag_smooth = G_mag.copy()

    # Low-frequency plateau: average lowest fraction
    n_low = max(1, int(len(G_mag) * FEATURE_CONFIG.plateau_percentile))
    # Suppress warnings for edge cases with small arrays (handled by validation later)
    with np.errstate(invalid="ignore"):
        low_plateau = np.mean(np.sort(G_mag_smooth)[:n_low])

    # High-frequency plateau: average highest fraction
    n_high = max(1, int(len(G_mag) * FEATURE_CONFIG.plateau_percentile))
    with np.errstate(invalid="ignore"):
        high_plateau = np.mean(np.sort(G_mag_smooth)[-n_high:])

    # Find transition frequency (steepest slope in log-log)
    eps = FEATURE_CONFIG.epsilon
    log_omega = np.log10(omega + eps)
    log_G = np.log10(G_mag_smooth + eps)
    d_log_G = np.gradient(log_G, log_omega)
    idx_mid = np.argmax(np.abs(d_log_G))
    omega_mid = omega[idx_mid]

    # Estimate alpha from slope at transition
    alpha_estimate = d_log_G[idx_mid]
    alpha_estimate = np.clip(
        alpha_estimate, PARAM_BOUNDS.min_alpha, PARAM_BOUNDS.max_alpha
    )

    # Check validity using configurable thresholds
    freq_range = np.log10((omega.max() + eps) / (omega.min() + eps))
    plateau_ratio = high_plateau / (low_plateau + eps)
    valid = (
        freq_range > FEATURE_CONFIG.min_frequency_decades
        and plateau_ratio > FEATURE_CONFIG.min_plateau_ratio
    )

    features = {
        "low_plateau": float(low_plateau),
        "high_plateau": float(high_plateau),
        "omega_mid": float(omega_mid),
        "alpha_estimate": float(alpha_estimate),
        "valid": bool(valid),
    }

    logger.debug(
        "Feature extraction complete",
        low_plateau=features["low_plateau"],
        high_plateau=features["high_plateau"],
        omega_mid=features["omega_mid"],
        alpha_estimate=features["alpha_estimate"],
        valid=features["valid"],
        freq_range_decades=float(freq_range),
        plateau_ratio=float(plateau_ratio),
    )

    return features


class BaseInitializer(ABC):
    """Abstract base class for fractional model parameter initialization.

    Implements the template method pattern to enforce consistent initialization
    algorithm across all 11 fractional models while allowing model-specific
    parameter extraction logic.

    The template method `initialize()` defines the algorithm skeleton:
    1. Extract frequency features (common logic)
    2. Validate features (common logic)
    3. Estimate model-specific parameters (abstract methods)
    4. Clip to parameter bounds (common logic)
    5. Set parameters in ParameterSet (abstract method)

    Subclasses must implement all abstract methods to provide model-specific logic.

    Examples
    --------
    >>> class MyInitializer(BaseInitializer):
    ...     def _estimate_equilibrium_modulus(self, features):
    ...         return features['low_plateau']
    ...     # ... implement other abstract methods
    >>> initializer = MyInitializer()
    >>> success = initializer.initialize(omega, G_star, parameters)
    """

    def initialize(self, omega: np.ndarray, G_star: np.ndarray, param_set) -> bool:
        """Template method defining initialization algorithm.

        This method enforces consistent algorithm structure across all initializers
        while delegating model-specific logic to abstract methods implemented by
        subclasses.

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

        Notes
        -----
        The template method follows these steps:
        1. Extract frequency features using common logic
        2. Validate extracted features
        3. Call abstract methods to estimate parameters
        4. Clip parameters to bounds (common logic)
        5. Set parameters in ParameterSet via abstract method
        """
        initializer_name = self.__class__.__name__
        logger.debug(
            "Starting parameter initialization",
            initializer=initializer_name,
            omega_len=len(omega),
            G_star_len=len(G_star),
        )

        # Step 1: Extract frequency features (common logic)
        features = extract_frequency_features(omega, G_star)

        # Step 2: Validate features (common validation)
        if not self._validate_data(features):
            logger.debug(
                "Feature validation failed, falling back to defaults",
                initializer=initializer_name,
                valid=features.get("valid", False),
            )
            return False  # Fall back to defaults

        logger.debug(
            "Features validated successfully",
            initializer=initializer_name,
        )

        # Step 3: Estimate model-specific parameters (abstract methods)
        try:
            estimated_params = self._estimate_parameters(features)
        except Exception as e:
            logger.debug(
                "Parameter estimation failed, falling back to defaults",
                initializer=initializer_name,
                error=str(e),
            )
            return False
        logger.debug(
            "Parameters estimated",
            initializer=initializer_name,
            estimated_params=estimated_params,
        )

        # Step 4: Clip to parameter bounds (common logic)
        clipped_params = self._clip_to_bounds(estimated_params, param_set)
        logger.debug(
            "Parameters clipped to bounds",
            initializer=initializer_name,
            clipped_params=clipped_params,
        )

        # Step 5: Set parameters in ParameterSet (abstract method)
        self._set_parameters(param_set, clipped_params)

        logger.info(
            "Parameter initialization complete",
            initializer=initializer_name,
            params=clipped_params,
        )

        return True

    def _validate_data(self, features: dict) -> bool:
        """Common validation logic for extracted features.

        Parameters
        ----------
        features : dict
            Dictionary of extracted features from extract_frequency_features()

        Returns
        -------
        bool
            True if features are valid for initialization
        """
        return features.get("valid", False)

    @abstractmethod
    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate all model-specific parameters from frequency features.

        Subclasses implement this method to extract parameters specific to their
        rheological model. This is the core model-specific logic.

        Parameters
        ----------
        features : dict
            Dictionary from extract_frequency_features() containing:
            - low_plateau: float
            - high_plateau: float
            - omega_mid: float
            - alpha_estimate: float
            - valid: bool

        Returns
        -------
        dict
            Dictionary mapping parameter names to estimated values.
            Keys must match parameter names in the model's ParameterSet.

        Examples
        --------
        For FractionalZenerSolidSolid:
        >>> return {
        ...     'Ge': features['low_plateau'],
        ...     'Gm': features['high_plateau'] - features['low_plateau'],
        ...     'alpha': features['alpha_estimate'],
        ...     'tau_alpha': 1.0 / features['omega_mid']
        ... }
        """
        pass

    def _clip_to_bounds(self, estimated_params: dict, param_set) -> dict:
        """Clip estimated parameters to their defined bounds.

        Common logic to ensure all estimated parameters respect the bounds
        defined in the model's ParameterSet.

        Parameters
        ----------
        estimated_params : dict
            Dictionary of estimated parameter values
        param_set : ParameterSet
            ParameterSet containing parameter bounds

        Returns
        -------
        dict
            Dictionary of clipped parameter values
        """
        clipped = {}
        eps = FEATURE_CONFIG.epsilon

        for param_name, value in estimated_params.items():
            # Ensure value is at least epsilon to prevent zeros
            value = max(value, eps)

            # Check if parameter exists in ParameterSet
            if param_name in param_set._parameters:
                bounds = param_set._parameters[param_name].bounds
                # Clip to bounds
                clipped[param_name] = np.clip(value, bounds[0], bounds[1])
            else:
                # Parameter not in set, keep value as-is (don't clip)
                clipped[param_name] = value

        return clipped

    @abstractmethod
    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set clipped parameters in the ParameterSet.

        Subclasses implement this method to update the ParameterSet with
        estimated and clipped parameter values.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet object to update
        clipped_params : dict
            Dictionary of clipped parameter values

        Examples
        --------
        >>> def _set_parameters(self, param_set, clipped_params):
        ...     param_set.set_value('Ge', clipped_params['Ge'])
        ...     param_set.set_value('Gm', clipped_params['Gm'])
        ...     param_set.set_value('alpha', clipped_params['alpha'])
        ...     param_set.set_value('tau_alpha', clipped_params['tau_alpha'])
        """
        pass

    def _safe_set_parameter(self, param_set, name: str, value: float) -> bool:
        """Safely set a parameter if it exists in ParameterSet.

        This helper method checks if a parameter exists before attempting to set it.
        Useful for making initializers robust to different ParameterSet configurations.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        name : str
            Parameter name
        value : float
            Parameter value

        Returns
        -------
        bool
            True if parameter was set, False if parameter doesn't exist
        """
        # Check if parameter exists in ParameterSet (dict check)
        if name not in param_set._parameters:
            logger.debug(
                "Parameter not found in ParameterSet, skipping",
                parameter=name,
                initializer=self.__class__.__name__,
            )
            return False

        param_set.set_value(name, value)
        logger.debug(
            "Parameter set successfully",
            parameter=name,
            value=value,
            initializer=self.__class__.__name__,
        )
        return True
