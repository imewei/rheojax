"""Shared helper functions for RheoJAX example notebooks.

Provides reusable utilities for:
- Extracting scipy-compatible bounds from model parameters
- Validating overshoot physics in startup shear experiments
"""

import warnings


def get_model_bounds(model, param_names):
    """Extract scipy-compatible bounds from model parameters.

    Returns a tuple of (lower_bounds, upper_bounds) lists that can be
    passed directly to ``scipy.optimize.curve_fit(bounds=...)``.

    Parameters
    ----------
    model : BaseModel
        A RheoJAX model instance with a ``parameters`` attribute.
    param_names : list of str
        Parameter names to extract bounds for. Must exist in
        ``model.parameters``.

    Returns
    -------
    bounds : tuple of (list, list)
        ``(lower_bounds, upper_bounds)`` suitable for scipy curve_fit.

    Examples
    --------
    >>> model = DMTLocal(closure="exponential", include_elasticity=True)
    >>> bounds = get_model_bounds(model, ["eta_0", "eta_inf", "G0"])
    >>> popt, pcov = curve_fit(fn, x, y, bounds=bounds)
    """
    lower = []
    upper = []
    for name in param_names:
        param = model.parameters[name]
        lo, hi = param.bounds
        lower.append(lo)
        upper.append(hi)
    return (lower, upper)


def validate_overshoot_ratios(ratios, warn_high=50.0, warn_low=1.01):
    """Validate stress overshoot ratios against thixotropic material physics.

    Checks that overshoot ratios (sigma_peak / sigma_ss) fall within
    physically reasonable ranges for soft thixotropic materials.

    Parameters
    ----------
    ratios : array-like
        Overshoot ratios (sigma_peak / sigma_ss) at different shear rates.
    warn_high : float, optional
        Upper threshold for warning. Default 50.0.
    warn_low : float, optional
        Lower threshold (no significant overshoot). Default 1.01.

    Returns
    -------
    max_ratio : float
        Maximum overshoot ratio found.
    is_valid : bool
        True if all ratios are within [warn_low, warn_high].
    """
    import numpy as np

    ratios = np.asarray(ratios, dtype=float)
    max_ratio = float(np.max(ratios))

    if max_ratio > warn_high:
        warnings.warn(
            f"Maximum overshoot ratio {max_ratio:.1f} exceeds {warn_high}. "
            "Typical values for soft materials: 1.5-50 depending on shear rate.",
            stacklevel=2,
        )
        return max_ratio, False
    elif not np.any(ratios >= warn_low):
        # No rate shows overshoot at all — likely missing elasticity
        warnings.warn(
            f"No significant overshoot detected (max ratio = {max_ratio:.3f}). "
            "Check that include_elasticity=True and shear rates are appropriate.",
            stacklevel=2,
        )
        return max_ratio, False
    else:
        # At least one rate shows overshoot; ratios near 1.0 at high shear
        # rates are physically expected (structure already broken)
        n_overshoot = int(np.sum(ratios >= warn_low))
        print(
            f"Overshoot ratios are physically reasonable "
            f"(max = {max_ratio:.1f}, {n_overshoot}/{len(ratios)} rates show overshoot)"
        )
        return max_ratio, True
