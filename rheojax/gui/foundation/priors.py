"""Prior format conversion utilities.

Converts between PriorsEditor's ``{"distribution", "params"}`` format and the
``{"type", ...}`` format expected by ``prior_dict_to_dist``.
"""

from __future__ import annotations

import math


def adapt_prior(editor_dict: dict) -> dict:
    """Convert one PriorsEditor entry to the ``prior_dict_to_dist`` format.

    Parameters
    ----------
    editor_dict:
        A single entry from ``PriorsEditor.get_all_priors()``, e.g.
        ``{"distribution": "lognormal", "params": {"loc": 0.0, "scale": 1.0}}``.

    Returns
    -------
    dict
        ``{"type": <name>, **params}`` as consumed by ``prior_dict_to_dist``.

    Notes
    -----
    PriorsEditor stores ``scale`` for the Exponential distribution, but NumPyro's
    ``Exponential`` uses ``rate = 1/scale``.  This conversion is applied here.
    """
    dist_name = editor_dict["distribution"].lower()
    params = dict(editor_dict.get("params", {}))
    t = dist_name
    # PriorsEditor stores scale for exponential, but NumPyro Exponential uses rate
    if dist_name == "exponential" and "scale" in params:
        scale = params.pop("scale")
        params["rate"] = 1.0 / scale if scale != 0.0 else 1.0
    return {"type": t, **params}


def map_centered_priors(map_estimate: dict[str, float]) -> dict[str, dict]:
    """Build LogNormal priors centered on MAP values plus a HalfNormal sigma prior.

    Parameters
    ----------
    map_estimate:
        ``{param_name: map_value}`` from a MAP optimisation run.

    Returns
    -------
    dict
        ``{param_name: {"type": "lognormal", "loc": log(val), "scale": 1.0}, ...,
        "sigma": {"type": "halfnormal", "scale": 1.0}}`` for non-negative MAP
        values. A strictly negative MAP value instead gets
        ``{"type": "normal", "loc": val, "scale": max(|val|, 1.0)}``, since
        LogNormal's support is ``(0, inf)`` and cannot represent a negative
        value at all -- not even with the "wrong" ``loc``.

    Notes
    -----
    ``loc = log(val)`` so the LogNormal median equals the MAP value.
    ``scale = 1.0`` is a broad default; callers may tighten it.
    Zero or None values fall back to ``loc = 0.0``.
    """
    priors: dict[str, dict] = {}
    for name, val in map_estimate.items():
        if val not in (0.0, None) and val < 0:
            priors[name] = {"type": "normal", "loc": val, "scale": max(abs(val), 1.0)}
        else:
            loc = math.log(val) if val not in (0.0, None) else 0.0
            priors[name] = {"type": "lognormal", "loc": loc, "scale": 1.0}
    priors["sigma"] = {"type": "halfnormal", "scale": 1.0}
    return priors
