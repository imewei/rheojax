"""Utilities for safely importing optional ArviZ dependency."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

from rheojax.logging import get_logger

logger = get_logger(__name__)


def import_arviz(*, required: Iterable[str] | None = None) -> ModuleType:
    """Return the ArviZ module or raise ImportError/RuntimeError as needed.

    Args:
        required: Optional iterable of attribute names that must exist on the
            ArviZ module (e.g., "plot_pair", "plot_forest", "from_numpyro").

    Raises:
        ImportError: When ArviZ itself cannot be imported or has been disabled,
            or when ArviZ imports but is missing required helpers.
    """

    if sys.modules.get("arviz", object()) is None:
        # Explicitly disabled (e.g., via monkeypatch during testing)
        raise ImportError(
            "ArviZ appears to be disabled. Re-import the module to restore plotting capabilities."
        )

    try:
        arviz = importlib.import_module("arviz")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when ArviZ absent
        raise ImportError(
            "ArviZ is required for Bayesian diagnostics. Install it with "
            '`pip install "arviz>=1.2.0,<2.0.0"`.'
        ) from exc

    if required:
        missing = sorted(name for name in required if not hasattr(arviz, name))
        if missing:
            missing_csv = ", ".join(missing)
            raise ImportError(
                "ArviZ is installed but missing required component(s) "
                f"[{missing_csv}]. This usually means an incompatible ArviZ "
                "version or a partial install -- try `pip install -U "
                '"arviz>=1.2.0,<2.0.0"`, or `pip install arviz[matplotlib]` '
                "if this is a plotting-backend gap."
            )

    return arviz


def inference_data_from_dict(
    groups: Mapping[str, Mapping[str, Any] | None],
) -> Any:
    """Build InferenceData from a group-oriented dict via ArviZ 1.x's API.

    ``groups`` uses the ArviZ group-oriented representation, for example
    ``{"posterior": {"tau": samples}}``.
    """

    arviz = import_arviz(required=("from_dict",))
    populated_groups = {name: data for name, data in groups.items() if data is not None}
    return arviz.from_dict(populated_groups)


def arviz_plot_kwargs(
    arviz: ModuleType, plot_name: str, /, **kwargs: Any
) -> dict[str, Any]:
    """Translate 0.x-shaped plotting kwargs to ArviZ 1.x's kwarg shape.

    ArviZ 1.x plotting functions accept unknown keywords through ``**pc_kwargs``.
    Passing removed 0.x options therefore fails later as an invalid aesthetic
    mapping instead of raising a useful signature error. This function keeps
    that translation policy in one place.
    """
    # ponytail: arviz param is now unused (0.x/1.x version dispatch was
    # removed -- only 1.x is supported) but kept for signature stability
    # across the 8+ existing call sites that already pass it.

    normalized = dict(kwargs)

    if plot_name == "plot_pair":
        kind = normalized.pop("kind", "scatter")
        if kind != "scatter":
            raise ValueError(
                "ArviZ 1.x plot_pair only supports scatter pair plots; "
                f"kind={kind!r} is unavailable"
            )

        if "marginals" in normalized:
            normalized.setdefault("marginal", normalized.pop("marginals"))
        else:
            normalized.setdefault("marginal", False)

        divergences = normalized.pop("divergences", None)
        if divergences is not None:
            visuals = dict(normalized.get("visuals") or {})
            visuals.setdefault("divergence", divergences)
            normalized["visuals"] = visuals

    elif plot_name == "plot_forest" and "hdi_prob" in normalized:
        hdi_prob = normalized.pop("hdi_prob")
        normalized.setdefault("ci_kind", "hdi")
        # ponytail: heuristic inner HDI band — 0.5 for prob>0.5, half otherwise;
        # replace with explicit ci_probs kwarg if ArviZ exposes a migration guide.
        inner_prob = 0.5 if hdi_prob > 0.5 else hdi_prob / 2
        normalized.setdefault("ci_probs", (inner_prob, hdi_prob))

    elif plot_name == "plot_posterior":
        # ArviZ 1.x has no plot_posterior; callers pass this name to look up
        # kwargs while calling az.plot_dist directly (see arviz_canvas.py).
        hdi_prob = normalized.pop("hdi_prob", None)
        if hdi_prob is not None:
            normalized.setdefault("ci_prob", hdi_prob)
        normalized.setdefault("ci_kind", "hdi")
        normalized.setdefault("point_estimate", "mean")
        normalized.setdefault("kind", "kde")

    elif plot_name == "plot_trace" and "combined" in normalized:
        combined = normalized.pop("combined")
        normalized.setdefault(
            "sample_dims", ("chain", "draw") if combined else ("draw",)
        )

    elif plot_name == "plot_autocorr":
        combined = normalized.pop("combined", False)
        normalized.setdefault(
            "sample_dims", ("chain", "draw") if combined else ("draw",)
        )

    return normalized


def arviz_figure(plot_result: Any) -> Any:
    """Extract the Matplotlib figure from ArviZ 0.x or 1.x plot output."""

    if hasattr(plot_result, "figure"):
        return plot_result.figure

    if hasattr(plot_result, "ravel"):
        flattened = plot_result.ravel()
        if len(flattened) and hasattr(flattened[0], "figure"):
            return flattened[0].figure

    if isinstance(plot_result, list) and plot_result:
        return arviz_figure(plot_result[0])

    # PlotCollection and PlotMatrix expose their rendered objects through the
    # public ``viz`` DataTree. Reading its scalar figure is concurrency-safe.
    viz = getattr(plot_result, "viz", None)
    if viz is not None:
        try:
            figure_data = viz["figure"]
            return figure_data.item()
        except (KeyError, TypeError, ValueError):
            pass

    raise RuntimeError(
        "ArviZ plot function returned unrecognized result type: "
        f"{type(plot_result).__name__}"
    )
