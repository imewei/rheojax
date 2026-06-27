"""Utilities for safely importing optional ArviZ dependency."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

from rheojax.logging import get_logger

logger = get_logger(__name__)


def _arviz_major_version(arviz: ModuleType) -> int:
    """Return ArviZ's major version, defaulting to the legacy API."""

    version = str(getattr(arviz, "__version__", "0"))
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return 0


def import_arviz(*, required: Iterable[str] | None = None) -> ModuleType:
    """Return the ArviZ module or raise ImportError/RuntimeError as needed.

    Args:
        required: Optional iterable of attribute names that must exist on the
            ArviZ module (e.g., "plot_pair", "plot_forest", "from_numpyro").

    Raises:
        ImportError: When ArviZ itself cannot be imported or has been disabled.
        RuntimeError: When ArviZ imports but is missing required helpers.
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
            "ArviZ is required for Bayesian diagnostics. Install it with `pip install arviz`."
        ) from exc

    if required:
        missing = sorted(name for name in required if not hasattr(arviz, name))
        if missing:
            missing_csv = ", ".join(missing)
            raise RuntimeError(
                "ArviZ is installed but missing required component(s) "
                f"[{missing_csv}]. Reinstall ArviZ with optional plotting extras "
                "(`pip install arviz[plots]`) to enable Bayesian diagnostics."
            )

    return arviz


def inference_data_from_dict(
    groups: Mapping[str, Mapping[str, Any] | None],
) -> Any:
    """Build inference data across the ArviZ 0.x and 1.x dictionary APIs.

    ``groups`` always uses the ArviZ group-oriented representation, for
    example ``{"posterior": {"tau": samples}}``. ArviZ 1.x accepts that
    mapping as its positional ``data`` argument, while ArviZ 0.x expects the
    groups as keyword arguments.
    """

    arviz = import_arviz(required=("from_dict",))
    populated_groups = {name: data for name, data in groups.items() if data is not None}
    if _arviz_major_version(arviz) >= 1:
        return arviz.from_dict(populated_groups)
    return arviz.from_dict(**populated_groups)


def arviz_plot_kwargs(
    arviz: ModuleType, plot_name: str, /, **kwargs: Any
) -> dict[str, Any]:
    """Translate supported ArviZ 0.x plotting options to the 1.x API.

    ArviZ 1.x plotting functions accept unknown keywords through ``**pc_kwargs``.
    Passing removed 0.x options therefore fails later as an invalid aesthetic
    mapping instead of raising a useful signature error. This function keeps
    that compatibility policy in one place and leaves 0.x calls untouched.
    """

    normalized = dict(kwargs)
    if _arviz_major_version(arviz) < 1:
        return normalized

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
        inner_prob = 0.5 if hdi_prob > 0.5 else hdi_prob / 2
        normalized.setdefault("ci_probs", (inner_prob, hdi_prob))

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
