"""Cox-Merz rule validation transform.

The Cox-Merz rule states that the complex viscosity magnitude equals the
steady shear viscosity at the same rate:

    |η*(ω)| = η(γ̇)  at  ω = γ̇

This transform takes two RheoData inputs (oscillation + flow curve),
interpolates to a common grid, and computes the deviation metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


@dataclass
class CoxMerzResult:
    """Result from Cox-Merz validation."""

    common_rates: np.ndarray
    eta_complex: np.ndarray
    eta_steady: np.ndarray
    deviation: np.ndarray
    mean_deviation: float
    max_deviation: float
    passes: bool


@TransformRegistry.register("cox_merz", type="analysis")
class CoxMerz(BaseTransform):
    """Cox-Merz rule validation.

    Compares |η*(ω)| from oscillation data with η(γ̇) from flow curve data
    to assess whether the Cox-Merz rule holds for a given material.

    Args:
        tolerance: Maximum mean relative deviation for the rule to "pass"
            (default: 0.1 = 10%).
        n_points: Number of interpolation points on the common grid.
    """

    def __init__(self, tolerance: float = 0.10, n_points: int = 50):
        super().__init__()
        self.tolerance = tolerance
        self.n_points = n_points
        self.result: CoxMerzResult | None = None

    def _transform(
        self, data: list[RheoData]
    ) -> tuple[RheoData, dict[str, Any]]:
        """Apply Cox-Merz comparison.

        Args:
            data: List of two RheoData objects:
                [0] = oscillation data (ω, G* = G' + iG'')
                [1] = flow curve data (γ̇, η or σ)

        Returns:
            Tuple of (RheoData with deviation, metadata dict).
        """
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            raise ValueError(
                "CoxMerz requires exactly 2 RheoData inputs: "
                "[oscillation, flow_curve]"
            )

        osc_data, flow_data = data[0], data[1]

        # Extract complex viscosity |η*(ω)| = |G*| / ω
        omega = np.asarray(osc_data.x)
        y_osc = np.asarray(osc_data.y)

        if np.iscomplexobj(y_osc):
            G_star_mag = np.abs(y_osc)
        elif y_osc.ndim == 2 and y_osc.shape[1] == 2:
            G_star_mag = np.sqrt(y_osc[:, 0] ** 2 + y_osc[:, 1] ** 2)
        else:
            G_star_mag = np.abs(y_osc)

        eta_star = G_star_mag / omega

        # Extract steady-shear viscosity η(γ̇)
        gamma_dot = np.asarray(flow_data.x)
        y_flow = np.asarray(flow_data.y)

        # Flow data might be σ(γ̇) or η(γ̇) — detect by metadata or magnitude
        flow_meta = getattr(flow_data, "metadata", {}) or {}
        if flow_meta.get("quantity") == "viscosity" or flow_meta.get("is_viscosity"):
            eta_steady_raw = y_flow
        else:
            # Assume stress → η = σ/γ̇
            eta_steady_raw = y_flow / gamma_dot

        # Build common log-spaced rate grid
        rate_min = max(np.min(omega), np.min(gamma_dot))
        rate_max = min(np.max(omega), np.max(gamma_dot))

        if rate_min >= rate_max:
            raise ValueError(
                f"No overlapping rate range: oscillation [{np.min(omega):.2g}, "
                f"{np.max(omega):.2g}], flow [{np.min(gamma_dot):.2g}, "
                f"{np.max(gamma_dot):.2g}]"
            )

        common_rates = np.logspace(
            np.log10(rate_min), np.log10(rate_max), self.n_points
        )

        # Interpolate in log-log space (np.interp requires sorted x-array)
        sort_o = np.argsort(omega)
        sort_g = np.argsort(gamma_dot)
        eta_c = np.exp(
            np.interp(np.log(common_rates), np.log(omega[sort_o]), np.log(eta_star[sort_o]))
        )
        eta_s = np.exp(
            np.interp(
                np.log(common_rates), np.log(gamma_dot[sort_g]), np.log(eta_steady_raw[sort_g])
            )
        )

        # Relative deviation: |η* - η| / η*
        deviation = np.abs(eta_c - eta_s) / np.maximum(eta_c, 1e-30)
        mean_dev = float(np.mean(deviation))
        max_dev = float(np.max(deviation))

        self.result = CoxMerzResult(
            common_rates=common_rates,
            eta_complex=eta_c,
            eta_steady=eta_s,
            deviation=deviation,
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            passes=mean_dev <= self.tolerance,
        )

        result_data = RheoData(
            x=common_rates,
            y=deviation,
            metadata={
                "source_transform": "cox_merz",
                "mean_deviation": mean_dev,
                "max_deviation": max_dev,
                "passes": mean_dev <= self.tolerance,
            },
        )
        return result_data, {"cox_merz_result": self.result}
