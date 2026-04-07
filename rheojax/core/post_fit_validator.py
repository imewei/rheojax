"""Post-fit validation: physics checks and uncertainty quantification.

Extracted from BaseModel.fit() to reduce its cyclomatic complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    import numpy as np

    type ArrayLike = np.ndarray

logger = get_logger(__name__)


class PostFitValidator:
    """Runs post-fit physics checks and uncertainty quantification."""

    @staticmethod
    def check_physics(model: Any) -> None:
        """Run physics validation and emit warnings for violations.

        Args:
            model: Fitted BaseModel instance.
        """
        try:
            import warnings as _warnings

            from rheojax.io._exceptions import RheoJaxPhysicsWarning
            from rheojax.utils.physics_checks import check_fit_physics

            violations = check_fit_physics(model)
            for v in violations:
                _warnings.warn(
                    f"{v.check}: {v.message} ({v.parameter}={v.value})",
                    RheoJaxPhysicsWarning,
                    stacklevel=3,
                )
        except Exception as exc:
            logger.debug(
                "Physics check failed",
                model=model.__class__.__name__,
                error=str(exc),
            )

    @staticmethod
    def compute_uncertainty(
        model: Any,
        X: ArrayLike,
        y: ArrayLike,
        uncertainty: str,
        test_mode: str | None,
    ) -> dict | None:
        """Compute post-fit uncertainty estimates.

        Args:
            model: Fitted BaseModel instance.
            X: Input data.
            y: Target data.
            uncertainty: Method name — ``"hessian"`` or ``"bootstrap"``.
            test_mode: Protocol string or None.

        Returns:
            Uncertainty result dict, or None on failure.
        """
        try:
            from rheojax.utils.uncertainty import bootstrap_ci, hessian_ci

            if uncertainty == "hessian":
                return hessian_ci(model, X, y, test_mode=test_mode)
            elif uncertainty == "bootstrap":
                return bootstrap_ci(model, X, y, test_mode=test_mode)
            else:
                logger.warning("Unknown uncertainty method", method=uncertainty)
                return None
        except Exception as exc:
            logger.warning(
                "Uncertainty computation failed",
                model=model.__class__.__name__,
                method=uncertainty,
                error=str(exc),
            )
            return None
