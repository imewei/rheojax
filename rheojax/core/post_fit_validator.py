"""Post-fit validation: physics checks and uncertainty quantification.

Extracted from BaseModel.fit() to reduce its cyclomatic complexity.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from rheojax.core.jax_config import safe_import_jax
from rheojax.io._exceptions import RheoJaxPhysicsWarning
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    import numpy as np

    type ArrayLike = np.ndarray

logger = get_logger(__name__)


class RheoJaxUncertaintyWarning(UserWarning):
    """Emitted when uncertainty computation (hessian/bootstrap) fails to run.

    Subclasses UserWarning so standard warning filters apply.
    """


class PostFitValidator:
    """Runs post-fit physics checks and uncertainty quantification."""

    @staticmethod
    def check_physics(model: Any) -> None:
        """Run physics validation and emit warnings for violations.

        Args:
            model: Fitted BaseModel instance.
        """
        try:
            from rheojax.utils.physics_checks import check_fit_physics

            violations = check_fit_physics(model)
            for v in violations:
                warnings.warn(
                    f"{v.check}: {v.message} ({v.parameter}={v.value})",
                    RheoJaxPhysicsWarning,
                    stacklevel=3,
                )
        except Exception as exc:
            # check_physics is advisory: a bug in one check rule must not crash
            # fit(), but check_physics=True still promises RheoJaxPhysicsWarning
            # on violations, so a failure to even run has to be visible too.
            logger.warning(
                "Physics check failed",
                model=model.__class__.__name__,
                error=str(exc),
                exc_info=True,
            )
            warnings.warn(
                f"check_physics failed to run: {exc}",
                RheoJaxPhysicsWarning,
                stacklevel=3,
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
        if uncertainty not in ("hessian", "bootstrap"):
            raise ValueError(
                f"Unknown uncertainty method: {uncertainty!r}. "
                "Expected 'hessian' or 'bootstrap'."
            )

        try:
            from rheojax.utils.uncertainty import bootstrap_ci, hessian_ci

            if uncertainty == "hessian":
                return hessian_ci(model, X, y, test_mode=test_mode)
            else:
                return bootstrap_ci(model, X, y, test_mode=test_mode)
        except Exception as exc:
            # Mirror check_physics's policy: a failure to even run has to be
            # visible, not just logged, otherwise the caller gets a FitResult
            # that silently lacks uncertainty metadata.
            logger.warning(
                "Uncertainty computation failed",
                model=model.__class__.__name__,
                method=uncertainty,
                error=str(exc),
                exc_info=True,
            )
            warnings.warn(
                f"Uncertainty computation ({uncertainty}) failed to run: {exc}",
                RheoJaxUncertaintyWarning,
                stacklevel=3,
            )
            return None
