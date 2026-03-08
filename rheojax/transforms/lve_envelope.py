"""Linear viscoelastic (LVE) envelope transform.

Computes the LVE startup stress envelope from a Prony series representation
of the relaxation modulus:

    σ_LVE⁺(t) = γ̇₀ Σ Gᵢτᵢ (1 − exp(−t/τᵢ))

This analytical expression is fully JIT-compilable and gives the linear
viscoelastic prediction for stress growth in a startup experiment.
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
class LVEEnvelopeResult:
    """Result from LVE envelope computation."""

    t: np.ndarray
    sigma_lve: np.ndarray
    G_i: np.ndarray
    tau_i: np.ndarray
    shear_rate: float


@TransformRegistry.register("lve_envelope", type="analysis")
class LVEEnvelope(BaseTransform):
    """Compute the linear viscoelastic startup stress envelope.

    The LVE envelope provides the theoretical stress growth response
    assuming linear viscoelasticity. Comparing experimental startup data
    with this envelope reveals nonlinear effects (strain hardening/softening).

    Args:
        shear_rate: Applied shear rate γ̇₀ (s⁻¹).
        G_i: Prony mode strengths (Pa). If None, must be in data metadata.
        tau_i: Prony relaxation times (s). If None, must be in data metadata.
        G_e: Equilibrium modulus (Pa, default 0).
        t_out: Time array for output. If None, auto-generated.
    """

    def __init__(
        self,
        shear_rate: float = 1.0,
        G_i: np.ndarray | None = None,
        tau_i: np.ndarray | None = None,
        G_e: float = 0.0,
        t_out: np.ndarray | None = None,
    ):
        super().__init__()
        self.shear_rate = shear_rate
        self.G_i = np.asarray(G_i) if G_i is not None else None
        self.tau_i = np.asarray(tau_i) if tau_i is not None else None
        self.G_e = G_e
        self.t_out = t_out
        self.result: LVEEnvelopeResult | None = None

    def _transform(self, data: RheoData | None = None) -> tuple[RheoData, dict[str, Any]]:
        """Compute LVE envelope.

        Args:
            data: Optional RheoData. If G_i/tau_i were not provided at
                construction, they are read from ``data.metadata``.
                If data has x values, those are used as the time array.

        Returns:
            Tuple of (RheoData with sigma_LVE(t), metadata dict).
        """
        G_i = self.G_i
        tau_i = self.tau_i
        G_e = self.G_e

        # Try reading from data metadata
        if data is not None:
            meta = getattr(data, "metadata", {}) or {}
            if G_i is None and "G_i" in meta:
                G_i = np.asarray(meta["G_i"])
            if tau_i is None and "tau_i" in meta:
                tau_i = np.asarray(meta["tau_i"])
            if abs(self.G_e) < 1e-30 and "G_e" in meta:
                G_e = float(meta["G_e"])

        if G_i is None or tau_i is None:
            raise ValueError(
                "Prony parameters G_i and tau_i must be provided either at "
                "construction or in data.metadata"
            )

        G_i = np.asarray(G_i, dtype=np.float64)
        tau_i = np.asarray(tau_i, dtype=np.float64)

        if len(G_i) != len(tau_i):
            raise ValueError(
                f"G_i ({len(G_i)}) and tau_i ({len(tau_i)}) must have same length"
            )

        # T-12: Guard against non-positive relaxation times
        if np.any(tau_i <= 0):
            raise ValueError(
                f"All relaxation times tau_i must be positive, "
                f"got min={np.min(tau_i)}"
            )

        # Time array
        t = self.t_out
        if t is None and data is not None:
            t = np.asarray(data.x)
        if t is None:
            t_max = 10.0 * np.max(tau_i)
            t = np.logspace(-2, np.log10(t_max), 200)

        # Compute σ_LVE⁺(t) = γ̇₀ [G_e * t + Σ Gᵢτᵢ (1 − exp(−t/τᵢ))]
        sigma_lve = lve_envelope(t, G_i, tau_i, G_e, self.shear_rate)

        self.result = LVEEnvelopeResult(
            t=t,
            sigma_lve=sigma_lve,
            G_i=G_i,
            tau_i=tau_i,
            shear_rate=self.shear_rate,
        )

        result_data = RheoData(
            x=t,
            y=sigma_lve,
            metadata={
                "test_mode": "startup",
                "source_transform": "lve_envelope",
                "shear_rate": self.shear_rate,
                "n_modes": len(G_i),
            },
        )
        return result_data, {"lve_result": self.result}


# ---------------------------------------------------------------------------
# Pure computation (JIT-safe)
# ---------------------------------------------------------------------------


@jax.jit
def _lve_envelope_jax(
    t: jnp.ndarray,
    G_i: jnp.ndarray,
    tau_i: jnp.ndarray,
    G_e: float,
    shear_rate: float,
) -> jnp.ndarray:
    """JIT-compiled LVE envelope computation."""
    # modulus_integral = G_e * t + Σ Gᵢτᵢ (1 − exp(−t/τᵢ))
    modulus_integral = G_e * t + jnp.sum(
        G_i[None, :] * tau_i[None, :] * (1.0 - jnp.exp(-t[:, None] / tau_i[None, :])),
        axis=1,
    )
    return shear_rate * modulus_integral


def lve_envelope(
    t: np.ndarray,
    G_i: np.ndarray,
    tau_i: np.ndarray,
    G_e: float = 0.0,
    shear_rate: float = 1.0,
) -> np.ndarray:
    """Compute LVE startup stress envelope.

    σ_LVE⁺(t) = γ̇₀ [G_e * t + Σ Gᵢτᵢ (1 − exp(−t/τᵢ))]

    Uses JAX JIT compilation for vectorized evaluation.

    Args:
        t: Time array (s).
        G_i: Prony mode strengths (Pa).
        tau_i: Prony relaxation times (s).
        G_e: Equilibrium modulus (Pa).
        shear_rate: Applied shear rate (s⁻¹).

    Returns:
        Stress envelope σ_LVE⁺(t) in Pa.
    """
    t_j = jnp.asarray(t, dtype=jnp.float64)
    G_i_j = jnp.asarray(G_i, dtype=jnp.float64)
    tau_i_j = jnp.asarray(tau_i, dtype=jnp.float64)

    return np.asarray(_lve_envelope_jax(t_j, G_i_j, tau_i_j, G_e, shear_rate))
