"""Prony series conversion transform.

Converts between time-domain and frequency-domain viscoelastic data using
Prony series decomposition:

- G(t) → G'(ω), G''(ω) via analytical Prony relations
- G'(ω), G''(ω) → G(t) via inverse Prony fitting

The Prony series representation is:

    G(t) = G_e + Σ G_i exp(-t/τ_i)

    G'(ω) = G_e + Σ G_i ω²τ_i² / (1 + ω²τ_i²)
    G''(ω) = Σ G_i ωτ_i / (1 + ω²τ_i²)
"""

from __future__ import annotations

import warnings
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
class PronyResult:
    """Result from Prony conversion."""

    G_i: np.ndarray
    tau_i: np.ndarray
    G_e: float
    n_modes: int
    r_squared: float | None = None


@TransformRegistry.register("prony_conversion", type="spectral")
class PronyConversion(BaseTransform):
    """Convert between time-domain and frequency-domain via Prony series.

    This transform uses the Prony series utilities in ``rheojax.utils.prony``
    for fitting and conversion.

    Args:
        n_modes: Number of Prony modes (default: auto-select).
        direction: ``"time_to_freq"`` or ``"freq_to_time"``.
        omega_out: Target frequency array for time→freq conversion.
        t_out: Target time array for freq→time conversion.
    """

    def __init__(
        self,
        n_modes: int | None = None,
        direction: str = "time_to_freq",
        omega_out: np.ndarray | None = None,
        t_out: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.direction = direction
        self.omega_out = omega_out
        self.t_out = t_out
        self._prony_result: PronyResult | None = None

    def _transform(self, data: RheoData) -> tuple[RheoData, dict[str, Any]]:
        """Apply Prony conversion.

        Args:
            data: Input RheoData. For ``time_to_freq``, expects relaxation
                data (t, G(t)). For ``freq_to_time``, expects oscillation
                data (ω, G* = G' + iG'').

        Returns:
            Tuple of (converted RheoData, metadata dict with PronyResult).
        """
        if self.direction == "time_to_freq":
            return self._time_to_freq(data)
        elif self.direction == "freq_to_time":
            return self._freq_to_time(data)
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

    def _time_to_freq(self, data: RheoData) -> tuple[RheoData, dict]:
        """G(t) → G'(ω), G''(ω) via Prony fit."""

        t = np.asarray(data.x)
        G_t = np.asarray(data.y)

        # Determine number of modes
        n_modes = self.n_modes or max(3, min(len(t) // 5, 20))

        # Fit Prony series to G(t)
        G_i, tau_i, G_e = _fit_prony_relaxation(t, G_t, n_modes)

        self._prony_result = PronyResult(
            G_i=G_i, tau_i=tau_i, G_e=G_e, n_modes=len(G_i)
        )

        # Convert to frequency domain
        omega = self.omega_out
        if omega is None:
            omega = np.logspace(
                np.log10(0.1 / np.max(tau_i)),
                np.log10(10.0 / np.min(tau_i)),
                100,
            )

        G_prime, G_double_prime = _prony_to_frequency(G_i, tau_i, G_e, omega)
        G_star = G_prime + 1j * G_double_prime

        result_data = RheoData(
            x=omega,
            y=G_star,
            metadata={
                "test_mode": "oscillation",
                "source_transform": "prony_conversion",
                "n_modes": len(G_i),
            },
        )
        return result_data, {"prony_result": self._prony_result}

    def _freq_to_time(self, data: RheoData) -> tuple[RheoData, dict]:
        """G'(ω), G''(ω) → G(t) via Prony fit to dynamic moduli."""
        omega = np.asarray(data.x)
        y = np.asarray(data.y)

        if np.iscomplexobj(y):
            G_prime = y.real
            G_double_prime = y.imag
        elif y.ndim == 2 and y.shape[1] == 2:
            G_prime = y[:, 0]
            G_double_prime = y[:, 1]
        else:
            raise ValueError(
                "freq_to_time requires complex G* or (N, 2) array [G', G'']"
            )

        n_modes = self.n_modes or max(3, min(len(omega) // 5, 20))

        G_i, tau_i, G_e = _fit_prony_oscillation(
            omega, G_prime, G_double_prime, n_modes
        )

        self._prony_result = PronyResult(
            G_i=G_i, tau_i=tau_i, G_e=G_e, n_modes=len(G_i)
        )

        t = self.t_out
        if t is None:
            t = np.logspace(
                np.log10(np.min(tau_i) / 10),
                np.log10(np.max(tau_i) * 10),
                100,
            )

        G_t = _prony_to_time(G_i, tau_i, G_e, t)

        result_data = RheoData(
            x=t,
            y=G_t,
            metadata={
                "test_mode": "relaxation",
                "source_transform": "prony_conversion",
                "n_modes": len(G_i),
            },
        )
        return result_data, {"prony_result": self._prony_result}


# ---------------------------------------------------------------------------
# Analytical Prony conversions (pure functions, JIT-safe)
# ---------------------------------------------------------------------------


@jax.jit
def _prony_to_frequency_jax(
    G_i: Any, tau_i: Any, G_e: float, omega: Any
) -> tuple[Any, Any]:
    """JIT-compiled Prony to frequency conversion."""
    wt2 = (omega[:, None] * tau_i[None, :]) ** 2
    G_prime = G_e + jnp.sum(G_i[None, :] * wt2 / (1.0 + wt2), axis=1)
    G_double_prime = jnp.sum(
        G_i[None, :] * omega[:, None] * tau_i[None, :] / (1.0 + wt2), axis=1
    )
    return G_prime, G_double_prime


def _prony_to_frequency(
    G_i: np.ndarray, tau_i: np.ndarray, G_e: float, omega: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute G'(ω) and G''(ω) from Prony parameters (JIT-accelerated)."""
    G_p, G_dp = _prony_to_frequency_jax(
        jnp.asarray(G_i, dtype=jnp.float64),
        jnp.asarray(tau_i, dtype=jnp.float64),
        G_e,
        jnp.asarray(omega, dtype=jnp.float64),
    )
    return np.asarray(G_p), np.asarray(G_dp)


@jax.jit
def _prony_to_time_jax(G_i: Any, tau_i: Any, G_e: float, t: Any) -> Any:
    """JIT-compiled Prony to time conversion."""
    return G_e + jnp.sum(G_i[None, :] * jnp.exp(-t[:, None] / tau_i[None, :]), axis=1)


def _prony_to_time(
    G_i: np.ndarray, tau_i: np.ndarray, G_e: float, t: np.ndarray
) -> np.ndarray:
    """Compute G(t) from Prony parameters (JIT-accelerated)."""
    return np.asarray(
        _prony_to_time_jax(
            jnp.asarray(G_i, dtype=jnp.float64),
            jnp.asarray(tau_i, dtype=jnp.float64),
            G_e,
            jnp.asarray(t, dtype=jnp.float64),
        )
    )


def _fit_prony_relaxation(
    t: np.ndarray, G_t: np.ndarray, n_modes: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Prony series to relaxation modulus G(t)."""
    from scipy.optimize import nnls

    # Log-spaced relaxation times (use t[0] if positive, else t[1])
    if len(t) < 2:
        raise ValueError("Prony fitting requires at least 2 time points.")
    t_positive = t[t > 0]
    if len(t_positive) == 0:
        raise ValueError("Prony fitting requires at least one positive time value.")
    t_min = float(np.min(t_positive))
    tau_i = np.logspace(np.log10(t_min), np.log10(t[-1]), n_modes)

    # Equilibrium modulus estimate
    G_e = max(float(G_t[-1]), 0.0)

    # Build kernel matrix: A_ij = exp(-t_j / tau_i)
    A = np.exp(-t[:, None] / tau_i[None, :])

    # Solve non-negative least squares: G(t) - G_e ≈ A @ G_i
    b = np.maximum(G_t - G_e, 0.0)
    G_i, _ = nnls(A, b)

    return G_i, tau_i, G_e


def _fit_prony_oscillation(
    omega: np.ndarray,
    G_prime: np.ndarray,
    G_double_prime: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Prony series to dynamic moduli G'(ω), G''(ω)."""
    from scipy.optimize import nnls

    # T-04: Ensure omega is sorted ascending for correct tau range computation
    omega = np.sort(omega)

    tau_i = np.logspace(np.log10(1.0 / omega[-1]), np.log10(1.0 / omega[0]), n_modes)

    G_e = max(float(np.min(G_prime)), 0.0)

    # Build kernel matrices
    n = len(omega)
    A = np.zeros((2 * n, n_modes))
    wt2 = (omega[:, None] * tau_i[None, :]) ** 2
    A[:n, :] = wt2 / (1.0 + wt2)  # G' kernel
    A[n:, :] = omega[:, None] * tau_i[None, :] / (1.0 + wt2)  # G'' kernel

    b = np.concatenate([G_prime - G_e, G_double_prime])
    neg_mask = b < 0
    if np.any(neg_mask):
        n_neg = int(np.sum(neg_mask))
        warnings.warn(
            f"Prony fit: {n_neg} negative target values clipped to zero "
            "(noisy data or overestimated G_e).",
            stacklevel=2,
        )
    b = np.maximum(b, 0.0)
    G_i, _ = nnls(A, b)

    return G_i, tau_i, G_e
