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


def _prony_to_frequency(
    G_i: np.ndarray, tau_i: np.ndarray, G_e: float, omega: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute G'(ω) and G''(ω) from Prony parameters."""
    omega = np.asarray(omega)
    G_prime = np.full_like(omega, G_e, dtype=np.float64)
    G_double_prime = np.zeros_like(omega, dtype=np.float64)

    for g, tau in zip(G_i, tau_i, strict=False):
        wt2 = (omega * tau) ** 2
        G_prime += g * wt2 / (1.0 + wt2)
        G_double_prime += g * omega * tau / (1.0 + wt2)

    return G_prime, G_double_prime


def _prony_to_time(
    G_i: np.ndarray, tau_i: np.ndarray, G_e: float, t: np.ndarray
) -> np.ndarray:
    """Compute G(t) from Prony parameters."""
    t = np.asarray(t)
    G_t = np.full_like(t, G_e, dtype=np.float64)
    for g, tau in zip(G_i, tau_i, strict=False):
        G_t += g * np.exp(-t / tau)
    return G_t


def _fit_prony_relaxation(
    t: np.ndarray, G_t: np.ndarray, n_modes: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Prony series to relaxation modulus G(t)."""
    from scipy.optimize import nnls

    # Log-spaced relaxation times (use t[0] if positive, else t[1])
    if len(t) < 2:
        raise ValueError("Prony fitting requires at least 2 time points.")
    t_min = t[0] if t[0] > 0 else t[1]
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

    tau_i = np.logspace(
        np.log10(1.0 / omega[-1]), np.log10(1.0 / omega[0]), n_modes
    )

    G_e = max(float(np.min(G_prime)), 0.0)

    # Build kernel matrices
    n = len(omega)
    A = np.zeros((2 * n, n_modes))
    for j, tau in enumerate(tau_i):
        wt2 = (omega * tau) ** 2
        A[:n, j] = wt2 / (1.0 + wt2)  # G' kernel
        A[n:, j] = omega * tau / (1.0 + wt2)  # G'' kernel

    b = np.concatenate([G_prime - G_e, G_double_prime])
    b = np.maximum(b, 0.0)
    G_i, _ = nnls(A, b)

    return G_i, tau_i, G_e
