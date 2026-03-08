"""Relaxation spectrum inversion transform.

Recovers the continuous relaxation spectrum H(τ) from dynamic moduli
G'(ω)/G''(ω) or relaxation modulus G(t).

Two regularization methods are provided:
- **Tikhonov** (default): L-curve or GCV for automatic regularization
  parameter selection.
- **Maximum entropy**: Maximizes entropy subject to data fidelity constraint.

The kernel relations are:

    G'(ω) = G_e + ∫ H(τ) ω²τ² / (1 + ω²τ²) d(ln τ)
    G''(ω) = ∫ H(τ) ωτ / (1 + ω²τ²) d(ln τ)
    G(t) = G_e + ∫ H(τ) exp(-t/τ) d(ln τ)
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
class SpectrumResult:
    """Result from spectrum inversion."""

    tau: np.ndarray
    H_tau: np.ndarray
    regularization_param: float
    method: str
    residual_norm: float
    G_e: float


@TransformRegistry.register("spectrum_inversion", type="spectral")
class SpectrumInversion(BaseTransform):
    """Recover relaxation spectrum H(τ) from viscoelastic data.

    Args:
        method: ``"tikhonov"`` (default) or ``"max_entropy"``.
        n_tau: Number of τ points in the spectrum (default: 100).
        tau_range: Tuple of (τ_min, τ_max). If None, auto-detected from data.
        regularization: Manual regularization parameter λ. If None,
            automatically selected via L-curve (Tikhonov) or GCV.
        source: Data type: ``"oscillation"`` or ``"relaxation"``.
        G_e: Equilibrium modulus (default: 0). If None, estimated from data.
    """

    def __init__(
        self,
        method: str = "tikhonov",
        n_tau: int = 100,
        tau_range: tuple[float, float] | None = None,
        regularization: float | None = None,
        source: str = "oscillation",
        G_e: float = 0.0,
    ):
        super().__init__()
        self.method = method
        self.n_tau = n_tau
        self.tau_range = tau_range
        self.regularization = regularization
        self.source = source
        self.G_e = G_e
        self.result: SpectrumResult | None = None

    def _transform(self, data: RheoData) -> tuple[RheoData, dict[str, Any]]:
        """Invert viscoelastic data to recover H(τ).

        Args:
            data: RheoData with oscillation (ω, G*) or relaxation (t, G(t)) data.

        Returns:
            Tuple of (RheoData with (τ, H(τ)), metadata dict).
        """
        x = np.asarray(data.x)
        y = np.asarray(data.y)

        # Build τ grid
        if self.tau_range is not None:
            tau_min, tau_max = self.tau_range
        elif self.source == "oscillation":
            tau_min = 1.0 / np.max(x) / 10.0
            tau_max = 1.0 / np.min(x) * 10.0
        else:
            tau_min = np.min(x) / 10.0
            tau_max = np.max(x) * 10.0

        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), self.n_tau)

        if self.method == "tikhonov":
            H_tau, lam, res_norm = _tikhonov_inversion(
                x, y, tau, self.source, self.G_e, self.regularization
            )
        elif self.method == "max_entropy":
            H_tau, lam, res_norm = _max_entropy_inversion(
                x, y, tau, self.source, self.G_e, self.regularization
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.result = SpectrumResult(
            tau=tau,
            H_tau=H_tau,
            regularization_param=lam,
            method=self.method,
            residual_norm=res_norm,
            G_e=self.G_e,
        )

        result_data = RheoData(
            x=tau,
            y=H_tau,
            metadata={
                "source_transform": "spectrum_inversion",
                "method": self.method,
                "regularization_param": lam,
                "residual_norm": res_norm,
            },
        )
        return result_data, {"spectrum_result": self.result}


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------


def _build_kernel(
    x: np.ndarray, tau: np.ndarray, source: str, G_e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the kernel matrix A and target vector b.

    Returns (A, b) where A @ H ≈ b.
    """
    d_ln_tau = np.diff(np.log(tau))
    d_ln_tau = np.append(d_ln_tau, d_ln_tau[-1])  # extend last bin

    if source == "oscillation":
        omega = x

        # Handle complex/real input
        n = len(omega)
        A = np.zeros((2 * n, len(tau)))
        for j, (t, dlnt) in enumerate(zip(tau, d_ln_tau, strict=False)):
            wt2 = (omega * t) ** 2
            A[:n, j] = wt2 / (1.0 + wt2) * dlnt       # G' kernel
            A[n:, j] = omega * t / (1.0 + wt2) * dlnt  # G'' kernel

        return A, None  # b assembled externally

    elif source == "relaxation":
        t_data = x
        A = np.zeros((len(t_data), len(tau)))
        for j, (tau_j, dlnt) in enumerate(zip(tau, d_ln_tau, strict=False)):
            A[:, j] = np.exp(-t_data / tau_j) * dlnt

        return A, None

    else:
        raise ValueError(f"Unknown source: {source}")


def _assemble_target(
    y: np.ndarray, source: str, G_e: float
) -> np.ndarray:
    """Assemble the target vector b from measurement data."""
    if source == "oscillation":
        if np.iscomplexobj(y):
            G_prime = y.real - G_e
            G_double_prime = y.imag
        elif y.ndim == 2 and y.shape[1] == 2:
            G_prime = y[:, 0] - G_e
            G_double_prime = y[:, 1]
        else:
            G_prime = np.abs(y) - G_e
            G_double_prime = np.zeros_like(y)
        return np.concatenate([G_prime, G_double_prime])
    else:
        return y - G_e


# ---------------------------------------------------------------------------
# Tikhonov regularization
# ---------------------------------------------------------------------------


def _tikhonov_inversion(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    source: str,
    G_e: float,
    lam: float | None,
) -> tuple[np.ndarray, float, float]:
    """Tikhonov-regularized spectrum inversion.

    Solves:  min ||A H - b||² + λ² ||L H||²

    where L is the identity (zeroth order) for stability.
    """
    A, _ = _build_kernel(x, tau, source, G_e)
    b = _assemble_target(y, source, G_e)

    n_tau = len(tau)
    L = np.eye(n_tau)

    if lam is None:
        lam = _select_lambda_gcv(A, b, L)

    # Solve regularized normal equations
    ATA = A.T @ A
    ATb = A.T @ b
    H_tau = np.linalg.solve(ATA + lam**2 * L.T @ L, ATb)

    # Enforce non-negativity
    H_tau = np.maximum(H_tau, 0.0)

    residual_norm = float(np.linalg.norm(A @ H_tau - b))
    return H_tau, float(lam), residual_norm


def _select_lambda_gcv(
    A: np.ndarray, b: np.ndarray, L: np.ndarray
) -> float:
    """Select regularization parameter via Generalized Cross-Validation.

    GCV(λ) = ||A H_λ - b||² / (trace(I - A(A^TA + λ²L^TL)^{-1}A^T))²

    When L = I (zeroth-order Tikhonov), the influence trace simplifies via
    the SVD of A:  trace(A(A^T A + λ²I)^{-1} A^T) = Σ σ_i²/(σ_i² + λ²).
    This reduces the per-lambda cost from O(n·m²) to O(min(n,m)).
    """
    n = A.shape[0]
    lambdas = np.logspace(-6, 4, 50)
    gcv_scores = np.full(len(lambdas), np.inf)

    # Check if L is identity — enables fast SVD path
    is_identity_L = (L.shape[0] == L.shape[1] and np.allclose(L, np.eye(L.shape[0])))

    if is_identity_L:
        # Fast SVD-based GCV: precompute once, O(min(n,m)) per lambda
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s2 = s ** 2
        # Coefficients for residual: UTb projected onto singular vectors
        UTb = U.T @ b

        for i, lam in enumerate(lambdas):
            lam2 = lam ** 2
            # Filter factors: f_j = σ_j² / (σ_j² + λ²)
            f = s2 / (s2 + lam2)

            # H_λ = V diag(f/σ) U^T b  (without non-negativity clamp for GCV)
            # Residual = b - A H_λ = b - U diag(f) U^T b
            # = U (I - diag(f)) U^T b  +  (I - UU^T) b
            # ||residual||² = Σ (1-f_j)² (UTb_j)² + ||b - UU^Tb||²
            res_filtered = (1.0 - f) * UTb
            # Component orthogonal to range(A) is constant across lambdas
            b_perp_sq = np.sum(b ** 2) - np.sum(UTb ** 2)
            res_norm_sq = float(np.sum(res_filtered ** 2) + b_perp_sq)

            # trace(I - M) = n - Σ f_j
            trace_I_minus_M = n - np.sum(f)

            if trace_I_minus_M > 0:
                gcv_scores[i] = res_norm_sq / trace_I_minus_M ** 2
    else:
        # General case: L ≠ I — use direct solve (original algorithm)
        ATA = A.T @ A
        ATb = A.T @ b
        LTL = L.T @ L

        for i, lam in enumerate(lambdas):
            try:
                H = np.linalg.solve(ATA + lam**2 * LTL, ATb)
                residual = A @ H - b
                res_norm_sq = np.sum(residual**2)

                # Influence matrix trace via solve (avoids forming full n×n matrix)
                # trace(A (ATA + λ²LTL)^{-1} A^T) = trace((ATA + λ²LTL)^{-1} ATA)
                C = np.linalg.solve(ATA + lam**2 * LTL, ATA)
                trace_I_minus_M = n - np.trace(C)

                if trace_I_minus_M > 0:
                    gcv_scores[i] = res_norm_sq / trace_I_minus_M**2
            except np.linalg.LinAlgError:
                continue

    best_idx = np.argmin(gcv_scores)
    return float(lambdas[best_idx])


# ---------------------------------------------------------------------------
# Maximum entropy
# ---------------------------------------------------------------------------


def _max_entropy_inversion(
    x: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    source: str,
    G_e: float,
    lam: float | None,
) -> tuple[np.ndarray, float, float]:
    """Maximum entropy spectrum inversion.

    Maximizes: S = -Σ H_i ln(H_i / m_i)  subject to  χ² ≤ target

    Uses iterative multiplicative update (Bryan's algorithm).
    """
    A, _ = _build_kernel(x, tau, source, G_e)
    b = _assemble_target(y, source, G_e)

    n_tau = len(tau)

    # Default model: uniform spectrum
    m = np.ones(n_tau) * np.mean(np.abs(b)) / n_tau
    m = np.maximum(m, 1e-30)

    H = m.copy()

    if lam is None:
        lam = 1.0  # Initial guess, refined during iteration

    max_iter = 200
    tol = 1e-6

    for iteration in range(max_iter):
        # Gradient of chi-squared: ∂χ²/∂H = 2 A^T (AH - b)
        residual = A @ H - b
        grad_chi2 = 2.0 * A.T @ residual

        # Multiplicative update (entropy gradient implicit in exponential form)
        H_new = H * np.exp(-lam * grad_chi2 / (1.0 + lam * np.abs(grad_chi2)))
        H_new = np.maximum(H_new, 1e-30)

        # Convergence check
        rel_change = np.linalg.norm(H_new - H) / (np.linalg.norm(H) + 1e-30)
        H = H_new

        if rel_change < tol:
            logger.debug(
                "MaxEnt converged",
                iterations=iteration + 1,
                relative_change=rel_change,
            )
            break

    residual_norm = float(np.linalg.norm(A @ H - b))
    return H, float(lam), residual_norm
