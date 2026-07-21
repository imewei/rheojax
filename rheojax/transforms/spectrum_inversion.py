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
from scipy.optimize import nnls

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
        G_e: Equilibrium modulus (default: 0.0). Automatic estimation from
            data is not implemented; passing ``None`` raises
            ``NotImplementedError`` rather than silently guessing a value.
    """

    def __init__(
        self,
        method: str = "tikhonov",
        n_tau: int = 100,
        tau_range: tuple[float, float] | None = None,
        regularization: float | None = None,
        source: str = "oscillation",
        G_e: float | None = 0.0,
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

        # G_e=None is not implemented (no data-driven estimator exists yet).
        # Fail loudly here rather than letting `y - None` raise an opaque
        # TypeError deep inside _assemble_target.
        if self.G_e is None:
            raise NotImplementedError(
                "SpectrumInversion: automatic estimation of G_e from data is "
                "not implemented. Pass an explicit float via G_e (default "
                "0.0 for a purely viscous/viscoelastic liquid, or the "
                "measured equilibrium/plateau modulus for a crosslinked "
                "network)."
            )

        # Validate n_tau — must be >=2 so that d_ln_tau has at least one element
        if self.n_tau < 2:
            raise ValueError(
                f"SpectrumInversion: n_tau must be >= 2 (got {self.n_tau}). "
                "The kernel matrix requires at least 2 τ points to compute d(ln τ) bin widths."
            )

        # Validate inputs — x must be strictly positive for log-space operations.
        # `~(x > 0)` (rather than `x <= 0`) also catches NaN, since NaN comparisons
        # are always False and would otherwise silently pass the `x <= 0` check.
        if np.any(~(x > 0)):
            raise ValueError(
                f"SpectrumInversion: x (frequency/time) must be strictly positive "
                f"and finite; got min(x) = {np.min(x):.4g}"
            )

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
) -> np.ndarray:
    """Build the kernel matrix A for the linear inverse problem A @ H ≈ b.

    The target vector b is assembled separately by :func:`_assemble_target`
    because it depends on data preprocessing (G_e subtraction, complex
    splitting) that is independent of the kernel structure.

    Returns:
        A: Kernel matrix with shape ``(M, n_tau)`` where M = N for relaxation
           and M = 2N for oscillation (stacked G'/G'' rows).
    """
    # Centered trapezoidal quadrature weights in log(tau): each interior point
    # gets the half-width to both neighbors, and both boundary points (not just
    # the last) get a half-width bin so the total weight equals the true
    # log(tau_max/tau_min) range.
    log_tau = np.log(tau)
    d_ln_tau = np.empty_like(log_tau)
    d_ln_tau[0] = (log_tau[1] - log_tau[0]) / 2.0
    d_ln_tau[-1] = (log_tau[-1] - log_tau[-2]) / 2.0
    d_ln_tau[1:-1] = (log_tau[2:] - log_tau[:-2]) / 2.0

    if source == "oscillation":
        omega = x

        # Handle complex/real input
        n = len(omega)
        A = np.zeros((2 * n, len(tau)))
        wt2 = (omega[:, None] * tau[None, :]) ** 2
        A[:n, :] = wt2 / (1.0 + wt2) * d_ln_tau[None, :]  # G' kernel
        A[n:, :] = (
            omega[:, None] * tau[None, :] / (1.0 + wt2) * d_ln_tau[None, :]
        )  # G'' kernel

        return A

    elif source == "relaxation":
        t_data = x
        A = np.exp(-t_data[:, None] / tau[None, :]) * d_ln_tau[None, :]

        return A

    else:
        raise ValueError(f"Unknown source: {source}")


def _assemble_target(y: np.ndarray, source: str, G_e: float) -> np.ndarray:
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

    Solves:  min ||A H - b||²  s.t. H >= 0, via NNLS on the Tikhonov-
    augmented stacked system  [A; λL] H ≈ [b; 0]  -- the non-negativity
    constraint is enforced directly by NNLS rather than by solving the
    unconstrained ridge normal equations and clipping negative values
    afterward (clip-after-solve is not the minimizer of the constrained
    problem).

    where L is the identity (zeroth order) for stability.
    """
    A = _build_kernel(x, tau, source, G_e)
    b = _assemble_target(y, source, G_e)

    n_tau = len(tau)
    L = np.eye(n_tau)

    if lam is None:
        lam = _select_lambda_gcv(A, b, L)

    A_aug = np.vstack([A, lam * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    H_tau, _ = nnls(A_aug, b_aug)

    residual_norm = float(np.linalg.norm(A @ H_tau - b))
    return H_tau, float(lam), residual_norm


def _select_lambda_gcv(A: np.ndarray, b: np.ndarray, L: np.ndarray) -> float:
    """Select regularization parameter via Generalized Cross-Validation.

    GCV(λ) = ||A H_λ - b||² / (trace(I - A(A^TA + λ²L^TL)^{-1}A^T))²

    H_λ is the non-negativity-constrained NNLS solution of the augmented
    system [A; λL] H ≈ [b; 0] -- the same solve used by the caller -- so
    the residual driving lambda selection matches the H_tau actually
    returned, rather than the unconstrained ridge residual. The
    degrees-of-freedom trace still uses the linear ridge influence
    operator (the standard GCV approximation); when L = I it simplifies
    via the SVD of A: trace(A(A^T A + λ²I)^{-1} A^T) = Σ σ_i²/(σ_i² + λ²).
    """
    n = A.shape[0]
    lambdas = np.logspace(-6, 4, 50)
    gcv_scores = np.full(len(lambdas), np.inf)

    # Check if L is identity — enables fast SVD path for the trace term
    is_identity_L = L.shape[0] == L.shape[1] and np.allclose(L, np.eye(L.shape[0]))

    if is_identity_L:
        # Fast SVD-based trace: precompute once, O(min(n,m)) per lambda
        _U, s, _Vt = np.linalg.svd(A, full_matrices=False)
        s2 = s**2

        for i, lam in enumerate(lambdas):
            # Filter factors: f_j = σ_j² / (σ_j² + λ²)
            f = s2 / (s2 + lam**2)
            # trace(I - M) = n - Σ f_j
            trace_I_minus_M = n - np.sum(f)

            if trace_I_minus_M <= 0:
                continue

            A_aug = np.vstack([A, lam * L])
            b_aug = np.concatenate([b, np.zeros(L.shape[0])])
            H_lam, _ = nnls(A_aug, b_aug)
            res_norm_sq = float(np.sum((A @ H_lam - b) ** 2))

            gcv_scores[i] = res_norm_sq / trace_I_minus_M**2
    else:
        # General case: L ≠ I — use direct solve for the trace term
        ATA = A.T @ A
        LTL = L.T @ L

        for i, lam in enumerate(lambdas):
            try:
                # Influence matrix trace via solve (avoids forming full n×n matrix)
                # trace(A (ATA + λ²LTL)^{-1} A^T) = trace((ATA + λ²LTL)^{-1} ATA)
                C = np.linalg.solve(ATA + lam**2 * LTL, ATA)
                trace_I_minus_M = n - np.trace(C)

                if trace_I_minus_M <= 0:
                    continue

                A_aug = np.vstack([A, lam * L])
                b_aug = np.concatenate([b, np.zeros(L.shape[0])])
                H_lam, _ = nnls(A_aug, b_aug)
                res_norm_sq = float(np.sum((A @ H_lam - b) ** 2))

                gcv_scores[i] = res_norm_sq / trace_I_minus_M**2
            except np.linalg.LinAlgError:
                continue

    finite_mask = np.isfinite(gcv_scores)
    if not np.any(finite_mask):
        # All GCV scores are inf (e.g. degenerate kernel / zero trace denominators).
        # Fall back to a moderate regularization parameter and warn.
        import warnings as _warnings

        _warnings.warn(
            "GCV: all regularization candidates produced infinite scores. "
            "Falling back to lambda=1e-3. Consider providing a manual "
            "regularization parameter via the 'regularization' argument.",
            UserWarning,
            stacklevel=2,
        )
        return 1e-3
    best_idx = int(np.argmin(gcv_scores))
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
    A = _build_kernel(x, tau, source, G_e)
    b = _assemble_target(y, source, G_e)

    n_tau = len(tau)
    n_data = len(b)

    # Default model: uniform spectrum
    m = np.ones(n_tau) * np.mean(np.abs(b)) / n_tau
    m = np.maximum(m, 1e-30)

    H = m.copy()

    auto_lam = lam is None
    if auto_lam:
        # Scale the initial entropy weight to the data. At H = m the entropy
        # gradient is the constant vector -1, so matching its magnitude to the
        # typical chi-squared gradient there makes the entropy term a genuine,
        # data-scaled competitor to the data-fidelity term (unlike a hardcoded
        # constant, which under- or over-regularizes depending on how large
        # G*/G(t) happens to be).
        grad_chi2_at_m = 2.0 * A.T @ (A @ m - b)
        lam = max(float(np.mean(np.abs(grad_chi2_at_m))), 1e-30)
        lam_lo, lam_hi = lam * 1e-3, lam * 1e3
    assert lam is not None  # guaranteed by the auto_lam branch above or the caller

    max_iter = 200
    tol = 1e-6

    for iteration in range(max_iter):
        # Gradient of chi-squared: ∂χ²/∂H = 2 A^T (AH - b)
        residual = A @ H - b
        grad_chi2 = 2.0 * A.T @ residual

        # Entropy gradient: S = -Σ H_i ln(H_i/m_i)  =>  ∂S/∂H_i = -(ln(H_i/m_i) + 1)
        grad_entropy = -(np.log(H / m) + 1.0)

        # Minimize F = χ² - λ·S (Lagrangian form of "maximize entropy subject to
        # data fidelity"): grad_F = grad_chi2 - λ·grad_entropy. This is what makes
        # `m`/entropy actually enter the update rule instead of being dead weight.
        grad_total = grad_chi2 - lam * grad_entropy

        # Multiplicative update, damped to a bounded step regardless of gradient scale.
        H_new = H * np.exp(-grad_total / (1.0 + np.abs(grad_total)))
        H_new = np.maximum(H_new, 1e-30)

        # Convergence check
        rel_change = np.linalg.norm(H_new - H) / (np.linalg.norm(H) + 1e-30)
        H = H_new

        if auto_lam:
            # Nudge lambda toward the classic MaxEnt target chi2 ~= n_data
            # (degrees of freedom): increase the entropy weight while
            # overfitting (chi2 < n_data), relax it while underfitting
            # (chi2 > n_data). Small step + clamp keep this a gentle
            # refinement rather than a divergent feedback loop.
            chi2 = float(np.sum((A @ H - b) ** 2))
            lam = lam * (n_data / max(chi2, 1e-30)) ** 0.02
            lam = float(np.clip(lam, lam_lo, lam_hi))

        if rel_change < tol:
            logger.debug(
                "MaxEnt converged",
                iterations=iteration + 1,
                relative_change=rel_change,
            )
            break

    residual_norm = float(np.linalg.norm(A @ H - b))
    return H, float(lam), residual_norm
