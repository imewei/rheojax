"""Fit utilities for HVM (Hybrid Vitrimer Model) tutorial notebooks.

Provides standardized NLSQ → NUTS workflow, posterior predictive checks,
diagnostics printing, and result persistence.

Usage:
    from examples.utils.hvm_fit import (
        FAST_MODE, get_bayesian_config,
        run_nlsq_saos, run_nuts,
        print_convergence, print_parameter_table,
        plot_trace_and_forest, posterior_predictive_saos,
        get_output_dir, save_results, save_figure,
    )
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# FAST_MODE Configuration
# =============================================================================
# True  → CI / quick demo (~1-2 min per notebook, 1 chain, few samples)
# False → publication-quality (~10-30 min, 4 chains, more samples)

FAST_MODE = True


def get_fast_mode() -> bool:
    """Return current FAST_MODE setting."""
    return FAST_MODE


def get_bayesian_config() -> dict[str, int]:
    """Return MCMC configuration based on FAST_MODE.

    Returns:
        Dict with num_warmup, num_samples, num_chains.
    """
    if FAST_MODE:
        return {"num_warmup": 50, "num_samples": 100, "num_chains": 1}
    return {"num_warmup": 500, "num_samples": 1000, "num_chains": 4}


# =============================================================================
# NLSQ Fitting
# =============================================================================


def run_nlsq_saos(
    model: Any,
    omega: np.ndarray,
    G_star: np.ndarray,
    **fit_kwargs: Any,
) -> dict[str, float]:
    """Run NLSQ fit on SAOS data (|G*| = sqrt(G'^2 + G''^2)).

    Args:
        model: HVMLocal instance.
        omega: Angular frequency (rad/s).
        G_star: Complex modulus magnitude (Pa).
        **fit_kwargs: Extra kwargs for model.fit().

    Returns:
        Dict of fitted parameter name → value.
    """
    defaults = {"test_mode": "oscillation", "use_log_residuals": True}
    defaults.update(fit_kwargs)
    model.fit(omega, G_star, **defaults)

    param_names = list(model.parameters.keys())
    return {p: float(model.parameters.get_value(p)) for p in param_names}


def run_nlsq_protocol(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    test_mode: str,
    **fit_kwargs: Any,
) -> dict[str, float]:
    """Run NLSQ fit for any protocol.

    Args:
        model: HVMLocal instance.
        x: Independent variable.
        y: Dependent variable.
        test_mode: Protocol name.
        **fit_kwargs: Extra kwargs for model.fit().

    Returns:
        Dict of fitted parameter name → value.
    """
    defaults = {"test_mode": test_mode}
    if test_mode in ("relaxation", "flow_curve"):
        defaults["use_log_residuals"] = True
    defaults.update(fit_kwargs)
    model.fit(x, y, **defaults)

    param_names = list(model.parameters.keys())
    return {p: float(model.parameters.get_value(p)) for p in param_names}


# =============================================================================
# Bayesian Inference (NUTS)
# =============================================================================


def run_nuts(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    test_mode: str,
    seed: int = 42,
    **bayes_kwargs: Any,
) -> Any:
    """Run NumPyro NUTS sampling with NLSQ warm-start.

    Assumes model.fit() has already been called so NLSQ parameters
    serve as initial values for the sampler.

    Args:
        model: Fitted HVMLocal instance.
        x: Independent variable.
        y: Dependent variable.
        test_mode: Protocol name.
        seed: Random seed for reproducibility.
        **bayes_kwargs: Override num_warmup, num_samples, num_chains.

    Returns:
        BayesianResult from model.fit_bayesian().
    """
    config = get_bayesian_config()
    config.update(bayes_kwargs)
    config["seed"] = seed
    config["test_mode"] = test_mode

    return model.fit_bayesian(x, y, **config)


# =============================================================================
# Diagnostics
# =============================================================================


def print_convergence(result: Any, param_names: list[str]) -> bool:
    """Print convergence diagnostics from a BayesianResult.

    Args:
        result: BayesianResult instance.
        param_names: Parameters to report.

    Returns:
        True if all R-hat < 1.05 and ESS > 100.
    """
    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 50)
    print(f"{'Parameter':>12s}  {'R-hat':>8s}  {'ESS':>8s}")
    print("-" * 50)

    all_ok = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))
        flag = ""
        if r_hat > 1.05 or ess < 100:
            flag = " *"
            all_ok = False
        print(f"{p:>12s}  {r_hat:8.4f}  {ess:8.0f}{flag}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")
    print(f"Convergence: {'PASSED' if all_ok else 'CHECK REQUIRED'}")
    return all_ok


def print_parameter_table(
    param_names: list[str],
    nlsq_vals: dict[str, float],
    posterior: dict[str, Any],
    true_vals: dict[str, float] | None = None,
) -> None:
    """Print NLSQ vs Bayesian parameter comparison.

    Args:
        param_names: Parameters to report.
        nlsq_vals: NLSQ fitted values.
        posterior: Posterior samples dict.
        true_vals: Optional ground-truth values.
    """
    header = f"{'Param':>12s}  "
    if true_vals:
        header += f"{'True':>12s}  "
    header += f"{'NLSQ':>12s}  {'Bayes (med)':>12s}  {'95% CI':>24s}"
    print("\nParameter Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for p in param_names:
        samples = np.array(posterior[p])
        med = np.median(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        row = f"{p:>12s}  "
        if true_vals and p in true_vals:
            row += f"{true_vals[p]:12.4g}  "
        row += f"{nlsq_vals[p]:12.4g}  {med:12.4g}  [{lo:.4g}, {hi:.4g}]"
        print(row)


# =============================================================================
# Posterior Predictive
# =============================================================================


def posterior_predictive_saos(
    model: Any,
    omega: np.ndarray,
    posterior: dict[str, Any],
    n_draws: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior predictive envelope for SAOS.

    Args:
        model: HVMLocal instance.
        omega: Angular frequency array.
        posterior: Posterior samples dict.
        n_draws: Number of posterior draws to use.

    Returns:
        (G_prime_draws, G_double_prime_draws) arrays of shape (n_draws, len(omega)).
    """
    param_names = list(model.parameters.keys())
    n_total = len(next(iter(posterior.values())))
    indices = np.random.default_rng(0).choice(
        n_total, size=min(n_draws, n_total), replace=False
    )

    G_prime_all = []
    G_double_prime_all = []

    for idx in indices:
        for p in param_names:
            if p in posterior:
                model.parameters.set_value(p, float(np.array(posterior[p])[idx]))
        try:
            G_p, G_pp = model.predict_saos(omega, return_components=True)
            G_prime_all.append(np.array(G_p))
            G_double_prime_all.append(np.array(G_pp))
        except Exception:
            continue

    return np.array(G_prime_all), np.array(G_double_prime_all)


def posterior_predictive_1d(
    model: Any,
    x: np.ndarray,
    posterior: dict[str, Any],
    test_mode: str,
    n_draws: int = 200,
    **predict_kwargs: Any,
) -> np.ndarray:
    """Compute posterior predictive envelope for 1D predictions.

    Args:
        model: HVMLocal instance.
        x: Independent variable.
        posterior: Posterior samples dict.
        test_mode: Protocol name.
        n_draws: Number of posterior draws.
        **predict_kwargs: Extra kwargs for model.predict().

    Returns:
        Array of shape (n_draws, len(x)).
    """
    param_names = list(model.parameters.keys())
    n_total = len(next(iter(posterior.values())))
    indices = np.random.default_rng(0).choice(
        n_total, size=min(n_draws, n_total), replace=False
    )

    draws = []
    for idx in indices:
        for p in param_names:
            if p in posterior:
                model.parameters.set_value(p, float(np.array(posterior[p])[idx]))
        try:
            y_pred = model.predict(x, test_mode=test_mode, **predict_kwargs)
            draws.append(np.array(y_pred))
        except Exception:
            continue

    return np.array(draws)


# =============================================================================
# Plotting Helpers
# =============================================================================


def plot_trace_and_forest(
    result: Any,
    param_names: list[str],
    figsize_trace: tuple[float, float] = (12, 5),
    figsize_forest: tuple[float, float] = (10, 3),
) -> tuple[plt.Figure, plt.Figure]:
    """Plot ArviZ trace and forest plots.

    Args:
        result: BayesianResult instance.
        param_names: Parameters to include.
        figsize_trace: Trace figure size.
        figsize_forest: Forest figure size.

    Returns:
        (trace_fig, forest_fig).
    """
    import arviz as az

    idata = result.to_inference_data()

    # Scale figure height to fit all parameters (2 subplots per param for trace)
    n = len(param_names)
    trace_h = max(figsize_trace[1], 2.0 * n)
    forest_h = max(figsize_forest[1], 1.2 * n)

    axes = az.plot_trace(
        idata, var_names=param_names, figsize=(figsize_trace[0], trace_h)
    )
    fig_trace = axes.ravel()[0].figure
    fig_trace.suptitle("HVM Trace Plots", fontsize=14, y=1.02)
    fig_trace.tight_layout(rect=[0, 0, 1, 0.98])

    axes = az.plot_forest(
        idata,
        var_names=param_names,
        combined=True,
        hdi_prob=0.95,
        figsize=(figsize_forest[0], forest_h),
    )
    fig_forest = axes.ravel()[0].figure
    fig_forest.tight_layout()

    return fig_trace, fig_forest


def plot_posterior_predictive_saos(
    omega: np.ndarray,
    G_prime_data: np.ndarray,
    G_double_prime_data: np.ndarray,
    G_prime_draws: np.ndarray,
    G_double_prime_draws: np.ndarray,
    G_prime_nlsq: np.ndarray | None = None,
    G_double_prime_nlsq: np.ndarray | None = None,
    omega_fit: np.ndarray | None = None,
    credibility: float = 0.95,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot SAOS data with posterior predictive bands.

    Args:
        omega: Angular frequency for data points.
        G_prime_data: Measured G'.
        G_double_prime_data: Measured G''.
        G_prime_draws: Posterior draws for G' (on omega_fit grid if provided).
        G_double_prime_draws: Posterior draws for G'' (on omega_fit grid if provided).
        G_prime_nlsq: NLSQ best-fit G' (on omega_fit grid if provided, optional).
        G_double_prime_nlsq: NLSQ best-fit G'' (on omega_fit grid if provided, optional).
        omega_fit: Smooth frequency grid for predictions (defaults to omega).
        credibility: Credible interval width.
        figsize: Figure size.

    Returns:
        Figure.
    """
    alpha = (1 - credibility) / 2 * 100
    omega_pred = omega_fit if omega_fit is not None else omega

    fig, ax = plt.subplots(figsize=figsize)

    # Data
    ax.loglog(omega, G_prime_data, "s", color="C0", label="G' (data)", markersize=6)
    ax.loglog(
        omega, G_double_prime_data, "o", color="C1", label="G'' (data)", markersize=6
    )

    # NLSQ
    if G_prime_nlsq is not None:
        ax.loglog(
            omega_pred, G_prime_nlsq, "-", color="C0", alpha=0.5, label="G' (NLSQ)"
        )
    if G_double_prime_nlsq is not None:
        ax.loglog(
            omega_pred,
            G_double_prime_nlsq,
            "-",
            color="C1",
            alpha=0.5,
            label="G'' (NLSQ)",
        )

    # Posterior bands
    if len(G_prime_draws) > 0:
        lo = np.percentile(G_prime_draws, alpha, axis=0)
        hi = np.percentile(G_prime_draws, 100 - alpha, axis=0)
        med = np.median(G_prime_draws, axis=0)
        ax.fill_between(
            omega_pred, lo, hi, color="C0", alpha=0.15, label=f"G' {credibility:.0%} CI"
        )
        ax.loglog(
            omega_pred, med, "--", color="C0", linewidth=1.5, label="G' (Bayes median)"
        )

    if len(G_double_prime_draws) > 0:
        lo = np.percentile(G_double_prime_draws, alpha, axis=0)
        hi = np.percentile(G_double_prime_draws, 100 - alpha, axis=0)
        med = np.median(G_double_prime_draws, axis=0)
        ax.fill_between(
            omega_pred,
            lo,
            hi,
            color="C1",
            alpha=0.15,
            label=f"G'' {credibility:.0%} CI",
        )
        ax.loglog(
            omega_pred, med, "--", color="C1", linewidth=1.5, label="G'' (Bayes median)"
        )

    ax.set_xlabel("Angular Frequency (rad/s)")
    ax.set_ylabel("Modulus (Pa)")
    ax.set_title("HVM SAOS: Posterior Predictive Check")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


# =============================================================================
# Save / Load
# =============================================================================


def get_output_dir(protocol: str) -> str:
    """Return path to output directory for a given protocol.

    Creates directory if it doesn't exist.
    """
    out = Path(__file__).parent / ".." / "outputs" / "hvm" / protocol
    out.mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def save_results(
    output_dir: str,
    model: Any,
    result: Any | None = None,
    param_names: list[str] | None = None,
    extra_meta: dict | None = None,
) -> None:
    """Save NLSQ params, posterior samples, and summary CSV.

    Args:
        output_dir: Directory to write into.
        model: Fitted HVMLocal instance.
        result: BayesianResult (optional).
        param_names: Parameter names (auto-detected if None).
        extra_meta: Extra metadata for the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if param_names is None:
        param_names = list(model.parameters.keys())

    # NLSQ parameters
    nlsq = {p: float(model.parameters.get_value(p)) for p in param_names}
    if extra_meta:
        nlsq.update(extra_meta)
    with open(os.path.join(output_dir, "fitted_params_nlsq.json"), "w") as f:
        json.dump(nlsq, f, indent=2)

    if result is None:
        return

    # Posterior samples
    posterior = result.posterior_samples
    np.savez(
        os.path.join(output_dir, "posterior_samples.npz"),
        **{k: np.array(v) for k, v in posterior.items()},
    )

    # Summary CSV
    rows = []
    for p in param_names:
        if p not in posterior:
            continue
        samples = np.array(posterior[p])
        rows.append(
            {
                "parameter": p,
                "nlsq": nlsq.get(p),
                "posterior_mean": float(np.mean(samples)),
                "posterior_median": float(np.median(samples)),
                "ci_2.5": float(np.percentile(samples, 2.5)),
                "ci_97.5": float(np.percentile(samples, 97.5)),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"Results saved to {output_dir}/")


def save_figure(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save a figure to the figures/ subdirectory.

    Args:
        output_dir: Base output directory.
        name: Filename (e.g., "posterior_predictive.png").
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
    print(f"Figure saved: {os.path.join(fig_dir, name)}")
