"""Reproducible HVM fitting demonstrations for example notebooks.

The experimental HVM notebooks intentionally include some datasets that are a
poor physical match for the Hybrid Vitrimer Model.  This module provides a
compact positive-control dataset generated from HVMLocal itself so examples can
demonstrate the fitting workflow with fitted curves overlapping the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheojax.models import HVMLocal

TRUE_PARAMS = {
    "G_P": 25.0,
    "G_E": 2_500.0,
    "G_D": 15_000.0,
    "nu_0": 1.0e8,
    "E_a": 5.0e4,
    "V_act": 1.0e-5,
    "T": 360.0,
    "k_d_D": 800.0,
}

INITIAL_PARAMS = {
    "G_P": 50.0,
    "G_E": 2_200.0,
    "G_D": 12_000.0,
    "nu_0": 8.0e7,
    "E_a": 5.0e4,
    "V_act": 1.0e-5,
    "T": 360.0,
    "k_d_D": 600.0,
}


@dataclass
class HVMFitDemoResult:
    """Container for one protocol's synthetic data and fitted overlay."""

    protocol: str
    x_data: np.ndarray
    y_data: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_data_fit: np.ndarray
    r_squared: float
    fit_model: HVMLocal
    true_model: HVMLocal
    predict_kwargs: dict[str, float]


def set_hvm_parameters(model: HVMLocal, values: dict[str, float]) -> HVMLocal:
    """Set HVM parameters by name and return the model for chaining."""

    for name, value in values.items():
        model.parameters.set_value(name, value)
    return model


def make_hvm_demo_model(values: dict[str, float] | None = None) -> HVMLocal:
    """Create an HVM model with the provided demo parameters."""

    model = HVMLocal(include_dissociative=True, kinetics="stress")
    set_hvm_parameters(model, TRUE_PARAMS if values is None else values)
    return model


def _protocol_grid(protocol: str) -> tuple[np.ndarray, dict[str, float]]:
    if protocol == "flow_curve":
        return np.logspace(-2, 2, 32), {}
    if protocol == "relaxation":
        return np.logspace(-2, 3, 60), {}
    if protocol == "creep":
        return np.logspace(-2, 3, 60), {"sigma_applied": 100.0}
    if protocol == "startup":
        return np.linspace(0.01, 10.0, 80), {"gamma_dot": 1.0}
    raise ValueError(f"Unsupported HVM demo protocol: {protocol!r}")


def _fit_grid(protocol: str) -> np.ndarray:
    if protocol == "startup":
        return np.linspace(0.01, 10.0, 200)
    return np.logspace(-2, 3 if protocol in {"relaxation", "creep"} else 2, 200)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray, *, log_space: bool) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if log_space:
        finite &= (y_true > 0) & (y_pred > 0)
        y_true = np.log(y_true[finite])
        y_pred = np.log(y_pred[finite])
    else:
        y_true = y_true[finite]
        y_pred = y_pred[finite]

    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def generate_hvm_demo_data(
    protocol: str,
    *,
    noise_level: float = 0.01,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], HVMLocal]:
    """Generate small reproducible rheology data from the HVM positive control."""

    x_data, predict_kwargs = _protocol_grid(protocol)
    true_model = make_hvm_demo_model()
    y_clean = np.array(
        true_model.predict(x_data, test_mode=protocol, **predict_kwargs),
        dtype=float,
        copy=True,
    )

    if noise_level:
        rng = np.random.default_rng(seed)
        y_clean *= np.exp(rng.normal(0.0, noise_level, size=y_clean.shape))

    return x_data, y_clean, predict_kwargs, true_model


def fit_hvm_demo_protocol(
    protocol: str,
    *,
    noise_level: float = 0.01,
    seed: int = 7,
) -> HVMFitDemoResult:
    """Fit one HVM protocol and return arrays for raw-data/fitted overlays."""

    x_data, y_data, predict_kwargs, true_model = generate_hvm_demo_data(
        protocol,
        noise_level=noise_level,
        seed=seed,
    )

    fit_model = make_hvm_demo_model(INITIAL_PARAMS)
    fit_model.fit(
        x_data,
        y_data,
        test_mode=protocol,
        use_log_residuals=protocol != "startup",
        max_iter=2_000,
        **predict_kwargs,
    )

    x_fit = _fit_grid(protocol)
    y_data_fit = np.asarray(fit_model.predict(x_data, test_mode=protocol, **predict_kwargs))
    y_fit = np.asarray(fit_model.predict(x_fit, test_mode=protocol, **predict_kwargs))

    return HVMFitDemoResult(
        protocol=protocol,
        x_data=x_data,
        y_data=y_data,
        x_fit=x_fit,
        y_fit=y_fit,
        y_data_fit=y_data_fit,
        r_squared=_r_squared(y_data, y_data_fit, log_space=protocol != "startup"),
        fit_model=fit_model,
        true_model=true_model,
        predict_kwargs=predict_kwargs,
    )


def run_hvm_demo_fits(
    *,
    protocols: tuple[str, ...] = ("flow_curve", "relaxation", "creep", "startup"),
    noise_level: float = 0.01,
    seed: int = 7,
) -> dict[str, HVMFitDemoResult]:
    """Run the positive-control HVM fit demonstration for several protocols."""

    return {
        protocol: fit_hvm_demo_protocol(
            protocol,
            noise_level=noise_level,
            seed=seed,
        )
        for protocol in protocols
    }
