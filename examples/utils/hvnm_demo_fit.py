"""Reproducible HVNM fitting demonstrations for example notebooks.

The experimental HVNM notebooks (09-15) currently load datasets that do not
exhibit nanoparticle-mediated transient-network physics. This module provides
positive-control datasets generated from HVNMLocal itself, so the NLSQ + NUTS
pipeline can demonstrate parameter recovery on data the model actually
represents.

This file mirrors examples/utils/hvm_demo_fit.py for the HVNM model
(15-parameter superset including beta_I, nu_0_int, E_a_int, V_act_int, phi,
R_NP, delta_m for the nanoparticle interface contribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rheojax.models import HVNMLocal

from .hvnm_tutorial_utils import ProtocolData

# Ground-truth parameters for HVNM positive-control data generation.
# Shared with HVM TRUE_PARAMS for the 8 backbone params + HVNM-specific
# nanoparticle/interface parameters set from HVNM_DEFAULT_PARAMS scale.
TRUE_PARAMS = {
    # Backbone HVM parameters (matches utils/hvm_demo_fit.TRUE_PARAMS)
    "G_P": 25.0,
    "G_E": 2_500.0,
    "G_D": 15_000.0,
    "nu_0": 1.0e8,
    "E_a": 5.0e4,
    "V_act": 1.0e-5,
    "T": 360.0,
    "k_d_D": 800.0,
    # HVNM-specific nanoparticle/interface parameters
    "beta_I": 3.0,
    "nu_0_int": 1.0e8,
    "E_a_int": 5.5e4,
    "V_act_int": 5.0e-6,
    "phi": 0.05,
    "R_NP": 20.0e-9,
    "delta_m": 10.0e-9,
}


# Perturbed initial values so NLSQ has work to do (mirrors HVM INITIAL_PARAMS).
INITIAL_PARAMS = {
    "G_P": 50.0,
    "G_E": 2_200.0,
    "G_D": 12_000.0,
    "nu_0": 8.0e7,
    "E_a": 5.0e4,
    "V_act": 1.0e-5,
    "T": 360.0,
    "k_d_D": 600.0,
    "beta_I": 3.0,
    "nu_0_int": 8.0e7,
    "E_a_int": 5.5e4,
    "V_act_int": 5.0e-6,
    "phi": 0.05,
    "R_NP": 20.0e-9,
    "delta_m": 10.0e-9,
}


@dataclass
class HVNMFitDemoResult:
    """Container for one protocol's synthetic data and fitted overlay."""

    protocol: str
    x_data: np.ndarray
    y_data: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_data_fit: np.ndarray
    r_squared: float
    fit_model: HVNMLocal
    true_model: HVNMLocal
    predict_kwargs: dict[str, float]


def set_hvnm_parameters(model: HVNMLocal, values: dict[str, float]) -> HVNMLocal:
    """Set HVNM parameters by name and return the model for chaining."""
    for name, value in values.items():
        if name in model.parameters.keys():
            model.parameters.set_value(name, value)
    return model


def make_hvnm_demo_model(values: dict[str, float] | None = None) -> HVNMLocal:
    """Create an HVNM model with the provided demo parameters."""
    model = HVNMLocal(include_dissociative=True, kinetics="stress")
    set_hvnm_parameters(model, TRUE_PARAMS if values is None else values)
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
    if protocol == "oscillation":
        return np.logspace(-2, 2, 32), {}
    raise ValueError(f"Unsupported HVNM demo protocol: {protocol!r}")


def _fit_grid(protocol: str) -> np.ndarray:
    if protocol == "startup":
        return np.linspace(0.01, 10.0, 200)
    if protocol == "oscillation":
        return np.logspace(-2, 2, 200)
    if protocol in {"relaxation", "creep"}:
        return np.logspace(-2, 3, 200)
    return np.logspace(-2, 2, 200)


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


def generate_hvnm_demo_data(
    protocol: str,
    *,
    noise_level: float = 0.01,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], HVNMLocal]:
    """Generate small reproducible HVNM rheology data.

    For ``protocol="oscillation"`` the returned y is the complex G* = G' + i G''
    (numpy complex array), suitable for direct use with ``run_nlsq_saos`` style
    fitters. All other protocols return real-valued y.
    """
    x_data, predict_kwargs = _protocol_grid(protocol)
    true_model = make_hvnm_demo_model()

    if protocol == "oscillation":
        Gp, Gdp = true_model.predict_saos(x_data)
        Gp = np.asarray(Gp, dtype=float)
        Gdp = np.asarray(Gdp, dtype=float)
        if noise_level:
            rng = np.random.default_rng(seed)
            Gp = Gp * np.exp(rng.normal(0.0, noise_level, size=Gp.shape))
            Gdp = Gdp * np.exp(rng.normal(0.0, noise_level, size=Gdp.shape))
        y_clean = Gp + 1j * Gdp
    else:
        y_clean = np.array(
            true_model.predict(x_data, test_mode=protocol, **predict_kwargs),
            dtype=float,
            copy=True,
        )
        if noise_level:
            rng = np.random.default_rng(seed)
            y_clean = y_clean * np.exp(rng.normal(0.0, noise_level, size=y_clean.shape))

    return x_data, y_clean, predict_kwargs, true_model


_PROTOCOL_LABELS = {
    "flow_curve": {
        "x_label": r"$\dot{\gamma}$ [1/s]",
        "y_label": r"$\sigma$ [Pa]",
        "y2_label": r"$\eta$ [Pa$\cdot$s]",
    },
    "relaxation": {"x_label": "Time [s]", "y_label": "G(t) [Pa]"},
    "creep": {"x_label": "Time [s]", "y_label": "J(t) [1/Pa]"},
    "startup": {"x_label": "Time [s]", "y_label": r"$\sigma$ [Pa]"},
    "oscillation": {
        "x_label": r"$\omega$ [rad/s]",
        "y_label": r"$|G^*|$ [Pa]",
        "y2_label": "G', G'' [Pa]",
    },
    "laos": {"x_label": "Time [s]", "y_label": r"$\sigma$ [Pa]"},
}


def make_synthetic_protocol_data(
    protocol: str,
    *,
    noise_level: float = 0.01,
    seed: int = 7,
    laos_gamma_0: float = 1.0,
    laos_omega: float = 1.0,
    laos_n_cycles: int = 4,
    laos_n_per_cycle: int = 100,
) -> ProtocolData:
    """Build a :class:`ProtocolData` carrying HVNM-synthetic positive-control data.

    Drop-in replacement for ``load_*`` loaders in HVNM notebooks 09-15.
    LAOS is generated as ``sigma(t)`` over a few cycles at fixed (gamma_0, omega).
    """
    labels = _PROTOCOL_LABELS[protocol]
    metadata = {
        "material": "HVNM-synthetic positive control",
        "source": "examples/utils/hvnm_demo_fit.py (generate_hvnm_demo_data)",
        "true_params": dict(TRUE_PARAMS),
    }

    if protocol == "laos":
        true_model = make_hvnm_demo_model()
        t_max = laos_n_cycles * 2.0 * np.pi / laos_omega
        time = np.linspace(0.0, t_max, laos_n_cycles * laos_n_per_cycle)
        sigma = np.array(
            true_model.predict(
                time,
                test_mode="laos",
                gamma_0=laos_gamma_0,
                omega=laos_omega,
            ),
            dtype=float,
        )
        if noise_level:
            rng = np.random.default_rng(seed)
            sigma = sigma + np.std(sigma) * rng.normal(
                0.0, noise_level, size=sigma.shape
            )
        gamma = laos_gamma_0 * np.sin(laos_omega * time)
        return ProtocolData(
            protocol="laos",
            x=time,
            y=sigma,
            y2=gamma,
            y2_label=r"$\gamma$",
            metadata={
                **metadata,
                "gamma_0": laos_gamma_0,
                "omega": laos_omega,
                "n_cycles": laos_n_cycles,
            },
            protocol_kwargs={"gamma_0": laos_gamma_0, "omega": laos_omega},
            **labels,
        )

    x_data, y_data, predict_kwargs, _ = generate_hvnm_demo_data(
        protocol,
        noise_level=noise_level,
        seed=seed,
    )

    if protocol == "flow_curve":
        viscosity = y_data / np.maximum(x_data, 1e-30)
        return ProtocolData(
            protocol="flow_curve",
            x=x_data,
            y=y_data,
            y2=viscosity,
            metadata=metadata,
            protocol_kwargs=predict_kwargs,
            **labels,
        )

    if protocol == "oscillation":
        G_prime = np.real(y_data)
        G_double_prime = np.imag(y_data)
        G_star_mag = np.sqrt(G_prime**2 + G_double_prime**2)
        return ProtocolData(
            protocol="oscillation",
            x=x_data,
            y=G_star_mag,
            y2=np.column_stack([G_prime, G_double_prime]),
            metadata=metadata,
            protocol_kwargs=predict_kwargs,
            **labels,
        )

    # creep / relaxation / startup — single response curve
    return ProtocolData(
        protocol=protocol,
        x=x_data,
        y=y_data,
        metadata=metadata,
        protocol_kwargs=predict_kwargs,
        **labels,
    )


def fit_hvnm_demo_protocol(
    protocol: str,
    *,
    noise_level: float = 0.01,
    seed: int = 7,
) -> HVNMFitDemoResult:
    """Fit one HVNM protocol and return arrays for raw-data/fitted overlays."""
    x_data, y_data, predict_kwargs, true_model = generate_hvnm_demo_data(
        protocol,
        noise_level=noise_level,
        seed=seed,
    )
    fit_model = make_hvnm_demo_model(INITIAL_PARAMS)

    fit_kwargs = {
        "test_mode": protocol,
        "use_log_residuals": protocol != "startup",
        "max_iter": 5_000,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "gtol": 1e-10,
    }
    fit_model.fit(x_data, y_data, **fit_kwargs, **predict_kwargs)

    x_fit = _fit_grid(protocol)
    if protocol == "oscillation":
        Gp_data, Gdp_data = fit_model.predict_saos(x_data)
        y_data_fit = np.asarray(Gp_data, dtype=float) + 1j * np.asarray(
            Gdp_data, dtype=float
        )
        Gp_fit, Gdp_fit = fit_model.predict_saos(x_fit)
        y_fit = np.asarray(Gp_fit, dtype=float) + 1j * np.asarray(Gdp_fit, dtype=float)
    else:
        y_data_fit = np.asarray(
            fit_model.predict(x_data, test_mode=protocol, **predict_kwargs)
        )
        y_fit = np.asarray(
            fit_model.predict(x_fit, test_mode=protocol, **predict_kwargs)
        )

    return HVNMFitDemoResult(
        protocol=protocol,
        x_data=x_data,
        y_data=y_data,
        x_fit=x_fit,
        y_fit=y_fit,
        y_data_fit=y_data_fit,
        r_squared=_r_squared(
            np.abs(y_data) if protocol == "oscillation" else y_data,
            np.abs(y_data_fit) if protocol == "oscillation" else y_data_fit,
            log_space=protocol != "startup",
        ),
        fit_model=fit_model,
        true_model=true_model,
        predict_kwargs=predict_kwargs,
    )
