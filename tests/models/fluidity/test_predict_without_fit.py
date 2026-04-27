"""Regression test: ``predict()`` must work without a prior ``fit()``.

Motivating bug (2026-04-14): both ``FluidityLocal._predict_transient``
and ``FluidityNonlocal._predict_transient`` read protocol inputs
(``gamma_dot``, ``sigma_applied``) only from instance attributes that are
populated exclusively inside ``_fit_*`` methods. Passing ``gamma_dot``
as a kwarg to ``predict()`` was silently dropped, so notebook-08 (a
tutorial that forward-simulates synthetic data via ``predict`` WITHOUT
fitting first) produced an all-zero stress trajectory with no warning.

This module enforces the invariant that forward simulation is a
first-class operation:

    model = Cls(...)                                  # no fit
    model.parameters.update({...})                    # set physical params
    σ = model.predict(t, test_mode=protocol, **kw)    # must be non-trivial

If this fails for a newly added transient-protocol model, the likely
cause is the same "kwarg dropped on the floor" pattern. Look at the
model's ``_predict`` dispatch and make sure protocol kwargs reach the
inner solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

safe_import_jax()  # float64 setup

from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry

# Force eager import of every model submodule so the ``@ModelRegistry.register``
# decorators fire BEFORE canary parametrization collects cases. Without this
# the registry only knows about models imported above (fluidity family), and
# the canary silently covers a handful of models instead of every
# transient-protocol implementation in the codebase.
from rheojax.models import _ensure_all_registered as _ensure_models
from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal
from rheojax.models.fluidity.saramito import (
    FluiditySaramitoLocal,
    FluiditySaramitoNonlocal,
)

_ensure_models()


# Physically-valid parameter seeds for each fluidity model family.
# Values chosen so the default HB flow curve produces O(10-100) Pa
# stress under γ̇ = 1 /s — any silent drop of protocol kwargs (→ γ̇ = 0)
# collapses the trajectory to zero and trips the ``nonzero_stress``
# assertion.
_FLUIDITY_HB_PARAMS = {
    "G": 1000.0,
    "tau_y": 30.0,
    "K": 10.0,
    "n_flow": 0.5,
    "f_eq": 1e-4,
    "f_inf": 1e-2,
    "theta": 10.0,
    "a": 1.0,
    "n_rejuv": 1.0,
}
_FLUIDITY_NONLOCAL_PARAMS = {**_FLUIDITY_HB_PARAMS, "xi": 5e-5}

_SARAMITO_MIN_PARAMS = {
    "G": 1000.0,
    "eta_s": 1.0,
    "tau_y0": 30.0,
    "K_HB": 10.0,
    "n_HB": 0.5,
    "f_age": 1e-6,
    "f_flow": 1e-2,
    "t_a": 10.0,
    "b": 1.0,
    "n_rej": 1.0,
}
_SARAMITO_NONLOCAL_PARAMS = {**_SARAMITO_MIN_PARAMS, "xi": 5e-5}


def _t_grid() -> np.ndarray:
    """Small (101-point) log-linear time grid covering early + late transient."""
    early = np.logspace(-2, 0, 30)
    late = np.linspace(1.0, 100.0, 71)
    return np.unique(np.concatenate([early, late]))


@pytest.mark.smoke
class TestFluidityPredictWithoutFit:
    """Direct regression tests for the four fluidity models.

    Each test constructs a model, sets a valid parameter dict, then
    calls ``predict()`` for each transient protocol passing the
    protocol-specific kwarg. A non-zero stress (or strain) trajectory
    is the only required invariant — if the protocol kwarg is dropped,
    the PDE runs with γ̇ = 0 (startup) / σ_applied = 0 (creep) /
    σ_0 = tau_y fallback (relaxation) and the observable collapses.
    """

    # --- FluidityLocal ---------------------------------------------------

    def test_local_startup(self):
        model = FluidityLocal()
        model.parameters.update(_FLUIDITY_HB_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="startup", gamma_dot=1.0)
        assert np.any(np.asarray(sigma) > 1.0), (
            "Startup predict returned ~0 stress — gamma_dot kwarg likely "
            "not forwarded to _simulate_transient."
        )

    def test_local_relaxation(self):
        model = FluidityLocal()
        model.parameters.update(_FLUIDITY_HB_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="relaxation", sigma_0=100.0)
        sigma_arr = np.asarray(sigma)
        assert sigma_arr[0] > 10.0, "Relaxation did not start from sigma_0."
        assert sigma_arr[-1] < sigma_arr[0], "Stress did not relax."

    def test_local_creep(self):
        model = FluidityLocal()
        model.parameters.update(_FLUIDITY_HB_PARAMS)
        gamma = model.predict(
            _t_grid(), test_mode="creep", sigma_applied=100.0
        )
        assert np.any(np.asarray(gamma) > 0.0), (
            "Creep predict returned ~0 strain — sigma_applied kwarg "
            "likely not forwarded."
        )

    # --- FluidityNonlocal ------------------------------------------------

    def test_nonlocal_startup(self):
        model = FluidityNonlocal(N_y=21, gap_width=1e-3)
        model.parameters.update(_FLUIDITY_NONLOCAL_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="startup", gamma_dot=1.0)
        assert np.any(np.asarray(sigma) > 1.0)

    def test_nonlocal_relaxation(self):
        model = FluidityNonlocal(N_y=21, gap_width=1e-3)
        model.parameters.update(_FLUIDITY_NONLOCAL_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="relaxation", sigma_0=100.0)
        sigma_arr = np.asarray(sigma)
        assert sigma_arr[0] > 10.0
        assert sigma_arr[-1] < sigma_arr[0]

    def test_nonlocal_creep(self):
        model = FluidityNonlocal(N_y=21, gap_width=1e-3)
        model.parameters.update(_FLUIDITY_NONLOCAL_PARAMS)
        gamma = model.predict(
            _t_grid(), test_mode="creep", sigma_applied=100.0
        )
        assert np.any(np.asarray(gamma) > 0.0)

    # --- FluiditySaramitoLocal (minimal coupling) ------------------------

    def test_saramito_local_startup(self):
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.update(_SARAMITO_MIN_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="startup", gamma_dot=1.0)
        assert np.any(np.asarray(sigma) > 1.0)

    # --- FluiditySaramitoNonlocal (minimal coupling) ---------------------

    def test_saramito_nonlocal_startup(self):
        model = FluiditySaramitoNonlocal(coupling="minimal", N_y=21, H=1e-3)
        model.parameters.update(_SARAMITO_NONLOCAL_PARAMS)
        sigma = model.predict(_t_grid(), test_mode="startup", gamma_dot=1.0)
        assert np.any(np.asarray(sigma) > 1.0)


# ---------------------------------------------------------------------------
# Registry-driven canary test — best-effort sweep over every model that
# declares support for a transient protocol. Construction or protocol
# setup that needs special kwargs beyond what this test can provide is
# gracefully skipped. The goal is *not* to exhaustively validate every
# model, but to catch the "kwarg dropped on the floor" antipattern for
# any future transient-protocol model that can be built with defaults.
# ---------------------------------------------------------------------------

_TRANSIENT_KWARGS: dict[Protocol, dict] = {
    # Drive amplitudes chosen to overshoot typical model-default yield
    # stresses (tau_y defaults range from O(10)–O(100) Pa) and to exceed
    # the longest relaxation timescale on the 10 s canary window so every
    # fluidity/saramito default exits the purely-elastic regime.
    Protocol.STARTUP: {"gamma_dot": 100.0},
    Protocol.RELAXATION: {"sigma_0": 500.0},
    Protocol.CREEP: {"sigma_applied": 1000.0},
}


# Models that are too slow to integrate at default parameters without a
# warm start — skip these in the canary to keep the suite under budget.
# Documented in CLAUDE.md memory under "Slow models (>120s)".
_SLOW_MODELS_DENYLIST: set[str] = {
    "itt_mct_schematic",  # PDE integrator chokes on default aging params
    "itt_mct_isotropic",  # same family
}


def _transient_model_cases() -> list[tuple[str, str]]:
    """Return (model_name, protocol_value) for every registered model
    supporting any transient protocol."""
    cases: list[tuple[str, str]] = []
    for proto in _TRANSIENT_KWARGS:
        for info in ModelRegistry.for_protocol(proto):
            if info.name in _SLOW_MODELS_DENYLIST:
                continue
            cases.append((info.name, proto.value))
    return cases


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_name, protocol", _transient_model_cases(),
    ids=lambda x: str(x),
)
def test_transient_predict_without_fit_canary(model_name: str, protocol: str):
    """Build + predict each transient-protocol model without fitting.

    Skips gracefully on construction errors, missing protocol kwargs,
    or models that legitimately require a prior fit (documented below).
    Any *hard* failure here indicates the same architectural bug the
    fluidity family had: protocol kwargs are being dropped between
    ``predict()`` and the inner simulator.
    """
    try:
        model = ModelRegistry.create(model_name)
    except Exception as exc:
        pytest.skip(f"cannot default-construct {model_name}: {exc}")

    proto = Protocol(protocol)
    kwargs = dict(_TRANSIENT_KWARGS[proto])

    # Small, model-agnostic time grid; avoid extreme ranges that may
    # destabilise stiff kernels without a warm-started parameter set.
    t = np.linspace(0.01, 10.0, 51)

    try:
        out = model.predict(t, test_mode=protocol, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"{model_name}.{protocol}: not implemented ({exc})")
    except Exception as exc:
        # Construction succeeded but predict path needs extra setup
        # (e.g. metadata, spatial grid, required protocol kwargs).
        # That's a different bug class — not the kwarg-forwarding one
        # we're guarding against — so we skip rather than fail.
        pytest.skip(f"{model_name}.{protocol}: predict needs more setup ({exc})")

    try:
        arr = np.asarray(out)
        finite_mask = np.isfinite(arr)
    except TypeError as exc:
        # Some models return structured / object outputs (e.g. EPM returns
        # a namedtuple of (stress, strain, metadata)). That's a different
        # contract from "predict returns a numeric array" and not what
        # this canary is testing.
        pytest.skip(
            f"{model_name}.{protocol}: predict returned non-array output "
            f"({type(out).__name__}); canary asserts invariants on numeric "
            f"arrays only ({exc})"
        )

    if not np.all(finite_mask):
        pytest.skip(
            f"{model_name}.{protocol}: non-finite output at default params "
            "(parameter warm-start would likely fix this — out of scope)."
        )

    # The core invariant: some non-trivial variation must reach the
    # output. Zero variance means the protocol input silently evaporated.
    arr_real = np.real(arr) if np.iscomplexobj(arr) else arr
    variation = float(np.ptp(arr_real))
    assert variation > 0.0, (
        f"{model_name}.{protocol}: predict() returned a constant trajectory. "
        "Check that the protocol kwarg (gamma_dot / sigma_applied / sigma_0) "
        "is forwarded from _predict() to the inner simulator instead of "
        "being read only from a self._*_applied attribute set during fit."
    )
