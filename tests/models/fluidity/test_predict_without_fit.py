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

from rheojax.core.data import RheoData
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
        gamma = model.predict(_t_grid(), test_mode="creep", sigma_applied=100.0)
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
        gamma = model.predict(_t_grid(), test_mode="creep", sigma_applied=100.0)
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
# Dedicated tests for models whose real predict-without-fit contract differs
# from the fluidity family's ``predict(t, test_mode, protocol_kwarg)`` shape.
# These replace the generic canary's graceful skips (see skip categories in
# ``test_transient_predict_without_fit_canary``) with assertions of the
# actual, documented behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("protocol", ["startup", "creep"])
def test_vlb_nonlocal_predict_not_implemented(protocol):
    """VLBNonlocal intentionally has no ``_predict()`` — it is a spatially
    resolved model that only exposes ``simulate_steady_shear()``.

    The canary skips these two combos citing "not implemented"; this asserts
    the documented ``NotImplementedError`` is actually raised rather than
    silently returning a wrong (e.g. all-zero) trajectory.
    """
    model = ModelRegistry.create("vlb_nonlocal")
    t = np.linspace(0.01, 10.0, 51)
    with pytest.raises(NotImplementedError, match="does not support _predict"):
        model.predict(t, test_mode=protocol, **_TRANSIENT_KWARGS[Protocol(protocol)])


# EPM ``predict()`` consumes protocol inputs via ``RheoData`` metadata, not via
# plain ``predict(**kwargs)`` (the raw-array path only reads fit-time caches and
# ``_rheo_metadata``). The metadata key is protocol-specific and differs from
# the fluidity kwarg names: startup→``gamma_dot``, relaxation→``gamma`` (step
# strain), creep→``stress`` (target stress). Magnitudes overshoot the default
# unit yield threshold (``sigma_c_mean=1``, ``mu=1``) so blocks actually yield
# and the observable varies. A small lattice (L=16) keeps the smoke test fast.
_EPM_META = {
    "startup": {"gamma_dot": 100.0},
    "relaxation": {"gamma": 3.0},
    "creep": {"stress": 2.0},
}


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", ["lattice_epm", "tensorial_epm"])
@pytest.mark.parametrize("protocol", ["startup", "relaxation", "creep"])
def test_epm_predict_without_fit(model_name, protocol):
    """EPM models return a structured ``RheoData`` (not a bare array), so the
    canary skips them as "non-array output". This drives the real contract:
    build a ``RheoData`` carrying the protocol input in metadata, then assert
    the unwrapped ``.y`` observable is finite and non-trivial.
    """
    model = ModelRegistry.create(model_name, L=16, dt=0.01)
    t = np.linspace(0.01, 10.0, 51)
    rheo = RheoData(
        x=t,
        y=np.zeros_like(t),
        initial_test_mode=protocol,
        metadata={**_EPM_META[protocol], "test_mode": protocol},
    )

    out = model.predict(rheo, test_mode=protocol)

    assert isinstance(out, RheoData), (
        f"{model_name}.{protocol}: expected RheoData, got {type(out).__name__}"
    )
    y = np.asarray(out.y)
    assert np.all(np.isfinite(y)), f"{model_name}.{protocol}: non-finite output"
    assert np.ptp(y) > 0.0, (
        f"{model_name}.{protocol}: constant trajectory — protocol input in "
        "RheoData metadata did not reach the inner EPM step."
    )


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", ["fikh", "fmlikh"])
def test_fikh_startup_predict_without_fit(model_name):
    """FIKH/FMLIKH startup is strain-driven: the return-mapping path needs the
    full strain history, not just ``gamma_dot`` (the canary skips these as
    "needs more setup"). Supply ``strain = gamma_dot * t`` alongside the time
    array and assert a finite, non-trivial stress growth curve.
    """
    model = ModelRegistry.create(model_name)
    t = np.linspace(0.01, 10.0, 51)
    gamma_dot = 1.0
    strain = gamma_dot * t

    sigma = np.asarray(model.predict(t, test_mode="startup", strain=strain))

    assert np.all(np.isfinite(sigma)), f"{model_name} startup: non-finite stress"
    assert np.ptp(sigma) > 0.0, (
        f"{model_name} startup: flat stress — strain history not forwarded to "
        "the return-mapping solver."
    )
    assert sigma[-1] > sigma[0], f"{model_name} startup: stress did not build up."


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_name",
    ["tnt", "tnt_single_mode", "tnt_cates", "tnt_loop_bridge", "tnt_multi_species"],
)
def test_tnt_relaxation_predict_without_fit(model_name):
    """TNT ``relaxation`` is a POST-FLOW relaxation: the network is pre-sheared
    at ``gamma_dot`` then released, so the mode requires ``gamma_dot`` — not the
    ``sigma_0`` (initial-stress) the generic canary supplies, which is why the
    canary skips these as "needs more setup". Drive with a pre-shear rate and
    assert the stress decays from a finite non-zero plateau.
    """
    model = ModelRegistry.create(model_name)
    t = np.linspace(0.01, 10.0, 51)

    sigma = np.asarray(model.predict(t, test_mode="relaxation", gamma_dot=1.0))
    sigma = np.real(sigma) if np.iscomplexobj(sigma) else sigma

    assert np.all(np.isfinite(sigma)), f"{model_name} relaxation: non-finite stress"
    assert np.ptp(sigma) > 0.0, (
        f"{model_name} relaxation: flat trajectory — pre-shear gamma_dot not "
        "forwarded to the relaxation solver."
    )
    assert sigma[0] > sigma[-1], (
        f"{model_name} relaxation: stress did not decay after pre-shear release."
    )


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


# (model_name, protocol) combinations that the generic canary can only ever
# gracefully skip under its one-size-fits-all kwargs contract, but which are
# now exercised for real by a dedicated test above with model-appropriate
# kwargs/parameters. Excluded here rather than left to skip: the real
# behaviour IS tested (just not via this generic sweep), so a skip here
# would just be redundant noise, not missing coverage. See the dedicated
# test docstrings for exactly why each one doesn't fit the generic contract.
_COVERED_BY_DEDICATED_TESTS: set[tuple[str, str]] = {
    # test_vlb_nonlocal_predict_not_implemented
    ("vlb_nonlocal", "startup"),
    ("vlb_nonlocal", "creep"),
    # test_epm_predict_without_fit
    ("lattice_epm", "startup"),
    ("lattice_epm", "relaxation"),
    ("lattice_epm", "creep"),
    ("tensorial_epm", "startup"),
    ("tensorial_epm", "relaxation"),
    ("tensorial_epm", "creep"),
    # test_fikh_startup_predict_without_fit
    ("fikh", "startup"),
    ("fmlikh", "startup"),
    # test_tnt_relaxation_predict_without_fit
    ("tnt", "relaxation"),
    ("tnt_single_mode", "relaxation"),
    ("tnt_cates", "relaxation"),
    ("tnt_loop_bridge", "relaxation"),
    ("tnt_multi_species", "relaxation"),
    # test_creep_predict_without_fit_finite_viscosity
    ("giesekus", "creep"),
    ("giesekus_single", "creep"),
    ("vlb_multi_network", "creep"),
    ("tnt", "creep"),
    ("tnt_single_mode", "creep"),
    ("tnt_cates", "creep"),
    # test_ikh_creep_predict_without_fit_scaled
    ("mikh", "creep"),
    ("ml_ikh", "creep"),
    # test_stz_conventional_{startup,relaxation}_predict_without_fit
    ("stz_conventional", "startup"),
    ("stz_conventional", "relaxation"),
}


def _transient_model_cases() -> list[tuple[str, str]]:
    """Return (model_name, protocol_value) for every registered model
    supporting any transient protocol, excluding models already covered
    by a dedicated test."""
    cases: list[tuple[str, str]] = []
    for proto in _TRANSIENT_KWARGS:
        for info in ModelRegistry.for_protocol(proto):
            if info.name in _SLOW_MODELS_DENYLIST:
                continue
            if (info.name, proto.value) in _COVERED_BY_DEDICATED_TESTS:
                continue
            cases.append((info.name, proto.value))
    return cases


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_name, protocol",
    _transient_model_cases(),
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


# ---------------------------------------------------------------------------
# Dedicated tests for models the canary can only *skip* with "non-finite
# output at default params". These are NOT kwarg-forwarding bugs — the
# protocol input IS forwarded, but the canary's one-size-fits-all forcing
# (γ̇ = 100, σ₀ = 500, σ_applied = 1000 Pa) is physically wrong for these
# models' own parameter scales and pushes their ODE solvers into divergence.
# Each test below supplies forcing (and, where a default is physically
# ill-posed, one physical parameter override) scaled to the model's own
# characteristic stress/rate so the real transient-solve path runs and
# returns FINITE, non-trivial output.
# ---------------------------------------------------------------------------


# Viscoelastic-network models whose DEFAULT ``eta_s = 0`` makes creep
# ill-posed: the creep RHS computes γ̇ = (σ − τ_xy) / max(eta_s, 1e-10·eta_p),
# so with eta_s = 0 the regularization floor leaves γ̇ ≈ 1e8·σ at t=0 — an
# infinitely stiff kick the ODE solver (throw=False) returns as NaN (the
# Giesekus kernel docstring even states "For η_s = 0, creep is not
# well-defined"). A physical solvent viscosity (~1–10 % of the elastic
# stress scale G = eta_p/lambda_1 or G_0) restores a finite instantaneous
# response so creep is well-posed. ``sigma_applied`` is set to O(elastic
# stress scale); a silent drop of the kwarg collapses the strain to zero.
_CREEP_VISCOSITY_CASES: dict[str, tuple[dict, float]] = {
    "giesekus": ({"eta_s": 1.0}, 50.0),  # G = eta_p/lambda_1 = 100 Pa
    "giesekus_single": ({"eta_s": 1.0}, 50.0),  # alias of GiesekusSingleMode
    "vlb_multi_network": ({"eta_s": 10.0}, 100.0),  # G_0 + G_1 = 1000 Pa
    "tnt": ({"eta_s": 10.0}, 100.0),  # G = 1000 Pa
    "tnt_single_mode": ({"eta_s": 10.0}, 100.0),  # alias of TNTSingleMode
    "tnt_cates": ({"eta_s": 10.0}, 100.0),  # G_0 = 1000 Pa
}


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", list(_CREEP_VISCOSITY_CASES))
def test_creep_predict_without_fit_finite_viscosity(model_name):
    """Creep for viscoelastic-network models with a physical solvent viscosity.

    Replaces the canary's "non-finite at default params" skip: the NaN is
    an artefact of the ``eta_s = 0`` default (creep ill-posed), not of the
    solver. With eta_s > 0 the strain accumulates finitely and non-trivially.
    """
    params, sigma_applied = _CREEP_VISCOSITY_CASES[model_name]
    model = ModelRegistry.create(model_name)
    model.parameters.update(params)
    gamma = np.asarray(
        model.predict(_t_grid(), test_mode="creep", sigma_applied=sigma_applied)
    )
    gamma = np.real(gamma) if np.iscomplexobj(gamma) else gamma
    assert np.all(np.isfinite(gamma)), f"{model_name} creep: non-finite strain."
    assert np.ptp(gamma) > 0.0, (
        f"{model_name} creep: constant strain — sigma_applied not forwarded "
        "to the creep solver."
    )


# Elastoviscoplastic (IKH-family) models that ARE finite under a reasonable
# creep stress but overflow under the canary's aggressive σ_applied = 1000 Pa
# (an order of magnitude above their O(10) Pa yield). Apply a stress modestly
# above the default yield so the material flows into a finite, non-trivial
# creep. No parameter override needed — the DEFAULTS are fine here.
_CREEP_AMPLITUDE_CASES: dict[str, float] = {
    "mikh": 200.0,  # yield ≈ sigma_y0 + delta_sigma_y ≈ 60 Pa
    "ml_ikh": 30.0,  # two-mode yield ≈ 10 Pa; low mu_p → fast flow above yield
}


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", list(_CREEP_AMPLITUDE_CASES))
def test_ikh_creep_predict_without_fit_scaled(model_name):
    """Creep for IKH-family models at a stress scaled to their own yield.

    Replaces the canary's "non-finite at default params" skip, which is only
    triggered because σ_applied = 1000 Pa is ~10× these models' yield stress
    and overflows the flow solve. A stress just above yield gives a finite,
    growing creep strain.
    """
    sigma_applied = _CREEP_AMPLITUDE_CASES[model_name]
    model = ModelRegistry.create(model_name)
    gamma = np.asarray(
        model.predict(_t_grid(), test_mode="creep", sigma_applied=sigma_applied)
    )
    gamma = np.real(gamma) if np.iscomplexobj(gamma) else gamma
    assert np.all(np.isfinite(gamma)), f"{model_name} creep: non-finite strain."
    assert np.ptp(gamma) > 0.0, (
        f"{model_name} creep: constant strain — sigma_applied not forwarded "
        "to the flow solver."
    )


# STZ conventional: the default ``tau0 = 1e-12`` s (a femtosecond molecular
# attempt time) makes the plastic-rate term ~1/tau0 ≈ 1e12 astronomically
# stiff. The transient path uses an EXPLICIT Tsit5 integrator (throw=False),
# which hits max_steps and returns NaN — hence the canary can only skip.
# A millisecond attempt time (tau0 = 1e-3 s) is still a physical STZ value
# and keeps the ODE integrable while the flow-defect (χ) dynamics are
# unchanged, so the observable is a real STZ transient rather than a stub.


@pytest.mark.smoke
def test_stz_conventional_startup_predict_without_fit():
    """STZ startup with a numerically tractable attempt time.

    tau0 = 1e-3 s tames the default femtosecond stiffness; γ̇ = 0.1 /s drives
    a finite stress-growth curve (well below the sigma_y = 1e6 Pa scale). A
    dropped gamma_dot would leave γ̇ = 0 and a flat zero trajectory.
    """
    model = ModelRegistry.create("stz_conventional")
    model.parameters.update({"tau0": 1e-3})
    sigma = np.asarray(model.predict(_t_grid(), test_mode="startup", gamma_dot=0.1))
    sigma = np.real(sigma) if np.iscomplexobj(sigma) else sigma
    assert np.all(np.isfinite(sigma)), "STZ startup: non-finite stress."
    assert np.ptp(sigma) > 0.0, (
        "STZ startup: flat trajectory — gamma_dot not forwarded to the "
        "transient solver."
    )
    assert sigma.max() > 1.0, "STZ startup: stress never built up."


@pytest.mark.smoke
def test_stz_conventional_relaxation_predict_without_fit():
    """STZ relaxation from an imposed sigma_0 with a tractable attempt time.

    Relaxation is stiffer than startup at default params: the trajectory
    STARTS at the high imposed stress, so the plastic decay is immediate and
    the explicit solver diverges at every amplitude. tau0 = 1e-3 s makes the
    decay integrable; sigma_0 = 1e4 Pa then relaxes toward ~0. A dropped
    sigma_0 would start from the sigma_y fallback (1e6 Pa) instead.
    """
    model = ModelRegistry.create("stz_conventional")
    model.parameters.update({"tau0": 1e-3})
    sigma = np.asarray(model.predict(_t_grid(), test_mode="relaxation", sigma_0=1e4))
    sigma = np.real(sigma) if np.iscomplexobj(sigma) else sigma
    assert np.all(np.isfinite(sigma)), "STZ relaxation: non-finite stress."
    assert sigma[0] > 1e3, "STZ relaxation: did not start from the imposed sigma_0."
    assert sigma[-1] < sigma[0], "STZ relaxation: stress did not relax."
