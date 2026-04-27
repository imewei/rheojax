"""Parameter-schema regression tests for the Fluidity family.

Freezes the exact parameter name list and bound tuple for each fluidity
model so that any rename, addition, removal, or bounds shift triggers a
loud, localised test failure instead of cascading into downstream
notebooks and example scripts that silently drift out of sync. Motivated
by the notebook-08 incident (2026-04-14) where the nonlocal fluidity
parameters were rewritten to the current Herschel-Bulkley + aging
schema, but the tutorial notebook kept calling ``set_value('tau_eq', …)``
and ``set_value('D_f', …)`` under a ``try/except logger.warning`` block.

If this test fails, the author changing the model parameters MUST decide
whether downstream notebooks, saved ``.npz`` files, and user-facing docs
need to be migrated in the same commit.
"""

from __future__ import annotations

import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled before importing any model module.
safe_import_jax()

from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal
from rheojax.models.fluidity.saramito import (
    FluiditySaramitoLocal,
    FluiditySaramitoNonlocal,
)

# ---------------------------------------------------------------------------
# Frozen parameter schemas — update here when a rename / addition is
# intentional, then migrate notebooks & docs in the SAME commit.
# ---------------------------------------------------------------------------

# FluidityLocal: HB flow curve + aging/rejuvenation fluidity
FLUIDITY_LOCAL_PARAMS: list[tuple[str, tuple[float, float]]] = [
    ("G", (1e3, 1e9)),
    ("tau_y", (1e-3, 1e6)),
    ("K", (1e-3, 1e6)),
    ("n_flow", (0.1, 2.0)),
    ("f_eq", (1e-12, 1e-3)),
    ("f_inf", (1e-6, 1.0)),
    ("theta", (0.1, 1e4)),
    ("a", (0.0, 100.0)),
    ("n_rejuv", (0.0, 2.0)),
]

# FluidityNonlocal: local schema + cooperativity length xi
FLUIDITY_NONLOCAL_PARAMS: list[tuple[str, tuple[float, float]]] = [
    *FLUIDITY_LOCAL_PARAMS,
    ("xi", (1e-9, 1e-3)),
]

# FluiditySaramitoLocal (coupling="minimal"): Saramito yield-Maxwell + aging
FLUIDITY_SARAMITO_LOCAL_MIN_PARAMS: list[tuple[str, tuple[float, float]]] = [
    ("G", (1e1, 1e8)),
    ("eta_s", (0.0, 1e3)),
    ("tau_y0", (1e-1, 1e5)),
    ("K_HB", (1e-2, 1e5)),
    ("n_HB", (0.1, 1.5)),
    ("f_age", (1e-12, 1e-2)),
    ("f_flow", (1e-6, 1.0)),
    ("t_a", (1e-2, 1e5)),
    ("b", (0.0, 1e3)),
    ("n_rej", (0.1, 3.0)),
]

# FluiditySaramitoLocal (coupling="full"): +2 coupling params
FLUIDITY_SARAMITO_LOCAL_FULL_EXTRAS: list[tuple[str, tuple[float, float]]] = [
    ("tau_y_coupling", (0.0, 1e4)),
    ("m_yield", (0.1, 2.0)),
]

# FluiditySaramitoNonlocal: Saramito base + xi
FLUIDITY_SARAMITO_NONLOCAL_MIN_PARAMS: list[tuple[str, tuple[float, float]]] = [
    *FLUIDITY_SARAMITO_LOCAL_MIN_PARAMS,
    ("xi", (1e-7, 1e-2)),
]


def _extract(model) -> list[tuple[str, tuple[float, float]]]:
    """Return [(name, bounds), ...] in registration order."""
    ps = model.parameters
    return [(name, ps[name].bounds) for name in ps.keys()]


@pytest.mark.smoke
class TestParameterSchema:
    """Lock the current parameter schema for every fluidity model.

    Changes to these lists should be deliberate and accompanied by
    migration of any notebooks / saved parameter files that reference
    the old names.
    """

    def test_fluidity_local_schema(self):
        actual = _extract(FluidityLocal())
        assert actual == FLUIDITY_LOCAL_PARAMS

    def test_fluidity_nonlocal_schema(self):
        actual = _extract(FluidityNonlocal())
        assert actual == FLUIDITY_NONLOCAL_PARAMS

    def test_saramito_local_minimal_schema(self):
        actual = _extract(FluiditySaramitoLocal(coupling="minimal"))
        assert actual == FLUIDITY_SARAMITO_LOCAL_MIN_PARAMS

    def test_saramito_local_full_schema(self):
        actual = _extract(FluiditySaramitoLocal(coupling="full"))
        expected = (
            FLUIDITY_SARAMITO_LOCAL_MIN_PARAMS + FLUIDITY_SARAMITO_LOCAL_FULL_EXTRAS
        )
        assert actual == expected

    def test_saramito_nonlocal_minimal_schema(self):
        actual = _extract(FluiditySaramitoNonlocal(coupling="minimal"))
        assert actual == FLUIDITY_SARAMITO_NONLOCAL_MIN_PARAMS


@pytest.mark.smoke
class TestLegacyNamesRejected:
    """Explicitly assert that the old pre-April-2026 names no longer exist.

    If any of these re-appear, the rename was partial and the migration
    is broken.
    """

    LEGACY_NAMES = ["tau_eq", "D_f", "c", "tau_age", "alpha", "f_0"]

    @pytest.mark.parametrize(
        "model_cls, extra_kwargs",
        [
            (FluidityLocal, {}),
            (FluidityNonlocal, {}),
            (FluiditySaramitoLocal, {"coupling": "minimal"}),
            (FluiditySaramitoLocal, {"coupling": "full"}),
            (FluiditySaramitoNonlocal, {"coupling": "minimal"}),
        ],
    )
    def test_no_legacy_names(self, model_cls, extra_kwargs):
        model = model_cls(**extra_kwargs)
        present = set(model.parameters.keys())
        leaked = present & set(self.LEGACY_NAMES)
        assert not leaked, (
            f"{model_cls.__name__}({extra_kwargs}) exposes legacy "
            f"names {sorted(leaked)} — migration incomplete."
        )
