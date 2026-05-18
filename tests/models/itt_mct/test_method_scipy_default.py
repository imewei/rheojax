"""Regression test: ITT-MCT _fit_* methods must force ``method='scipy'``.

ITT-MCT models use ``diffrax`` ODE solvers in their ``_predict_*`` paths. The
default NLSQ optimizer attempts forward-mode autodiff on the residual function,
which JIT-traces ``ResidualFunction``. The residual closures in ``_base.py``
call ``self.parameters.set_values(param_dict)`` which performs ``float(value)``
on JAX tracers — that triggers ``jax.errors.ConcretizationTypeError``.

``_fit_flow_curve`` was already pinned to ``method='scipy'`` (line 380), but the
other five protocols (oscillation, startup, creep, relaxation, laos) accepted
``**kwargs`` straight through, allowing the parent ``BaseModel.fit()`` default
of ``method='nlsq'`` to flow in and re-enter the broken NLSQ-AD path. This test
pins the contract that ITT-MCT always forces ``method='scipy'`` regardless of
caller preference.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from rheojax.models.itt_mct import ITTMCTSchematic


@pytest.mark.smoke
def test_itt_mct_fit_routes_to_scipy_for_all_protocols(monkeypatch):
    """Every _fit_* protocol path must hand method='scipy' to fit_with_nlsq."""
    import rheojax.utils.optimization as opt_mod

    captured = {"calls": []}

    def fake_fit_with_nlsq(residual_fn, x0, bounds=None, y_data=None, **kwargs):
        captured["calls"].append(dict(kwargs))
        x0_arr = np.asarray(x0)
        y_d = y_data

        class Stub:
            x = x0_arr
            r_squared = 0.0
            cost = 0.0
            success = True
            x_scale = None
            n_iter = 0
            message = "stub"
            nfev = 0
            njev = 0
            fun = np.array([0.0])

        result = Stub()
        result.y_data = y_d
        return result

    monkeypatch.setattr(opt_mod, "fit_with_nlsq", fake_fit_with_nlsq)

    model = ITTMCTSchematic(epsilon=0.05)
    gd = np.array([0.1, 1.0, 10.0])
    y = np.array([10.0, 50.0, 200.0])

    protocols = [
        dict(test_mode="flow_curve"),
        dict(test_mode="oscillation"),
        dict(test_mode="startup", gamma_dot=1.0),
        dict(test_mode="creep", sigma_applied=10.0),
        dict(test_mode="relaxation", gamma_pre=0.05),
        dict(test_mode="laos", gamma_0=0.1, omega=1.0),
    ]

    for fkw in protocols:
        captured["calls"].clear()
        model.fit(gd, y, **fkw)
        assert captured["calls"], f"fit_with_nlsq not invoked for {fkw['test_mode']}"
        last = captured["calls"][-1]
        assert last.get("method") == "scipy", (
            f"Protocol {fkw['test_mode']}: expected method='scipy', got {last.get('method')!r}. "
            "ITT-MCT must force scipy because NLSQ JIT-traces residual_func which "
            "calls float() on tracers via set_values()."
        )


@pytest.mark.smoke
def test_itt_mct_fit_overrides_user_method_nlsq(monkeypatch):
    """Even when caller asks for method='nlsq', ITT-MCT must override to 'scipy'."""
    import rheojax.utils.optimization as opt_mod

    captured = {"calls": []}

    def fake_fit_with_nlsq(residual_fn, x0, bounds=None, y_data=None, **kwargs):
        captured["calls"].append(dict(kwargs))
        x0_arr = np.asarray(x0)

        class Stub:
            x = x0_arr
            r_squared = 0.0
            cost = 0.0
            success = True
            x_scale = None
            n_iter = 0
            message = "stub"
            nfev = 0
            njev = 0
            fun = np.array([0.0])

        result = Stub()
        result.y_data = y_data
        return result

    monkeypatch.setattr(opt_mod, "fit_with_nlsq", fake_fit_with_nlsq)

    model = ITTMCTSchematic(epsilon=0.05)
    model.fit(
        np.array([0.01, 1.0]),
        np.array([5.0, 100.0]),
        test_mode="relaxation",
        gamma_pre=0.05,
        method="nlsq",
    )
    assert captured["calls"][-1]["method"] == "scipy"
