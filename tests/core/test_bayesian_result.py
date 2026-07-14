"""Regression tests for BayesianResult.to_inference_data() fast-path assembly.

Covers the sample_stats exception-handling fix: a failure while extracting
NUTS extra fields must not (a) leave a partially-populated, internally
inconsistent sample_stats group silently attached, or (b) be logged only at
DEBUG level where it is invisible at normal verbosity.
"""

from __future__ import annotations

import logging

import numpy as np

from rheojax.core.bayesian_result import BayesianResult


class _PartialFailExtraFields(dict):
    """Simulates MCMC.get_extra_fields() failing partway through iteration.

    Yields one field successfully (so ``stats_dict`` would be non-empty if
    not reset) then raises, mirroring an unexpected error in the middle of
    the per-field coercion loop rather than the documented
    ``group_by_chain=True`` TypeError case.
    """

    def items(self):
        yield "potential_energy", np.ones((1, 5))
        raise RuntimeError("simulated mid-loop failure")


class _FakeMCMC:
    def __init__(self, extra_fields_factory):
        self._extra_fields_factory = extra_fields_factory

    def get_samples(self, group_by_chain=True):
        return {"a": np.ones((1, 5))}

    def get_extra_fields(self, group_by_chain=True):
        return self._extra_fields_factory()


def _make_result(mcmc) -> BayesianResult:
    return BayesianResult(
        posterior_samples={"a": np.ones(5)},
        summary={},
        diagnostics={},
        num_samples=5,
        num_chains=1,
        mcmc=mcmc,
    )


def test_sample_stats_failure_drops_partial_stats_not_silently(caplog) -> None:
    """A mid-loop sample_stats failure must clear stats_dict, not leak a
    partially-built subset of fields into the returned InferenceData."""
    result = _make_result(_FakeMCMC(_PartialFailExtraFields))

    with caplog.at_level(logging.WARNING, logger="rheojax.core.bayesian_result"):
        idata = result.to_inference_data()

    # Posterior group must survive (unrelated to the sample_stats failure).
    assert hasattr(idata, "posterior")
    assert "a" in idata.posterior.data_vars

    # sample_stats must be entirely absent, not partially populated with
    # just the "lp"/"energy" entries collected before the exception fired.
    assert not hasattr(idata, "sample_stats")

    # The drop must be visible at WARNING level (not silently swallowed at
    # DEBUG), so callers/log aggregators can detect the degradation.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("sample_stats" in r.getMessage() for r in warnings)


def test_sample_stats_success_path_still_populates_stats() -> None:
    """Sanity check: a well-behaved get_extra_fields() still produces a
    populated sample_stats group (fix must not regress the happy path)."""

    def good_extra_fields():
        return {"potential_energy": np.ones((1, 5)), "diverging": np.zeros((1, 5))}

    result = _make_result(_FakeMCMC(good_extra_fields))
    idata = result.to_inference_data()

    assert hasattr(idata, "sample_stats")
    assert "lp" in idata.sample_stats.data_vars
    assert "energy" in idata.sample_stats.data_vars
    assert "diverging" in idata.sample_stats.data_vars


def test_lp_is_not_negated_potential_energy() -> None:
    """Fix #2: the fast path must copy potential_energy into "lp" verbatim,
    matching the real ArviZ NumPyroConverter (no sign flip)."""

    potential = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    def extra_fields():
        return {"potential_energy": potential.copy()}

    result = _make_result(_FakeMCMC(extra_fields))
    idata = result.to_inference_data()

    np.testing.assert_array_equal(idata.sample_stats["lp"].values, potential)


def test_real_energy_field_is_not_overwritten_by_potential_energy_proxy() -> None:
    """Fix #1: when NumPyro provides the genuine "energy" extra field (the
    true Hamiltonian, potential + kinetic), it must be passed through as-is
    rather than fabricated as a copy of potential_energy."""

    potential = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    true_energy = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])

    def extra_fields():
        return {
            "potential_energy": potential.copy(),
            "energy": true_energy.copy(),
        }

    result = _make_result(_FakeMCMC(extra_fields))
    idata = result.to_inference_data()

    np.testing.assert_array_equal(idata.sample_stats["energy"].values, true_energy)
    # Must differ from the potential-energy-only proxy that would have been
    # fabricated pre-fix.
    assert not np.array_equal(idata.sample_stats["energy"].values, potential)


def test_missing_energy_field_falls_back_with_warning(caplog) -> None:
    """When only potential_energy is available (no real "energy" extra
    field, e.g. an older cached MCMC object), the fast path must still
    populate "energy" as a fallback proxy but log a WARNING noting BFMI
    will be approximate."""

    def extra_fields():
        return {"potential_energy": np.ones((1, 5))}

    result = _make_result(_FakeMCMC(extra_fields))
    with caplog.at_level(logging.WARNING, logger="rheojax.core.bayesian_result"):
        idata = result.to_inference_data()

    assert "energy" in idata.sample_stats.data_vars
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("approximat" in r.getMessage().lower() for r in warnings)


class _PartialFailGetSamples(dict):
    """Simulates get_samples(group_by_chain=True) failing partway through
    iteration, after at least one deterministic site has already been added
    to posterior_dict."""

    def items(self):
        yield "deterministic_site", np.ones((1, 5))
        raise RuntimeError("simulated mid-loop failure")


class _FakeMCMCPosteriorFail(_FakeMCMC):
    def get_samples(self, group_by_chain=True):
        return _PartialFailGetSamples()


def test_posterior_fallback_resets_partial_dict_and_warns(caplog) -> None:
    """A mid-loop failure in the posterior fast path must clear
    posterior_dict before falling back to posterior_samples, not leak
    entries collected before the exception, and log at WARNING."""

    def extra_fields():
        return {"potential_energy": np.ones((1, 5))}

    result = _make_result(_FakeMCMCPosteriorFail(extra_fields))

    with caplog.at_level(logging.WARNING, logger="rheojax.core.bayesian_result"):
        idata = result.to_inference_data()

    assert hasattr(idata, "posterior")
    # Fallback-derived entry present.
    assert "a" in idata.posterior.data_vars
    # Entry added before the exception must not survive into the fallback.
    assert "deterministic_site" not in idata.posterior.data_vars

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("posterior_samples" in r.getMessage() for r in warnings)


def test_eq_does_not_crash_with_multi_element_arrays() -> None:
    """Default dataclass __eq__ raises ValueError on ndarray-valued dict
    fields once posterior_samples has more than one element; the custom
    __eq__ must compare by value (np.array_equal) instead."""
    r1 = _make_result(None)
    r2 = _make_result(None)
    assert r1 == r2

    r3 = BayesianResult(
        posterior_samples={"a": np.array([9.0, 9.0, 9.0, 9.0, 9.0])},
        summary={},
        diagnostics={},
        num_samples=5,
        num_chains=1,
    )
    assert r1 != r3
    assert r1 != "not a bayesian result"


def test_dropped_extra_field_types_are_logged_at_debug(caplog) -> None:
    """Fix #3: dict/tuple-valued and wrong-ndim extra fields must be logged
    (not silently dropped), mirroring the visibility standard already
    applied to the outer exception handler in this function."""

    def extra_fields():
        return {
            "potential_energy": np.ones((1, 5)),
            "adapt_state": {"step_size": 0.1},  # dict -> hits dict/tuple branch
            "bad_ndim": np.ones((1, 5, 5)),  # ndim == 3 -> hits ndim branch
        }

    result = _make_result(_FakeMCMC(extra_fields))
    with caplog.at_level(logging.DEBUG, logger="rheojax.core.bayesian_result"):
        idata = result.to_inference_data()

    assert "lp" in idata.sample_stats.data_vars
    assert "adapt_state" not in idata.sample_stats.data_vars
    assert "bad_ndim" not in idata.sample_stats.data_vars

    # Structured kwargs (stat name, reason) are merged onto record.extra by
    # RheoJAXLogger rather than interpolated into the message text.
    def _stat_of(record):
        return getattr(record, "extra", {}).get("stat")

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "container type" in m and _stat_of(r) == "adapt_state"
        for r, m in zip(caplog.records, messages)
    )
    assert any(
        "ndim" in m and _stat_of(r) == "bad_ndim"
        for r, m in zip(caplog.records, messages)
    )
