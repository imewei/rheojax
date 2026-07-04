"""Task 7: target_accept passthrough — signature-level test, no MCMC runs."""

import inspect

import jax  # noqa: F401
import nlsq  # noqa: F401 — must precede any rheojax.core imports

from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.process_adapter import make_bayesian_worker
from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated


def test_target_accept_param_present():
    assert "target_accept" in inspect.signature(run_bayesian_isolated).parameters
    assert "target_accept" in inspect.signature(make_bayesian_worker).parameters
    assert "target_accept" in inspect.signature(BayesianWorker.__init__).parameters


def test_target_accept_defaults():
    params = inspect.signature(run_bayesian_isolated).parameters
    assert params["target_accept"].default == 0.8

    params = inspect.signature(make_bayesian_worker).parameters
    assert params["target_accept"].default == 0.8

    params = inspect.signature(BayesianWorker.__init__).parameters
    assert params["target_accept"].default == 0.8
