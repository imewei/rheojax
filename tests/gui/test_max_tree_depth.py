"""max_tree_depth passthrough — signature-level checks.

Regression test: config["target_accept_prob"]/config["max_tree_depth"] were
computed from the Advanced Options dialog but never reached
make_bayesian_worker() in _continue_bayesian_with_dataset -- silently
dropped before the sampler ever saw them.
"""

import inspect

import jax  # noqa: F401
import nlsq  # noqa: F401 — must precede any rheojax.core imports

from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.process_adapter import make_bayesian_worker
from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated


def test_max_tree_depth_param_present():
    assert "max_tree_depth" in inspect.signature(run_bayesian_isolated).parameters
    assert "max_tree_depth" in inspect.signature(make_bayesian_worker).parameters
    assert "max_tree_depth" in inspect.signature(BayesianWorker.__init__).parameters


def test_max_tree_depth_defaults_to_none():
    assert (
        inspect.signature(run_bayesian_isolated).parameters["max_tree_depth"].default
        is None
    )
    assert (
        inspect.signature(make_bayesian_worker).parameters["max_tree_depth"].default
        is None
    )
    assert (
        inspect.signature(BayesianWorker.__init__).parameters["max_tree_depth"].default
        is None
    )
