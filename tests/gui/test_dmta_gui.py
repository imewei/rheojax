import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = REPO_ROOT / "rheojax" / "gui"
FORBIDDEN_TOKENS = (
    "deformation_mode",
    "poisson_ratio",
    "get_supported_deformation_modes",
    "SET_DEFORMATION_MODE",
    "SET_POISSON_RATIO",
)


def test_gui_clean():
    assert GUI_ROOT.is_dir()
    python_files = sorted(GUI_ROOT.rglob("*.py"))
    assert python_files

    bad = [
        str(path.relative_to(REPO_ROOT))
        for path in python_files
        if any(token in path.read_text() for token in FORBIDDEN_TOKENS)
    ]
    assert bad == [], bad


def test_worker_interfaces_are_shear_only():
    from rheojax.gui.jobs.bayesian_worker import BayesianWorker
    from rheojax.gui.jobs.fit_worker import FitWorker
    from rheojax.gui.jobs.process_adapter import (
        make_bayesian_worker,
        make_fit_worker,
    )
    from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated
    from rheojax.gui.jobs.subprocess_fit import run_fit_isolated

    removed_parameters = {"deformation_mode", "poisson_ratio"}
    interfaces = (
        FitWorker,
        BayesianWorker,
        make_fit_worker,
        make_bayesian_worker,
        run_fit_isolated,
        run_bayesian_isolated,
    )

    for interface in interfaces:
        assert removed_parameters.isdisjoint(inspect.signature(interface).parameters)
