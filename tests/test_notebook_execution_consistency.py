"""Pytest regression for the notebook silent-skip QA scanner.

Two responsibilities:

1. **Scanner unit behaviour.** Synthetic notebooks exercise the three states
   the scanner needs to distinguish: (a) a clean partial-execution notebook
   with no fit cells, (b) a fully-cleared notebook (intentional, must be
   ignored), (c) the silent-skip pattern (must be flagged).

2. **Repo invariant.** All notebooks under ``examples/`` must pass the
   scanner. This catches future occurrences of the same bug class that hit
   STZ NB04/NB05 (2026-05-18 RCA — see memory file
   ``stz-notebook-rca-2026-05-18.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_notebook_execution import (
    _allowlist_matches,
    _load_allowlist,
    check_notebook,
)
from check_notebook_execution import (
    main as scanner_main,
)

BASELINE_FILE = REPO_ROOT / "scripts" / "notebook_execution_baseline.txt"


def _make_nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code(source: str, execution_count: int | None, idx: int = 0) -> dict:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "id": f"cell-{idx}",
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


@pytest.mark.smoke
def test_scanner_clean_notebook(tmp_path: Path) -> None:
    """Partial-execution notebook with no fit cells must pass."""
    p = tmp_path / "clean.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("import numpy as np", execution_count=1, idx=0),
                    _code("print('hi')", execution_count=2, idx=1),
                ]
            )
        )
    )
    assert check_notebook(p) == []


@pytest.mark.smoke
def test_scanner_skips_fully_cleared(tmp_path: Path) -> None:
    """Fully-cleared notebooks (e.g. clean_notebook_outputs.py output) are intentional."""
    p = tmp_path / "cleared.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("model.fit(x, y)", execution_count=None, idx=0),
                    _code(
                        "compute_fit_quality(y, y_pred)", execution_count=None, idx=1
                    ),
                ]
            )
        )
    )
    assert check_notebook(p) == []


@pytest.mark.smoke
def test_scanner_flags_silent_skip_model_fit(tmp_path: Path) -> None:
    """Unexecuted ``.fit(`` cell alongside executed cells = silent-skip."""
    p = tmp_path / "skip.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("import numpy as np", execution_count=1, idx=0),
                    _code(
                        "model.fit(x, y, test_mode='creep')",
                        execution_count=None,
                        idx=1,
                    ),
                    _code("model.predict(x)", execution_count=2, idx=2),
                ]
            )
        )
    )
    violations = check_notebook(p)
    assert len(violations) == 1
    assert "cell[1]" in violations[0]
    assert "model.fit" in violations[0]


@pytest.mark.smoke
def test_scanner_flags_silent_skip_fit_bayesian(tmp_path: Path) -> None:
    """The fit_bayesian marker is also caught."""
    p = tmp_path / "skip_bayes.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("model = STZConventional()", execution_count=1, idx=0),
                    _code(
                        "result = model.fit_bayesian(t, y)", execution_count=None, idx=1
                    ),
                    _code("print(result.posterior_samples)", execution_count=2, idx=2),
                ]
            )
        )
    )
    violations = check_notebook(p)
    assert len(violations) == 1
    assert "fit_bayesian" in violations[0]


@pytest.mark.smoke
def test_scanner_flags_compute_fit_quality(tmp_path: Path) -> None:
    """compute_fit_quality is the second marker on the STZ tutorial cells."""
    p = tmp_path / "skip_quality.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("model.fit(x, y)", execution_count=1, idx=0),
                    _code(
                        "q = compute_fit_quality(y, y_pred)",
                        execution_count=None,
                        idx=1,
                    ),
                ]
            )
        )
    )
    violations = check_notebook(p)
    assert len(violations) == 1
    assert "compute_fit_quality" in violations[0]


@pytest.mark.smoke
def test_scanner_ignores_comments(tmp_path: Path) -> None:
    """A commented-out fit call must NOT be flagged."""
    p = tmp_path / "comment.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("import numpy as np", execution_count=1, idx=0),
                    _code(
                        "# model.fit(x, y) - example only", execution_count=None, idx=1
                    ),
                ]
            )
        )
    )
    assert check_notebook(p) == []


@pytest.mark.smoke
def test_scanner_cli_returns_nonzero_on_violation(tmp_path: Path) -> None:
    """Driver returns exit code 1 when a violation is present."""
    p = tmp_path / "skip.ipynb"
    p.write_text(
        json.dumps(
            _make_nb(
                [
                    _code("x = 1", execution_count=1, idx=0),
                    _code("model.fit(x, y)", execution_count=None, idx=1),
                ]
            )
        )
    )
    assert scanner_main([str(p)]) == 1


@pytest.mark.smoke
def test_examples_notebooks_have_no_new_silent_skips() -> None:
    """examples/*.ipynb outside the baseline must pass the scanner.

    Regression guard: STZ NB04 / NB05 silently shipped with unexecuted fit
    cells in 2026-05-18. Forty other notebooks in the repo also carry the
    same pattern as pre-existing tech debt; they are listed in
    ``scripts/notebook_execution_baseline.txt`` and excluded here. Any
    *new* notebook with the silent-skip pattern fails this test.
    """
    examples_dir = REPO_ROOT / "examples"
    if not examples_dir.is_dir():
        pytest.skip("examples/ directory not present")
    allowed = _load_allowlist(BASELINE_FILE)
    notebooks = [
        p
        for p in sorted(examples_dir.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in p.parts
        and not _allowlist_matches(p, allowed, REPO_ROOT)
    ]
    assert notebooks, "expected example notebooks under examples/"

    all_violations: list[str] = []
    for nb in notebooks:
        all_violations.extend(check_notebook(nb))

    assert not all_violations, (
        "New notebooks have unexecuted fit/quality cells while otherwise "
        "executed (silent-skip pattern). Either re-execute end-to-end "
        "before committing, or add the path to "
        "scripts/notebook_execution_baseline.txt if intentional:\n  "
        + "\n  ".join(all_violations)
    )


@pytest.mark.smoke
def test_baseline_entries_still_exist() -> None:
    """Stale baseline rot: every listed notebook must still exist on disk.

    If a notebook is renamed or deleted, its baseline entry becomes a
    silent allowlist for nothing — and a real future regression at a new
    path could slip through. Catch that immediately.
    """
    allowed = _load_allowlist(BASELINE_FILE)
    missing = [entry for entry in allowed if not (REPO_ROOT / entry).is_file()]
    assert not missing, (
        "Baseline references notebooks that no longer exist. "
        "Remove these lines from scripts/notebook_execution_baseline.txt:\n  "
        + "\n  ".join(missing)
    )
