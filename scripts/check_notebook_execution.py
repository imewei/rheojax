#!/usr/bin/env python3
"""Detect notebooks committed with unexecuted fit cells.

A common silent-failure pattern: an author iterates on a fit cell, hits an
error or convergence issue, then saves and commits the notebook with that
cell's ``execution_count`` cleared while downstream cells (which rely on
``model`` in stale kernel state) keep their old outputs. The committed
notebook *looks* fine on casual inspection — a reviewer only catches the
gap by counting execution_count==None against the cell content.

This script scans .ipynb files and reports cells where:
    - cell_type == "code"
    - execution_count is None
    - the source mentions one of the fit/quality calls below

Notebooks with **all** code cells unexecuted are assumed to be intentionally
cleared (e.g. by ``scripts/clean_notebook_outputs.py``) and are skipped.

Usage::

    python scripts/check_notebook_execution.py examples/stz/04_stz_creep.ipynb
    python scripts/check_notebook_execution.py examples/**/*.ipynb

Exit code 0: clean; exit code 1: at least one notebook has the silent-skip
signature.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Substrings that mark a cell as a "must execute" cell. Match by raw substring,
# not regex, so the list stays easy to extend.
FIT_CALL_MARKERS = (
    ".fit(",
    ".fit_bayesian(",
    "compute_fit_quality(",
    "fit_orchestrator",
    "nlsq_optimize(",
)


def _is_fit_cell(source: str) -> bool:
    for marker in FIT_CALL_MARKERS:
        if marker in source:
            return True
    return False


def _strip_comments_and_strings(source: str) -> str:
    """Remove # comments and triple-quoted strings so we don't false-match on docs."""
    # Remove triple-quoted strings (docstrings)
    source = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", "", source, flags=re.DOTALL)
    # Remove # comments to end of line
    source = re.sub(r"#[^\n]*", "", source)
    return source


def check_notebook(path: Path) -> list[str]:
    """Return list of violation messages for one notebook (empty = clean)."""
    try:
        with path.open(encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot parse ({exc})"]

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        return []

    # Notebooks intentionally cleared (every code cell unexecuted) are not
    # silent-skip cases. Only flag partial execution states.
    n_executed = sum(1 for c in code_cells if c.get("execution_count") is not None)
    if n_executed == 0:
        return []

    violations: list[str] = []
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        stripped = _strip_comments_and_strings(source)
        if _is_fit_cell(stripped):
            first_line = source.strip().split("\n", 1)[0][:100]
            violations.append(
                f"{path}:cell[{idx}] (id={cell.get('id', '?')!r}) is "
                f"unexecuted but contains a fit/quality call: {first_line!r}"
            )
    return violations


def _load_allowlist(path: Path) -> set[str]:
    """Return the set of repo-relative paths listed in *path* (POSIX style).

    Lines that are blank or start with '#' are ignored. Trailing whitespace
    is stripped. The file is optional — a missing path returns an empty set.
    """
    if not path.is_file():
        return set()
    allow: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            allow.add(entry)
    return allow


def _allowlist_matches(nb_path: Path, allowed: set[str], repo_root: Path) -> bool:
    """True if *nb_path* matches any entry in *allowed*."""
    if not allowed:
        return False
    try:
        rel = nb_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel = nb_path.as_posix()
    return rel in allowed or nb_path.as_posix() in allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag notebooks where a fit cell is unexecuted while "
        "the notebook otherwise has executed cells — the silent-skip pattern."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Notebook files (.ipynb) or directories to scan.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Text file with one repo-relative notebook path per line. "
        "Listed notebooks are tolerated (existing tech debt). "
        "'#' starts a comment.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repo root used to resolve allowlist paths "
        "(default: parent of scripts/).",
    )
    args = parser.parse_args(argv)

    allowed = _load_allowlist(args.allowlist) if args.allowlist else set()

    notebooks: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            notebooks.extend(sorted(p.rglob("*.ipynb")))
        elif p.suffix == ".ipynb":
            notebooks.append(p)
    # Filter out .ipynb_checkpoints and allow-listed paths.
    notebooks = [
        p
        for p in notebooks
        if ".ipynb_checkpoints" not in p.parts
        and not _allowlist_matches(p, allowed, args.repo_root)
    ]

    all_violations: list[str] = []
    for nb_path in notebooks:
        all_violations.extend(check_notebook(nb_path))

    if all_violations:
        print("Notebook silent-skip violations detected:", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nFix: re-execute the affected cells "
            "(e.g. `jupyter nbconvert --to notebook --execute --inplace <file>`) "
            "and commit the result.\n"
            "If the notebook is known pre-existing tech debt, add its path "
            "to scripts/notebook_execution_baseline.txt.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
