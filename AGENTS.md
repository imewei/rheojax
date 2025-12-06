# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `rheojax/` (pipelines, rheology models, transforms, I/O utilities). Tests are under `tests/`, mirroring package structure with unit, integration, and GPU-specific suites defined via pytest markers. Documentation sources reside in `docs/` (Sphinx) and user-facing notebooks are grouped inside `examples/` (`basic`, `transforms`, `bayesian`, `advanced`). Build metadata and shared configuration sit at the repo root (`pyproject.toml`, `setup.py`, `Makefile`, `requirements*.txt`) with `pyproject.toml` declaring `requires-python = ">=3.12"`; keep feature work aligned with that minimum version. Keep large assets or generated plots out of version control unless they belong in `examples/`.

## Build, Test & Development Commands
- `make install-dev` — editable install plus development extras (ruff, black, mypy, pytest, docs).
- `make test` / `make test-fast` — full suite vs. marker-filtered run that skips `slow` tests.
- `make test-coverage` — executes pytest with HTML + XML coverage reports (`htmlcov/index.html`).
- `make quick` — formats code then executes the fast test subset for tight loops.
- `make docs` — builds Sphinx docs into `docs/_build/html` (open locally to spot visual issues).
- `make install-jax-gpu` — Linux-only helper that reinstalls JAX with CUDA 12 support; runs `make gpu-check` afterward.

## Coding Style & Naming Conventions
Python 3.12+, 4-space indentation, and strict type hints are expected. Favor a “JAX-first” approach—prefer `jax`, `jax.numpy as jnp`, and other accelerator-safe utilities before falling back to NumPy/SciPy. Always use explicit imports grouped by stdlib/third-party/local; avoid wildcard imports so call sites stay traceable. Run `black` and `ruff` (via `make format`) before submitting; CI also runs `mypy` (`make type-check`). Follow snake_case for modules/functions, PascalCase for classes, keep notebook filenames descriptive (`examples/bayesian/fractional_maxwell.ipynb`), and ensure public APIs include docstrings detailing parameter units and tensor shapes.

## Testing Guidelines
Pytest is configured in `pytest.ini` with enforced naming (`tests/test_*.py`, classes starting with `Test`). Use the provided markers (`slow`, `integration`, `gpu`, `benchmark`, etc.) so collaborators can select the right subsets, and skip GPU-reliant cases on macOS by guarding with the `gpu` marker. Add regression tests near the affected module and update snapshot data under `tests/data/` if formats change. Validate coverage with `make test-coverage` and ensure new features maintain or improve existing percentages for touched files.

## Commit & Pull Request Guidelines
History follows Conventional Commits (`fix(models):`, `chore(git):`, `docs(examples):`). Write imperative, present-tense summaries (<75 chars) and include a scope when practical. Each PR should describe the problem, outline the solution, and list validation commands (e.g., `make test`, `make docs`). Link related GitHub issues and attach screenshots or rendered plots when you change notebooks or docs outputs. Avoid bundling unrelated features; keep GPU-focused changes separate from doc-only updates.

## Security & Configuration Tips
Store credentials (e.g., private TRIOS datasets) outside the repo and load paths via environment variables or `.env` files ignored by git. GPU tooling only targets Linux with CUDA 12.1–12.9; on macOS, leave `make install-jax-gpu` untouched to avoid dependency churn. When adding new config files, update `.gitignore` and document required environment variables in `docs/configuration.md`.

## GUI State Store Notes
- `StateStore.dispatch` accepts either an action dict with a `type` key or a string action plus an optional payload dict (e.g., `dispatch("SET_THEME", {"theme": "dark"})`). Prefer these two shapes to keep signals consistent and avoid TypeError crashes.
