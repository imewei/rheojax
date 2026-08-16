<!-- Generated: 2026-07-18 | Refreshed 2026-08-16 against pyproject.toml directly (optimistix,
     optax, lineax, interpax removed from the actual dependency list since this file was last
     generated) | Files scanned: pyproject.toml | Token estimate: ~450 -->

# Dependencies

## Numerical Core

- **JAX / jaxlib** (`>=0.8.3`) — computational core, x64 forced at import. GPU via
  `jax[cuda12/cuda13]` extras (never both installed together).
- **NLSQ** (`>=0.6.10`) — GPU-accelerated non-linear least squares; must be imported before JAX.
- **diffrax** (`>=0.7.1`) — ODE solving (used instead of `jax.experimental.ode`).
- **numpy** / **scipy** — array core and non-JIT-path numerics.

Not actual dependencies (despite historically being listed here / in root CLAUDE.md's stated
stack): **optimistix**, **optax**, **lineax** are not in `pyproject.toml`. **interpax** was
removed and replaced by `rheojax.utils.jax_cubic_spline` — a pure-JAX cubic spline with no JAX
version ceiling (see the `jax` dependency comment in `pyproject.toml` and
`tests/utils/test_jax_cubic_spline.py` for the scipy-validated parity check).

## Bayesian

- **NumPyro** — preferred Bayesian engine (NUTS).
- **ArviZ** (1.x: `arviz-base`/`arviz-stats`/`arviz-plots` split) — diagnostics (R-hat, ESS,
  BFMI); `core/arviz_utils.py` shims kwarg differences.

## GUI

- **PySide6** — Qt bindings, desktop shell.
- **PyQtGraph** — interactive plotting canvas.
- **matplotlib** — diagnostic canvases (ArviZ, residuals panel).

## I/O

- **pandas** — CSV/Excel read/write.
- **h5py** (or equivalent) — HDF5 read/write.

## Dev/Test

- **uv** — package/environment manager, `uv.lock` is the lockfile source of truth.
- **pytest** (+ **pytest-xdist**, **pytest-qt**, **pytest-timeout**, **pytest-cov**,
  **pytest-image-diff**, **hypothesis**) — test runner; `--dist=loadgroup` preserves NUTS-test
  isolation.
- **ruff** — lint (line-length 88, E/W/F/I/C/B/UP/S rule sets).
- **mypy** — type check (`rheojax.gui.*`/`rheojax.cli.*` ignored for PySide6 stub gaps).

## Cross-Cutting Constraint

Never install `gpu_cuda12` and `gpu_cuda13` extras simultaneously (`pyproject.toml` optional-deps
groups are mutually exclusive at the plugin level).

## External Services

None — this is a local-only library/CLI/desktop app. No network calls except optional GPU
detection and GitHub-hosted docs (`Help > Documentation`/`Tutorials` in the GUI open
`readthedocs.io` via `webbrowser.open()`).
