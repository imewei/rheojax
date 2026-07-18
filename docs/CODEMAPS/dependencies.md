<!-- Generated: 2026-07-18 | Files scanned: pyproject.toml (not re-parsed; from CLAUDE.md/known stack) | Token estimate: ~450 -->

# Dependencies

## Numerical Core

- **JAX / jaxlib** — computational core, x64 forced at import. GPU via `jax[cuda12/cuda13]`
  extras (never both installed together).
- **NLSQ** — GPU-accelerated non-linear least squares.
- **optimistix** — root-finding.
- **optax** — optimizer schedules.
- **diffrax** — ODE solving (used instead of `jax.experimental.ode`).
- **interpax** — JIT-safe interpolation (never `scipy.interpolate` in JIT paths).
- **lineax** — linear algebra (recently upgraded, see git log `06f42c25`).

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
- **pytest** (+ **pytest-xdist**, **pytest-qt**, **pytest-timeout**, **pytest-benchmark**) — test
  runner; `--dist=loadgroup` preserves NUTS-test isolation.
- **ruff** — lint (line-length 88, E/W/F/I/C/B/UP/S rule sets).
- **mypy** — type check (`rheojax.gui.*`/`rheojax.cli.*` ignored for PySide6 stub gaps).

## Cross-Cutting Constraint

Never install `gpu_cuda12` and `gpu_cuda13` extras simultaneously (`pyproject.toml` optional-deps
groups are mutually exclusive at the plugin level).

## External Services

None — this is a local-only library/CLI/desktop app. No network calls except optional GPU
detection and GitHub-hosted docs (`Help > Documentation`/`Tutorials` in the GUI open
`readthedocs.io` via `webbrowser.open()`).
