# Coding Style Guide

> 此文件定义团队编码规范，所有 LLM 工具在修改代码时必须遵守。
> 提交到 Git，团队共享。

## General
- Prefer small, reviewable changes; avoid unrelated refactors.
- Keep functions short (<50 lines); avoid deep nesting (≤3 levels).
- Name things explicitly; no single-letter variables except loop counters.
- Handle errors explicitly; never swallow errors silently.

## Language-Specific

### Python
- Python 3.12+, `uv` for deps (`uv sync`, `uv run`).
- `ruff check .` (line-length 88, rules E/W/F/I/C/B/UP/S) + `mypy .` (gui/cli modules ignored — PySide6 stub gaps).
- JAX-first: minimize Host↔Device transfers; import JAX only via `safe_import_jax()` (`rheojax.core.jax_config`), never `import jax` directly.
- Float64 forced project-wide; no non-JIT-safe interpolation (`interpax`, not `scipy.interpolate`); ODEs via `diffrax`, not `jax.experimental.ode`.
- Bayesian: NumPyro preferred, NLSQ warm-start → NUTS, ArviZ diagnostics mandatory (R-hat, ESS, BFMI).

### Julia
- Julia 1.12, 5-env split (`@v1.12`, `@sciml`, `@bayes`, `@pinn`, `@gnn`) — see root CLAUDE.md for which env owns what.
- Type-stable, allocation-free hot paths; leverage multiple dispatch at API boundaries.
- Never `Pkg.add`/`Pkg.free` dev-overridden packages (`Pigeons`, `GNNLux`/`GNNGraphs`/`GNNlib`) without confirming.

## Git Commits
- Conventional Commits, imperative mood.
- Atomic commits: one logical change per commit.

## Testing
- Every feat/fix MUST include corresponding tests.
- Coverage must not decrease.
- Fix flow: write failing test FIRST, then fix code.
- `pytest` markers: smoke/unit/integration/validation/benchmark; slow/gpu/macos_only/crash_test.

## Security
- Never log secrets (tokens/keys/cookies/JWT).
- Validate inputs at trust boundaries (I/O boundaries: shape, dtype, NaN, monotonicity).
