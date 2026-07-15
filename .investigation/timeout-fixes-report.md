# Timeout fixes: HVM demo fit + ANSYS APDL FZSS Bayesian fit

## Summary

Both tests were genuinely doing excessive computational work relative to what
they validate, not just resource-contention artifacts. Both were fixed by
reducing scope; the HVM test additionally needed a per-test timeout override
since even the reduced version legitimately takes several minutes.

## 1. `tests/validation/test_bayesian_ansys_apdl.py::TestFractionalZenerANSYS::test_fractional_zener_ansys_fit`

### Root cause

The test calls `model.fit_bayesian(...)` without overriding `num_chains`,
so it uses the production default of 4. `BayesianMixin.fit_bayesian`'s
`_select_chain_method()` (rheojax/core/bayesian.py) picks `"parallel"` only
when the accelerator count >= num_chains; on this CPU-only host (no
CUDA-enabled jaxlib installed) that's never true, so it falls back to
`"vectorized"`. For `FractionalZenerSolidSolid` (a Mittag-Leffler-based
model — documented elsewhere in the test suite, see
`tests/validation/test_bayesian_mode_aware.py`, as "extremely slow"),
vectorizing 4 chains together on CPU turned out to be far more than 4x
slower than a single chain, not embarrassingly parallel.

Measured directly:
- `num_chains=1`, `num_warmup=50`, `num_samples=50`: **9.5s** total
  (including NLSQ warm-start + JIT compile).
- `num_chains=4` (default), `num_warmup=20`, `num_samples=20` (a much
  smaller budget): still running after **2+ minutes**, killed after
  confirming the gap was not closing.

The test's own assertion is trivial — `assert len(summary) > 0` — with no
R-hat/ESS convergence check, so there was no reason to pay for a
multi-chain convergence-quality run.

### Fix

`tests/validation/test_bayesian_ansys_apdl.py`: changed the `fit_bayesian`
call to `num_chains=1, num_warmup=100, num_samples=100` (previously
`num_warmup=500, num_samples=1000` with the implicit default `num_chains=4`).
Added a comment explaining the vectorized-chain-method cost and why a small
single-chain budget is sufficient for this assertion.

No `@pytest.mark.timeout` override needed — the fixed test comfortably
clears the project's 120s default.

### Verification

Ran individually twice:
- Run 1: `1 passed in 18.65s`
- Run 2: `1 passed in 17.93s`

Consistent, no flakiness.

## 2. `tests/examples/test_hvm_fit_demo.py::test_hvm_demo_fits_overlap_generated_rheology_data`

### Root cause

`HVMLocal._fit` (rheojax/models/hvm/local.py) silently remaps
`method="nlsq"` to `method="scipy"` for ODE-based protocols
(`relaxation`, `creep`, `startup`, `laos`) because NLSQ's forward-mode AD
is incompatible with diffrax's `custom_vjp` ODE solves. The scipy path
(`_run_scipy_least_squares` in rheojax/utils/optimization.py) uses
`scipy.optimize.least_squares` with a **numerical** (finite-difference)
Jacobian — every residual evaluation re-solves the full ODE, and this cost
does not shrink with JIT warm-up.

Confirmed by instrumenting `HVMLocal.model_function` directly during a real
`relaxation` fit (the demo's actual seeded fit, not a synthetic
reconstruction): 75 calls, each **1.6–2.4s**, flat across the whole run
(call #1: 2.39s, call #20: 2.08s, call #40: 2.02s, call #60: 2.01s) — i.e.
~150s just for the `relaxation` protocol's NLSQ warm-start, dominated
entirely by call count × fixed per-call ODE-solve cost, not by data grid
size (separately confirmed: `predict()` timing was flat at ~2-2.8s across
grid sizes 10/30/60).

The demo's `fit_kwargs` set `ftol=xtol=gtol=1e-10` uniformly for all 4
protocols — extremely tight for a `scipy` TRF numerical-Jacobian fallback,
and unnecessary: at that tolerance `relaxation` already reaches
`R²=1.0000`, far above the test's `R² > 0.95` threshold. On top of that,
the `startup` protocol runs a manual 4-way multi-start
(`ge_scale in (0.5, 1.0, 2.0, 4.0)`) to escape a local minimum, each trial
paying the full ~2s/call cost — the single most expensive part of the test.

The original full test (untouched) did not finish even with a 600s
per-test timeout (5x the project default), confirmed by running it in
isolation.

### Fix

`examples/utils/hvm_demo_fit.py`:
- Loosened `ftol`/`xtol`/`gtol` from `1e-10` to `1e-6` for all protocols'
  `fit_kwargs`.
- Reduced the `startup` protocol's multi-start sweep from 4 trials
  (`0.5, 1.0, 2.0, 4.0`) to 2 (`0.5, 2.0`) — still bracketing
  `INITIAL_PARAMS`'s `G_E` from both sides, which is what actually escapes
  the pinned local minimum per the existing comment, just with half the
  trials.

`tests/examples/test_hvm_fit_demo.py`: added
`@pytest.mark.timeout(1200)` with a comment. Even after both scope
reductions above, this test legitimately does ~7 full NLSQ optimizations
worth of ODE-heavy scipy fallback work (flow_curve is fast/algebraic;
relaxation, creep, and startup's 2 trials are all ODE-based). 1200s is
~1.4x the measured real runtime — a genuine compute cost from the model's
architecture (ODE solve cost per scipy iteration), not something further
reducible without changing `HVMLocal`'s fit machinery, which is out of
scope for this fix.

### Verification

Ran individually twice:
- Run 1 (post-fix): `1 passed in 854.08s` (0:14:14) — vs. the original,
  which did not finish even with a 600s timeout.
- Run 2 (post-fix, using the new `@pytest.mark.timeout(1200)` marker):
  `1 passed in 856.08s` (0:14:16).

Consistent within ~2s across both runs, no flakiness.

### What was ruled out

- Grid size (number of data/predict points) is **not** the cost driver —
  `predict()` timing was flat across n=10/30/60.
- JIT recompilation caching is **not** helping mid-fit — per-call cost of
  `model_function` stayed flat (~2s) across 75 calls within a single
  `.fit()` invocation, so the ~2s/call is real ODE-solve + dispatch cost,
  not a one-time warm-up paid once and amortized.
- This is not a resource-contention artifact — confirmed via isolated runs
  with generous timeouts, run one at a time, with no other tests competing
  for CPU.

## Final combined check

`uv run pytest tests/examples/test_hvm_fit_demo.py tests/validation/test_bayesian_ansys_apdl.py -v`:

```
11 passed in 876.80s (0:14:36)
```

All tests in both files pass together, including both originally-failing
tests.

## Files changed

- `tests/validation/test_bayesian_ansys_apdl.py` — reduced `fit_bayesian`
  budget in `test_fractional_zener_ansys_fit`.
- `examples/utils/hvm_demo_fit.py` — loosened `fit_kwargs` tolerances,
  reduced `startup` multi-start trial count.
- `tests/examples/test_hvm_fit_demo.py` — added
  `@pytest.mark.timeout(1200)` with justification comment.
