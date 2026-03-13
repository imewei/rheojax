# Optimization Ledger — RheoJAX

## Baseline Measurements (2026-03-12)

### Model Fit Timings (CPU, Apple Silicon M-series)

| Model | Protocol | Cold (ms) | Warm (ms) | JIT Ratio | Category |
|-------|----------|-----------|-----------|-----------|----------|
| Maxwell | relaxation | 764 | 118 | 6.5x | Analytical |
| Maxwell | oscillation | 890 | 159 | 5.6x | Analytical |
| Zener | relaxation | 1132 | 182 | 6.2x | Analytical |
| PowerLaw | flow_curve | 0.2 | 0.2 | 1x | Analytical (numpy) |
| Giesekus | oscillation | 1091 | 226 | 4.8x | Analytical kernel |
| FracMaxwell | oscillation | 1552 | 301 | 5.2x | Mittag-Leffler |
| Giesekus | flow_curve | 1584 | 465 | 3.4x | ODE (scipy fallback) |
| Giesekus | relaxation | **60734** | **6827** | 8.9x | ODE (diffrax) |

### Micro-Benchmarks

| Metric | Value | Assessment |
|--------|-------|------------|
| JIT first predict (cold) | 142ms | Expected |
| JIT same instance (warm) | 0.1ms | Excellent |
| JIT new instance (warm cache) | 0.4ms | Excellent, cache shared |
| H2D np→jnp (N=100K) | 2.3ms | Not a bottleneck |
| D2H jnp→np (N=100K) | 0.1ms | Not a bottleneck |
| Model instantiation (Maxwell) | 5.4ms | Acceptable |
| Model instantiation (SGR) | 7.2ms | Acceptable |
| safe_import_jax + model import | 620ms | One-time, acceptable |
| Mittag-Leffler (N=500) | 6.6ms | Not a bottleneck |

### Pipeline Phase Breakdown (Maxwell, warm)

| Phase | Time (ms) | % |
|-------|-----------|---|
| NLSQ fit | 118 | 87% |
| predict() | 28 | 13% |
| score() | 0.2 | <1% |
| Instantiation | 0.1 | <1% |

---

## Bottleneck Registry

| ID | File:Line | Category | Severity | Owner | Status | Speedup |
|----|-----------|----------|----------|-------|--------|---------|
| B001 | models/giesekus/single_mode.py:726 | ODE | CRITICAL | jax | MITIGATED | ~0.5s saved (skip failed NLSQ) |
| B002 | utils/optimization.py:1382 | JIT | HIGH | jax | RESOLVED | Compilation cache enabled (GPU) |
| B003 | utils/optimization.py:2444 | CPU | MEDIUM | python | WONTFIX | <0.1ms/iter, negligible |
| B004 | core/bayesian.py:1671 | JIT | HIGH | systems | RESOLVED | jax.eval_shape eliminates 1-5s JIT probe |
| B005 | core/bayesian.py:1765 | MEMORY | HIGH | jax | RESOLVED | jax.checkpoint on 10 ODE RHS (6 files) |
| B006 | utils/optimization.py:211 | CPU | MEDIUM | python | RESOLVED | Dead branch removed |
| B007 | models/*/_kernels.py | VECTORIZATION | MEDIUM | jax | WONTFIX | Already vectorized (GMM, Maxwell) |
| B008 | core/base.py:489 | CPU | LOW | python | PREVIOUSLY_FIXED | R6-OPT-001 reuses NLSQ residual |
| B009 | models/*/single_mode.py | ODE | HIGH | jax | RESOLVED | Skip NLSQ→scipy for ODE protocols |
| B010 | utils/optimization.py:85 | ODE | HIGH | jax | RESOLVED (GPU) | FD-JVP enables NLSQ for ODE (GPU only) |

---

## Bottleneck Details

### B001 — ODE-based model fitting via diffrax is 50-350x slower than analytical

**Severity: CRITICAL** — Giesekus relaxation: 60.7s cold, 6.8s warm (vs Maxwell 118ms warm)

The diffrax ODE solver path (Tsit5 + PIDController + checkpointed_while_loop) is extremely expensive:
- Each NLSQ iteration requires a full ODE solve
- Forward-mode AD (jvp) incompatible with diffrax custom_vjp → falls back to scipy (no JAX AD)
- `_simulate_relaxation_internal` at line 726 is the hotspot
- Same pattern affects: HVM, HVNM, IKH, FIKH, Fluidity, VLB, STZ, SGR

**Optimization opportunities:**
1. Use `adjoint=diffrax.RecursiveCheckpointAdjoint()` for memory-efficient reverse-mode AD
2. Tune PIDController tolerances (rtol/atol from 1e-8 to 1e-6 for fitting)
3. Pre-warm diffrax JIT compilation with dummy solve
4. Consider analytical approximations for NLSQ initial guess, then ODE for refinement

### B002 — Cold JIT compilation overhead (764-1552ms per model)

**Severity: HIGH** — First fit on each model triggers XLA compilation

cProfile shows `backend_compile_and_load` = 57-111ms per compiled function, with 3-5 functions per fit. Complex oscillation triggers additional compilations for the G'/G'' split.

**Optimization opportunities:**
1. `model.precompile()` API already exists for ITT-MCT — extend to all models
2. Persistent JIT cache (XLA compilation cache on disk)
3. AOT compilation for common model signatures

### B003 — params array conversion on every residual evaluation

**Severity: MEDIUM** — `jnp.asarray(params, dtype=jnp.float64)` at line 2444

Called on every NLSQ iteration (17-45 iterations per fit). Low cost individually (~0.1ms) but adds up for ODE models with many iterations.

### B004 — Bayesian model_function probe triggers extra JIT compilation

**Severity: HIGH** — RESOLVED

Replaced `self.model_function(dummy_X, dummy_params, test_mode)` probe with `jax.eval_shape(functools.partial(self.model_function, test_mode=...), X_shape, params_shape)`. This traces the function's shape logic without XLA compilation, eliminating 1-5s overhead on first `fit_bayesian()` call. Also removes the fragile `_test_mode` save/restore dance (R12-B-018).

### B005 — NUTS VJP through ODE solvers consumes excessive memory

**Severity: HIGH** — RESOLVED

Added `jax.checkpoint` wrapping to ODE RHS functions in 6 model files (10 ODETerm locations), matching the existing pattern in HVM/HVNM `_kernels_diffrax.py`. This trades ~2x ODE RHS compute for O(1) memory per RHS evaluation during reverse-mode AD (VJP), reducing peak memory from O(solver_steps) to O(sqrt(solver_steps)).

**Files modified:**
- `models/itt_mct/_kernels_diffrax.py` — 2 vector field factories (flow curve, flow curve with stress)
- `models/sgr/sgr_conventional.py` — 1 inline vector field
- `models/stz/conventional.py` — 2 ODETerm locations (startup/relaxation + LAOS)
- `models/vlb/local.py` — 2 ODETerm locations (LAOS internal × 2)
- `models/vlb/nonlocal_model.py` — 2 ODETerm locations (PDE RHS + creep RHS)
- `models/vlb/multi_network.py` — 2 ODETerm locations (creep + LAOS)

### B006 — Double np.asarray conversion in optimization result

**Severity: MEDIUM** — Lines 214-215 convert residuals twice

```python
res = np.asarray(res)  # line 214 (inside conditional)
res = np.asarray(res)  # line 215 (unconditional)
```

### B007 — Multi-mode models use Python loops instead of vectorized operations

**Severity: MEDIUM** — Several `_kernels.py` files have `for i in range(n_modes)` loops

These should be vectorized with `jnp.sum(G_modes * jnp.exp(-t[:, None] / tau_modes))` but some models have complex mode interactions that make vectorization non-trivial.

### B008 — fit() calls score() which calls predict() redundantly

**Severity: LOW** — PREVIOUSLY FIXED by R6-OPT-001

`BaseModel.fit()` at line 489 now extracts R² from `self._nlsq_result.fun` (the residual vector from NLSQ) instead of calling `self.score()` which would trigger a redundant `predict()` call. Falls back to `score()` only when `_nlsq_result` is unavailable AND DEBUG logging is active.

### B009 — ODE models waste time attempting NLSQ before falling back to scipy

**Severity: HIGH** — Saves ~0.5s per ODE fit by skipping failed NLSQ attempt

diffrax ODE solvers use `custom_vjp` which is incompatible with NLSQ's forward-mode AD (jvp). ODE protocols (relaxation, startup, creep, LAOS) would attempt NLSQ, fail with TypeError, then fall back to scipy. The fix adds protocol-aware method routing in each ODE model's `_fit()`:

```python
_ode_protocols = {"relaxation", "startup", "creep", "laos"}
_method = kwargs.get("method", "auto")
if test_mode in _ode_protocols and _method in ("auto", "nlsq", "trf", "lm"):
    _method = "scipy"
```

Applied to 12 model files: Giesekus (single/multi), HVM, HVNM, TNT (5 files), VLB (2 files), Fluidity, Saramito.

Note: `BaseModel.fit()` passes `method="nlsq"` by default, so `kwargs.setdefault("method", "scipy")` is INEFFECTIVE — must check and override the actual value.

### B010 — Finite-difference JVP wrapper for GPU-accelerated ODE fitting

**Severity: HIGH** — GPU: expected 3-5x speedup over scipy; CPU: no benefit (vmap overhead)

`make_fd_differentiable(fn)` wraps a function with `jax.custom_jvp` that computes JVPs via central differences. When NLSQ uses `jax.jacfwd`, this becomes vmap'd perturbations — a single batched XLA call computing all N_params+1 ODE solves in parallel on GPU.

**CPU benchmark (4-param Giesekus relaxation, 30 points):**
- NLSQ + FD-JVP: ~20.6s warm (vmap sequentializes on CPU)
- Scipy sequential FD: ~6.8s warm (faster on CPU)

**Applied to:** Giesekus single/multi mode with backend detection (`jax.default_backend()`). Other ODE models retain the `method='scipy'` CPU fallback from B009 and can be upgraded when GPU testing is available.

---

## Categories
- MEMORY: Peak RAM, allocation patterns, GC pressure
- CPU: Python-level algorithm efficiency
- GPU: JAX device utilization, XLA graph optimization
- IO: File reading/writing, serialization
- ALGORITHMIC: Mathematical algorithm choice
- JIT: XLA compilation overhead, cache behavior
- VECTORIZATION: Loop → vmap/scan conversion
- ODE: diffrax solver choice, step-size, checkpointing, VJP memory
