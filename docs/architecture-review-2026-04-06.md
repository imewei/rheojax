# RheoJAX Architecture Review — 2026-04-06

## Executive Summary

RheoJAX is a well-structured JAX-accelerated rheological analysis library with 53 models across 19 model directories, 11 transforms, a fluent Pipeline API, Bayesian inference via NumPyro, and a PyQt GUI. The codebase totals ~185k lines of Python. The architecture is fundamentally sound — clean layering, good use of registry/mixin patterns, and consistent scikit-learn-style APIs. However, there are five architectural concerns that warrant attention, documented as ADRs below.

---

## Dependency Analysis

Package dependency flow (non-GUI):

```
cli -> [core, io, logging, models, pipeline, transforms]
pipeline -> [core, io, logging, models, parallel, transforms, utils, visualization]
models -> [core, logging, transforms, utils]
transforms -> [core, logging, utils]
visualization -> [core, logging, utils]
io -> [core, logging]
utils -> [core, io, logging]
core -> [io, logging, utils, models*]
```

*`core -> models` is a lazy import in `registry.py` for auto-discovery only (lines 790, 850). No circular import at module load time.

**Verdict:** Layering is clean. `pipeline` is the integration layer as expected. The `core -> models` lazy import for registry discovery is acceptable but worth documenting.

---

## ADR-001: BaseModel God Class (1,610 lines, 44cc fit())

### Context

`rheojax/core/base.py:34` — `BaseModel` is 1,262 lines (before `BaseTransform`) and serves as:
1. Abstract model interface (`_fit`, `_predict`)
2. NLSQ optimization orchestrator (`fit()` at 368 lines, 44cc)
3. Bayesian inference bridge (`fit_bayesian()` overrides `BayesianMixin`)
4. DMTA/deformation mode converter
5. Serialization (`to_dict`, `from_dict`)
6. Scoring (`score`, `R2`)
7. Parameter management (`get_params`, `set_params`)
8. Compatibility checking
9. Auto-initialization
10. Uncertainty quantification
11. Physics validation
12. Precompilation

`fit()` alone (lines 263-630) handles: RheoData unpacking, deformation mode conversion, optimization strategy auto-detection, auto-initialization, compatibility checking, method remapping, NLSQ delegation, R2 computation, physics checks, uncertainty quantification, and FitResult construction.

### Decision

**Status:** PROPOSED

Split `BaseModel.fit()` into a composition of strategies:

1. **Extract `FitOrchestrator`** — move the 368-line `fit()` body into a separate class that composes: data preparation, strategy detection, optimization dispatch, post-fit validation. BaseModel delegates to it.
2. **Extract `DeformationModeConverter`** — the E*/G* conversion logic (repeated in both `fit()` and `fit_bayesian()`) becomes a standalone utility called at the boundary.
3. **Extract `PostFitValidator`** — physics checks + uncertainty quantification become a composable post-processor.

### Consequences

- Reduces BaseModel to ~400 lines (interface + delegation)
- Each concern becomes independently testable
- `fit()` cyclomatic complexity drops from 44 to ~10
- Risk: over-abstraction if taken too far; keep it to 3-4 extractions max

---

## ADR-002: _fit() Boilerplate Duplication Across 41 Model Files

### Context

Every model's `_fit()` method repeats a near-identical pattern:
1. Parse RheoData vs raw arrays
2. Detect/resolve test_mode
3. Cache `self._test_mode` for Bayesian
4. Define a local `model_fn(x, params)` closure
5. Call `create_least_squares_objective(model_fn, x, y)`
6. Call `nlsq_optimize(objective, self.parameters)`
7. Check `result.success`, raise on failure
8. Set `self.fitted_ = True`

This pattern appears in **41 model files** with 294 combined occurrences of `create_least_squares_objective`/`nlsq_optimize`. The boilerplate ranges from 50-120 lines per model.

Some models (e.g., flow models like `PowerLaw`, `Carreau`) also duplicate heuristic initialization logic.

### Decision

**Status:** PROPOSED

Introduce a `_standard_nlsq_fit()` template method in `BaseModel`:

```python
def _standard_nlsq_fit(self, X, y, model_fn, *, test_mode=None, **kwargs):
    """Standard NLSQ fitting pipeline for models with a stateless model_fn."""
    # Handles: RheoData unpacking, test_mode resolution, caching,
    # objective creation, optimization, result validation, fitted_ flag
```

Models that follow the standard pattern reduce their `_fit()` to:

```python
def _fit(self, X, y, **kwargs):
    def model_fn(x, params):
        G0, eta = params[0], params[1]
        return self._predict_relaxation(x, G0, eta)
    return self._standard_nlsq_fit(X, y, model_fn, **kwargs)
```

Models with non-standard fitting (ODE-based, multi-protocol) continue to override `_fit()` directly.

### Consequences

- Eliminates ~2,000 lines of duplicated boilerplate across 30+ simple models
- Single point of maintenance for RheoData handling, test_mode caching, error handling
- Models with custom fitting (SGR, ITT-MCT, HL, STZ, EPM) remain unaffected
- Risk: some models have subtle per-model tweaks in the boilerplate; need careful migration

---

## ADR-003: BayesianMixin at 2,163 Lines — Mixed Abstraction Levels

### Context

`rheojax/core/bayesian.py` is 2,163 lines containing:
- `BayesianResult` dataclass with ArviZ conversion (lines 80-400, including fast-path xarray assembly)
- `BayesianMixin` with NUTS orchestration (lines ~400-1200)
- `_build_numpyro_model()` at 300 lines, 49cc — the most complex function in core
- Prior distribution builders
- Diagnostics computation
- Posterior summary statistics

`_build_numpyro_model()` (line 1691, 49cc, 300 lines) handles: parameter iteration, prior selection (uniform/normal/lognormal/halfnormal), transform application, model function invocation, NaN guarding, likelihood construction, and finite-check factors.

### Decision

**Status:** PROPOSED

1. **Extract `BayesianResult` to `rheojax/core/bayesian_result.py`** — it's a self-contained dataclass with ArviZ integration, no dependency on BayesianMixin internals.
2. **Extract `_build_numpyro_model` to `rheojax/core/numpyro_model_builder.py`** — a pure function that takes parameters + model_fn and returns a NumPyro model callable. This is the highest-complexity function in core and deserves isolation.
3. **Extract diagnostics to `rheojax/core/bayesian_diagnostics.py`** — R-hat, ESS, divergence counting.

### Consequences

- BayesianMixin reduces to ~500 lines of orchestration
- `_build_numpyro_model` becomes independently testable with mock parameters
- ArviZ fast-path logic in BayesianResult is already self-contained
- No API changes — all public methods stay on BayesianMixin/BayesianResult

---

## ADR-004: predict() Defensive kwargs Stripping (Lines 984-1021)

### Context

`BaseModel.predict()` (line 928) contains a defensive try/except/retry pattern for `_predict()` calls:

```python
try:
    result = self._predict(X, **kwargs)
except TypeError as e:
    if "unexpected keyword argument" in str(e):
        # Strip injected kwargs and retry
        # Then retry with no kwargs at all
```

This 3-level fallback (full kwargs -> stripped kwargs -> bare call) exists because:
1. `predict()` injects `test_mode` into kwargs
2. Some models' `_predict()` signatures don't accept `**kwargs`
3. 13+ models have `_predict(self, X)` only

This is a Liskov Substitution Principle violation: the base class contract (`_predict(X, **kwargs)`) is not honored by subclasses. The string-parsing error recovery is fragile.

### Decision

**Status:** PROPOSED

1. **Standardize `_predict()` signature** to always accept `**kwargs` (even if ignored). This is a one-line change per model (`def _predict(self, X, **kwargs):`).
2. **Remove the try/except/retry** in `predict()` — it becomes unnecessary once all subclasses accept kwargs.
3. **Add a protocol check** in the registry decorator or a test that validates all registered models accept `**kwargs` in `_predict`.

### Consequences

- Eliminates 38 lines of fragile error-recovery code in the hot path
- Enforces LSP across all 53 models
- Migration: grep for `def _predict(self, X)` (no kwargs) and add `**kwargs`
- Low risk — adding `**kwargs` to a function signature is backward-compatible

---

## ADR-005: GUI Complexity Hotspots

### Context

The GUI layer contains the most complex code in the entire codebase:

| File | Lines | Max CC | Function |
|------|-------|--------|----------|
| `gui/state/store.py` | 2,297 | 127cc | `_reduce_action` (836 lines) |
| `gui/pages/export_page.py` | — | 74cc | `_perform_export` (371 lines) |
| `gui/pages/bayesian_page.py` | 2,110 | 43cc | `_on_run_clicked` (329 lines) |
| `gui/app/main_window.py` | 3,420 | — | God module |
| `gui/services/model_service.py` | 1,323 | 42cc | `fit` (378 lines) |

`_reduce_action` at 127cc and 836 lines is the single most complex function in the codebase. It's a monolithic Redux-style reducer handling every possible UI action in one giant if/elif chain.

### Decision

**Status:** PROPOSED

1. **Split `_reduce_action` into action-specific reducers** — one function per action group (data_actions, model_actions, bayesian_actions, export_actions, ui_actions). Register them in a dispatch table.
2. **Extract `_perform_export` and `_on_run_clicked`** into dedicated command/handler classes.
3. **Consider the Command pattern** for GUI actions — each action becomes a class with `execute()` and `undo()`.

### Consequences

- `_reduce_action` drops from 127cc/836L to ~10cc/50L dispatch + individual reducers at 10-20cc each
- Each reducer becomes independently testable
- Undo/redo becomes cleaner with Command pattern
- Risk: GUI refactoring is high-risk if not well-tested; ensure smoke tests cover all action types first

---

## Additional Observations

### Strengths

1. **Registry pattern** is well-implemented — thread-safe singleton, decorator-based registration, protocol/deformation-mode metadata, lazy discovery.
2. **BayesianMixin composition** is elegant — every model gets Bayesian inference for free via MRO.
3. **Float64 discipline** via `safe_import_jax()` is consistently enforced across all 105 model files.
4. **Pipeline fluent API** is clean and well-separated from core logic.
5. **Structured logging** with `get_logger`/`log_fit`/`log_bayesian` context managers throughout.
6. **Only 12 TODO/FIXME/HACK comments** in 185k lines — low tech debt signal.

### Known Duplication (from TODOs)

- `FL-010`: `_predict_saos_jit` duplicated between `FluidityLocal` and `FluidityNonlocal` — should be extracted to a shared `_kernels.py` or base class.

### Interface Segregation

`BaseModel` currently forces all 53 models to inherit both NLSQ and Bayesian capabilities. Models that only support a single protocol (e.g., flow models supporting only ROTATION) still carry the full multi-protocol, multi-deformation-mode machinery. This is acceptable given the mixin approach, but worth monitoring as the model count grows.

### Test Coverage

With ~3,652 tests and tiered execution (smoke/non-slow/full), the test infrastructure is mature. The identified complexity hotspots (store reducer, export handler) should be prioritized for additional unit test coverage before any refactoring.

---

## Priority Ranking

| Priority | ADR | Impact | Effort | Risk |
|----------|-----|--------|--------|------|
| 1 | ADR-002 | High (41 files) | Medium | Low |
| 2 | ADR-004 | Medium (LSP fix) | Low | Low |
| 3 | ADR-001 | High (maintainability) | Medium | Medium |
| 4 | ADR-003 | Medium (testability) | Medium | Low |
| 5 | ADR-005 | High (GUI quality) | High | High |

**Recommended first action:** ADR-002 + ADR-004 together — they're low-risk, high-impact, and can be done incrementally (model by model).
