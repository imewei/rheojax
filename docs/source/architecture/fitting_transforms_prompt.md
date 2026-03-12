# RheoJAX Fitting & Transform Workflows — Optimization & Enhancement

## Pre-Task: Mandatory Reads Before Writing Any Code

Read ALL of the following in order. Do not write any implementation until all reads
are complete and you have produced the audit summary described below.

1. `docs/source/architecture/io_architecture.rst`
2. The **entire existing fitting and transform codebase**:
   - `rheojax/core/base.py` — `BaseModel` (`.fit()`/`.predict()`) and `BaseTransform`
   - `rheojax/core/registry.py` — `ModelRegistry`, `TransformRegistry`, `PluginInfo`
   - `rheojax/core/parameters.py` — `Parameter`, `ParameterSet`, `ParameterConstraint`
   - `rheojax/core/data.py` — `RheoData` container
   - `rheojax/core/bayesian.py` — `BayesianMixin`, `BayesianResult`
   - `rheojax/core/test_modes.py` — `TestModeEnum`, `DeformationMode`, `Protocol`
   - `rheojax/core/inventory.py` — `Protocol`, `TransformType` enums
   - `rheojax/utils/optimization.py` — `create_least_squares_objective()`,
     `nlsq_optimize()`, `OptimizationResult`
   - `rheojax/pipeline/base.py` — `Pipeline` fluent API
   - `rheojax/pipeline/bayesian.py` — `BayesianPipeline`
   - Every model class across `rheojax/models/` (53 models, 20 families)
   - Every transform class in `rheojax/transforms/` (7 transforms)
   - All existing tests in `tests/`

Once all reads are complete, produce an **audit summary block** (as a top-level
code comment in a new file `rheojax/fitting/_audit.py`) containing:
```python
# AUDIT SUMMARY — generated before any implementation
# ─────────────────────────────────────────────────────
# Protocol → Model registry (current state):
#   RELAXATION  : [list all N models currently registered]
#   CREEP       : [...]
#   OSCILLATION : [...]
#   FLOW_CURVE  : [...]
#   STARTUP     : [...]
#   LAOS        : [...]
#
# Transform registry (current state):
#   [list all 7 transforms: FFTAnalysis, Mastercurve, SRFS,
#    MutationNumber, OWChirp, SmoothDerivative, SPPDecomposer]
#   — their inputs, outputs, and any protocol dependencies
#
# Objective function interfaces (current):
#   create_least_squares_objective() signature and behavior
#   nlsq_optimize() method dispatch ("auto"/"nlsq"/"scipy")
#   Complex G* dispatch: complex(N,) → 2N residuals, real(N,2) → element-wise, real(N,) → |G*|
#
# BaseModel.fit() workflow (current):
#   fit() → _fit() → create_least_squares_objective() → nlsq_optimize()
#   model_function(x, params) closure captures test_mode
#   Protocol kwargs cached as self._attr for NUTS reuse
#
# Performance bottlenecks identified:
#   [list any O(N²) loops, Python-level for-loops over data points,
#    non-JIT-compiled paths, redundant recompilation triggers]
#
# Breaking vs. non-breaking changes required:
#   [explicit list — preserve BaseModel API, Pipeline API, etc.]
#
# Ambiguities requiring resolution before proceeding:
#   [explicit list — do not guess, flag for user review]
```

---

## Objective

Optimize and enhance the end-to-end fitting and transform workflows **within the
existing RheoJAX architecture**. Do NOT create a parallel system. All changes must
extend, not replace, the current `BaseModel`, `Pipeline`, `ModelRegistry`, and
`TransformRegistry` infrastructure.

### Existing architecture to preserve (non-negotiable):

| Component | Location | API |
|---|---|---|
| `BaseModel` | `core/base.py` | `.fit(X, y, **kwargs)` → self, `.predict(X)` → array |
| `BayesianMixin` | `core/bayesian.py` | `.fit_bayesian(X, y, seed, num_chains, ...)` → `BayesianResult` |
| `ModelRegistry` | `core/registry.py` | `@ModelRegistry.register("name", protocols=[...])` |
| `TransformRegistry` | `core/registry.py` | `@TransformRegistry.register("name")` |
| `ParameterSet` | `core/parameters.py` | `.add()`, `.get_value()`, `.set_value()`, `.get_bounds()` |
| `RheoData` | `core/data.py` | `.x`, `.y`, `.to_jax()`, `.to_numpy()`, `.test_mode` |
| `Pipeline` | `pipeline/base.py` | `.load().transform().fit().plot().save()` |
| `create_least_squares_objective` | `utils/optimization.py` | `model_fn, x_data, y_data → residual_fn` |
| `nlsq_optimize` | `utils/optimization.py` | `objective, parameters → OptimizationResult` |

### Target improvements:

- Enhanced initial parameter estimation (`auto_p0`) for all 53 models
- Model comparison and selection (AIC/BIC) across compatible models
- Extended uncertainty quantification (Hessian CI, bootstrap)
- Per-protocol fitting optimizations (auto-preprocessing, diagnostics)
- New convenience transforms (spectrum inversion, Cox-Merz, LVE envelope)
- Richer `FitResult` container with serialization and plotting

---

## Critical Invariants

### Float64 (MANDATORY)
```python
# CORRECT — always use this in every module
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

# NEVER import jax directly — float64 won't be configured
```

### model_function Pattern
All models implement a stateless `model_function(X, params, test_mode, **kwargs)` closure:
```python
def model_function(self, X, params, test_mode=None, **kwargs):
    """Stateless prediction for NLSQ and NUTS."""
    if test_mode == TestMode.RELAXATION:
        return self._predict_relaxation(X, *params)
    elif test_mode == TestMode.OSCILLATION:
        return self._predict_oscillation(X, *params)
    ...
```

### Protocol Kwargs Bridge
Every `_fit_*()` method must cache protocol kwargs as `self._attr` for
`model_function` to use during NUTS:
```python
def _fit_transient(self, X, y, gamma_dot=1.0, **kwargs):
    self._gamma_dot_applied = gamma_dot  # bridge for model_function
    ...
```

### Oscillation Data Dispatch
`create_least_squares_objective` handles three input formats for G*:
- **Complex `G*` (N,)**: complex128, returns 2N stacked residuals [re₁..reₙ, im₁..imₙ]
- **Real `(N, 2)` [G', G'']**: element-wise residuals on both columns
- **Real `(N,)` |G*|**: magnitude-only fallback

Model `_fit` methods must NOT cast complex y to float64 before passing to the objective.

---

## Functional Requirements

### F1 — Enhanced Model Discovery

Extend `ModelRegistry` (do NOT replace it) with query methods:

```python
# Already exists:
ModelRegistry.register("name", protocols=[Protocol.RELAXATION, ...])
ModelRegistry.find(protocol=Protocol.RELAXATION)  # -> list[PluginInfo]
ModelRegistry.get("model_name")  # -> PluginInfo

# NEW — add to registry.py:
ModelRegistry.for_protocol(protocol: str | Protocol) -> list[PluginInfo]
ModelRegistry.compatible_models(data: RheoData) -> list[PluginInfo]
ModelRegistry.model_info(name: str) -> ModelInfo  # rich metadata
```

Add a `ModelInfo` dataclass that aggregates existing `PluginInfo` with runtime
metadata from the model class:
```python
@dataclass
class ModelInfo:
    name: str                           # canonical name from registry
    model_class: type[BaseModel]        # the actual class
    protocols: list[Protocol]           # from registration
    deformation_modes: list[DeformationMode]
    param_names: list[str]              # from ParameterSet.keys()
    param_bounds: dict[str, tuple]      # from ParameterSet.get_bounds()
    param_units: dict[str, str]         # from Parameter.units
    n_params: int
    supports_bayesian: bool             # has model_function
    doc: str                            # from docstring
```

### F2 — Initial Parameter Estimation Engine

Add `rheojax/utils/initialization/auto_p0.py`:

```python
def auto_p0(
    X: ArrayLike,
    y: ArrayLike,
    model: BaseModel,
    test_mode: TestModeEnum | str | None = None,
) -> dict[str, float]:
    """Data-driven initial parameter estimation for any registered model.

    Returns dict of {param_name: estimated_value} that can be applied via
    model.parameters.set_value(name, value) before model.fit().
    """
```

| Parameter family | Auto-estimation strategy |
|---|---|
| Viscosity plateau (η₀) | Mean of lowest 10% γ̇ points on η(γ̇) |
| Relaxation time (λ, τ) | Frequency at G'=G'' crossover; or peak in G''(ω) |
| Plateau modulus (G_N⁰, G0) | G' value at minimum tan δ in rubbery plateau |
| Yield stress (σ_y) | Extrapolation of low-γ̇ σ(γ̇) to γ̇=0 via log-log |
| Power-law index (n) | Slope of log σ vs log γ̇ in power-law region |
| Stretch exponent (β) | Fixed at 0.5 as universal initial guess |
| Spring extensibility (L²) | Estimated from overshoot ratio σ_peak/σ_ss |
| Gel strength (S) | G(t) at t=1 s on power-law gel |

Integrate with `BaseModel.fit()` via a new kwarg:
```python
model.fit(X, y, auto_init=True)  # calls auto_p0 before NLSQ
```

If `auto_p0` cannot estimate a parameter (insufficient data range, missing field),
it must emit `RheoJaxInitWarning` with the specific reason — never silently use
an arbitrary default.

### F3 — Extended FitResult Container

Add `rheojax/core/fit_result.py`:

```python
@dataclass
class FitResult:
    model_name     : str
    protocol       : str
    params         : dict[str, float]          # optimal parameters
    params_ci      : dict[str, tuple] | None   # confidence intervals
    params_units   : dict[str, str]
    fitted_curve   : jnp.ndarray               # model prediction on data grid
    residuals      : jnp.ndarray
    loss_value     : float
    r_squared      : float
    aic            : float | None
    bic            : float | None
    n_params       : int
    n_points       : int
    converged      : bool
    n_iterations   : int
    optimizer_used : str                       # "nlsq", "scipy", "auto"
    covariance     : jnp.ndarray | None        # from Jacobian
    optimization_result: OptimizationResult    # raw result from nlsq_optimize
    input_data     : RheoData                  # reference back to source data
    timestamp      : str                       # ISO 8601
```

`FitResult` must:
- Serialise to HDF5, JSON, `.npz` (extend existing writers in `rheojax/io/writers/`)
- Support `result.plot()` → protocol-appropriate figure with data + fitted curve
  + residuals panel
- Support `result.summary()` → formatted parameter table (rich / plain text)
- Support `result.to_latex()` → LaTeX parameter table for papers
- Support `result.prediction_interval(X_new, alpha=0.95)` → confidence bands

Integrate with `BaseModel.fit()`:
```python
model.fit(X, y, return_result=True)  # returns FitResult instead of self
```

### F4 — Model Selection & Comparison

Add `rheojax/utils/model_selection.py`:

```python
def compare_models(
    X: ArrayLike,
    y: ArrayLike,
    models: list[str | BaseModel],
    test_mode: str | TestModeEnum | None = None,
    criterion: Literal["aic", "bic", "aicc"] = "aic",
    **fit_kwargs,
) -> ModelComparison:
    """Fit multiple models and rank by information criterion."""
```

```python
@dataclass
class ModelComparison:
    results: list[FitResult]           # one per model, sorted by criterion
    rankings: dict[str, int]           # model_name → rank
    delta_criterion: dict[str, float]  # Δ from best model
    weights: dict[str, float]          # Akaike weights
    best_model: str
```

Uses existing `ModelRegistry.find(protocol=...)` to discover compatible models.
Fits sequentially (not `jax.vmap` — models have different param counts).

### F5 — Extended Uncertainty Quantification

Extend `BayesianMixin` and add `rheojax/utils/uncertainty.py`:

```python
def hessian_ci(
    model: BaseModel,
    X: ArrayLike,
    y: ArrayLike,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Confidence intervals from Hessian approximation (Cramér-Rao bound)."""

def bootstrap_ci(
    model: BaseModel,
    X: ArrayLike,
    y: ArrayLike,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap confidence intervals via resampling."""
```

Integrate with `BaseModel.fit()`:
```python
model.fit(X, y, uncertainty="hessian")   # compute_diagnostics already does Jacobian
model.fit(X, y, uncertainty="bootstrap") # resample + refit
```

### F6 — Per-Protocol Fitting Optimizations

Add protocol-specific preprocessing to `BaseModel.fit()` via hooks, NOT by
replacing the fit API. New module: `rheojax/utils/protocol_preprocessing.py`:

**RELAXATION**
- Auto-apply inertia/ringing cutoff at t_min before fitting
- Prony series: enforce log-spacing of relaxation times τᵢ
- Winter-Chambon gel point convenience: `fit_gel_point(t, G_t)` → (S, n)

**CREEP**
- Auto-detect creep regime (viscoelastic plateau vs viscous flow) from long-time slope
- Recovery segment: if present, fit jointly with creep segment using shared
  parameters (same J_e⁰, η₀)

**OSCILLATION**
- Kramers-Kronig consistency check: `check_kramers_kronig(omega, G_prime, G_double_prime)`
- Relaxation spectrum H(τ) inversion: Tikhonov + maximum-entropy options

**FLOW_CURVE**
- Auto-detect yield stress presence from low-γ̇ slope
- Shear-banding detection: flag non-monotonic σ(γ̇) before fitting
- Auto-estimate η₀ from Newtonian plateau

**STARTUP**
- Compute LVE envelope σ_LVE⁺(t) = γ̇₀ ∫₀ᵗ G(s)ds from Prony series
- Overshoot diagnostics: extract (t_peak, σ_peak, σ_peak/σ_ss)

**LAOS**
- LAOS classification per Ewoldt (2008): strain-stiffening/softening/thickening/thinning
- Q₀ nonlinearity coefficient extraction

### F7 — New Convenience Transforms

Add to `rheojax/transforms/` (register via existing `TransformRegistry`):

| Transform | File | Input → Output |
|---|---|---|
| `SpectrumInversion` | `spectrum_inversion.py` | G'(ω)/G''(ω) or G(t) → H(τ) or L(τ) |
| `CoxMerz` | `cox_merz.py` | η*(ω) + η(γ̇) → overlay + validation |
| `LVEEnvelope` | `lve_envelope.py` | G(t) Prony series → σ_LVE⁺(t) for startup |
| `PronyConversion` | `prony_conversion.py` | G(t) ↔ G'(ω)/G''(ω) interconversion |

These supplement the existing 7 transforms (FFT, Mastercurve, SRFS, MutationNumber,
OWChirp, SmoothDerivative, SPP). All must:
- Inherit `BaseTransform` from `rheojax/core/base.py`
- Register via `@TransformRegistry.register("name")`
- Use `safe_import_jax()` for float64
- Be pure JAX functions; `jit`-compilable
- Work in the `Pipeline` fluent API: `pipeline.transform("spectrum_inversion", method="tikhonov")`

### F8 — Physics Sanity Checks

Add `rheojax/utils/physics_checks.py`:

```python
def check_fit_physics(
    model: BaseModel,
    result: FitResult | None = None,
) -> list[PhysicsWarning]:
    """Post-fit physics sanity checks. Returns list of warnings."""
```

Checks:
- η₀ > 0, G > 0, τ > 0 for all modes
- Power-law index 0 < n < 1 for shear-thinning models
- Prony series: all Gᵢ > 0, all τᵢ > 0
- Chebyshev e₁ > 0 (elastic dominated at small strain)
- Parameter values within physically plausible ranges

Integrate as opt-in post-fit hook:
```python
model.fit(X, y, check_physics=True)  # emits RheoJaxPhysicsWarning if violated
```

### F9 — Performance Requirements

- All single-dataset NLSQ fits must complete in < 2 s on CPU for N ≤ 1000.
- JIT compilation overhead (first call) must not exceed 10 s for any model.
- `jax.lax.scan` must be used for all recurrence relations
  (Maxwell chain, retardation spectrum summation, convolution integrals).
- Zero Python-level loops over data points in hot paths.
- Profile all 53 model `model_function` calls; flag any that fail the < 2 s target
  in `_audit.py` with proposed remediation.

### F10 — Robustness

- `BaseModel.fit()` with `method='auto'` already falls back scipy → NLSQ.
  Extend with `method='auto_global'` to add differential evolution as final fallback.
- Log-transform bounded parameters internally for better optimizer convergence.
- `fit()` must never raise an unhandled exception for any combination of valid
  `RheoData` + registered model; catch, log, and return `FitResult(converged=False)`.

---

## Implementation Requirements

- **JAX-first**: all numerical computation in `jax.numpy`; no `numpy` in hot paths.
- **Float64**: every new module must use `safe_import_jax()` — never `import jax`.
- **Pure functions**: all residual, loss, and transform functions must be pure
  (no side effects); `jit` and `vmap` must work without `static_argnums` hacks.
- **Typed exceptions**: extend existing hierarchy in `rheojax/io/_exceptions.py`
  with `RheoJaxFitError`, `RheoJaxInitWarning`, `RheoJaxPhysicsWarning`,
  `RheoJaxConvergenceWarning`.
- **Logging**: use `rheojax.logging` module; never `print`.
- **Docstrings**: NumPy-style on all public classes and functions.
- **Type annotations**: full, Python ≥ 3.12 (`X | None`, `list[X]`, etc.).
- **No global mutable state**: `ModelRegistry` is a singleton but immutable
  after import.
- **ParameterSet API**: Use `keys()` for names, `[name]` for Parameter access,
  `get_bounds()` (no args), `set_value()`/`get_value()`. NO `.names`,
  `.get_parameter()`, `.summary()`, `.to_jax()`.

---

## Deliverables

Produce exactly this file tree (all within existing `rheojax/`):

```
rheojax/
├── core/
│   └── fit_result.py                # FitResult, ModelInfo, ModelComparison
├── utils/
│   ├── initialization/
│   │   └── auto_p0.py               # auto_p0() engine + per-family estimators
│   ├── model_selection.py           # compare_models(), AIC/BIC/AICc
│   ├── uncertainty.py               # hessian_ci(), bootstrap_ci()
│   ├── physics_checks.py           # check_fit_physics(), PhysicsWarning
│   └── protocol_preprocessing.py   # per-protocol auto-preprocessing
├── transforms/
│   ├── spectrum_inversion.py       # H(τ), L(τ) inversion (Tikhonov + MaxEnt)
│   ├── cox_merz.py                 # Cox-Merz overlay and verification
│   ├── lve_envelope.py             # LVE startup envelope from Prony series
│   └── prony_conversion.py         # G(t) ↔ G'(ω)/G''(ω) interconversion
tests/
├── utils/
│   ├── test_auto_p0.py             # auto_p0 coverage for all 53 models
│   ├── test_model_selection.py     # compare_models, AIC/BIC
│   ├── test_uncertainty.py         # hessian_ci, bootstrap_ci
│   └── test_physics_checks.py     # physics sanity checks
├── transforms/
│   ├── test_spectrum_inversion.py  # new transform
│   ├── test_cox_merz.py           # new transform
│   ├── test_lve_envelope.py       # new transform
│   └── test_prony_conversion.py   # new transform
└── test_fit_result.py              # FitResult serialization, plotting, summary
```

---

## Execution Order

Implement strictly in this sequence. Do not proceed to the next step until the
current step's outputs are complete and internally consistent:

1. **Audit summary** → `_audit.py` (read existing code; do not write anything else first)
2. **Typed exceptions** → extend `rheojax/io/_exceptions.py` with fit/physics warnings
3. **`FitResult` + `ModelInfo` + `ModelComparison`** → `rheojax/core/fit_result.py`
4. **`auto_p0` engine** → `rheojax/utils/initialization/auto_p0.py`
5. **Extended `ModelRegistry` queries** → extend `rheojax/core/registry.py`
6. **Physics sanity checks** → `rheojax/utils/physics_checks.py`
7. **Protocol preprocessing** → `rheojax/utils/protocol_preprocessing.py`
8. **Uncertainty quantification** → `rheojax/utils/uncertainty.py`
9. **Model selection** → `rheojax/utils/model_selection.py`
10. **New transforms** (one at a time): spectrum_inversion → cox_merz → lve_envelope → prony_conversion
11. **Integrate with BaseModel.fit()** → extend `rheojax/core/base.py`
    (add `auto_init`, `return_result`, `check_physics`, `uncertainty` kwargs)
12. **Integrate with Pipeline** → extend `rheojax/pipeline/base.py`
    (add `compare_models()` step, `get_fit_result()` method)
13. **FitResult serialization** → extend `rheojax/io/writers/` for FitResult export
14. **Full test suite** → `tests/`
15. **Performance profiling pass** → flag any model failing < 2 s in `_audit.py`
