# RheoJAX Architecture Overview

> JAX-accelerated rheological analysis — 2-10x faster than SciPy.

## Package Structure

```
rheojax/                          # v0.6.0, Python 3.12+
├── core/           # Foundation: BaseModel, RheoData, Parameter, BayesianMixin, Registry
├── models/         # 53 models across 20 families (see Model Inventory)
├── transforms/     # 12 transforms (FFT, mastercurve, OWChirp, SPP, Prony, etc.)
├── pipeline/       # Fluent API: Pipeline, BayesianPipeline, BatchPipeline
├── io/             # TRIOS, CSV, Excel, Anton Paar readers; HDF5/Excel/NPZ writers
├── visualization/  # Auto plot selection, 3 styles (seaborn, matplotlib, ggplot)
├── logging/        # Structured logging (config, formatters, JAX-safe tracing)
├── utils/          # Optimization (NLSQ), Prony, MCT kernels, modulus conversion
├── parallel/       # PersistentProcessPool, xdist-aware auto-configuration
├── cli/            # CLI entry point (rheojax command)
├── gui/            # PySide6 desktop application (rheojax-gui)
└── fitting/        # Fitting engine internals
```

---

## Core Abstractions

### BaseModel (`core/base.py`)

Abstract base class with a **scikit-learn-compatible API**. All 53 models inherit from it.

```
BaseModel (+ BayesianMixin)
├── .fit(X, y, test_mode=None, **kwargs)        → self
├── .predict(X, test_mode=None, **kwargs)        → np.ndarray
├── .fit_bayesian(X, y, seed=42, num_chains=4, num_warmup=1000, num_samples=2000, ...)
│                                                → BayesianResult
├── .score(X, y)                                 → R²
├── .precompile(test_mode='relaxation')          → warm JIT cache
├── .parameters                                  → ParameterSet
├── .get_nlsq_result()                           → FitResult
└── .get_bayesian_result()                       → BayesianResult
```

**Required overrides for new models:**
- `_fit(X, y, **kwargs)` — NLSQ fitting implementation
- `_predict(X, **kwargs)` — Forward model evaluation

**model_function bridge:** Stateless `model_function(X, params, test_mode, **kwargs)` used by both NLSQ and NUTS. Protocol kwargs (gamma_dot, sigma_init, etc.) are cached as `self._attr` in `_fit_*()` methods so `model_function` can access them during NUTS sampling.

### RheoData (`core/data.py`)

JAX-native data container with automatic test mode detection.

```
RheoData
├── .x, .y                  → arrays (JAX float64)
├── .test_mode               → auto-detected from data structure
├── .deformation_mode        → SHEAR | TENSION
├── .to_jax() / .to_numpy() → array conversion
├── .interpolate() / .resample() / .smooth() / .derivative() / .integral()
├── .storage_modulus / .loss_modulus / .tan_delta  (oscillation properties)
└── .slice(start, end)       → sub-range by x-value
```

**Test mode detection:** `initial_test_mode` (InitVar) auto-caches in `metadata["test_mode"]` and `metadata["detected_test_mode"]`.

### Parameter / ParameterSet (`core/parameters.py`)

```
Parameter
├── .value, .bounds, .units, .description, .constraints, .prior
├── .bounds setter → auto-syncs ParameterConstraint objects
└── .validate(value, context={})

ParameterSet (dict-like)
├── .keys() / .values() / .items()
├── [name] → Parameter access
├── .set_value(name, val) / .get_value(name)
├── .get_bounds() → list of (lo, hi) tuples (all params)
├── .to_dict() → {name: param.to_dict()}
└── NO: .names, .get_parameter(), .summary(), .to_jax()
```

### BayesianMixin (`core/bayesian.py`)

Mixed into BaseModel, provides full Bayesian inference via NumPyro NUTS.

```
BayesianMixin
├── .fit_bayesian(X, y, test_mode, seed, num_chains, num_warmup, num_samples, ...)
│   → BayesianResult(posterior_samples, divergences, diagnostics, fit_result)
├── .get_credible_intervals(posterior_samples, credibility=0.95) → dict
├── .sample_prior(n_samples, seed) → dict
└── .precompile_bayesian(test_mode)
```

**Diagnostics (mandatory):** Split R-hat < 1.01, ESS > 400, BFMI checked. ArviZ InferenceData conversion via `.to_inference_data()`.

### FitResult (`core/fit_result.py`)

```
FitResult
├── .r_squared, .adj_r_squared, .rmse, .mae, .aic, .bic, .aicc
├── .success, .converged, .n_iterations, .optimizer_used
├── .residuals, .covariance, .params_ci → {name: (lo, hi)}
├── .prediction_interval(x_new, alpha=0.95) → (mu, lower, upper)
├── .summary() → str, .to_latex(), .to_dict()
└── .save(path) / .load(path) — JSON or NPZ serialization
```

---

## Registry System (`core/registry.py`)

Singleton pattern with decorator-based registration.

```python
@ModelRegistry.register(
    "maxwell",
    protocols=[Protocol.OSCILLATION, Protocol.RELAXATION],
    deformation_modes=[DeformationMode.SHEAR, DeformationMode.TENSION]
)
class Maxwell(BaseModel):
    ...
```

**PluginInfo** dataclass stores: `name`, `plugin_class`, `plugin_type`, `protocols`, `deformation_modes`, `metadata`, `doc`.

**Registry API:**
- `register(name, plugin_class, plugin_type, protocols=[], deformation_modes=[], ...)`
- `get_plugin(name)` → PluginInfo
- `list_models(protocol=None, deformation_mode=None)` → filtered list
- `list_transforms(transform_type=None)` → filtered list

---

## Model Inventory (53 Models, 20 Families)

| Family | Count | Key Pattern | Protocols |
|--------|-------|-------------|-----------|
| Classical | 3 | Maxwell, Zener, SpringPot | Oscillation, Relaxation, Creep |
| Flow | 6 | PowerLaw, Carreau, Cross, HerschelBulkley, Bingham, CarreauYasuda | Flow curve, Startup |
| Fractional Maxwell | 4 | Fractional viscoelastic variants | Oscillation, Relaxation |
| Fractional Zener | 4 | Fractional solid-like variants | Oscillation, Relaxation |
| Fractional Advanced | 3 | Extended fractional models | Oscillation, Relaxation |
| Multimode/GMM | 1 | Generalized Maxwell (Prony series) | Oscillation, Relaxation |
| SGR | 2 | Soft Glassy Rheology (conventional + GENERIC) | Oscillation, Creep |
| STZ | 1 | Shear Transformation Zone | Flow curve, Startup, LAOS |
| EPM | 2 | Entanglement Packing (scalar + tensorial) | Oscillation, SAOS |
| Fluidity | 2 | Local (0D) + Nonlocal (1D PDE) | Flow curve, Oscillation |
| Fluidity-Saramito | 2 | Saramito yield stress fluidity | Flow curve, Oscillation |
| IKH | 2 | Isotropic Kinematic Hardening (ODE) | All protocols |
| FIKH | 2 | Fractional IKH + FMLIKH (Caputo + diffrax) | All protocols |
| HL | 1 | Hierarchical Liquid (PDE) | Flow curve, Oscillation |
| SPP | 1 | Sequence of Physical Processes | LAOS |
| Giesekus | 2 | Single + Multi-mode nonlinear | Oscillation, Relaxation, LAOS |
| DMT | 2 | Derec-Ajdari-Lequeux thixotropic | Flow curve, Oscillation |
| ITT-MCT | 2 | F₁₂ schematic + isotropic k-resolved | Oscillation, Flow curve |
| TNT | 5 | SingleMode, Cates, LoopBridge, MultiSpecies, StickyRouse | All protocols |
| VLB | 4 | local, multi-network, variant, nonlocal PDE | All protocols |
| HVM | 1 | Hybrid Vitrimer Model (11-comp ODE) | All protocols |
| HVNM | 1 | Hybrid Vitrimeric Network Model | All protocols |

### File Pattern per Model Family

```
rheojax/models/<family>/
├── __init__.py          # Exports + registration
├── _kernels.py          # JAX-compiled pure functions (JIT-safe)
├── _base.py             # Shared abstract base for family
├── local.py             # 0D (spatially homogeneous) model
├── multi_*.py           # Multi-mode/network/species variants
└── _kernels_diffrax.py  # ODE-specific kernels (if applicable)
```

---

## Test Modes & Protocols

| Test Mode | Input | Output | Model Count |
|-----------|-------|--------|-------------|
| `OSCILLATION` | ω (frequency) | G* = G' + iG'' | 41 models |
| `RELAXATION` | t (time) | G(t) modulus | ~20 models |
| `CREEP` | t (time) | J(t) compliance / γ(t) strain | ~10 models |
| `FLOW_CURVE` | γ̇ (shear rate) | σ (stress) | ~15 models |
| `STARTUP` | t at const γ̇ | σ(t) transient stress | ~10 models |
| `LAOS` | ω + γ₀ | Nonlinear response | ~10 models |
| `SAOS` | ω (amplitude sweep) | Nonlinear G', G'' | Special models |
| `SRFS` | Spectral decomposition | Relaxation spectrum | Transforms |

### Oscillation Data Dispatch (Complex G*)

Models handle three input formats:
1. **Complex `G*` (N,):** `jnp.complex128` — `create_least_squares_objective` fits G' and G'' independently
2. **Real `(N, 2)` [G', G'']:** Element-wise residuals on both columns
3. **Real `(N,)` |G*|:** Magnitude-only fallback

### DMTA / DMA Support

49 oscillation-capable models auto-convert E* ↔ G* at BaseModel boundary:
```python
model.fit(omega, E_star, test_mode="oscillation", deformation_mode="tension", poisson_ratio=0.5)
```

---

## Transforms (12)

| Transform | Purpose |
|-----------|---------|
| `fft_analysis` | FFT spectral decomposition |
| `mastercurve` | Time-temperature superposition |
| `owchirp` | Optimized White-noise Chirp analysis |
| `spp_decomposer` | SPP LAOS decomposition |
| `prony_conversion` | Spectrum → Prony (Generalized Maxwell) |
| `spectrum_inversion` | Inverse Laplace → relaxation spectrum |
| `srfs` | Stress Relaxation From SAOS |
| `cox_merz` | Cox-Merz superposition (SAOS → flow) |
| `lve_envelope` | Linear viscoelastic envelope |
| `mutation_number` | TTT diagram mutation numbers |
| `smooth_derivative` | Smooth numerical differentiation |

**Transform API:** `BaseTransform` with `.transform()`, `.inverse_transform()`, `.fit()`, `.fit_transform()`, `.batch_transform()`.

---

## Pipeline Architecture (`pipeline/`)

### Fluent API

```python
# Basic fitting
Pipeline().load('data.csv').fit('maxwell').plot().save('result.hdf5')

# Bayesian workflow
BayesianPipeline().load(data).fit_nlsq('maxwell').fit_bayesian(num_samples=2000).plot_posterior()

# Model comparison
ModelComparisonPipeline(['maxwell', 'zener']).run(data)

# Batch processing
BatchPipeline().fit_batch(datasets, n_workers=4)

# Programmatic construction
PipelineBuilder().add_load_step().add_transform_step().add_fit_step().build()
```

### Pipeline Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Pipeline` | Core fluent API | `.load()`, `.fit()`, `.predict()`, `.plot()`, `.save()` |
| `BayesianPipeline` | NLSQ warm-start → NUTS | `.fit_bayesian()`, `.plot_posterior()` |
| `BatchPipeline` | Multi-dataset parallel processing | `.fit_batch(datasets, n_workers)` |
| `PipelineBuilder` | Programmatic construction | `.add_stage()`, `.build()` |

### Specialized Workflows (`workflows.py`)

- `MastercurvePipeline` — Temperature superposition
- `ModelComparisonPipeline` — Side-by-side evaluation
- `CreepToRelaxationPipeline` — Interconversion
- `FrequencyToTimePipeline` — FFT inverse
- `SPPAmplitudeSweepPipeline` — LAOS harmonics

---

## I/O System

### Readers (`io/readers/`)

| Reader | Format | Source |
|--------|--------|--------|
| `trios/` | TRIOS XML | TA Instruments rotational rheometer |
| `anton_paar.py` | Anton Paar binary/XML | Anton Paar MCR/DSR |
| `csv_reader.py` | CSV/TSV | Generic with column mapping |
| `excel_reader.py` | .xlsx | Multi-sheet with selection |
| `auto.py` | Auto-detect | Extension-based dispatch |
| `multi_file.py` | Batch import | Directory + metadata |

### Writers (`io/writers/`)

| Writer | Format | Features |
|--------|--------|----------|
| `hdf5_writer.py` | HDF5 | Full serialization (models, results, metadata) |
| `excel_writer.py` | .xlsx | Multi-sheet (params, results, predictions, diagnostics) |
| `npz_writer.py` | NumPy NPZ | Lightweight JAX-native export |

---

## Parallel Processing (`parallel/`)

### PersistentProcessPool

```python
with PersistentProcessPool(n_workers=4, timeout=300, warmup_jax=True) as pool:
    futures = [pool.submit(fit_fn, data) for data in datasets]
    results = [f.result() for f in futures]
```

- **Subprocess isolation** prevents xdist × JAX OOM
- **JAX warm-up** per worker process
- **Auto-cleanup** via `atexit` handler + WeakSet tracking
- **Sequential mode** auto-detected in pytest-xdist workers (`PYTEST_XDIST_WORKER` env)

### Worker Configuration (`config.py`)

```
Priority: PYTEST_XDIST_WORKER detected → 1 worker (sequential)
         Config override → use value
         RHEOJAX_PARALLEL_WORKERS env → parse
         Default → cpu_count // 2 (max 8), GPU-aware cap to 4
```

---

## Visualization (`visualization/`)

| Component | Purpose |
|-----------|---------|
| `FitPlotter` | Main plotting with uncertainty bands |
| `TransformPlotter` | Transform result visualization |
| `epm_plots` | EPM 3D stress/modulus animations |
| `spp_plots` | SPP LAOS decomposition plots |
| `templates` | Publication-quality templates (light/dark) |

**Auto-detection:** Oscillation → Bode/Cole-Cole, Relaxation → log-log/semilog.

**Styles:** `seaborn` (default, publication-ready), `matplotlib` (classic), `ggplot`.

---

## Logging (`logging/`)

Structured, JAX-safe logging system.

| Module | Purpose |
|--------|---------|
| `config.py` | LogConfig, configure_logging() |
| `context.py` | Contextual loggers (log_bayesian, log_fit, log_pipeline_stage) |
| `formatters.py` | Standard, detailed, JSON output formats |
| `handlers.py` | File + console handlers |
| `jax_utils.py` | JAX tracing-safe utilities |
| `metrics.py` | Performance metrics tracking |
| `exporters.py` | BatchingExporter, CallbackExporter |

**Environment variables:** `RHEOJAX_LOG_LEVEL`, `RHEOJAX_LOG_FILE`, `RHEOJAX_LOG_FORMAT`.

---

## Critical Design Invariants

### 1. Float64 Enforcement (MANDATORY)

```python
# CORRECT — every module must use this
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

# NEVER import jax directly — float64 won't be configured
```

`safe_import_jax()` ensures:
1. NLSQ imported first (runtime check)
2. `jax.config.update("jax_enable_x64", True)`
3. XLA compilation cache at `~/.cache/rheojax/jax_cache/`
4. Suppresses matplotlib glyph warnings

### 2. Model Function Bridge

Protocol kwargs must be cached as `self._attr` in `_fit_*()` for NUTS:
```python
# In _fit_transient():
self._gamma_dot_applied = gamma_dot  # NUTS reads this via model_function
```

### 3. test_mode Kwarg Filtering

Always filter `test_mode` from kwargs before forwarding to `model_function`:
```python
kwargs.pop("test_mode", None)  # Prevents "multiple values" TypeError
```

### 4. Bounds-Constraints Sync

Setting `param.bounds = (lo, hi)` auto-updates `ParameterConstraint` objects — no manual sync needed.
