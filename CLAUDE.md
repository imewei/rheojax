# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RheoJAX is a JAX-accelerated rheological analysis package providing a unified framework for analyzing experimental rheology data. It integrates classical rheological models with modern data transforms, offering 2-10x performance improvements through JAX + GPU acceleration.

**Technology Stack:**
- Python 3.12+ (3.8-3.11 NOT supported)
- JAX 0.8.0 for automatic differentiation and GPU acceleration
- NLSQ 0.1.6+ for GPU-accelerated nonlinear least squares optimization
- NumPyro for Bayesian inference (MCMC NUTS sampling)
- ArviZ 0.15.0+ for Bayesian visualization and diagnostics
- NumPy, SciPy for numerical operations
- Matplotlib for visualization
- h5py, pandas, openpyxl for I/O

**Recent Major Features (v0.2.0):**
- **Model-Data Compatibility Checking**: Intelligent detection of physics mismatches between models and data
- **Smart Initialization**: Automatic parameter initialization for fractional models in oscillation mode
- See [Recent Features](#recent-features-updated-2025-11-07) section below for details

## Common Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests excluding slow ones
pytest -m "not slow"

# Run with coverage report
pytest --cov=rheo --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m validation    # Validation against original pyRheo/hermes-rheo
pytest -m benchmark     # Performance benchmarks

# Run specific test file
pytest tests/core/test_parameters.py

# Run feature-specific tests (NLSQ + NumPyro workflow)
pytest tests/core/test_float64_precision.py tests/utils/test_nlsq_optimization.py \
       tests/core/test_bayesian.py tests/integration/test_nlsq_numpyro_workflow.py \
       tests/pipeline/test_bayesian_pipeline.py
```

### Code Quality
```bash
# Format code (Black + Ruff)
black rheo tests
ruff check --fix rheo tests

# Linting only
ruff check rheo tests

# Type checking
mypy rheo

# Quick iteration (format + fast tests)
make quick
```

### Using Makefile
```bash
make test           # Run all tests
make test-fast      # Skip slow tests
make test-coverage  # Generate coverage report
make format         # Format code
make lint           # Run linting
make type-check     # Type checking
make check          # All quality checks
make quick          # Format + fast tests
make clean          # Remove build artifacts (preserves .venv, .claude, agent-os)
```

### GPU Installation (Linux + CUDA 12.1-12.9 only)
```bash
# Quick install (recommended)
make install-jax-gpu

# Manual install
pip uninstall -y jax jaxlib
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Expected: [cuda(id=0)]
```

**Note:** GPU acceleration provides 20-100x speedup for large datasets. CPU-only JAX works on all platforms (Linux, macOS, Windows). JAX versions must match exactly (0.8.0).

## Architecture

### Core Package Structure

```
rheojax/
├── core/               # Core abstractions and common functionality
│   ├── base.py        # BaseModel, BaseTransform abstract classes
│   ├── data.py        # RheoData wrapper (JAX-compatible)
│   ├── parameters.py  # Parameter, ParameterSet, ParameterOptimizer
│   ├── bayesian.py    # BayesianMixin for NumPyro NUTS sampling + ArviZ integration
│   ├── jax_config.py  # Float64 precision enforcement
│   ├── test_modes.py  # TestMode detection (relaxation, creep, oscillation, rotation)
│   └── registry.py    # Plugin registry for models/transforms
├── models/            # 20 rheological models
│   ├── Classical: maxwell.py, zener.py, springpot.py, bingham.py
│   ├── Fractional (11): fractional_maxwell_*.py, fractional_zener_*.py
│   └── Flow (6): power_law.py, carreau*.py, cross.py, herschel_bulkley.py
├── transforms/        # 5 data transforms
│   ├── FFT analysis
│   ├── mastercurve.py (time-temperature superposition)
│   ├── mutation_number.py (viscoelastic character)
│   ├── owchirp.py (LAOS analysis)
│   └── smooth_derivative.py
├── pipeline/          # Fluent API for workflows
│   ├── base.py        # Pipeline class (load → fit → plot → save)
│   ├── bayesian.py    # BayesianPipeline (NLSQ → NUTS workflow + ArviZ diagnostics)
│   ├── workflows.py   # Pre-configured pipelines (mastercurve, model comparison)
│   ├── builder.py     # Programmatic pipeline construction
│   └── batch.py       # Batch processing multiple datasets
├── io/
│   ├── readers/       # TRIOS, CSV, Excel, Anton Paar, auto-detection
│   └── writers/       # HDF5 (full fidelity), Excel
├── visualization/     # Publication-quality plotting
│   ├── plotter.py     # Automatic plot type selection
│   └── templates.py   # Three styles: default, publication, presentation
└── utils/
    ├── optimization.py     # NLSQ-based optimization (5-270x speedup)
    ├── mittag_leffler.py   # Mittag-Leffler functions (1 and 2-parameter)
    ├── compatibility.py    # Model-data compatibility checking (NEW in v0.2.0)
    ├── initialization.py   # Backward-compatible facade (471 lines, refactored in v0.2.0)
    └── initialization/     # Template Method pattern (NEW in v0.2.0)
        ├── __init__.py                      # Re-exports facade functions
        ├── base.py                          # BaseInitializer + extract_frequency_features()
        ├── fractional_zener_ss.py           # FZSS initializer
        ├── fractional_maxwell_liquid.py     # FML initializer
        ├── fractional_maxwell_gel.py        # FMG initializer
        ├── fractional_maxwell_model.py      # FMM initializer
        ├── fractional_zener_ll.py           # FZLL initializer
        ├── fractional_zener_sl.py           # FZSL initializer
        ├── fractional_burgers.py            # Burgers initializer
        ├── fractional_jeffreys.py           # Jeffreys initializer
        ├── fractional_kelvin_voigt.py       # FKV initializer
        ├── fractional_kv_zener.py           # FKV-Zener initializer
        └── fractional_poynting_thomson.py   # Poynting-Thomson initializer
```

### Key Design Patterns

**1. BaseModel Pattern**
- All models inherit from `BaseModel` (rheojax/core/base.py)
- Implements scikit-learn compatible API: `.fit(X, y)`, `.predict(X)`
- JAX-compatible: accepts both NumPy and JAX arrays
- Internal methods: `_fit()`, `_predict()` for subclass implementation
- Bayesian capabilities via BayesianMixin: `.fit_bayesian()`, `.sample_prior()`, `.get_credible_intervals()`

**2. RheoData Container**
- Wraps data with metadata (units, domain, test_mode)
- JAX-compatible via `.to_jax()` method
- Operations: `.smooth()`, `.resample()`, `.derivative()`
- Automatic test mode detection: relaxation, creep, oscillation, rotation

**3. Parameter System**
- `Parameter`: Individual parameter with value, bounds, units
- `ParameterSet`: Collection of parameters for a model
- Used for optimization bounds, Bayesian priors, and JAX conversion
- Type-safe with bounds validation

**4. Pipeline API**
- Fluent interface: `Pipeline().load().fit().plot().save()`
- Pre-configured workflows: `MastercurvePipeline`, `ModelComparisonPipeline`, `BayesianPipeline`
- Batch processing: `BatchPipeline` for multiple datasets
- Programmatic construction: `PipelineBuilder`

**5. Plugin Registry**
- Models and transforms register via `@ModelRegistry.register()` decorator
- Enables dynamic discovery and instantiation
- Supports custom plugins

**6. Template Method Pattern for Initialization (v0.2.0 Refactoring)**

Smart parameter initialization for fractional models now uses the Template Method design pattern to eliminate code duplication while maintaining 100% backward compatibility.

**Architecture:**
- `BaseInitializer` abstract class (`rheojax/utils/initialization/base.py`) enforces consistent 5-step algorithm
- 11 concrete initializer classes (one per fractional model)
- Backward-compatible facade in `initialization.py` delegates to concrete initializers

**Template Method Algorithm (`initialize()`):**
1. **Extract frequency features** - Common logic using `extract_frequency_features(omega, G_star)`
   - Identifies low/high frequency plateaus from |G*| data
   - Finds transition frequency (steepest slope in log-log plot)
   - Estimates fractional order α from slope
2. **Validate features** - Common validation checking frequency range and plateau ratio
3. **Estimate model-specific parameters** - Abstract method `_estimate_parameters(features)` implemented by each model
4. **Clip to parameter bounds** - Common logic ensuring estimates respect ParameterSet bounds
5. **Set parameters** - Abstract method `_set_parameters(param_set, clipped_params)` for safe parameter setting

**Benefits:**
- **Code reduction**: 49% (932 → 471 lines in `initialization.py`)
- **Performance**: <0.01% overhead (187 μs initialization vs 1.76s total fitting time)
- **Maintainability**: Single algorithm definition, no duplication across 11 models
- **Backward compatible**: All public functions preserved (`initialize_fractional_zener_ss()`, etc.)

**Location**: `rheojax/utils/initialization/` (modular structure with one file per model)

### Test Mode System

Located in `rheojax/core/test_modes.py`. Auto-detects:
- **Relaxation**: stress decay over time
- **Creep**: strain increase under constant stress
- **Oscillation**: frequency-domain (G', G", tan δ)
- **Rotation**: flow curves (viscosity vs shear rate)

Detection based on data characteristics (monotonicity, domain, column names). Used for automatic plot selection and model validation.

### Float64 Precision Enforcement

**Critical Requirement:** RheoJAX enforces float64 precision throughout the entire JAX stack to ensure numerical stability for rheological calculations. This is accomplished through mandatory import order.

#### Import Order Requirement

**NLSQ must be imported BEFORE JAX to enable float64 globally.**

The package automatically handles this in `rheojax/__init__.py`:
```python
# rheojax/__init__.py (automatic - no user action needed)
import nlsq  # MUST come before any JAX imports
# ... JAX is imported later by models/utils
```

#### Safe JAX Import Pattern for Developers

**DO NOT import JAX directly in RheoJAX modules.** Always use the safe import mechanism:

```python
# CORRECT - Safe import (enforces float64)
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

# INCORRECT - Never do this in RheoJAXJAX modules
import jax  # Will raise ImportError if NLSQ not imported first
import jax.numpy as jnp
```

All 20 models in `rheojax/models/` use the safe import pattern. If adding a new model or transform, always use `safe_import_jax()`.

#### Float64 Verification

To verify JAX is operating in float64 mode:
```python
from rheojax.core.jax_config import verify_float64
verify_float64()  # Raises exception if float64 not enabled
```

#### Rationale

Float64 precision is essential for:
- **Numerical stability** in optimization with tight tolerances
- **Accurate parameter estimation** for rheological models
- **Reliable MCMC sampling** in Bayesian inference
- **Gradient computation** in JAX automatic differentiation

NLSQ automatically configures JAX for float64 when imported, making the import order critical.

## Recent Features (Updated: 2025-11-08)

### Template Method Refactoring (v0.2.0 - 3 Phases Complete)

**Phase 1: Template Method Pattern** - Eliminated code duplication via BaseInitializer
**Phase 2: Constants Extraction** - Centralized configuration in dedicated module
**Phase 3: FractionalModelMixin** - Reduced model class duplication

### Template Method Refactoring for Initialization (v0.2.0)

**Location:** `rheojax/utils/initialization/` (modular package structure)

Refactored smart parameter initialization to use the Template Method design pattern, eliminating code duplication across all 11 fractional models while maintaining 100% backward compatibility.

**Key Changes:**
- **49% code reduction**: 932 → 471 lines in `initialization.py` facade
- **Modular architecture**: 11 separate initializer files (one per model)
- **Consistent algorithm**: BaseInitializer enforces 5-step template across all models
- **Performance**: <0.01% overhead (187 μs initialization vs 1.76s total)
- **Test coverage**: 22/22 tests passing (3 test files covering algorithm, models, and backward compatibility)

**Benefits for Developers:**
- Single source of truth for initialization algorithm
- Easy to extend with new fractional models
- Comprehensive test suite validates all initializers
- Zero breaking changes to public API

**Phase 2: Constants Extraction** - Centralized magic numbers into `constants.py`:
- `FEATURE_CONFIG`: Savitzky-Golay window, plateau percentile, epsilon
- `PARAM_BOUNDS`: min/max fractional order constraints
- `DEFAULT_PARAMS`: fallback values when initialization fails
- Benefits: Tunable configuration, reduced coupling, better testability

**Phase 3: FractionalModelMixin** - Reduced code duplication in model classes:
- `_apply_smart_initialization()`: Delegated initialization for all 11 models
- `_validate_fractional_parameters()`: Common validation logic
- Automatic initializer mapping via class name lookup
- Benefits: DRY principle, consistent error handling, easier maintenance

See [Template Method Pattern](#6-template-method-pattern-for-initialization-v020-refactoring) section for implementation details.

### Example Notebooks (24 Total)

**New in v0.2.0:**
- `examples/advanced/07-trios_chunked_reading_example.ipynb` - Chunked reading for large TRIOS files (>1GB)
- `examples/bayesian/06-bayesian_workflow_demo.ipynb` - Complete Bayesian workflow (NLSQ → NUTS → diagnostics)

**All Example Categories:**
- **Basic** (5 notebooks): Maxwell, Zener, SpringPot, Bingham, PowerLaw fitting
- **Transforms** (6 notebooks): FFT, Mastercurve, Mutation Number, OWChirp, Derivatives
- **Bayesian** (6 notebooks): Basics, Priors, Diagnostics, Model Comparison, Uncertainty, Workflow
- **Advanced** (7 notebooks): Multi-technique, Batch, Custom Models, Fractional Deep-Dive, Performance, Model Selection, TRIOS Chunked

### Model-Data Compatibility Checking System (v0.2.0)

**Location:** `rheojax/utils/compatibility.py`

A new intelligent system that detects when rheological models are inappropriate for experimental data based on underlying physics. This helps users understand why certain model-data combinations fail and recommends alternatives.

#### Key Capabilities

1. **Decay Type Detection (`detect_decay_type`)**: Automatically identifies relaxation patterns using statistical regression on log-transformed data
   - Exponential (Maxwell-like): G(t) = G₀ exp(-t/τ)
   - Power-law (gel-like): G(t) = G₀ t^(-α)
   - Stretched exponential: G(t) = G₀ exp(-(t/τ)^β)
   - Mittag-Leffler (fractional): G(t) = G₀ E_α(-(t/τ)^α)
   - Multi-mode relaxation

2. **Material Type Classification (`detect_material_type`)**: Classifies materials from relaxation or oscillation data
   - Solid-like (finite equilibrium modulus)
   - Liquid-like (zero equilibrium modulus, flows)
   - Gel-like (power-law relaxation)
   - Viscoelastic solid/liquid

3. **Compatibility Checking (`check_model_compatibility`)**: Compares model physics with data characteristics
   - Configurable confidence levels
   - Physics-based compatibility rules for 6 major model families (FZSS, FML, FMG, Maxwell, Zener, FKV)
   - Returns detailed warnings and recommendations

4. **Enhanced Error Messages**: When optimization fails, provides physics-based explanations
   ```python
   try:
       model.fit(t, G_data, check_compatibility=True)
   except RuntimeError as e:
       # Error message includes compatibility analysis and alternative models
       print(e)
   ```

5. **Minimal Overhead**: Fast detection (< 1 ms for typical datasets)

#### Usage Example

```python
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
from rheojax.utils.compatibility import check_model_compatibility, format_compatibility_message
import numpy as np

# Test FZSS with exponential decay (incompatible)
t = np.logspace(-2, 2, 50)
G_t = 1e5 * np.exp(-t / 1.0)

model = FractionalZenerSolidSolid()
compat = check_model_compatibility(model, t=t, G_t=G_t, test_mode='relaxation')

print(format_compatibility_message(compat))
# Output: WARNING - FZSS expects power-law decay but detected exponential
#         Recommended alternatives: Maxwell, Zener, FractionalMaxwellLiquid
```

#### Integration with BaseModel

The compatibility checking is integrated into `BaseModel._fit()` with optional `check_compatibility=True` parameter:

```python
model.fit(X, y, test_mode='relaxation', check_compatibility=True)
```

When enabled and optimization fails, enhanced error messages include:
- Detected decay type and material type
- Model-specific compatibility warnings
- Recommended alternative models
- Physics-based explanation of failure

See [Model Selection Guide](docs/model_selection_guide.md) for comprehensive decision flowcharts.

### Smart Initialization for Fractional Models (Oscillation Mode) (v0.2.0)

**Location:** `rheojax/utils/initialization.py`

Automatic intelligent parameter initialization for fractional models when fitting oscillation data. This significantly improves convergence and parameter recovery for models that were previously unstable in oscillation mode (resolves Issue #9).

#### Affected Models

All 11 fractional models now support smart initialization:
- `FractionalZenerSolidSolid` (FZSS)
- `FractionalMaxwellLiquid` (FML)
- `FractionalMaxwellGel` (FMG)
- `FractionalMaxwellModel`, `FractionalBurgers`, `FractionalJeffreys`
- `FractionalKelvinVoigt`, `FractionalKVZener`, `FractionalZenerLL`, `FractionalZenerSL`
- `FractionalPoyntingThomson`

#### How It Works

When `test_mode='oscillation'`, the initialization system:

1. **Estimates Initial Moduli**: From low/high-frequency plateau regions of G' and G"
   - Low-frequency G' → equilibrium modulus (Ge)
   - High-frequency G' → glassy modulus (Gm or G0)

2. **Estimates Fractional Order (α)**: From frequency-dependence of loss tangent slope
   - Analyzes slope of tan(δ) = G"/G' in intermediate frequency range
   - Maps slope to fractional order α ∈ [0, 1]

3. **Estimates Characteristic Time (τ)**: From crossover frequency (tan δ peak)
   - Identifies frequency where G' = G" (or tan δ maximum)
   - Converts to characteristic relaxation time: τ = 1/ω_crossover

#### Usage

Smart initialization is **automatic** when fitting oscillation data:

```python
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
import numpy as np

# Generate oscillation data
omega = np.logspace(-2, 2, 50)
G_star = ...  # Complex modulus [G', G"]

# Fit automatically uses smart initialization
model = FractionalZenerSolidSolid()
model.fit(omega, G_star, test_mode='oscillation')  # Smart init applied transparently

# Check fitted parameters
print(f"Ge = {model.parameters.get_value('Ge'):.2e} Pa")
print(f"alpha = {model.parameters.get_value('alpha'):.4f}")
```

No user action required - initialization happens transparently during `fit()`.

#### Benefits

- **Improved Convergence**: Reduces optimization failures in oscillation mode
- **Better Parameter Recovery**: Starting from physics-based estimates
- **Faster Optimization**: Fewer iterations needed with better initial guess
- **Issue #9 Resolution**: Fixes long-standing instability in fractional model fitting

#### Implementation Details

**Template Method Architecture (v0.2.0):**

The initialization system uses the Template Method design pattern for consistency across all 11 fractional models:

```python
# File: rheojax/utils/initialization/base.py
from abc import ABC, abstractmethod

class BaseInitializer(ABC):
    """Template method pattern for fractional model initialization."""

    def initialize(self, omega, G_star, param_set) -> bool:
        """5-step template method: extract → validate → estimate → clip → set."""
        features = extract_frequency_features(omega, G_star)
        if not self._validate_data(features):
            return False
        estimated = self._estimate_parameters(features)
        clipped = self._clip_to_bounds(estimated, param_set)
        self._set_parameters(param_set, clipped)
        return True

    @abstractmethod
    def _estimate_parameters(self, features: dict) -> dict:
        """Model-specific: extract parameters from features."""
        pass

    @abstractmethod
    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Model-specific: safely set parameters in ParameterSet."""
        pass
```

**Concrete Initializer Example (FZSS):**

```python
# rheojax/utils/initialization/fractional_zener_ss.py
class FractionalZenerSSInitializer(BaseInitializer):
    def _estimate_parameters(self, features: dict) -> dict:
        return {
            'Ge': features['low_plateau'],
            'Gm': features['high_plateau'] - features['low_plateau'],
            'tau_alpha': 1.0 / features['omega_mid'],
            'alpha': features['alpha_estimate']
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        for name in ['Ge', 'Gm', 'tau_alpha', 'alpha']:
            if param_set.has_parameter(name):
                param_set.set_value(name, clipped_params[name])
```

**Model Integration:** Fractional models use initializers in `_fit()` when `test_mode='oscillation'`.

**Backward Compatibility:** Facade in `initialization.py` delegates to initializers (49% code reduction: 932 → 471 lines).

**Performance:** 187 μs overhead (<0.01% of 1.76s total), 22/22 tests passing.

### NLSQ + NumPyro Workflow

RheoJAX implements a two-step optimization workflow combining fast NLSQ point estimation with Bayesian NUTS inference.

#### Step 1: NLSQ Optimization (Fast Point Estimation)

NLSQ provides GPU-accelerated nonlinear least squares with 5-270x speedup over scipy:

```python
from rheojax.models.maxwell import Maxwell
import numpy as np

# Generate data
t = np.linspace(0.1, 10, 50)
G_true = 1e5 * np.exp(-t / 0.01)

# Fit with NLSQ (default)
model = Maxwell()
model.fit(t, G_true)

# Access fitted parameters
G0 = model.parameters.get_value('G0')
eta = model.parameters.get_value('eta')
print(f"Fitted: G0={G0:.3e}, eta={eta:.3e}")
```

#### Step 2: Bayesian Inference with Warm-Start

Use NLSQ point estimates to warm-start NUTS for faster convergence:

```python
# Bayesian inference with warm-start from NLSQ
result = model.fit_bayesian(
    t, G_true,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1
)

# Access posterior samples
posterior_G0 = result.posterior_samples['G0']
posterior_eta = result.posterior_samples['eta']

# Check convergence diagnostics
print(f"R-hat: G0={result.diagnostics['r_hat']['G0']:.4f}")
print(f"ESS: G0={result.diagnostics['ess']['G0']:.0f}")

# Get credible intervals
intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
print(f"G0 95% CI: [{intervals['G0'][0]:.3e}, {intervals['G0'][1]:.3e}]")
```

#### Complete Workflow Example

```python
from rheojax.models.maxwell import Maxwell
import numpy as np

# 1. Create model and data
model = Maxwell()
t = np.linspace(0.1, 10, 50)
G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, size=t.shape)

# 2. NLSQ point estimation (fast)
model.fit(t, G_data)
print(f"NLSQ: G0={model.parameters.get_value('G0'):.3e}")

# 3. Bayesian inference (warm-start from NLSQ)
initial_values = {
    'G0': model.parameters.get_value('G0'),
    'eta': model.parameters.get_value('eta')
}

result = model.fit_bayesian(
    t, G_data,
    num_warmup=1000,
    num_samples=2000,
    initial_values=initial_values
)

# 4. Analyze results
print(f"Posterior mean: G0={result.summary['G0']['mean']:.3e} ± {result.summary['G0']['std']:.3e}")
print(f"Convergence: R-hat={result.diagnostics['r_hat']['G0']:.4f}, ESS={result.diagnostics['ess']['G0']:.0f}")
```

#### BayesianPipeline for Fluent API

For complex workflows, use BayesianPipeline with comprehensive ArviZ integration:

```python
from rheojax.pipeline.bayesian import BayesianPipeline

pipeline = BayesianPipeline()

# Fluent API: load → fit_nlsq → fit_bayesian → plot → save
(pipeline
    .load('data.csv', x_col='time', y_col='stress')
    .fit_nlsq('maxwell')
    .fit_bayesian(num_samples=2000, num_warmup=1000)
    .plot_posterior()
    .plot_trace()
    .save('results.hdf5'))

# Access diagnostics
diagnostics = pipeline.get_diagnostics()
summary = pipeline.get_posterior_summary()

# ArviZ diagnostic plots (comprehensive MCMC quality assessment)
(pipeline
    .plot_pair(divergences=True)        # Parameter correlations with divergences
    .plot_forest(hdi_prob=0.95)         # Credible intervals comparison
    .plot_energy()                       # NUTS energy diagnostic
    .plot_autocorr()                     # Mixing diagnostic
    .plot_rank()                         # Convergence diagnostic
    .plot_ess(kind='local'))            # Effective sample size
```

#### ArviZ Diagnostic Plots

BayesianPipeline provides 6 MCMC diagnostics via ArviZ with fluent API:

1. **`plot_pair()`** - Parameter correlations (detect non-identifiability, funnels)
2. **`plot_forest()`** - Credible intervals (HDI: 0.68, 0.95, 0.997)
3. **`plot_energy()`** - NUTS diagnostic (requires multi-chain)
4. **`plot_autocorr()`** - Mixing quality (goal: drops to ~0 within few lags)
5. **`plot_rank()`** - Convergence (goal: uniform histogram)
6. **`plot_ess()`** - Effective sample size (goal: ESS > 400)

All methods support `show=False` and `.save_figure()` chaining. Convert to ArviZ InferenceData: `pipeline._bayesian_result.to_inference_data()`.

#### BayesianPlotter: Direct Access (v0.2.0)

BayesianPlotter can be accessed via `.plotter` property or standalone:

```python
# Via pipeline (delegated API)
pipeline.plot_pair().plot_forest()

# Direct access (for finer control)
pipeline.plotter.plot_pair(var_names=['G0', 'eta'])

# Standalone usage
from rheojax.visualization import BayesianPlotter
plotter = BayesianPlotter(result)
plotter.plot_pair().save_figure('diagnostics.png')
```

### Data Transforms: Mastercurve API

Time-temperature superposition (TTS) via `Mastercurve` transform:

```python
from rheojax.transforms.mastercurve import Mastercurve

# Create mastercurve with WLF or Arrhenius shifting
mc = Mastercurve(reference_temp=298.15, method='wlf', C1=17.44, C2=51.6)
mastercurve, shift_factors = mc.transform(datasets)

# Get model parameters and shift factors
wlf_params = mc.get_wlf_parameters()  # Returns C1, C2, T_ref
temps, shifts = mc.get_shift_factors_array()  # For plotting

# Optimize and validate
C1_opt, C2_opt = mc.optimize_wlf_parameters(datasets)
error = mc.compute_overlap_error(datasets)
```

**Key Methods:** `transform()`, `get_wlf_parameters()`, `get_arrhenius_parameters()`, `get_shift_factors_array()`, `optimize_wlf_parameters()`, `compute_overlap_error()`

### Troubleshooting

#### Float64 Precision Issues

**Error: "NLSQ must be imported before JAX"**
```
ImportError: NLSQ must be imported before JAX to enable float64 precision.
Import order: import nlsq -> import jax
```

**Solution:** You're trying to import JAX directly. Use safe_import_jax():
```python
# Change this:
import jax
import jax.numpy as jnp

# To this:
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()
```

**Error: Arrays are float32 instead of float64**

**Check:** Verify NLSQ was imported first:
```python
import sys
assert 'nlsq' in sys.modules, "NLSQ must be imported before JAX"
```

**Check:** Verify JAX default dtype:
```python
from rheojax.core.jax_config import verify_float64
verify_float64()  # Will raise exception if not float64
```

#### NLSQ Optimization Issues

**Error: "Optimization did not converge"**

**Solutions:**
- Increase max_iter: `model.fit(X, y, max_iter=5000)`
- Adjust tolerance: `model.fit(X, y, ftol=1e-6, xtol=1e-6)`
- Check data quality and scale
- Try better initial parameter guesses

**Issue: Optimization is slow**

**Solutions:**
- Ensure JAX is using GPU: `print(jax.devices())`
- Enable JIT compilation (enabled by default)
- Check if data is unnecessarily large

#### Bayesian Inference Issues

**Warning: "R-hat > 1.1" (poor convergence)**

**Solutions:**
- Increase warmup: `fit_bayesian(..., num_warmup=2000)`
- Increase samples: `fit_bayesian(..., num_samples=5000)`
- Check if NLSQ warm-start is being used
- Verify data quality and model appropriateness

**Warning: "Low ESS" (effective sample size < 400)**

**Solutions:**
- Increase num_samples
- Use warm-start from NLSQ fit
- Check for multimodal posterior (may need longer chains)

**Error: "Too many divergences"**

**Solutions:**
- Use NLSQ warm-start (crucial for avoiding divergences)
- Increase adapt_step_size
- Check parameter bounds are reasonable
- Verify model is appropriate for data

### JAX Integration

**Key Points:**
- All numerical operations JAX-compatible (use `jax.numpy` instead of `numpy` internally)
- Models use `@jax.jit` for compilation
- Optimization uses `jax.grad` for automatic differentiation
- Convert data: `data.to_jax()` or `RheoData(..., use_jax=True)`
- 2-10x speedup on CPU, 20-100x with GPU
- **Always use safe_import_jax() in RheoJAX modules**

**Example:**
```python
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

@jax.jit
def objective(params):
    predictions = model_function(params)
    return jnp.sum((predictions - data)**2)

# NLSQ automatically uses JAX gradients
from rheojax.utils.optimization import nlsq_optimize
result = nlsq_optimize(objective, params, use_jax=True)
```

## Development Workflow

### Adding a New Model
1. Create file in `rheojax/models/` (e.g., `my_model.py`)
2. **Use safe JAX imports:**
   ```python
   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   ```
3. Inherit from `BaseModel` (`rheojax/core/base.py`)
4. Implement `_fit()` and `_predict()` methods
5. Register with `@ModelRegistry.register("my_model")`
6. Add tests in `tests/models/test_my_model.py`
7. Include numerical validation tests
8. Add docstring with equations and references

### Adding a New Transform
1. Create file in `rheojax/transforms/` (e.g., `my_transform.py`)
2. **Use safe JAX imports if needed**
3. Inherit from `BaseTransform` (`rheojax/core/base.py`)
4. Implement `transform()` method
5. Register with `@TransformRegistry.register("my_transform")`
6. Add tests in `tests/transforms/test_my_transform.py`

### Test Organization
- `tests/core/`: Core functionality (base, data, parameters, bayesian, float64 precision)
- `tests/models/`: Model implementations
- `tests/transforms/`: Transform implementations
- `tests/io/`: File readers/writers
- `tests/pipeline/`: Pipeline API (including BayesianPipeline)
- `tests/utils/`: Optimization utilities (NLSQ integration, compatibility checking, initialization)
- `tests/integration/`: End-to-end workflows (NLSQ → NUTS)
- `tests/validation/`: Validation against legacy packages (pyrheo, hermes-rheo)

### Markers for Test Selection
- `@pytest.mark.slow`: Long-running tests (skip with `-m "not slow"`)
- `@pytest.mark.gpu`: Requires GPU (skip if GPU unavailable)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.validation`: Validation against original packages
- `@pytest.mark.benchmark`: Performance benchmarks

## Important Notes

### Python Version
- **Requires Python 3.12+** (specified in pyproject.toml and CLAUDE.md)
- JAX 0.8.0 compatibility requires modern Python
- Runtime check in `rheojax/__init__.py` enforces version

### JAX Version Pinning
- JAX and jaxlib must be **exactly 0.8.0** and match versions
- CPU-only by default (works all platforms)
- GPU requires manual install to avoid platform conflicts

### NLSQ Dependency
- NLSQ >= 0.1.6 required for GPU-accelerated optimization
- Automatically configures JAX for float64 when imported first
- Provides 5-270x speedup over scipy-based optimization

### Float64 Precision (Critical)
- **NLSQ must be imported before JAX** - automatic in rheojax package
- **Always use safe_import_jax() in RheoJAX modules** - never import JAX directly
- Runtime checks enforce import order with helpful error messages
- Float64 precision essential for numerical stability

### Bayesian Inference Support (20/20 Models - 100%)

**All 20 models support complete Bayesian inference workflows**:
- All models inherit from BaseModel → BayesianMixin
- Full API: `fit()`, `fit_bayesian()`, `sample_prior()`, `get_credible_intervals()`
- Warm-start workflow: `fit()` (NLSQ) → `fit_bayesian()` (NUTS)
- Every model has `model_function()` for NumPyro NUTS sampling

**Complete model list with Bayesian support:**

Classical Viscoelastic (3):
- Maxwell, Zener, SpringPot

Flow Models (6):
- Bingham, PowerLaw, Herschel-Bulkley
- Carreau, Carreau-Yasuda, Cross

Fractional Models (11):
- fractional_burgers, fractional_jeffreys
- fractional_kelvin_voigt, fractional_kv_zener
- fractional_maxwell_gel, fractional_maxwell_liquid, fractional_maxwell_model
- fractional_poynting_thomson
- fractional_zener_ll, fractional_zener_sl, fractional_zener_ss

### Clean Commands Safety
- `make clean` and `make clean-all` preserve:
  - `.venv/` and `venv/` (virtual environments)
  - `.claude/` (Claude Code settings)
  - `.specify/` (Specify agent settings)
  - `agent-os/` (agent operating system files)
- Use `make clean-venv` to remove virtual environment (requires confirmation)

### Import Style
- **Always use explicit imports** (from user's global CLAUDE.md)
- Example: `from rheojax.core.base import BaseModel` (not `from rheojax.core import *`)
- **Never import JAX directly** - use `from rheojax.core.jax_config import safe_import_jax`

### Package Manager Support
- Makefile auto-detects: uv → conda/mamba → pip
- GPU installation uses pip regardless (conda JAX extras not supported)
- Works in conda environments using pip for JAX

### Protected Directories
Build/clean commands preserve:
- `.venv/`, `venv/`: Virtual environments
- `.claude/`: Claude Code configuration
- `.specify/`: Specify agent configuration
- `agent-os/`: Agent OS files (standards, specs, product docs)

## Performance Characteristics

### NLSQ Optimization
- **5-270x speedup** over scipy on CPU
- **Additional GPU acceleration** on CUDA-enabled systems
- Warm-start from good initial guesses improves convergence
- Typical convergence: 50-500 iterations for rheological models

### Bayesian Inference (NUTS)
- **2-5x faster convergence** with NLSQ warm-start vs cold start
- Typical settings: num_warmup=1000, num_samples=2000
- Good convergence: R-hat < 1.01, ESS > 400
- Warm-start reduces divergences significantly

### Model Validation
- All 20 models validated with NLSQ + NUTS workflow
- Float64 precision maintained throughout stack
- Convergence diagnostics computed for all parameters
- Credible intervals quantify parameter uncertainty
