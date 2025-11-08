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
    ├── initialization.py   # Smart parameter initialization facade (refactored in v0.2.0)
    └── initialization/     # Template Method initialization (NEW in v0.2.0)
        ├── base.py         # BaseInitializer abstract class + extract_frequency_features()
        └── fractional_*.py # 11 concrete initializers (one per fractional model)
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

**6. Template Method Pattern for Initialization**
- `BaseInitializer` abstract class enforces consistent initialization algorithm (NEW in v0.2.0)
- 11 concrete initializers for fractional models (one per model)
- Template method `initialize()` defines 5-step algorithm skeleton:
  1. Extract frequency features (common logic)
  2. Validate features (common validation)
  3. Estimate model-specific parameters (abstract methods)
  4. Clip to parameter bounds (common logic)
  5. Set parameters in ParameterSet (abstract method)
- Eliminates code duplication across models (49% code reduction in `initialization.py`)
- Performance overhead: <0.01% of total fitting time
- Location: `rheojax/utils/initialization/` (modular structure)
- Backward compatible: All public functions in `initialization.py` preserved

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

## Recent Features (Updated: 2025-11-07)

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
# Abstract base class enforces 5-step algorithm
from rheojax.utils.initialization.base import BaseInitializer

class BaseInitializer(ABC):
    def initialize(self, omega, G_star, param_set) -> bool:
        # Step 1: Extract frequency features (common logic)
        features = extract_frequency_features(omega, G_star)

        # Step 2: Validate features (common validation)
        if not self._validate_data(features):
            return False

        # Step 3: Estimate parameters (model-specific - abstract)
        estimated_params = self._estimate_parameters(features)

        # Step 4: Clip to bounds (common logic)
        clipped_params = self._clip_to_bounds(estimated_params, param_set)

        # Step 5: Set parameters (model-specific - abstract)
        self._set_parameters(param_set, clipped_params)

        return True

    @abstractmethod
    def _estimate_parameters(self, features: dict) -> dict:
        """Model-specific parameter estimation logic."""
        pass

    @abstractmethod
    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Model-specific parameter setting logic."""
        pass
```

**Concrete Initializer Example:**

```python
# rheojax/utils/initialization/fractional_zener_ss.py
class FractionalZenerSSInitializer(BaseInitializer):
    """Concrete initializer for FZSS model."""

    def _estimate_parameters(self, features: dict) -> dict:
        # Extract FZSS-specific parameters from features
        return {
            'Ge': features['low_plateau'],
            'Gm': features['high_plateau'] - features['low_plateau'],
            'tau_alpha': 1.0 / features['omega_mid'],
            'alpha': features['alpha_estimate']
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        # Set parameters safely (checks existence first)
        self._safe_set_parameter(param_set, 'Ge', clipped_params['Ge'])
        self._safe_set_parameter(param_set, 'Gm', clipped_params['Gm'])
        self._safe_set_parameter(param_set, 'tau_alpha', clipped_params['tau_alpha'])
        self._safe_set_parameter(param_set, 'alpha', clipped_params['alpha'])
```

**Model Integration:**

Each fractional model instantiates its concrete initializer in `_fit()`:

```python
# rheojax/models/fractional_zener_ss.py (updated in v0.2.0)
def _fit(self, X, y, **kwargs):
    test_mode = kwargs.get('test_mode', 'relaxation')

    # Smart initialization for oscillation mode (Issue #9)
    if test_mode == TestMode.OSCILLATION:
        from rheojax.utils.initialization.fractional_zener_ss import (
            FractionalZenerSSInitializer
        )
        initializer = FractionalZenerSSInitializer()
        success = initializer.initialize(np.array(X), np.array(y), self.parameters)
        if success:
            logging.debug("Smart initialization applied from frequency-domain features")

    # ... continue with optimization using initialized parameters
```

**Backward Compatibility:**

The public API in `rheojax/utils/initialization.py` is preserved for 100% backward compatibility:

```python
# rheojax/utils/initialization.py (facade)
def initialize_fractional_zener_ss(omega, G_star, param_set) -> bool:
    """Backward-compatible facade that delegates to concrete initializer."""
    initializer = FractionalZenerSSInitializer()
    return initializer.initialize(omega, G_star, param_set)
```

**Performance:**
- Initialization overhead: <0.01% of total fitting time (187 μs vs 1.76s total)
- Code reduction: 49% (932 → 471 lines in `initialization.py`)
- All 11 fractional models covered with 22/22 tests passing

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

BayesianPipeline provides comprehensive MCMC diagnostics through ArviZ integration. All 6 diagnostic methods support the fluent API pattern with `show` parameter and `.save_figure()` chaining.

**1. Pair Plot (`plot_pair`)** - Parameter Correlations
```python
# Identify parameter dependencies and non-identifiability
pipeline.plot_pair(
    var_names=['G0', 'eta'],    # Specific parameters (or None for all)
    kind='scatter',              # 'scatter', 'kde', or 'hexbin'
    divergences=True             # Highlight problematic regions
)
```
Use for: Detecting correlations, funnel geometry, multimodal posteriors

**2. Forest Plot (`plot_forest`)** - Credible Intervals
```python
# Compare parameter magnitudes and uncertainties
pipeline.plot_forest(
    hdi_prob=0.95,              # 0.68 (1σ), 0.95 (2σ), 0.997 (3σ)
    combined=True                # Combine multiple chains
)
```
Use for: Quick parameter comparison, uncertainty assessment

**3. Energy Plot (`plot_energy`)** - NUTS Sampler Diagnostic
```python
# Identify problematic posterior geometry
pipeline.plot_energy()
```
Use for: Detecting heavy tails, funnels, poor parameterizations
Note: Requires multi-chain MCMC (not single-chain)

**4. Autocorrelation Plot (`plot_autocorr`)** - Mixing Diagnostic
```python
# Check MCMC mixing quality
pipeline.plot_autocorr(
    max_lag=100,                # Lag length to display
    combined=False               # Per-chain or combined
)
```
Use for: Assessing mixing, determining if more samples needed
Goal: Autocorrelation drops to ~0 within few dozen lags

**5. Rank Plot (`plot_rank`)** - Convergence Diagnostic
```python
# Modern convergence diagnostic (alternative to trace plots)
pipeline.plot_rank()
```
Use for: Detecting non-convergence, chain sticking, insufficient mixing
Goal: Uniform histogram across all bins

**6. ESS Plot (`plot_ess`)** - Effective Sample Size
```python
# Quantify sampling efficiency
pipeline.plot_ess(
    kind='local'                # 'local', 'quantile', or 'evolution'
)
```
Use for: Assessing which parameters need more sampling
Goal: ESS > 400 for bulk and tail estimates

**Converting to ArviZ InferenceData**
```python
# Access ArviZ InferenceData directly for advanced analysis
idata = pipeline._bayesian_result.to_inference_data()

# Use any ArviZ function
import arviz as az
az.plot_trace(idata)
az.summary(idata)
```

#### BayesianPlotter: Direct Access to Plotting

**New in v0.2.0:** BayesianPlotter can be accessed directly via the `.plotter` property for advanced plotting control, while maintaining 100% backward compatibility with the original pipeline API.

**Old API (backward compatible):**
```python
# Traditional fluent pipeline API - still works exactly as before
(pipeline
    .fit_bayesian(num_samples=2000, num_warmup=1000)
    .plot_pair(show=False)          # Returns pipeline for chaining
    .plot_forest(show=False)
    .save_figure('diagnostics.png'))
```

**New API (direct plotter access):**
```python
# Access BayesianPlotter directly for finer control
pipeline.fit_bayesian(num_samples=2000, num_warmup=1000)

plotter = pipeline.plotter  # Lazy-loaded BayesianPlotter instance

# Fluent API on plotter itself
(plotter
    .plot_pair(var_names=['G0', 'eta'], show=False)
    .plot_forest(hdi_prob=0.95, show=False)
    .save_figure('diagnostics.png'))  # Returns plotter for chaining
```

**When to use which API:**
- **Old API (`pipeline.plot_*`)**: When chaining with pipeline methods like `.load()`, `.fit_nlsq()`, `.fit_bayesian()`, `.save()`
- **New API (`pipeline.plotter.plot_*`)**: When you need multiple diagnostic plots without mixing pipeline operations, or when importing from `rheojax.visualization import BayesianPlotter` for standalone use

**Standalone BayesianPlotter usage:**
```python
from rheojax.visualization import BayesianPlotter

# After running fit_bayesian() on any model
result = model.fit_bayesian(t, G_data, num_samples=2000)

# Create standalone plotter
plotter = BayesianPlotter(result)

# Generate diagnostic plots
(plotter
    .plot_pair(divergences=True, show=False)
    .plot_autocorr(max_lag=100, show=False)
    .save_figure('model_diagnostics.png'))
```

### Data Transforms: Mastercurve API

Rheo provides comprehensive time-temperature superposition (TTS) capabilities through the `Mastercurve` transform with enhanced API for plotting and analysis.

#### Basic Mastercurve Creation

```python
from rheojax.transforms.mastercurve import Mastercurve
from rheojax.io.readers import auto_read

# Load multi-temperature datasets
temps = [273.15, 298.15, 323.15]  # Kelvin
datasets = []
for temp_file in temp_files:
    data = auto_read(temp_file)
    # Ensure temperature is in metadata
    data.metadata['temperature'] = get_temperature(temp_file)
    datasets.append(data)

# Create Mastercurve transform
mc = Mastercurve(
    reference_temp=298.15,  # Reference temperature (K)
    method='wlf',           # 'wlf', 'arrhenius', or 'manual'
    C1=17.44,               # WLF parameter (optional, uses default)
    C2=51.6,                # WLF parameter in K (optional)
    vertical_shift=False,   # Apply vertical shifting
    optimize_shifts=True    # Optimize shift factors
)
```

#### Creating Mastercurves (Two API Options)

**Option 1: Using `create_mastercurve()` (Explicit)**
```python
# Create mastercurve without returning shift factors
mastercurve = mc.create_mastercurve(datasets)

# Access shift factors from metadata
shift_factors = mastercurve.metadata['shift_factors']
print(f"Shift factors: {shift_factors}")
```

**Option 2: Using `transform()` (Recommended for Plotting)**
```python
# Transform returns both mastercurve and shift factors
mastercurve, shift_factors = mc.transform(datasets)

# shift_factors is a dict: {temperature: shift_factor}
print(f"At 273.15 K: a_T = {shift_factors[273.15]}")
print(f"At 298.15 K: a_T = {shift_factors[298.15]}")  # Should be ~1.0
```

#### Retrieving Model Parameters

**Get WLF Parameters**
```python
# After creating WLF-based mastercurve
wlf_params = mc.get_wlf_parameters()

C1 = wlf_params['C1']       # WLF constant C1
C2 = wlf_params['C2']       # WLF constant C2 (K)
T_ref = wlf_params['T_ref'] # Reference temperature (K)

print(f"WLF Parameters: C1={C1:.2f}, C2={C2:.2f} K, T_ref={T_ref:.2f} K")
```

**Get Arrhenius Parameters**
```python
# For Arrhenius-based mastercurve
mc_arr = Mastercurve(reference_temp=298.15, method='arrhenius', E_a=50000)
mastercurve, shifts = mc_arr.transform(datasets)

arr_params = mc_arr.get_arrhenius_parameters()
E_a = arr_params['E_a']     # Activation energy (J/mol)
T_ref = arr_params['T_ref'] # Reference temperature (K)

print(f"Arrhenius: E_a={E_a:.0f} J/mol, T_ref={T_ref:.2f} K")
```

#### Getting Shift Factors as Arrays (For Plotting)

**After Mastercurve Creation**
```python
# Get shift factors as sorted NumPy arrays
temps_array, shifts_array = mc.get_shift_factors_array()

# temps_array: temperatures in Kelvin (sorted)
# shifts_array: corresponding shift factors

# Convert to Celsius for plotting
temps_C = temps_array - 273.15

# Plot WLF fit
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(temps_C, np.log10(shifts_array), 'o-', label='WLF fit')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=mc.T_ref - 273.15, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Temperature (°C)')
plt.ylabel('log₁₀(aₜ)')
plt.title('Time-Temperature Shift Factors')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

**Generate Smooth WLF Curve**
```python
# Create smooth temperature range for plotting
temps_range = np.linspace(250, 350, 100)  # Kelvin
temps_smooth, shifts_smooth = mc.get_shift_factors_array(temps_range)

# Plot smooth WLF curve with data points
plt.plot(temps_smooth - 273.15, np.log10(shifts_smooth), '-',
         label='WLF model', linewidth=2)
plt.plot(temps_array - 273.15, np.log10(shifts_array), 'o',
         label='Data points', markersize=8)
plt.xlabel('Temperature (°C)')
plt.ylabel('log₁₀(aₜ)')
plt.legend()
```

#### Complete Workflow Example

```python
from rheojax.transforms.mastercurve import Mastercurve
import numpy as np
import matplotlib.pyplot as plt

# 1. Create mastercurve
mc = Mastercurve(reference_temp=298.15, method='wlf')
mastercurve, shift_factors = mc.transform(datasets)

# 2. Get WLF parameters
wlf_params = mc.get_wlf_parameters()
print(f"Fitted WLF: C1={wlf_params['C1']:.2f}, C2={wlf_params['C2']:.2f} K")

# 3. Get shift factors for plotting
temps, shifts = mc.get_shift_factors_array()

# 4. Create publication-quality plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Shift factors vs temperature
temps_C = temps - 273.15
ax1.plot(temps_C, np.log10(shifts), 'o-', markersize=8, linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=wlf_params['T_ref'] - 273.15, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('log₁₀(aₜ)')
ax1.set_title('Time-Temperature Shift Factors')
ax1.grid(True, alpha=0.3)

# Right: Mastercurve
ax2.loglog(mastercurve.x, mastercurve.y, 'o', alpha=0.6)
ax2.set_xlabel('Shifted Frequency (rad/s)')
ax2.set_ylabel("G' (Pa)")
ax2.set_title(f'Master Curve (T_ref = {wlf_params["T_ref"] - 273.15:.1f}°C)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Save results
mastercurve.save('mastercurve.hdf5')
```

#### Optimizing WLF Parameters

```python
# Optimize WLF parameters to minimize overlap error
C1_opt, C2_opt = mc.optimize_wlf_parameters(
    datasets,
    initial_C1=17.0,
    initial_C2=50.0
)

print(f"Optimized WLF: C1={C1_opt:.2f}, C2={C2_opt:.2f} K")

# Compute overlap error
overlap_error = mc.compute_overlap_error(datasets)
print(f"Overlap error: {overlap_error:.2e}")
```

#### Key Features Summary

**`Mastercurve` Transform Methods:**
- `transform(data)`: Transform single dataset or create mastercurve from list
- `create_mastercurve(datasets, merge=True, return_shifts=False)`: Explicit mastercurve creation
- `get_wlf_parameters()`: Get WLF constants (C1, C2, T_ref)
- `get_arrhenius_parameters()`: Get Arrhenius parameters (E_a, T_ref)
- `get_shift_factors_array(temperatures=None)`: Get shift factors as NumPy arrays
- `optimize_wlf_parameters(datasets)`: Optimize WLF constants
- `compute_overlap_error(datasets)`: Quantify mastercurve quality
- `set_manual_shifts(shift_factors)`: Set manual shift factors

**Best Practices:**
1. Always ensure `temperature` is in dataset metadata
2. Use `transform()` when you need shift factors for plotting
3. Use `get_shift_factors_array()` for creating smooth plots
4. Verify `a_T ≈ 1.0` at reference temperature
5. Check overlap error to assess mastercurve quality
6. Consider optimizing WLF parameters for better fit

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
