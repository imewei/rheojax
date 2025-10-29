# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rheo is a JAX-accelerated rheological analysis package providing a unified framework for analyzing experimental rheology data. It integrates classical rheological models with modern data transforms, offering 2-10x performance improvements through JAX + GPU acceleration.

**Technology Stack:**
- Python 3.12+ (3.8-3.11 NOT supported)
- JAX 0.8.0 for automatic differentiation and GPU acceleration
- NLSQ 0.1.6+ for GPU-accelerated nonlinear least squares optimization
- NumPyro for Bayesian inference (MCMC NUTS sampling)
- NumPy, SciPy for numerical operations
- Matplotlib for visualization
- h5py, pandas, openpyxl for I/O

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
pytest -m validation    # Validation against legacy packages
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
rheo/
├── core/               # Core abstractions and common functionality
│   ├── base.py        # BaseModel, BaseTransform abstract classes
│   ├── data.py        # RheoData wrapper (JAX-compatible)
│   ├── parameters.py  # Parameter, ParameterSet, ParameterOptimizer
│   ├── bayesian.py    # BayesianMixin for NumPyro NUTS sampling
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
│   ├── bayesian.py    # BayesianPipeline (NLSQ → NUTS workflow)
│   ├── workflows.py   # Pre-configured pipelines (mastercurve, model comparison)
│   ├── builder.py     # Programmatic pipeline construction
│   └── batch.py       # Batch processing multiple datasets
├── io/
│   ├── readers/       # TRIOS, CSV, Excel, Anton Paar, auto-detection
│   └── writers/       # HDF5 (full fidelity), Excel
├── visualization/     # Publication-quality plotting
│   ├── plotter.py     # Automatic plot type selection
│   └── templates.py   # Three styles: default, publication, presentation
├── utils/
│   ├── optimization.py     # NLSQ-based optimization (5-270x speedup)
│   └── mittag_leffler.py  # Mittag-Leffler functions (1 and 2-parameter)
└── legacy/            # Original pyrheo and hermes-rheo code (for validation)
```

### Key Design Patterns

**1. BaseModel Pattern**
- All models inherit from `BaseModel` (rheo/core/base.py)
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

### Test Mode System

Located in `rheo/core/test_modes.py`. Auto-detects:
- **Relaxation**: stress decay over time
- **Creep**: strain increase under constant stress
- **Oscillation**: frequency-domain (G', G", tan δ)
- **Rotation**: flow curves (viscosity vs shear rate)

Detection based on data characteristics (monotonicity, domain, column names). Used for automatic plot selection and model validation.

### Float64 Precision Enforcement

**Critical Requirement:** Rheo enforces float64 precision throughout the entire JAX stack to ensure numerical stability for rheological calculations. This is accomplished through mandatory import order.

#### Import Order Requirement

**NLSQ must be imported BEFORE JAX to enable float64 globally.**

The package automatically handles this in `rheo/__init__.py`:
```python
# rheo/__init__.py (automatic - no user action needed)
import nlsq  # MUST come before any JAX imports
# ... JAX is imported later by models/utils
```

#### Safe JAX Import Pattern for Developers

**DO NOT import JAX directly in Rheo modules.** Always use the safe import mechanism:

```python
# CORRECT - Safe import (enforces float64)
from rheo.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

# INCORRECT - Never do this in Rheo modules
import jax  # Will raise ImportError if NLSQ not imported first
import jax.numpy as jnp
```

All 20 models in `rheo/models/` use the safe import pattern. If adding a new model or transform, always use `safe_import_jax()`.

#### Float64 Verification

To verify JAX is operating in float64 mode:
```python
from rheo.core.jax_config import verify_float64
verify_float64()  # Raises exception if float64 not enabled
```

#### Rationale

Float64 precision is essential for:
- **Numerical stability** in optimization with tight tolerances
- **Accurate parameter estimation** for rheological models
- **Reliable MCMC sampling** in Bayesian inference
- **Gradient computation** in JAX automatic differentiation

NLSQ automatically configures JAX for float64 when imported, making the import order critical.

### NLSQ + NumPyro Workflow

Rheo implements a two-step optimization workflow combining fast NLSQ point estimation with Bayesian NUTS inference.

#### Step 1: NLSQ Optimization (Fast Point Estimation)

NLSQ provides GPU-accelerated nonlinear least squares with 5-270x speedup over scipy:

```python
from rheo.models.maxwell import Maxwell
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
from rheo.models.maxwell import Maxwell
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

For complex workflows, use BayesianPipeline:

```python
from rheo.pipeline.bayesian import BayesianPipeline

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
```

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
from rheo.core.jax_config import safe_import_jax
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
from rheo.core.jax_config import verify_float64
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
- **Always use safe_import_jax() in Rheo modules**

**Example:**
```python
from rheo.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

@jax.jit
def objective(params):
    predictions = model_function(params)
    return jnp.sum((predictions - data)**2)

# NLSQ automatically uses JAX gradients
from rheo.utils.optimization import nlsq_optimize
result = nlsq_optimize(objective, params, use_jax=True)
```

## Development Workflow

### Adding a New Model
1. Create file in `rheo/models/` (e.g., `my_model.py`)
2. **Use safe JAX imports:**
   ```python
   from rheo.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   ```
3. Inherit from `BaseModel` (`rheo/core/base.py`)
4. Implement `_fit()` and `_predict()` methods
5. Register with `@ModelRegistry.register("my_model")`
6. Add tests in `tests/models/test_my_model.py`
7. Include numerical validation tests
8. Add docstring with equations and references

### Adding a New Transform
1. Create file in `rheo/transforms/` (e.g., `my_transform.py`)
2. **Use safe JAX imports if needed**
3. Inherit from `BaseTransform` (`rheo/core/base.py`)
4. Implement `transform()` method
5. Register with `@TransformRegistry.register("my_transform")`
6. Add tests in `tests/transforms/test_my_transform.py`

### Test Organization
- `tests/core/`: Core functionality (base, data, parameters, bayesian, float64 precision)
- `tests/models/`: Model implementations
- `tests/transforms/`: Transform implementations
- `tests/io/`: File readers/writers
- `tests/pipeline/`: Pipeline API (including BayesianPipeline)
- `tests/utils/`: Optimization utilities (NLSQ integration)
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
- Runtime check in `rheo/__init__.py` enforces version

### JAX Version Pinning
- JAX and jaxlib must be **exactly 0.8.0** and match versions
- CPU-only by default (works all platforms)
- GPU requires manual install to avoid platform conflicts

### NLSQ Dependency
- NLSQ >= 0.1.6 required for GPU-accelerated optimization
- Automatically configures JAX for float64 when imported first
- Provides 5-270x speedup over scipy-based optimization

### Float64 Precision (Critical)
- **NLSQ must be imported before JAX** - automatic in rheo package
- **Always use safe_import_jax() in Rheo modules** - never import JAX directly
- Runtime checks enforce import order with helpful error messages
- Float64 precision essential for numerical stability

### All Models Support Bayesian Inference
- All 20 models inherit from BaseModel → BayesianMixin
- Every model gains: `fit_bayesian()`, `sample_prior()`, `get_credible_intervals()`
- Warm-start workflow: `fit()` (NLSQ) then `fit_bayesian()` (NUTS)
- No model-specific changes needed for Bayesian capabilities

### Clean Commands Safety
- `make clean` and `make clean-all` preserve:
  - `.venv/` and `venv/` (virtual environments)
  - `.claude/` (Claude Code settings)
  - `.specify/` (Specify agent settings)
  - `agent-os/` (agent operating system files)
- Use `make clean-venv` to remove virtual environment (requires confirmation)

### Import Style
- **Always use explicit imports** (from user's global CLAUDE.md)
- Example: `from rheo.core.base import BaseModel` (not `from rheo.core import *`)
- **Never import JAX directly** - use `from rheo.core.jax_config import safe_import_jax`

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
