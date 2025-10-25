# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rheo is a JAX-accelerated rheological analysis package providing a unified framework for analyzing experimental rheology data. It integrates classical rheological models with modern data transforms, offering 2-10x performance improvements through JAX + GPU acceleration.

**Technology Stack:**
- Python 3.12+ (3.8-3.11 NOT supported)
- JAX 0.8.0 for automatic differentiation and GPU acceleration
- NumPy, SciPy for numerical operations
- NumPyro for probabilistic programming
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
│   ├── optimization.py     # JAX-compatible optimization with automatic gradients
│   └── mittag_leffler.py  # Mittag-Leffler functions (1 and 2-parameter)
└── legacy/            # Original pyrheo and hermes-rheo code (for validation)
```

### Key Design Patterns

**1. BaseModel Pattern**
- All models inherit from `BaseModel` (rheo/core/base.py)
- Implements scikit-learn compatible API: `.fit(X, y)`, `.predict(X)`
- JAX-compatible: accepts both NumPy and JAX arrays
- Internal methods: `_fit()`, `_predict()` for subclass implementation

**2. RheoData Container**
- Wraps data with metadata (units, domain, test_mode)
- JAX-compatible via `.to_jax()` method
- Operations: `.smooth()`, `.resample()`, `.derivative()`
- Automatic test mode detection: relaxation, creep, oscillation, rotation

**3. Parameter System**
- `Parameter`: Individual parameter with value, bounds, units
- `ParameterSet`: Collection of parameters for a model
- `ParameterOptimizer`: JAX-accelerated optimization with auto-gradients
- Type-safe with bounds validation

**4. Pipeline API**
- Fluent interface: `Pipeline().load().fit().plot().save()`
- Pre-configured workflows: `MastercurvePipeline`, `ModelComparisonPipeline`
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

### JAX Integration

**Key Points:**
- All numerical operations JAX-compatible (use `jax.numpy` instead of `numpy` internally)
- Models use `@jax.jit` for compilation
- Optimization uses `jax.grad` for automatic differentiation
- Convert data: `data.to_jax()` or `RheoData(..., use_jax=True)`
- 2-10x speedup on CPU, 20-100x with GPU

**Example:**
```python
import jax
import jax.numpy as jnp
from rheo.utils.optimization import nlsq_optimize

@jax.jit
def objective(params):
    predictions = model_function(params)
    return jnp.sum((predictions - data)**2)

result = nlsq_optimize(objective, params, use_jax=True)
```

## Development Workflow

### Adding a New Model
1. Create file in `rheo/models/` (e.g., `my_model.py`)
2. Inherit from `BaseModel` (`rheo/core/base.py`)
3. Implement `_fit()` and `_predict()` methods
4. Register with `@ModelRegistry.register("my_model")`
5. Add tests in `tests/models/test_my_model.py`
6. Include numerical validation tests
7. Add docstring with equations and references

### Adding a New Transform
1. Create file in `rheo/transforms/` (e.g., `my_transform.py`)
2. Inherit from `BaseTransform` (`rheo/core/base.py`)
3. Implement `transform()` method
4. Register with `@TransformRegistry.register("my_transform")`
5. Add tests in `tests/transforms/test_my_transform.py`

### Test Organization
- `tests/core/`: Core functionality (base, data, parameters)
- `tests/models/`: Model implementations
- `tests/transforms/`: Transform implementations
- `tests/io/`: File readers/writers
- `tests/pipeline/`: Pipeline API
- `tests/integration/`: End-to-end workflows
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
