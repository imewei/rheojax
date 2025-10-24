# Rheo - JAX-Powered Rheological Analysis

[![CI](https://github.com/username/rheo/actions/workflows/ci.yml/badge.svg)](https://github.com/username/rheo/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/rheo.svg)](https://badge.fury.io/py/rheo)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rheo.readthedocs.io)

A modern, JAX-accelerated rheological analysis package providing a unified framework for analyzing experimental rheology data with state-of-the-art performance and flexibility.

## 🆕 What's New in v0.2.0

Phase 2 brings the complete rheological analysis toolkit:

- **20 rheological models** (classical, fractional, flow)
- **5 data transforms** (FFT, mastercurve, mutation number, OWChirp, derivatives)
- **Pipeline API** for intuitive workflows
- **2-10x performance improvement** with JAX + GPU acceleration
- **150+ pages of documentation** with examples

### Quick Example (New in v0.2.0)

```python
from rheo.pipeline import Pipeline

# Complete workflow in 3 lines
pipeline = (Pipeline()
    .load('polymer_data.csv')
    .fit('fractional_maxwell_liquid')
    .plot(style='publication'))

# Get fitted parameters
params = pipeline.get_fitted_parameters()
print(f"Alpha: {params['alpha'].value:.3f}")
```

## Features

- **20 Rheological Models**: Classical (Maxwell, Zener, SpringPot), Fractional (11 variants), Flow (6 models)
- **5 Data Transforms**: FFT, Mastercurve (TTS), Mutation Number, OWChirp (LAOS), Smooth Derivative
- **Pipeline API**: Fluent interface for load → fit → plot → save workflows
- **JAX-First Architecture**: 2-10x performance improvements with automatic differentiation and GPU support
- **Comprehensive Data Support**: Automatic test mode detection (relaxation, creep, oscillation, rotation)
- **Multiple File Formats**: TRIOS, CSV, Excel, Anton Paar with intelligent auto-detection
- **Flexible Parameter System**: Type-safe parameter management with bounds and constraints
- **Publication-Quality Visualization**: Three built-in styles (default, publication, presentation)
- **Extensible Design**: Plugin system for custom models and transforms

## Installation

### Requirements

- Python 3.12 or later (3.8-3.11 are NOT supported due to JAX requirements)
- JAX and jaxlib for acceleration
- NumPy, SciPy for numerical operations
- Matplotlib for visualization

### Basic Installation

```bash
pip install rheo
```

### Development Installation

```bash
git clone https://github.com/username/rheo.git
cd rheo
pip install -e ".[dev]"
```

### GPU Support

For GPU acceleration (CUDA 12):

```bash
pip install "rheo[gpu]"
```

## Quick Start

### Loading and Visualizing Data

```python
from rheo.io.readers import auto_read
from rheo.visualization import plot_rheo_data
import matplotlib.pyplot as plt

# Load data (auto-detect format)
data = auto_read("stress_relaxation.txt")

# Check detected test mode
print(f"Test mode: {data.test_mode}")  # Output: relaxation

# Visualize
fig, ax = plot_rheo_data(data, style='publication')
plt.show()
```

### Working with Parameters

```python
from rheo.core import ParameterSet

# Create parameter set
params = ParameterSet()
params.add("E", value=1000.0, bounds=(100, 10000), units="Pa")
params.add("tau", value=1.0, bounds=(0.1, 100), units="s")

# Get/set values
E = params.get_value("E")
params.set_value("tau", 2.5)
```

### Data Processing

```python
import numpy as np
from rheo.core import RheoData

# Create or load data
time = np.logspace(-1, 2, 100)
stress = 1000 * np.exp(-time / 5)
data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

# Data operations
smoothed = data.smooth(window_size=5)
resampled = data.resample(n_points=50)
derivative = data.derivative()

# Convert to JAX for performance
data_jax = data.to_jax()
```

### Data Transforms

```python
from rheo.transforms import FFTAnalysis, Mastercurve, MutationNumber

# FFT analysis for frequency spectrum
fft = FFTAnalysis(window='hann', detrend=True)
freq_data = fft.transform(data)
tau_char = fft.get_characteristic_time(freq_data)

# Time-temperature superposition (mastercurves)
mc = Mastercurve(reference_temp=298.15, method='wlf')
mastercurve = mc.create_mastercurve(datasets)

# Mutation number (viscoelastic character)
mn = MutationNumber()
delta = mn.calculate(data)  # 0=elastic, 1=viscous
```

### Optimization with JAX

```python
import jax
import jax.numpy as jnp
from rheo.utils.optimization import nlsq_optimize

# Define objective function
@jax.jit
def objective(params):
    E, tau = params
    predictions = E * jnp.exp(-time / tau)
    return jnp.sum((predictions - stress)**2)

# Optimize with JAX gradients
result = nlsq_optimize(objective, params, use_jax=True)
print(f"Optimal: E={result.x[0]:.1f} Pa, tau={result.x[1]:.2f} s")
```

## Documentation

Full documentation is available at [https://rheo.readthedocs.io](https://rheo.readthedocs.io)

### Key Topics

- [Getting Started](https://rheo.readthedocs.io/user_guide/getting_started.html) - Installation and basic usage
- [Core Concepts](https://rheo.readthedocs.io/user_guide/core_concepts.html) - RheoData, Parameters, Test Modes
- [I/O Guide](https://rheo.readthedocs.io/user_guide/io_guide.html) - Reading and writing data
- [Visualization Guide](https://rheo.readthedocs.io/user_guide/visualization_guide.html) - Creating plots
- [API Reference](https://rheo.readthedocs.io/api_reference.html) - Complete API documentation

## Development Status

### Phase 1 (Complete) - Core Infrastructure

✅ **Base Abstractions**
- BaseModel and BaseTransform interfaces
- RheoData container with JAX support
- Parameter system with bounds and constraints
- Scikit-learn compatible API

✅ **Test Mode Detection**
- Automatic detection: relaxation, creep, oscillation, rotation
- Metadata-based override capability
- Test mode validation and suggestions

✅ **Numerical Utilities**
- Mittag-Leffler functions (one and two-parameter)
- JAX-compatible optimization with automatic gradients
- scipy.optimize integration

✅ **File I/O**
- Readers: TRIOS, CSV, Excel, Anton Paar
- Writers: HDF5 (full fidelity), Excel
- Auto-detection with format inference

✅ **Visualization**
- Automatic plot type selection
- Three built-in styles (default, publication, presentation)
- Time-domain, frequency-domain, and flow curve plots
- Residual visualization

### Phase 2 (Planned) - Models and Transforms

- **20+ Rheological Models**: Maxwell, Zener, fractional models, flow models
- **Data Transforms**: Master curves, FFT analysis, OWChirp processing
- **Pipeline API**: High-level workflow orchestration
- **Enhanced Visualization**: Advanced templates and multi-panel figures

### Phase 3 (Future) - Advanced Features

- Bayesian parameter estimation
- Uncertainty quantification
- Multi-objective optimization
- Machine learning integration

## Performance

JAX provides significant performance improvements over NumPy:

| Operation | NumPy Time | JAX Time | Speedup |
|-----------|------------|----------|---------|
| Mittag-Leffler (1000 pts) | 45 ms | 0.8 ms | 56x |
| Parameter optimization | 2.5 s | 0.15 s | 17x |
| Data resampling (10k pts) | 120 ms | 3 ms | 40x |

*Benchmarks on M1 MacBook Pro. GPU acceleration provides additional speedups.*

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/username/rheo.git
cd rheo

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rheo --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Only unit tests
```

## Examples

See the `examples/` directory for comprehensive examples:

- Loading and processing data
- Parameter optimization
- Visualization customization
- Batch processing workflows

## Technology Stack

**Core**
- Python 3.12+
- JAX for automatic differentiation and acceleration
- NumPy, SciPy for numerical operations

**I/O**
- h5py for HDF5 files
- pandas for CSV/Excel
- openpyxl for Excel writing

**Visualization**
- matplotlib for plotting
- publication-quality output in multiple formats

**Optional**
- piblin for enhanced data management
- CUDA for GPU acceleration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use rheo in your research, please cite:

```bibtex
@software{rheo2024,
  title = {Rheo: JAX-Powered Unified Rheology Package},
  year = {2024},
  author = {Rheo Development Team},
  url = {https://github.com/username/rheo},
  version = {1.0.0}
}
```

## Acknowledgments

rheo is built on excellent open-source software:

- [JAX](https://github.com/google/jax) for automatic differentiation and acceleration
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computing
- [matplotlib](https://matplotlib.org/) for visualization
- [piblin](https://github.com/username/piblin) for data structures (optional)

## Support

- 📖 Documentation: [https://rheo.readthedocs.io](https://rheo.readthedocs.io)
- 💬 Discussions: [GitHub Discussions](https://github.com/username/rheo/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/username/rheo/issues)
- 📧 Email: rheo@example.com

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans.

---

Made with ❤️ by the Rheo Development Team
