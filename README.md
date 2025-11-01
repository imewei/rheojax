# RheoJAX - JAX-Powered Rheological Analysis

[![CI](https://github.com/imewei/rheojax/actions/workflows/ci.yml/badge.svg)](https://github.com/imewei/rheojax/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/rheojax.svg)](https://badge.fury.io/py/rheojax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rheojax.readthedocs.io)

A modern, JAX-accelerated rheological analysis package providing a unified framework for analyzing experimental rheology data with state-of-the-art performance, Bayesian inference, and comprehensive tutorial notebooks.

## 🆕 What's New in v0.2.0

Complete rheological analysis toolkit with Bayesian inference:

- **20 rheological models** (classical, fractional, flow)
- **5 data transforms** (FFT, mastercurve, mutation number, OWChirp, derivatives)
- **Bayesian inference** with NumPyro NUTS sampling + ArviZ diagnostics
- **Pipeline API** for intuitive workflows with NLSQ → NUTS warm-start
- **20 tutorial notebooks** covering basic to advanced workflows
- **5-270x performance improvement** with NLSQ + GPU acceleration

### Quick Example (New in v0.2.0)

```python
from rheojax.pipeline import BayesianPipeline

# Complete Bayesian workflow in 4 lines
pipeline = (BayesianPipeline()
    .load('polymer_data.csv')
    .fit_nlsq('fractional_maxwell_liquid')  # Fast point estimate
    .fit_bayesian(num_samples=2000))        # NUTS with warm-start

# Get credible intervals
intervals = pipeline.get_credible_intervals()
print(f"Alpha 95% CI: {intervals['alpha']}")

# Comprehensive diagnostics
pipeline.plot_pair().plot_trace().plot_forest()
```

## Features

### Core Capabilities
- **20 Rheological Models**: Classical (Maxwell, Zener, SpringPot), Fractional (11 variants), Flow (6 models)
- **5 Data Transforms**: FFT, Mastercurve (TTS), Mutation Number, OWChirp (LAOS), Smooth Derivative
- **Bayesian Inference**: All 20 models support NumPyro NUTS sampling with NLSQ warm-start
- **Pipeline API**: Fluent interface for load → fit → plot → save workflows
- **JAX-First Architecture**: 5-270x performance improvements with automatic differentiation and GPU support

### Data & I/O
- **Comprehensive Data Support**: Automatic test mode detection (relaxation, creep, oscillation, rotation)
- **Multiple File Formats**: TRIOS, CSV, Excel, Anton Paar with intelligent auto-detection
- **Flexible Parameter System**: Type-safe parameter management with bounds and constraints

### Visualization & Diagnostics
- **Publication-Quality Visualization**: Three built-in styles (default, publication, presentation)
- **ArviZ Diagnostic Suite**: 6 plot types (pair, forest, energy, autocorr, rank, ESS) for MCMC quality
- **Extensible Design**: Plugin system for custom models and transforms

### Tutorial Notebooks (New!)
- **20 Comprehensive Notebooks**: Organized in 4 learning phases
  - `examples/basic/` - 5 notebooks covering fundamental models
  - `examples/transforms/` - 5 notebooks for data transforms and analysis
  - `examples/bayesian/` - 5 notebooks for Bayesian inference workflows
  - `examples/advanced/` - 5 notebooks for production-ready patterns

## Installation

### Requirements

- Python 3.12 or later (3.8-3.11 are NOT supported due to JAX 0.8.0 requirements)
- JAX and jaxlib for acceleration
- NLSQ for GPU-accelerated optimization
- NumPyro for Bayesian inference
- ArviZ for Bayesian diagnostics

### Basic Installation

```bash
pip install rheojax
```

### Development Installation

```bash
git clone https://github.com/imewei/rheojax.git
cd rheojax
pip install -e ".[dev]"
```

### GPU Installation (Linux Only) ⚡

**Performance Impact:** 20-100x speedup for large datasets (>10K points)

#### Option 1: Quick Install via Makefile (Recommended)

From the repository:

```bash
git clone https://github.com/imewei/rheojax.git
cd rheojax
make install-jax-gpu  # Automatically handles uninstall + GPU install
```

This single command:
- Uninstalls CPU-only JAX
- Installs GPU-enabled JAX with CUDA 12 support
- Verifies GPU detection

#### Option 2: Manual Installation

For GPU-accelerated computation on Linux systems with CUDA 12+:

```bash
# Step 1: Uninstall CPU-only version
pip uninstall -y jax jaxlib

# Step 2: Install JAX with CUDA support
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Step 3: Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Should show: [cuda(id=0)] instead of [CpuDevice(id=0)]
```

**Why separate installation?** JAX with CUDA support is Linux-specific and requires system CUDA 12.1-12.9 pre-installed. Separating the installation avoids dependency conflicts on macOS/Windows.

#### GPU Troubleshooting

**Issue:** Warning "An NVIDIA GPU may be present... but a CUDA-enabled jaxlib is not installed"

**Solution:**
```bash
# 1. Check GPU hardware
nvidia-smi  # Should show your GPU

# 2. Check CUDA version
nvcc --version  # Should show CUDA 12.1-12.9

# 3. Reinstall JAX with GPU support
pip uninstall -y jax jaxlib
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# 4. Verify JAX detects GPU
python -c "import jax; print(jax.devices())"
# Expected: [cuda(id=0)]
# If still showing [CpuDevice(id=0)], check CUDA installation
```

**Issue:** ImportError or CUDA library not found

**Solution:**
```bash
# Set CUDA library path (add to ~/.bashrc for permanent fix)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Platform Support

- ✅ **Linux + NVIDIA GPU + CUDA 12.1-12.9:** Full GPU acceleration (20-100x speedup)
- ❌ **macOS:** CPU-only (Apple Silicon/Intel, no NVIDIA GPU support)
- ❌ **Windows:** CPU-only (CUDA support experimental/unstable)

**Requirements (Linux GPU):**
- System CUDA 12.1-12.9 pre-installed
- NVIDIA driver >= 525
- Linux x86_64 or aarch64

#### Conda/Mamba Users

The package works seamlessly in conda environments using pip:

```bash
conda create -n rheojax python=3.12
conda activate rheojax
pip install rheojax

# For GPU acceleration (Linux only)
git clone https://github.com/imewei/rheojax.git
cd rheojax
make install-jax-gpu
```

**Note:** Conda extras syntax (`conda install rheo[gpu]`) is not supported. Use the Makefile or manual pip installation method above.

## Quick Start

### Loading and Visualizing Data

```python
from rheojax.io.readers import auto_read
from rheojax.visualization import plot_rheo_data
import matplotlib.pyplot as plt

# Load data (auto-detect format)
data = auto_read("stress_relaxation.txt")

# Check detected test mode
print(f"Test mode: {data.test_mode}")  # Output: relaxation

# Visualize
fig, ax = plot_rheo_data(data, style='publication')
plt.show()
```

### Basic Model Fitting

```python
from rheojax.models.maxwell import Maxwell
import numpy as np

# Generate or load data
t = np.logspace(-2, 2, 50)
G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, 50)

# Fit with NLSQ (5-270x faster than scipy)
model = Maxwell()
model.fit(t, G_data)

print(f"G0 = {model.parameters.get_value('G0'):.3e} Pa")
print(f"eta = {model.parameters.get_value('eta'):.3e} Pa·s")
```

### Bayesian Inference Workflow

```python
from rheojax.models.maxwell import Maxwell
import numpy as np

# Create model and data
model = Maxwell()
t = np.logspace(-2, 2, 50)
G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, 50)

# Step 1: NLSQ optimization (fast point estimate)
model.fit(t, G_data)
print(f"NLSQ: G0={model.parameters.get_value('G0'):.3e}")

# Step 2: Bayesian inference with warm-start
result = model.fit_bayesian(
    t, G_data,
    num_warmup=1000,
    num_samples=2000
)

# Step 3: Analyze results
print(f"Posterior mean: G0={result.summary['G0']['mean']:.3e} ± {result.summary['G0']['std']:.3e}")
print(f"Convergence: R-hat={result.diagnostics['r_hat']['G0']:.4f}, ESS={result.diagnostics['ess']['G0']:.0f}")

# Get credible intervals
intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
print(f"G0 95% CI: [{intervals['G0'][0]:.3e}, {intervals['G0'][1]:.3e}]")
```

### Complete Bayesian Pipeline with ArviZ Diagnostics

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

# ArviZ diagnostic plots (comprehensive MCMC quality assessment)
(pipeline
    .plot_pair(divergences=True)        # Parameter correlations with divergences
    .plot_forest(hdi_prob=0.95)         # Credible intervals comparison
    .plot_energy()                       # NUTS energy diagnostic
    .plot_autocorr()                     # Mixing diagnostic
    .plot_rank()                         # Convergence diagnostic
    .plot_ess(kind='local'))            # Effective sample size
```

### Working with Parameters

```python
from rheojax.core import ParameterSet

# Create parameter set
params = ParameterSet()
params.add("E", value=1000.0, bounds=(100, 10000), units="Pa")
params.add("tau", value=1.0, bounds=(0.1, 100), units="s")

# Get/set values
E = params.get_value("E")
params.set_value("tau", 2.5)
```

### Data Transforms

```python
from rheojax.transforms import FFTAnalysis, Mastercurve, MutationNumber

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

## Tutorial Notebooks

Comprehensive learning path with 20 tutorial notebooks:

```
examples/
├── basic/                       # 5 notebooks: Fundamental models
│   ├── 01-maxwell-fitting.ipynb
│   ├── 02-zener-fitting.ipynb
│   ├── 03-springpot-fitting.ipynb
│   ├── 04-bingham-fitting.ipynb
│   └── 05-power-law-fitting.ipynb
├── transforms/                  # 5 notebooks: Data analysis workflows
│   ├── 01-fft-analysis.ipynb
│   ├── 02-mastercurve-tts.ipynb
│   ├── 03-mutation-number.ipynb
│   ├── 04-owchirp-laos-analysis.ipynb
│   └── 05-smooth-derivative.ipynb
├── bayesian/                    # 5 notebooks: Bayesian inference
│   ├── 01-bayesian-basics.ipynb
│   ├── 02-prior-selection.ipynb
│   ├── 03-convergence-diagnostics.ipynb
│   ├── 04-model-comparison.ipynb
│   └── 05-uncertainty-propagation.ipynb
└── advanced/                    # 5 notebooks: Production patterns
    ├── 01-multi-technique-fitting.ipynb
    ├── 02-batch-processing.ipynb
    ├── 03-custom-models.ipynb
    ├── 04-fractional-models-deep-dive.ipynb
    └── 05-performance-optimization.ipynb
```

See `examples/README.md` for complete learning path guide.

## Documentation

Full documentation is available at [https://rheojax.readthedocs.io](https://rheojax.readthedocs.io)

### Key Topics

- [Getting Started](https://rheojax.readthedocs.io/user_guide/getting_started.html) - Installation and basic usage
- [Core Concepts](https://rheojax.readthedocs.io/user_guide/core_concepts.html) - RheoData, Parameters, Test Modes
- [Bayesian Inference](https://rheojax.readthedocs.io/user_guide/bayesian_inference.html) - NLSQ → NUTS workflow, ArviZ diagnostics
- [Pipeline API](https://rheojax.readthedocs.io/user_guide/pipeline_api.html) - High-level workflows
- [I/O Guide](https://rheojax.readthedocs.io/user_guide/io_guide.html) - Reading and writing data
- [Visualization Guide](https://rheojax.readthedocs.io/user_guide/visualization_guide.html) - Creating plots
- [API Reference](https://rheojax.readthedocs.io/api_reference.html) - Complete API documentation

## Development Status

### Phase 1 (Complete) - Core Infrastructure ✅

**Base Abstractions**
- BaseModel and BaseTransform interfaces
- RheoData container with JAX support
- Parameter system with bounds and constraints
- Scikit-learn compatible API

**Test Mode Detection**
- Automatic detection: relaxation, creep, oscillation, rotation
- Metadata-based override capability
- Test mode validation and suggestions

**Numerical Utilities**
- Mittag-Leffler functions (one and two-parameter)
- NLSQ-based optimization with JAX gradients (5-270x faster)
- Float64 precision enforcement via safe_import_jax()

**File I/O**
- Readers: TRIOS, CSV, Excel, Anton Paar
- Writers: HDF5 (full fidelity), Excel
- Auto-detection with format inference

**Visualization**
- Automatic plot type selection
- Three built-in styles (default, publication, presentation)
- Time-domain, frequency-domain, and flow curve plots
- Residual visualization

### Phase 2 (Complete) - Models and Transforms ✅

- ✅ **20 Rheological Models**: Maxwell, Zener, fractional models (11 variants), flow models (6)
- ✅ **5 Data Transforms**: FFT analysis, Mastercurve (TTS), Mutation Number, OWChirp (LAOS), Smooth Derivative
- ✅ **Pipeline API**: Complete workflow orchestration with fluent interface
- ✅ **Enhanced Visualization**: Advanced templates and multi-panel figures
- ✅ **Comprehensive Testing**: 150+ tests with 90%+ coverage

### Phase 3 (Complete) - Bayesian Inference ✅

- ✅ **Bayesian Parameter Estimation**: NumPyro NUTS sampling with NLSQ warm-start (2-5x faster convergence)
- ✅ **Uncertainty Quantification**: Credible intervals, posterior distributions, parameter correlations
- ✅ **ArviZ Integration**: Complete diagnostic suite (pair, forest, energy, autocorr, rank, ESS plots)
- ✅ **Model Comparison**: WAIC/LOO for Bayesian model selection
- ✅ **BayesianMixin**: All 20 models automatically support Bayesian inference
- ✅ **BayesianPipeline**: Fluent API for NLSQ → NUTS workflows
- ✅ **Tutorial Notebooks**: 5 comprehensive Bayesian inference examples

### Phase 4 (Future) - Advanced Features

- Multi-objective optimization
- Machine learning integration for model selection
- Automated hyperparameter tuning
- Real-time data processing and streaming
- Enhanced custom model framework
- Advanced sensitivity analysis

## Performance

### NLSQ Optimization Performance

NLSQ provides significant performance improvements over scipy:

| Dataset Size | scipy (Powell) | NLSQ (JAX) | Speedup |
|--------------|----------------|------------|---------|
| 50 points    | 180 ms        | 35 ms      | 5x      |
| 500 points   | 920 ms        | 48 ms      | 19x     |
| 5000 points  | 8.2 s         | 95 ms      | 86x     |
| 50000 points | 94 s          | 350 ms     | 270x    |

### Bayesian Warm-Start Performance

NLSQ → NUTS warm-start dramatically improves MCMC convergence:

| Method | Convergence Time | Divergences | ESS/sec |
|--------|------------------|-------------|---------|
| Cold start (random init) | 45s | 15% | 44 |
| NLSQ warm-start | 18s | 0.2% | 111 |
| **Improvement** | **2.5x faster** | **75x fewer** | **2.5x higher** |

*Benchmarks on M1 MacBook Pro. GPU acceleration provides additional 5-20x speedups for large datasets.*

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/imewei/rheojax.git
cd rheojax

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
pytest --cov=rheojax --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Only unit tests
pytest -m integration # Integration tests
```

## Technology Stack

**Core**
- Python 3.12+
- JAX 0.8.0 for automatic differentiation and acceleration
- NLSQ 0.1.6+ for GPU-accelerated optimization
- NumPyro for Bayesian inference (MCMC NUTS)
- ArviZ 0.15.0+ for Bayesian visualization and diagnostics
- NumPy, SciPy for numerical operations

**I/O**
- h5py for HDF5 files
- pandas for CSV/Excel
- openpyxl for Excel writing

**Visualization**
- matplotlib for plotting
- publication-quality output in multiple formats

**Optional**
- CUDA 12+ for GPU acceleration (Linux only)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use rheo in your research, please cite:

```bibtex
@software{rheo2024,
  title = {Rheo: JAX-Powered Unified Rheology Package with Bayesian Inference},
  year = {2024},
  author = {Rheo Development Team},
  url = {https://github.com/imewei/rheojax},
  version = {0.2.0}
}
```

## Acknowledgments

rheo is built on excellent open-source software:

- [JAX](https://github.com/google/jax) for automatic differentiation and acceleration
- [NLSQ](https://github.com/rdyro/nlsq) for GPU-accelerated nonlinear least squares
- [NumPyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [ArviZ](https://github.com/arviz-devs/arviz) for Bayesian visualization
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computing
- [matplotlib](https://matplotlib.org/) for visualization

## Support

- 📖 Documentation: [https://rheojax.readthedocs.io](https://rheojax.readthedocs.io)
- 💬 Discussions: [GitHub Discussions](https://github.com/imewei/rheojax/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/imewei/rheojax/issues)
- 📧 Email: rheo@example.com

## Roadmap

See [CHANGELOG.md](CHANGELOG.md) for detailed development history and [examples/](examples/) for comprehensive tutorial notebooks.

---

Made with ❤️ by the Rheo Development Team
