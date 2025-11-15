# RheoJAX - JAX-Powered Rheological Analysis

[![CI](https://github.com/imewei/rheojax/actions/workflows/ci.yml/badge.svg)](https://github.com/imewei/rheojax/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/rheojax.svg)](https://badge.fury.io/py/rheojax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rheojax.readthedocs.io)

JAX-accelerated package for rheological data analysis. Provides 21 rheological models, 5 data transforms, Bayesian inference via NumPyro, and 27 tutorial notebooks.

## Features

Rheological analysis toolkit with Bayesian inference and parameter optimization:

### Core Capabilities
- **21 Rheological Models**: Classical (Maxwell, Zener, SpringPot), Fractional (11 variants), Flow (6 models), Multi-Mode (Generalized Maxwell)
- **5 Data Transforms**: FFT, Mastercurve (TTS), Mutation Number, OWChirp (LAOS), Smooth Derivative
- **Model-Data Compatibility Checking**: Detects when models are inappropriate for data based on physics (exponential vs power-law decay, material type classification)
- **Bayesian Inference**: All 21 models support NumPyro NUTS sampling with NLSQ warm-start
- **Pipeline API**: Fluent interface for load → fit → plot → save workflows
- **Automatic Initialization**: Parameter initialization for fractional models in oscillation mode
- **JAX-First Architecture**: 5-270x performance improvement with automatic differentiation and GPU support

### Data & I/O
- **Data Support**: Automatic test mode detection (relaxation, creep, oscillation, rotation)
- **File Formats**: TRIOS, CSV, Excel, Anton Paar with format auto-detection
- **Parameter System**: Type-safe parameter management with bounds and constraints

### Visualization & Diagnostics
- **Visualization**: Three built-in styles (default, publication, presentation)
- **ArviZ Diagnostic Suite**: 6 plot types (pair, forest, energy, autocorr, rank, ESS) for MCMC quality
- **Plugin System**: Support for custom models and transforms

### Tutorial Notebooks & Examples
- **27 Tutorial Notebooks**: Organized in 4 categories
  - `examples/basic/` - 5 notebooks covering fundamental models
  - `examples/transforms/` - 7 notebooks for data transforms and analysis
  - `examples/bayesian/` - 7 notebooks for Bayesian inference workflows
  - `examples/advanced/` - 8 notebooks for production patterns
- **I/O Examples**: TRIOS complex modulus handling and plotting

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

### GPU Installation (Linux Only)

**Performance Impact:** 20-100x speedup for large datasets (>10K points)

#### Option 1: Install via Makefile

From the repository:

```bash
git clone https://github.com/imewei/rheojax.git
cd rheojax
make install-jax-gpu  # Handles uninstall + GPU install
```

This command:
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

- **Linux + NVIDIA GPU + CUDA 12.1-12.9:** Full GPU acceleration (20-100x speedup)
- **macOS:** CPU-only (Apple Silicon/Intel, no NVIDIA GPU support)
- **Windows:** CPU-only (CUDA support experimental/unstable)

**Requirements (Linux GPU):**
- System CUDA 12.1-12.9 pre-installed
- NVIDIA driver >= 525
- Linux x86_64 or aarch64

#### Conda/Mamba Users

The package works in conda environments using pip:

```bash
conda create -n rheojax python=3.12
conda activate rheojax
pip install rheojax

# For GPU acceleration (Linux only)
git clone https://github.com/imewei/rheojax.git
cd rheojax
make install-jax-gpu
```

**Note:** Conda extras syntax (`conda install rheojax[gpu]`) is not supported. Use the Makefile or manual pip installation method above.

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

### Bayesian Pipeline with ArviZ Diagnostics

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

# ArviZ diagnostic plots (MCMC quality assessment)
(pipeline
    .plot_pair(divergences=True)        # Parameter correlations with divergences
    .plot_forest(hdi_prob=0.95)         # Credible intervals comparison
    .plot_energy()                       # NUTS energy diagnostic
    .plot_autocorr()                     # Mixing diagnostic
    .plot_rank()                         # Convergence diagnostic
    .plot_ess(kind='local'))            # Effective sample size
```

**Reference:** See [Bayesian Quick Start Guide](docs/BAYESIAN_QUICK_START.md) for:
- When and why to use Bayesian inference
- NLSQ → NUTS → ArviZ workflow walkthrough
- Troubleshooting convergence issues
- Best practices checklist
- Runnable demo: `python examples/bayesian_workflow_demo.py`

### Model-Data Compatibility Checking

RheoJAX detects when models are inappropriate for data based on physics:

```python
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
from rheojax.utils.compatibility import check_model_compatibility, format_compatibility_message
import numpy as np

# Generate exponential decay data
t = np.logspace(-2, 2, 50)
G_t = 1e5 * np.exp(-t / 1.0)

# Check compatibility before fitting
model = FractionalZenerSolidSolid()
compat = check_model_compatibility(
    model, t=t, G_t=G_t, test_mode='relaxation'
)

# Get report
print(format_compatibility_message(compat))
# Output:
# ⚠ Model may not be appropriate for this data
#   Confidence: 90%
#   Detected decay: exponential
#   Material type: viscoelastic_liquid
#
# Warnings:
#   • FZSS model expects Mittag-Leffler (power-law) relaxation,
#     but data shows exponential decay.
#
# Recommended alternative models:
#   • Maxwell
#   • Zener

# Or enable checking during fit
model.fit(t, G_t, check_compatibility=True)  # Warns if incompatible
```

**Features:**
- Detects decay type (exponential, power-law, stretched, Mittag-Leffler)
- Classifies material type (solid, liquid, gel, viscoelastic)
- Provides model recommendations when incompatible
- Error messages explain physics mismatches

**Reference:** [Model Selection Guide](docs/model_selection_guide.md) for decision flowcharts and model characteristics.

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

# Option 1: Create mastercurve (basic)
mastercurve = mc.create_mastercurve(datasets)

# Option 2: Transform with shift factors (for plotting)
mastercurve, shift_factors = mc.transform(datasets)

# Get parameters and arrays for analysis
wlf_params = mc.get_wlf_parameters()
temps, shifts = mc.get_shift_factors_array()

# Mutation number (viscoelastic character)
mn = MutationNumber()
delta = mn.calculate(data)  # 0=elastic, 1=viscous
```

## Tutorial Notebooks

27 tutorial notebooks organized by topic:

```
examples/
├── basic/                       # 5 notebooks: Fundamental models
│   ├── 01-maxwell-fitting.ipynb
│   ├── 02-zener-fitting.ipynb
│   ├── 03-springpot-fitting.ipynb
│   ├── 04-bingham-fitting.ipynb
│   └── 05-power-law-fitting.ipynb
├── transforms/                  # 7 notebooks: Data analysis workflows
│   ├── 01-fft-analysis.ipynb
│   ├── 02-mastercurve-tts.ipynb
│   ├── 02b-mastercurve-wlf-validation.ipynb
│   ├── 03-mutation-number.ipynb
│   ├── 04-owchirp-laos-analysis.ipynb
│   ├── 05-smooth-derivative.ipynb
│   └── 07-mastercurve_auto_shift.ipynb
├── bayesian/                    # 7 notebooks: Bayesian inference
│   ├── 01-bayesian-basics.ipynb
│   ├── 02-prior-selection.ipynb
│   ├── 03-convergence-diagnostics.ipynb
│   ├── 04-model-comparison.ipynb
│   ├── 05-uncertainty-propagation.ipynb
│   ├── 06-bayesian_workflow_demo.ipynb
│   └── 07-gmm_bayesian_workflow.ipynb
├── advanced/                    # 8 notebooks: Production patterns
│   ├── 01-multi-technique-fitting.ipynb
│   ├── 02-batch-processing.ipynb
│   ├── 03-custom-models.ipynb
│   ├── 04-fractional-models-deep-dive.ipynb
│   ├── 05-performance-optimization.ipynb
│   ├── 06-frequentist-model-selection.ipynb
│   ├── 07-trios_chunked_reading_example.ipynb
│   └── 08-generalized_maxwell_fitting.ipynb
└── io/                          # I/O demonstrations
    └── plot_trios_complex_modulus.ipynb
```

See `examples/README.md` for learning path guide.

## Documentation

Documentation: [https://rheojax.readthedocs.io](https://rheojax.readthedocs.io)

### Key Topics

- [Getting Started](https://rheojax.readthedocs.io/user_guide/getting_started.html) - Installation and basic usage
- [Core Concepts](https://rheojax.readthedocs.io/user_guide/core_concepts.html) - RheoData, Parameters, Test Modes
- [Bayesian Inference](https://rheojax.readthedocs.io/user_guide/bayesian_inference.html) - NLSQ → NUTS workflow, ArviZ diagnostics
- [Pipeline API](https://rheojax.readthedocs.io/user_guide/pipeline_api.html) - High-level workflows
- [I/O Guide](https://rheojax.readthedocs.io/user_guide/io_guide.html) - Reading and writing data
- [Visualization Guide](https://rheojax.readthedocs.io/user_guide/visualization_guide.html) - Creating plots
- [API Reference](https://rheojax.readthedocs.io/api_reference.html) - API documentation

## Performance

### NLSQ Optimization Performance

NLSQ performance compared to scipy:

| Dataset Size | scipy (Powell) | NLSQ (JAX) | Speedup |
|--------------|----------------|------------|---------|
| 50 points    | 180 ms        | 35 ms      | 5x      |
| 500 points   | 920 ms        | 48 ms      | 19x     |
| 5000 points  | 8.2 s         | 95 ms      | 86x     |
| 50000 points | 94 s          | 350 ms     | 270x    |

### Bayesian Warm-Start Performance

NLSQ → NUTS warm-start improves MCMC convergence:

| Method | Convergence Time | Divergences | ESS/sec |
|--------|------------------|-------------|---------|
| Cold start (random init) | 45s | 15% | 44 |
| NLSQ warm-start | 18s | 0.2% | 111 |
| **Improvement** | **2.5x faster** | **75x fewer** | **2.5x higher** |

*Benchmarks on M1 MacBook Pro. GPU acceleration provides additional 5-20x speedups for large datasets.*

## Contributing

Contributions are accepted. See [Contributing Guide](CONTRIBUTING.md) for details.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use rheo in your research, please cite:

```bibtex
@software{rheo2024,
  title = {Rheo: JAX-Powered Unified Rheology Package with Bayesian Inference},
  year = {2024},
  author = {Wei Chen},
  url = {https://github.com/imewei/rheojax},
  version = {0.2.2}
}
```

## Acknowledgments

Built on open-source software:

- [JAX](https://github.com/google/jax) for automatic differentiation and acceleration
- [NLSQ](https://github.com/rdyro/nlsq) for GPU-accelerated nonlinear least squares
- [NumPyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [ArviZ](https://github.com/arviz-devs/arviz) for Bayesian visualization
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computing
- [matplotlib](https://matplotlib.org/) for visualization

## Support

- Documentation: [https://rheojax.readthedocs.io](https://rheojax.readthedocs.io)
- Discussions: [GitHub Discussions](https://github.com/imewei/rheojax/discussions)
- Issues: [GitHub Issues](https://github.com/imewei/rheojax/issues)
- Email: wchen@anl.gov

## Roadmap

See [CHANGELOG.md](CHANGELOG.md) for development history and [examples/](examples/) for tutorial notebooks.

---

Wei Chen
