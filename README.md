# RheoJAX - JAX-Powered Rheological Analysis

[![CI](https://github.com/imewei/rheojax/actions/workflows/ci.yml/badge.svg)](https://github.com/imewei/rheojax/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/rheojax.svg)](https://badge.fury.io/py/rheojax)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://rheojax.readthedocs.io)

JAX-accelerated package for rheological data analysis. Provides 53 rheological models across 22 families (including TNT, VLB, HVM, HVNM, Giesekus, DMT, ITT-MCT, EPM, SGR, STZ, Fluidity-Saramito, IKH, and SPP), 7 data transforms (including SRFS and SPP), Bayesian inference via NumPyro, and 56 tutorial notebooks.

## Features

Rheological analysis toolkit with Bayesian inference and parameter optimization:

### Core Capabilities
- **53 Rheological Models**: Classical (3), Flow (6), Fractional Maxwell (4), Fractional Zener (4), Fractional Advanced (3), Multi-Mode (1), SGR (2), STZ (1), EPM (2), Fluidity (2), Fluidity-Saramito (2), IKH (2), FIKH (2), Hébraud-Lequeux (1), SPP LAOS (1), Giesekus (2), DMT (2), ITT-MCT (2), TNT (5), VLB (4), HVM (1), HVNM (1)
- **7 Data Transforms**: FFT, Mastercurve (TTS), Mutation Number, OWChirp (LAOS), Smooth Derivative, SRFS (Strain-Rate Frequency Superposition), SPP (Sequence of Physical Processes)
- **Model-Data Compatibility Checking**: Detects when models are inappropriate for data based on physics (exponential vs power-law decay, material type classification)
- **Bayesian Inference**: All 53 models support NumPyro NUTS sampling with NLSQ warm-start
- **Pipeline API**: Fluent interface for load → fit → plot → save workflows
- **Automatic Initialization**: Parameter initialization for fractional models in oscillation mode
- **JAX-First Architecture**: 5-270x performance improvement with automatic differentiation and GPU support

### Model Protocol Support Matrix

| Model Type | Model Name | Flow Curve (Steady Shear) | Creep | Relaxation | Start-up | SAOS (Oscillation) | LAOS (Large Amplitude) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical** | **Maxwell** | ✅ (Newtonian) | ✅ | ✅ | ❌ | ✅ | ❌ |
| | **Zener** (SLS) | ✅ (Newtonian) | ✅ | ✅ | ❌ | ✅ | ❌ |
| | **SpringPot** | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Flow** | **Carreau** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Power Law** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Herschel-Bulkley** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Bingham** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| | **Cross** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fractional** | **Fractional Maxwell** | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| | **Fractional Kelvin-Voigt**| ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Multi-mode** | **Generalized Maxwell** | ✅ (Newtonian) | ✅ | ✅ | ✅ (Linear) | ✅ | ✅ (Linear Only) |
| **SGR** | **SGR Conventional** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **SGR Generic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **STZ** | **STZ Conventional** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **EPM** | **Lattice EPM** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| | **Tensorial EPM** | ✅ (+ N₁) | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Fluidity** | **Fluidity Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Fluidity Nonlocal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **IKH** | **MIKH** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **ML-IKH** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SPP** | **SPP Yield Stress** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ (Amp. Sweep) |
| **Fluidity-Saramito** | **F-S Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **F-S Nonlocal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FIKH** | **FIKH** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **FMLIKH** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HL** | **Hébraud-Lequeux** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Giesekus** | **Single-Mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Multi-Mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **DMT** | **DMT Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **DMT Nonlocal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ITT-MCT** | **F₁₂ Schematic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **ISM Isotropic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **TNT** | **Single-Mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Loop-Bridge** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Sticky Rouse** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Cates** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **Multi-Species** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **VLB** | **VLB Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **VLB Multi-Network** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **VLB Variant** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **VLB Nonlocal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HVM** | **HVM Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HVNM** | **HVNM Local** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Data & I/O
- **Data Support**: Automatic test mode detection (relaxation, creep, oscillation, rotation)
- **File Formats**: TRIOS, CSV, Excel, Anton Paar with format auto-detection
- **Parameter System**: Type-safe parameter management with bounds and constraints

### Visualization & Diagnostics
- **Visualization**: Three built-in styles (default, publication, presentation)
- **ArviZ Diagnostic Suite**: 6 plot types (pair, forest, energy, autocorr, rank, ESS) for MCMC quality
- **Plugin System**: Support for custom models and transforms

### Tutorial Notebooks & Examples
- **235 Tutorial Notebooks**: Organized in 20 categories
  - `examples/basic/` - 5 notebooks covering fundamental models
  - `examples/transforms/` - 8 notebooks for data transforms and analysis (including SRFS)
  - `examples/bayesian/` - 9 notebooks for Bayesian inference workflows (including SPP LAOS)
  - `examples/advanced/` - 10 notebooks for production patterns (including SGR and SPP)
  - `examples/io/` - 1 notebook for TRIOS complex modulus handling
  - `examples/dmt/` - 6 notebooks: DMT thixotropic model (6 protocols)
  - `examples/epm/` - 6 notebooks: Elasto-plastic models (5 protocols + visualization)
  - `examples/fikh/` - 12 notebooks: FIKH + FMLIKH models (6 protocols each)
  - `examples/fluidity/` - 24 notebooks: Fluidity local/nonlocal + Saramito local/nonlocal (6 protocols each)
  - `examples/giesekus/` - 7 notebooks: Giesekus model (6 protocols + normal stresses)
  - `examples/hl/` - 6 notebooks: Hebraud-Lequeux model (6 protocols)
  - `examples/hvm/` - 13 notebooks: Hybrid Vitrimer Model (6 basic + 7 advanced tutorials)
  - `examples/hvnm/` - 15 notebooks: Hybrid Vitrimer Nanocomposite (7 basic + 8 NLSQ/NUTS)
  - `examples/ikh/` - 12 notebooks: MIKH + MLIKH models (6 protocols each)
  - `examples/itt_mct/` - 12 notebooks: ITT-MCT Schematic + Isotropic (6 protocols each)
  - `examples/sgr/` - 6 notebooks: Soft Glassy Rheology (6 protocols)
  - `examples/stz/` - 6 notebooks: Shear Transformation Zone (6 protocols)
  - `examples/tnt/` - 30 notebooks: 5 TNT sub-models (6 protocols each)
  - `examples/vlb/` - 16 notebooks: 6 protocols + Bayesian + Bell + FENE + Nonlocal + 6 NLSQ-to-NUTS
  - `examples/verification/` - 31 notebooks: Cross-model validation (6 protocol validators + 25 material-specific)

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

### GPU Installation (Linux + System CUDA)

**Performance Impact:** 20-100x speedup for large datasets (>10K points)

**Prerequisites:**
- NVIDIA GPU with SM >= 5.2 (Maxwell or newer)
- System CUDA 12.x or 13.x installed
- `nvcc` in PATH

#### Verify Prerequisites

```bash
# Check CUDA installation
nvcc --version
# Should show: Cuda compilation tools, release 12.x or 13.x

# Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Should show: GPU name and SM version (e.g., "8.9" for RTX 4090)
```

#### Option 1: Quick Install via Makefile (Recommended)

```bash
git clone https://github.com/imewei/rheojax.git
cd rheojax

# Auto-detect system CUDA version and install matching JAX
make install-jax-gpu

# Or explicitly choose CUDA version:
make install-jax-gpu-cuda13  # Requires system CUDA 13.x + SM >= 7.5
make install-jax-gpu-cuda12  # Requires system CUDA 12.x + SM >= 5.2
```

This:
- Detects your system CUDA version (via nvcc)
- Validates GPU compatibility
- Installs the matching `jax[cudaXX-local]` package
- Verifies GPU detection

#### Option 2: Manual Installation

**For System CUDA 13.x (Turing and newer GPUs):**

```bash
# Verify you have CUDA 13.x
nvcc --version  # Should show release 13.x

# Verify GPU supports CUDA 13 (SM >= 7.5)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader  # Should be >= 7.5

# Install
pip uninstall -y jax jaxlib
pip install "jax[cuda13-local]"

# Verify
python -c "import jax; print('Backend:', jax.default_backend())"
# Should show: Backend: gpu
```

**For System CUDA 12.x (Maxwell and newer GPUs):**

```bash
# Verify you have CUDA 12.x
nvcc --version  # Should show release 12.x

# Install
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"

# Verify
python -c "import jax; print('Backend:', jax.default_backend())"
```

**Why separate installation?** JAX with CUDA support is Linux-specific and requires system CUDA pre-installed. Separating the installation avoids dependency conflicts on macOS/Windows.

#### GPU Compatibility Guide

| GPU Generation | Example GPUs | SM Version | CUDA 13 | CUDA 12 |
|----------------|--------------|------------|---------|---------|
| Blackwell | B100, B200 | 10.0 | Yes | Yes |
| Hopper | H100, H200 | 9.0 | Yes | Yes |
| Ada Lovelace | RTX 40xx, L40 | 8.9 | Yes | Yes |
| Ampere | RTX 30xx, A100 | 8.x | Yes | Yes |
| Turing | RTX 20xx, T4 | 7.5 | Yes | Yes |
| Volta | V100, Titan V | 7.0 | No | Yes |
| Pascal | GTX 10xx, P100 | 6.x | No | Yes |
| Maxwell | GTX 9xx, Titan X | 5.x | No | Yes |
| Kepler | GTX 7xx, K80 | 3.x | No | No |

**Recommendation:** SM >= 7.5 (RTX 20xx or newer) → install CUDA 13 for best performance. SM 5.2–7.4 (GTX 9xx/10xx, V100) → install CUDA 12.

#### GPU Troubleshooting

**Issue: "nvcc not found"**

CUDA toolkit not installed or not in PATH:
```bash
# Option 1: Install CUDA toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Option 2: Add existing CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Add to ~/.bashrc for permanent fix
```

**Issue: "CUDA version mismatch"**

JAX package must match your system CUDA version:
```bash
# Check your system CUDA version
nvcc --version
# Shows: release 12.6 -> use cuda12-local
# Shows: release 13.x -> use cuda13-local

# Reinstall with correct package
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"  # or cuda13-local
```

**Issue: "GPU SM version doesn't support CUDA 13"**

Your GPU is older than Turing architecture:
```bash
# Check SM version
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# If < 7.5, you need CUDA 12

# Install CUDA 12.x toolkit, then:
pip install "jax[cuda12-local]"
```

**Issue: "libcuda.so not found" or similar library errors**

CUDA libraries not in LD_LIBRARY_PATH:
```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

#### Platform Support

| Platform | GPU Support | Notes |
|----------|-------------|-------|
| Linux x86_64/aarch64 | Full | Requires system CUDA 12.x or 13.x |
| Windows WSL2 | Experimental | Use Linux wheels |
| macOS (any) | CPU-only | No NVIDIA support |
| Windows native | CPU-only | No pre-built wheels |

**Requirements (Linux GPU):**
- System CUDA 12.1+ or 13.x pre-installed
- NVIDIA driver >= 525 (CUDA 12) or >= 560 (CUDA 13)
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

# Step 2: Bayesian inference with warm-start (4 chains by default)
result = model.fit_bayesian(
    t, G_data,
    num_warmup=1000,
    num_samples=2000,
    # num_chains=4 (default), use num_chains=1 for quick demos
    # seed=42 for reproducibility
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
    .fit_bayesian(num_samples=2000, num_warmup=1000)  # num_chains=4 by default
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

### Soft Glassy Rheology (SGR) Models

```python
from rheojax.models import SGRConventional, SGRGeneric

# Conventional SGR model (Sollich 1998)
# For soft glassy materials: foams, emulsions, pastes, colloidal suspensions
sgr = SGRConventional()
sgr.fit(omega, G_star, test_mode='oscillation')
# Parameters: x (noise temp 0.5-3), G0 (modulus), tau0 (attempt time)
# x < 1: glass, 1 < x < 2: power-law fluid, x >= 2: Newtonian

# GENERIC framework SGR (thermodynamically consistent, Fuereder & Ilg 2013)
sgr_gen = SGRGeneric()
sgr_gen.fit(omega, G_star, test_mode='oscillation')
```

### Elasto-Plastic Models (EPM)

```python
from rheojax.models import LatticeEPM, TensorialEPM
from rheojax.visualization.epm_plots import plot_lattice_fields, plot_normal_stress_field

# Lattice EPM for amorphous solids (mesoscopic simulation)
# Spatial heterogeneity, avalanches, shear banding
epm = LatticeEPM(L=64, yielding_mode='smooth')

# Fit steady-state flow curve
epm.fit(shear_rate, stress, test_mode='rotation')

# Transient creep simulation
epm.fit(time, strain, test_mode='creep', stress_0=1.5)

# Visualize stress and threshold fields
state = epm.get_state()
plot_lattice_fields(state)

# Tensorial EPM for normal stress predictions
# Full stress tensor [σ_xx, σ_yy, σ_xy] with N₁, N₂ predictions
tensorial = TensorialEPM(L=32, yield_criterion='von_mises')

# Predict flow curve with normal stresses
result = tensorial.predict(data, smooth=True)
sigma_xy = result.y  # Shear stress
N1 = result.metadata['N1']  # First normal stress difference

# Visualize normal stress fields
plot_normal_stress_field(stress_tensor, nu=0.48)
```

### Strain-Rate Frequency Superposition (SRFS)

```python
from rheojax.transforms import SRFS

# Collapse flow curves at different shear rates (analogous to TTS)
srfs = SRFS(reference_gamma_dot=1.0, auto_shift=True)
master_curve, shifts = srfs.transform(datasets)

# Power-law shift: a(γ̇) ~ (γ̇)^(2-x)
# Includes thixotropy detection and shear banding analysis
from rheojax.transforms import detect_shear_banding, compute_shear_band_coexistence
is_banding = detect_shear_banding(flow_curve)
```

### Fluidity Models (Cooperative Flow)

```python
from rheojax.models import FluidityLocal, FluidityNonlocal

# Local fluidity model (Picard et al. 2002)
# For yield stress fluids with cooperative rearrangements
fluidity = FluidityLocal()
fluidity.fit(gamma_dot, sigma, test_mode='flow_curve')
# Parameters: sigma_y (yield stress), f0 (microstructural relaxation rate),
#             n (flow index), mu_inf (high-shear viscosity)

# Nonlocal fluidity model with spatial correlations
# For shear-banding and spatial heterogeneity in YSFs
nonlocal = FluidityNonlocal(xi=0.1)  # Cooperativity length
nonlocal.fit(gamma_dot, sigma, test_mode='flow_curve')

# Predict velocity profiles (flow heterogeneity)
velocity_profile = nonlocal.predict_velocity_profile(stress, gap_width=1e-3)
```

### Isotropic-Kinematic Hardening (IKH) Models

```python
from rheojax.models import MIKH, MLIKH

# MIKH: Maxwell-Isotropic-Kinematic Hardening (Dimitriou & McKinley 2014)
# For thixotropic elasto-viscoplastic materials: waxy crude oils, drilling fluids
mikh = MIKH()
mikh.fit(t, sigma, test_mode='startup', gamma_dot=1.0)
# Parameters: G (modulus), eta (Maxwell viscosity), C (kinematic hardening),
#             sigma_y0 (yield stress), tau_thix (thixotropic time), etc.

# Predict stress overshoot in startup flow
sigma_startup = mikh.predict_startup(t, gamma_dot=1.0)

# ML-IKH: Multi-mode extension for distributed thixotropic timescales
# Captures stretched-exponential recovery
mlikh = MLIKH(n_modes=3, yield_mode='weighted_sum')
mlikh.fit(t, sigma, test_mode='startup')
```

### Shear Transformation Zone (STZ) Model

```python
from rheojax.models import STZConventional

# STZ model for amorphous solids: metallic glasses, colloidal suspensions
# Based on Langer 2008 effective temperature formulation
stz = STZConventional(variant='standard')  # 'minimal', 'standard', or 'full'

# Steady-state flow curve fitting
stz.fit(gamma_dot, stress, test_mode='rotation')

# Transient startup flow (stress overshoot)
stz.fit(time, stress, test_mode='transient', gamma_dot=1.0)

# Small-amplitude oscillatory shear (SAOS)
stz.fit(omega, G_star, test_mode='oscillation')

# Large-amplitude oscillatory shear (LAOS) simulation
result = stz.simulate_laos(gamma_0=0.1, omega=1.0, n_cycles=5)
harmonics = stz.extract_harmonics(result)
```

**Key Features:**
- Three variants: minimal (steady-state), standard (aging/thixotropy), full (LAOS/back-stress)
- Captures yield stress, stress overshoot, aging, and rejuvenation
- State variables: effective temperature (χ), STZ density (Λ), orientation (m)

### Sequence of Physical Processes (SPP) for LAOS Analysis

```python
from rheojax.transforms import SPPDecomposer
from rheojax.models import SPPYieldStress

# Time-domain LAOS analysis (no Fourier decomposition)
spp = SPPDecomposer(
    omega=1.0,          # Angular frequency (rad/s)
    gamma_0=1.0,        # Strain amplitude
    n_harmonics=39,     # Rogers-parity default
)
result = spp.transform(rheo_data)

# Extract SPP quantities
G_cage = result.cage_modulus           # Cage modulus
sigma_y_static = result.static_yield   # Static yield stress
sigma_y_dynamic = result.dynamic_yield # Dynamic yield stress

# Fit yield stress model with Bayesian inference
model = SPPYieldStress()
model.fit(gamma_0_array, sigma_y_data, test_mode='oscillation')
bayes_result = model.fit_bayesian(gamma_0_array, sigma_y_data, num_samples=2000)
```

### TNT Transient Network Models

```python
from rheojax.models import TNTSingleMode, TNTCates

# Single-mode transient network
tnt = TNTSingleMode()
tnt.fit(omega, G_star, test_mode='oscillation')

# Cates living polymer model (wormlike micelles)
cates = TNTCates()
cates.fit(omega, G_star, test_mode='oscillation')

# Startup shear with stress overshoot
result = cates.simulate_startup(t, gamma_dot=1.0)
```

### VLB Transient Network Models

```python
from rheojax.models import VLBLocal, VLBVariant, VLBNonlocal

# Basic VLB transient network
vlb = VLBLocal()
vlb.fit(omega, G_star, test_mode='oscillation')

# VLB Variant with Bell force-activated dissociation
vlb_bell = VLBVariant(force_model="bell")
vlb_bell.parameters.set_value("F_c", 10.0)  # Characteristic force

# VLB Nonlocal for shear banding prediction
vlb_nl = VLBNonlocal(n_points=51)
result = vlb_nl.simulate_steady_shear(gamma_dot_avg=10.0)
```

### HVM (Hybrid Vitrimer Model)

```python
from rheojax.models import HVMLocal

# Full HVM: permanent + exchangeable + dissociative networks
model = HVMLocal(kinetics="stress", include_dissociative=True)
model.parameters.set_value("G_P", 5000.0)   # Permanent (covalent)
model.parameters.set_value("G_E", 3000.0)   # Exchangeable (vitrimer BER)
model.parameters.set_value("G_D", 1000.0)   # Dissociative (physical)

# SAOS: two Maxwell modes + G_P plateau
omega = np.logspace(-3, 3, 100)
G_prime, G_double_prime = model.predict_saos(omega)

# Factory methods for limiting cases
partial = HVMLocal.partial_vitrimer(G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3)
```

### HVNM (Hybrid Vitrimer Nanocomposite Model)

```python
from rheojax.models import HVNMLocal

# Full HVNM: 4 subnetworks (P + E + D + I interphase)
model = HVNMLocal(kinetics="stress", include_dissociative=True)
model.parameters.set_value("phi", 0.1)      # NP volume fraction
model.parameters.set_value("beta_I", 3.0)   # Interphase reinforcement

# Guth-Gold strain amplification: X(phi) = 1 + 2.5*phi + 14.1*phi^2
omega = np.logspace(-3, 3, 100)
G_prime, G_double_prime = model.predict_saos(omega)

# phi=0 recovers HVM exactly
unfilled = HVNMLocal.unfilled_vitrimer(G_P=5000, G_E=3000, G_D=1000)
```

## Tutorial Notebooks

235 tutorial notebooks organized by topic:

```
examples/
├── basic/                       # 5 notebooks: Fundamental models
│   ├── 01-maxwell-fitting.ipynb
│   ├── 02-zener-fitting.ipynb
│   ├── 03-springpot-fitting.ipynb
│   ├── 04-bingham-fitting.ipynb
│   └── 05-power-law-fitting.ipynb
├── transforms/                  # 8 notebooks: Data analysis workflows
│   ├── 01-fft-analysis.ipynb
│   ├── 02-mastercurve-tts.ipynb
│   ├── 02b-mastercurve-wlf-validation.ipynb
│   ├── 03-mutation-number.ipynb
│   ├── 04-owchirp-laos-analysis.ipynb
│   ├── 05-smooth-derivative.ipynb
│   ├── 06-mastercurve_auto_shift.ipynb
│   └── 07-srfs-strain-rate-superposition.ipynb
├── bayesian/                    # 9 notebooks: Bayesian inference (including SPP)
│   ├── 01-bayesian-basics.ipynb
│   ├── 02-prior-selection.ipynb
│   ├── 03-convergence-diagnostics.ipynb
│   ├── 04-model-comparison.ipynb
│   ├── 05-uncertainty-propagation.ipynb
│   ├── 06-bayesian_workflow_demo.ipynb
│   ├── 07-gmm_bayesian_workflow.ipynb
│   ├── 08-spp-laos.ipynb
│   └── 09-spp-rheojax-workflow.ipynb
├── advanced/                    # 10 notebooks: Production patterns
│   ├── 01-multi-technique-fitting.ipynb
│   ├── 02-batch-processing.ipynb
│   ├── 03-custom-models.ipynb
│   ├── 04-fractional-models-deep-dive.ipynb
│   ├── 05-performance-optimization.ipynb
│   ├── 06-frequentist-model-selection.ipynb
│   ├── 07-trios_chunked_reading_example.ipynb
│   ├── 08-generalized_maxwell_fitting.ipynb
│   ├── 09-sgr-soft-glassy-rheology.ipynb
│   └── 10-spp-laos-tutorial.ipynb
├── io/                          # 1 notebook: I/O demonstrations
│   └── plot_trios_complex_modulus.ipynb
├── dmt/                         # 6 notebooks: DMT thixotropic model
│   ├── 01_dmt_flow_curve.ipynb
│   ├── 02_dmt_startup_shear.ipynb
│   ├── 03_dmt_stress_relaxation.ipynb
│   ├── 04_dmt_creep.ipynb
│   ├── 05_dmt_saos.ipynb
│   └── 06_dmt_laos.ipynb
├── epm/                         # 6 notebooks: Elasto-plastic models
│   ├── 01_epm_flow_curve.ipynb
│   ├── 02_epm_saos.ipynb
│   ├── 03_epm_startup.ipynb
│   ├── 04_epm_creep.ipynb
│   ├── 05_epm_relaxation.ipynb
│   └── 06_epm_visualization.ipynb
├── fikh/                        # 12 notebooks: FIKH + FMLIKH models
│   ├── 01_fikh_flow_curve.ipynb
│   ├── 02_fikh_startup_shear.ipynb
│   ├── 03_fikh_stress_relaxation.ipynb
│   ├── 04_fikh_creep.ipynb
│   ├── 05_fikh_saos.ipynb
│   ├── 06_fikh_laos.ipynb
│   ├── 07_fmlikh_flow_curve.ipynb
│   ├── 08_fmlikh_startup_shear.ipynb
│   ├── 09_fmlikh_stress_relaxation.ipynb
│   ├── 10_fmlikh_creep.ipynb
│   ├── 11_fmlikh_saos.ipynb
│   └── 12_fmlikh_laos.ipynb
├── fluidity/                    # 24 notebooks: Fluidity + Saramito (local & nonlocal)
│   ├── 01_fluidity_local_flow_curve.ipynb
│   ├── 02_fluidity_local_startup.ipynb
│   ├── 03_fluidity_local_creep.ipynb
│   ├── 04_fluidity_local_relaxation.ipynb
│   ├── 05_fluidity_local_saos.ipynb
│   ├── 06_fluidity_local_laos.ipynb
│   ├── 07_fluidity_nonlocal_flow_curve.ipynb
│   ├── 08_fluidity_nonlocal_startup.ipynb
│   ├── 09_fluidity_nonlocal_creep.ipynb
│   ├── 10_fluidity_nonlocal_relaxation.ipynb
│   ├── 11_fluidity_nonlocal_saos.ipynb
│   ├── 12_fluidity_nonlocal_laos.ipynb
│   ├── 13_saramito_local_flow_curve.ipynb
│   ├── 14_saramito_local_startup.ipynb
│   ├── 15_saramito_local_creep.ipynb
│   ├── 16_saramito_local_relaxation.ipynb
│   ├── 17_saramito_local_saos.ipynb
│   ├── 18_saramito_local_laos.ipynb
│   ├── 19_saramito_nonlocal_flow_curve.ipynb
│   ├── 20_saramito_nonlocal_startup.ipynb
│   ├── 21_saramito_nonlocal_creep.ipynb
│   ├── 22_saramito_nonlocal_relaxation.ipynb
│   ├── 23_saramito_nonlocal_saos.ipynb
│   └── 24_saramito_nonlocal_laos.ipynb
├── giesekus/                    # 7 notebooks: Giesekus constitutive model
│   ├── 01_giesekus_flow_curve.ipynb
│   ├── 02_giesekus_saos.ipynb
│   ├── 03_giesekus_startup.ipynb
│   ├── 04_giesekus_normal_stresses.ipynb
│   ├── 05_giesekus_creep.ipynb
│   ├── 06_giesekus_relaxation.ipynb
│   └── 07_giesekus_laos.ipynb
├── hl/                          # 6 notebooks: Hebraud-Lequeux model
│   ├── 01_hl_flow_curve.ipynb
│   ├── 02_hl_relaxation.ipynb
│   ├── 03_hl_creep.ipynb
│   ├── 04_hl_saos.ipynb
│   ├── 05_hl_startup.ipynb
│   └── 06_hl_laos.ipynb
├── hvm/                         # 13 notebooks: Hybrid Vitrimer Model
│   ├── 01_hvm_saos.ipynb
│   ├── 02_hvm_stress_relaxation.ipynb
│   ├── 03_hvm_startup_shear.ipynb
│   ├── 04_hvm_creep.ipynb
│   ├── 05_hvm_flow_curve.ipynb
│   ├── 06_hvm_laos.ipynb
│   ├── 07_hvm_overview.ipynb               # Overview tutorial
│   ├── 08_hvm_flow_curve.ipynb             # Advanced flow curve
│   ├── 09_hvm_creep.ipynb                  # Advanced creep
│   ├── 10_hvm_relaxation.ipynb             # Advanced relaxation
│   ├── 11_hvm_startup.ipynb                # Advanced startup
│   ├── 12_hvm_saos.ipynb                   # Advanced SAOS
│   └── 13_hvm_laos.ipynb                   # Advanced LAOS
├── hvnm/                        # 15 notebooks: Hybrid Vitrimer Nanocomposite
│   ├── 01_hvnm_saos.ipynb
│   ├── 02_hvnm_stress_relaxation.ipynb
│   ├── 03_hvnm_startup_shear.ipynb
│   ├── 04_hvnm_creep.ipynb
│   ├── 05_hvnm_flow_curve.ipynb
│   ├── 06_hvnm_laos.ipynb
│   ├── 07_hvnm_limiting_cases.ipynb        # phi=0 → HVM recovery
│   ├── 08_data_intake_and_qc.ipynb         # Data intake & QC
│   ├── 09_flow_curve_nlsq_nuts.ipynb       # NLSQ → NUTS workflow
│   ├── 10_creep_compliance_nlsq_nuts.ipynb
│   ├── 11_stress_relaxation_nlsq_nuts.ipynb
│   ├── 12_startup_shear_nlsq_nuts.ipynb
│   ├── 13_saos_nlsq_nuts.ipynb
│   ├── 14_laos_nlsq_nuts.ipynb
│   └── 15_global_multi_protocol.ipynb      # Multi-protocol fitting
├── ikh/                         # 12 notebooks: MIKH + MLIKH models
│   ├── 01_mikh_flow_curve.ipynb
│   ├── 02_mikh_startup_shear.ipynb
│   ├── 03_mikh_stress_relaxation.ipynb
│   ├── 04_mikh_creep.ipynb
│   ├── 05_mikh_saos.ipynb
│   ├── 06_mikh_laos.ipynb
│   ├── 07_mlikh_flow_curve.ipynb
│   ├── 08_mlikh_startup_shear.ipynb
│   ├── 09_mlikh_stress_relaxation.ipynb
│   ├── 10_mlikh_creep.ipynb
│   ├── 11_mlikh_saos.ipynb
│   └── 12_mlikh_laos.ipynb
├── itt_mct/                     # 12 notebooks: ITT-MCT Schematic + Isotropic
│   ├── 01_schematic_flow_curve.ipynb
│   ├── 02_schematic_startup_shear.ipynb
│   ├── 03_schematic_stress_relaxation.ipynb
│   ├── 04_schematic_creep.ipynb
│   ├── 05_schematic_saos.ipynb
│   ├── 06_schematic_laos.ipynb
│   ├── 07_isotropic_flow_curve.ipynb
│   ├── 08_isotropic_startup_shear.ipynb
│   ├── 09_isotropic_stress_relaxation.ipynb
│   ├── 10_isotropic_creep.ipynb
│   ├── 11_isotropic_saos.ipynb
│   └── 12_isotropic_laos.ipynb
├── sgr/                         # 6 notebooks: Soft Glassy Rheology
│   ├── 01_sgr_flow_curve.ipynb
│   ├── 02_sgr_stress_relaxation.ipynb
│   ├── 03_sgr_saos.ipynb
│   ├── 04_sgr_creep.ipynb
│   ├── 05_sgr_startup.ipynb
│   └── 06_sgr_laos.ipynb
├── stz/                         # 6 notebooks: Shear Transformation Zone
│   ├── 01_stz_flow_curve.ipynb
│   ├── 02_stz_startup_shear.ipynb
│   ├── 03_stz_stress_relaxation.ipynb
│   ├── 04_stz_creep.ipynb
│   ├── 05_stz_saos.ipynb
│   └── 06_stz_laos.ipynb
├── tnt/                         # 30 notebooks: 5 TNT sub-models × 6 protocols
│   ├── 01-06: SingleMode (flow, startup, relaxation, creep, SAOS, LAOS)
│   ├── 07-12: Cates (flow, startup, relaxation, creep, SAOS, LAOS)
│   ├── 13-18: LoopBridge (flow, startup, relaxation, creep, SAOS, LAOS)
│   ├── 19-24: MultiSpecies (flow, startup, relaxation, creep, SAOS, LAOS)
│   └── 25-30: StickyRouse (flow, startup, relaxation, creep, SAOS, LAOS)
├── vlb/                         # 16 notebooks: VLB transient network models
│   ├── 01_vlb_flow_curve.ipynb
│   ├── 02_vlb_startup_shear.ipynb
│   ├── 03_vlb_stress_relaxation.ipynb
│   ├── 04_vlb_creep.ipynb
│   ├── 05_vlb_saos.ipynb
│   ├── 06_vlb_laos.ipynb
│   ├── 07_vlb_bayesian_workflow.ipynb       # Bayesian inference
│   ├── 08_vlb_bell_shear_thinning.ipynb     # Bell model extension
│   ├── 09_vlb_fene_extensional.ipynb        # FENE extensibility
│   ├── 10_vlb_nonlocal_banding.ipynb        # Shear banding PDE
│   ├── 11_vlb_flow_curve_nlsq_to_nuts.ipynb # NLSQ → NUTS workflows
│   ├── 12_vlb_creep_nlsq_to_nuts.ipynb
│   ├── 13_vlb_stress_relaxation_nlsq_to_nuts.ipynb
│   ├── 14_vlb_startup_shear_nlsq_to_nuts.ipynb
│   ├── 15_vlb_saos_nlsq_to_nuts.ipynb
│   └── 16_vlb_laos_nlsq_to_nuts.ipynb
└── verification/                # 31 notebooks: Cross-model validation
    ├── 00_verification_index.ipynb
    ├── 01-06: Protocol validators (flow, creep, relaxation, startup, SAOS, LAOS)
    ├── creep/                   # 3 notebooks (mucus, perihepatic abscess, polystyrene)
    ├── oscillation/             # 13 notebooks (mastercurves, model evaluation, material-specific)
    ├── relaxation/              # 7 notebooks (fish muscle, laponite, foams, polyethylene, etc.)
    └── rotation/                # 1 notebook (emulsion)
```

See `examples/README.md` for learning path guide.

## Graphical User Interface (GUI)

RheoJAX includes an optional GUI built with PySide6/Qt6 for interactive analysis:

### Installation

```bash
pip install rheojax[gui]
```

### Launching

```bash
# From command line
rheojax-gui

# Start maximized (useful on high-DPI desktops)
rheojax-gui --maximized

# Or from Python
from rheojax.gui import main
main()
```

### Features

- **Data Loading**: Import CSV, Excel, TRIOS, and Anton Paar formats with preview
- **Model Fitting**: Interactive NLSQ curve fitting with real-time visualization
- **Bayesian Inference**: MCMC sampling with progress tracking
- **Diagnostics**: ArviZ plots (trace, forest, pair, energy, ESS, rank, autocorr)
- **Transforms**: Apply mastercurve, FFT, and derivative transforms
- **Export**: Save results, figures, and reports in multiple formats

See [GUI Reference Guide](https://rheojax.readthedocs.io/user_guide/06_gui/index.html) for detailed documentation.

## Documentation

Documentation: [https://rheojax.readthedocs.io](https://rheojax.readthedocs.io)

### Key Topics

- [Getting Started](https://rheojax.readthedocs.io/user_guide/getting_started.html) - Installation and basic usage
- [Core Concepts](https://rheojax.readthedocs.io/user_guide/core_concepts.html) - RheoData, Parameters, Test Modes
- [Bayesian Inference](https://rheojax.readthedocs.io/user_guide/03_advanced_topics/bayesian_inference.html) - NLSQ → NUTS workflow, ArviZ diagnostics
- [SGR Analysis](https://rheojax.readthedocs.io/user_guide/03_advanced_topics/sgr_analysis.html) - Soft Glassy Rheology framework
- [SPP Analysis](https://rheojax.readthedocs.io/user_guide/03_advanced_topics/spp_analysis.html) - Sequence of Physical Processes for LAOS
- [GUI Reference](https://rheojax.readthedocs.io/user_guide/06_gui/index.html) - Graphical user interface
- [Pipeline API](https://rheojax.readthedocs.io/user_guide/pipeline_api.html) - High-level workflows
- [I/O Guide](https://rheojax.readthedocs.io/user_guide/io_guide.html) - Reading and writing data
- [Visualization Guide](https://rheojax.readthedocs.io/user_guide/visualization_guide.html) - Creating plots
- [API Reference](https://rheojax.readthedocs.io/api_reference.html) - API documentation

### Model Handbooks

- [SGR Models](https://rheojax.readthedocs.io/models/sgr/sgr_conventional.html) - SGR Conventional and GENERIC models
- [STZ Models](https://rheojax.readthedocs.io/models/stz/stz_conventional.html) - Shear Transformation Zone (Langer 2008)
- [EPM Models](https://rheojax.readthedocs.io/models/epm/lattice_epm.html) - Elasto-Plastic lattice and tensorial models
- [Fluidity Models](https://rheojax.readthedocs.io/models/fluidity/fluidity_local.html) - Local and nonlocal cooperative flow
- [Fluidity-Saramito Models](https://rheojax.readthedocs.io/models/fluidity/saramito.html) - Tensorial EVP with thixotropy
- [IKH Models](https://rheojax.readthedocs.io/models/ikh/index.html) - MIKH and ML-IKH for thixotropic EVP materials
- [SPP Models](https://rheojax.readthedocs.io/models/spp/spp_decomposer.html) - SPP Decomposer and Yield Stress models
- [Giesekus Models](https://rheojax.readthedocs.io/models/giesekus/index.html) - Nonlinear viscoelastic polymer solutions
- [DMT Models](https://rheojax.readthedocs.io/models/dmt/index.html) - de Souza Mendes-Thompson thixotropic models
- [ITT-MCT Models](https://rheojax.readthedocs.io/models/itt_mct/index.html) - Mode-Coupling Theory for dense colloids
- [TNT Models](https://rheojax.readthedocs.io/models/tnt/index.html) - Transient network theory (5 variants)
- [VLB Models](https://rheojax.readthedocs.io/models/vlb/index.html) - Vernerey-Long-Brighenti transient networks
- [HVM Models](https://rheojax.readthedocs.io/models/hvm/index.html) - Hybrid Vitrimer Model
- [HVNM Models](https://rheojax.readthedocs.io/models/hvnm/index.html) - Hybrid Vitrimer Nanocomposite Model

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
  title = {RheoJAX: JAX-Powered Rheological Analysis with Bayesian Inference},
  year = {2024-2026},
  author = {Wei Chen},
  url = {https://github.com/imewei/rheojax},
  version = {0.6.0}
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
