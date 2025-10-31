# Example Datasets

This directory contains experimental datasets used in Rheo example notebooks. Synthetic datasets are generated programmatically within notebooks rather than stored as files.

## Directory Structure

```
data/
└── experimental/    # Real instrument files (TRIOS, CSV)
    ├── cellulose_hydrogel_flow.csv
    ├── creep_experiment.txt
    ├── frequency_sweep_tts.txt
    ├── multi_technique.txt
    ├── owchirp_tcs.txt
    ├── owchirp_tts.txt
    ├── polypropylene_relaxation.csv
    └── polystyrene_creep.csv
```

## Privacy and Anonymization

**All experimental data files have been fully anonymized for privacy protection:**

- **Instrument identifiers**: Serial numbers, instrument names → `ANONYMIZED`
- **Personal information**: Operators, users, project names → `Anonymous`
- **Document metadata**: File paths, document IDs → `ANONYMIZED`
- **Dates**: Specific dates → `MM/DD/YYYY`
- **Sample details**: Sample names, notes → `Anonymous`

The original TRIOS file structure is preserved to ensure compatibility with `rheo.io.readers.TRIOSReader` for educational and testing purposes.

## Data Paradigms

### Synthetic Data (Generated In-Notebook)

**All synthetic datasets are generated programmatically within notebooks** using known parameters. This design choice provides:

- **Known ground truth** for validation (relative error < 1e-6)
- **Reproducibility** via fixed random seed (42)
- **Educational transparency** (students see generation code)
- **Easy modification** for experimentation
- **No external dependencies** (notebooks fully self-contained)

**Example from `01-maxwell-fitting.ipynb`:**
```python
import numpy as np

# Generate time array (logarithmically spaced for relaxation)
np.random.seed(42)  # Reproducibility
t = np.logspace(-2, 2, 50)  # 0.01 to 100 seconds

# Known parameters for validation
G0 = 1e5  # Pa
eta = 1e3  # Pa·s

# Generate relaxation data
G_t = G0 * np.exp(-G0 * t / eta)

# Add realistic noise (1.5% relative error)
noise = np.random.normal(0, 0.015 * G_t)
G_t_noisy = G_t + noise
```

**Parameter Values Used Across Notebooks:**

| Notebook | Model | Parameters | Time/Frequency Range | Noise Level |
|----------|-------|------------|---------------------|-------------|
| 01-maxwell-fitting | Maxwell | G₀=1×10⁵ Pa, η=1×10³ Pa·s | 0.01 - 100 s (50 points) | 1.5% relative |
| 02-zener-fitting | Zener/SLS | Generated in notebook | Frequency-domain | 1-2% relative |
| 03-springpot-fitting | SpringPot | Generated in notebook | Power-law decay | 1.5% relative |
| 04-bingham-fitting | Bingham | Generated in notebook | Flow curve | 1.5% relative |
| 05-power-law-fitting | Power-Law | Generated in notebook | Shear-thinning | 1.5% relative |

**Transform Notebooks:**
- **01-fft-analysis**: Maxwell relaxation with known analytical FFT solution
- **02-mastercurve-tts**: Uses `frequency_sweep_tts.txt` (experimental)
- **03-mutation-number**: Three synthetic materials (solid, viscoelastic, fluid)
- **04-owchirp-laos-analysis**: Synthetic LAOS data with controlled nonlinearity
- **05-smooth-derivative**: Synthetic noisy function with known analytical derivative

### Experimental Data (Real Instrument Files)

Real instrument data demonstrates authentic noise patterns, artifacts, and practical data handling. All files are located directly in `experimental/` directory.

## Experimental Dataset Catalog

### Basic Model Fitting Datasets (Phase 1.1.4)

Extracted from pyRheo legacy demos for educational notebooks. All files are CSV format with headers.

**1. `experimental/polypropylene_relaxation.csv`**
- **Material**: Polypropylene polymer
- **Test Mode**: Stress Relaxation
- **Data**: Time [s] vs Relaxation Modulus [Pa]
- **Points**: ~600 pts, 0.001-600 s
- **Use Cases**: Maxwell, Zener, Fractional Maxwell models
- **Source**: pyRheo demos (Puente-Córdova)
- **Notes**: Clean dataset, well-characterized material

**2. `experimental/polystyrene_creep.csv`**
- **Material**: Polystyrene melt (160°C)
- **Test Mode**: Creep Compliance
- **Data**: Time [s] vs Creep Compliance [Pa⁻¹]
- **Points**: ~50 pts, 0.2-400 s
- **Use Cases**: Kelvin-Voigt, Burgers, creep models
- **Source**: pyRheo demos
- **Notes**: High-temperature melt data

**3. `experimental/cellulose_hydrogel_flow.csv`**
- **Material**: Cellulose nanofiber (CNF) hydrogel
- **Test Mode**: Steady Shear Flow
- **Data**: Shear Rate [1/s] vs Viscosity [Pa·s]
- **Points**: ~40 pts, 0.1-100 1/s
- **Use Cases**: Power-Law, Bingham, Herschel-Bulkley models
- **Source**: Miranda-Valdez et al. (2024) Cellulose 31:1545-1558
- **Notes**: Shear-thinning behavior, possible yield stress

### Transform Workflow Datasets (Phase 2)

Extracted from hermes-rheo tutorial notebooks. All files are TRIOS instrument format (tab-delimited .txt with extensive metadata headers).

**4. `experimental/frequency_sweep_tts.txt`**
- **Format**: TA Instruments TRIOS .txt export
- **Test**: Oscillation temperature sweep (150°C → -50°C, 10°C steps)
- **Data**: G'(ω), G"(ω) vs frequency (0.1-20 Hz, 7 pts/decade)
- **Points**: 20 temperature steps × 7 pts/decade
- **Size**: 318 KB
- **Use Cases**: Mastercurve construction, WLF shift factor calculation
- **Source**: hermes-rheo tutorial_4
- **Notes**: Clean frequency sweeps, ideal for TTS demonstration

**5. `experimental/owchirp_tts.txt`**
- **Format**: TA Instruments TRIOS .txt export (OWChirp protocol)
- **Test**: Optimally Windowed Chirp temperature sweep
- **Data**: Time-domain strain/stress → FFT → G'(ω), G"(ω) (156 frequencies)
- **Points**: 20 temps × 156 freqs
- **Size**: 80 MB
- **Use Cases**: OWChirp/LAOS analysis, FFT demonstration, mastercurve with chirp data
- **Source**: hermes-rheo tutorial_4
- **Notes**: Includes raw time-domain waveforms + FFT results

**6. `experimental/owchirp_tcs.txt`**
- **Format**: TA Instruments TRIOS .txt export (OWChirp protocol)
- **Test**: OWChirp during gel curing (time-evolving material)
- **Data**: Time-series of G'(ω), G"(ω) during gelation
- **Points**: 20 steps × 156 freqs
- **Size**: 66 MB
- **Use Cases**: Mutation number calculation, time-curing superposition (TCS), gel evolution
- **Source**: hermes-rheo tutorial_5
- **Notes**: Demonstrates viscoelastic transition from sol → gel

**7. `experimental/creep_experiment.txt`**
- **Format**: TA Instruments TRIOS .txt export
- **Test**: Creep compliance measurement
- **Data**: J(t) vs time under constant stress
- **Size**: 203 KB
- **Use Cases**: Basic I/O, creep modeling, FFT conversion to frequency domain
- **Source**: hermes-rheo tutorial_1
- **Notes**: Multi-step creep protocol

**8. `experimental/multi_technique.txt`**
- **Format**: TA Instruments TRIOS .txt export
- **Test**: Multiple rheological tests on same material
- **Data**: Relaxation + oscillation + flow on same sample
- **Size**: 151 KB
- **Use Cases**: Multi-technique workflows, data comparison, format handling
- **Source**: hermes-rheo tutorial_1
- **Notes**: Demonstrates combining multiple test modes

## TRIOS Format Notes

TRIOS .txt files have this structure:
```
[Version]
TA Instruments LIMS export format version 0

[Header]
Document ID, File path, Instrument serial number...

[Sample]
Name, Operator, Project, Notes...

[Geometry]
Name, Diameter, Gap, Material...

[Procedure: Steps]
Step 1: Conditioning...
Step 2: Temperature equilibration...
Step 3: Data acquisition...

[Data]
<tab-delimited data columns>
```

## Loading Data Examples

### Loading Experimental Data

```python
from rheo.io.readers import read_csv, TriosReader
from pathlib import Path

# Path to data directory
data_dir = Path('/Users/b80985/Projects/Rheo/examples/data/experimental')

# CSV files (basic datasets)
polypropylene = read_csv(
    data_dir / 'polypropylene_relaxation.csv',
    x_col='Time',
    y_col='Relaxation Modulus'
)

# TRIOS files (transform datasets)
tts_data = TriosReader().read(data_dir / 'frequency_sweep_tts.txt')

# Or use Pipeline API with auto-detection
from rheo.pipeline import Pipeline
pipeline = Pipeline().load(str(data_dir / 'frequency_sweep_tts.txt'))
```

### Error Handling Best Practices

**Always check for file existence before loading to provide helpful error messages:**

```python
from pathlib import Path

data_path = Path('../data/experimental/frequency_sweep_tts.txt')

if not data_path.exists():
    print(f"❌ Data file not found: {data_path}")
    print("\nOptions:")
    print("  1. Check that you're running from the correct directory")
    print("  2. Verify the relative path is correct")
    print("  3. Download from: https://github.com/imewei/Rheo/tree/main/examples/data")
    raise FileNotFoundError(f"Required data file missing: {data_path}")

# File exists - proceed with loading
data = read_trios(data_path)
```

**For notebooks that can fall back to synthetic data:**

```python
from pathlib import Path

data_path = Path('../data/experimental/multi_technique.txt')

if data_path.exists():
    # Use real experimental data
    print("✓ Loading experimental data")
    data = read_trios(data_path)
else:
    # Generate synthetic data as fallback
    print("⚠ Experimental data not found - using synthetic data")
    import numpy as np
    np.random.seed(42)
    t = np.logspace(-2, 2, 50)
    G_t = 1e5 * np.exp(-t / 0.01)
    # ... generate synthetic dataset
```

### Generating Synthetic Data

See individual notebooks for complete examples. Basic pattern:

```python
import numpy as np
from rheo.models.maxwell import Maxwell

# Set seed for reproducibility
np.random.seed(42)

# Create model with known parameters
model = Maxwell()
model.parameters.set_value('G0', 1e5)  # Pa
model.parameters.set_value('eta', 1e3)  # Pa·s

# Generate time array
t = np.logspace(-2, 2, 50)  # 0.01 to 100 s

# Generate clean data
G_t_clean = model.predict(t)

# Add realistic noise (1.5% relative error)
noise_level = 0.015
noise = np.random.normal(0, noise_level * G_t_clean)
G_t_noisy = G_t_clean + noise
```

## Dataset Usage by Notebook

| Notebook | Primary Dataset | Data Type | Purpose |
|----------|-----------------|-----------|---------|
| **Basic Model Fitting** |
| 01-maxwell-fitting | Synthetic (generated) | In-notebook | Parameter estimation validation |
| 02-zener-fitting | Synthetic (generated) | In-notebook | Complex modulus fitting |
| 03-springpot-fitting | Synthetic (generated) | In-notebook | Fractional calculus models |
| 04-bingham-fitting | Synthetic (generated) | In-notebook | Yield stress materials |
| 05-power-law-fitting | Synthetic (generated) | In-notebook | Shear-thinning fluids |
| **Transform Workflows** |
| 01-fft-analysis | Synthetic (generated) | In-notebook | FFT validation with known solution |
| 02-mastercurve-tts | frequency_sweep_tts.txt | Experimental TRIOS | TTS mastercurve, WLF fitting |
| 03-mutation-number | Synthetic (generated) | In-notebook | Material classification (solid/VE/fluid) |
| 04-owchirp-laos-analysis | Synthetic (generated) | In-notebook | Harmonic extraction validation |
| 05-smooth-derivative | Synthetic (generated) | In-notebook | Derivative accuracy comparison |
| **Bayesian Inference** |
| 01-bayesian-basics | Synthetic (generated) | In-notebook | NLSQ → NUTS workflow demo |
| 02-prior-selection | Synthetic (generated) | In-notebook | Prior sensitivity analysis |
| 03-convergence-diagnostics | Synthetic (generated) | In-notebook | ArviZ diagnostic suite |
| 04-model-comparison | Synthetic (generated) | In-notebook | Bayesian model selection (WAIC/LOO) |
| 05-uncertainty-propagation | Synthetic (generated) | In-notebook | Credible intervals, predictions |
| **Advanced Workflows** |
| 01-multi-technique-fitting | Synthetic (generated) | In-notebook | Multiple test modes, same material |
| 02-batch-processing | Synthetic (generated) | In-notebook | Multiple datasets pipeline |
| 03-custom-models | Synthetic (generated) | In-notebook | User-defined model implementation |
| 04-fractional-deep-dive | Synthetic (generated) | In-notebook | Fractional calculus models |
| 05-performance-optimization | Synthetic (generated) | In-notebook | JAX/GPU acceleration benchmarks |

## Data Quality Notes

### Noise Levels in Synthetic Data

Standard noise levels used across notebooks:
- **1.5%**: Typical instrument precision (most notebooks)
- **1-2%**: Realistic experimental conditions
- **Clean (no noise)**: Validation only (analytical comparisons)

### Experimental Data Characteristics

Real experimental files may have:
- **Truncated ranges**: Limited frequency/time coverage
- **Missing points**: Gaps in data collection
- **Outliers**: Instrument artifacts
- **Drift**: Temperature-dependent variations

Handle with RheoData operations:
```python
from rheo.core.data import RheoData

data = RheoData(x, y)
data_clean = data.smooth(window_length=5)      # Smoothing
data_interp = data.resample(new_x=desired_x)   # Resampling
```

## Contributing Datasets

To add new experimental datasets:

1. **Place file in `experimental/` directory**
2. **Add entry to this README** with:
   - Material description
   - Test mode and conditions
   - Data format and column names
   - Use cases and applications
   - Source/citation
3. **Create loading example** in relevant notebook
4. **Update dataset usage table** above

## Data Availability and Licensing

### Synthetic Data
All synthetic data generation code is public domain - use freely. Generation code is embedded in notebooks with documented parameters.

### Experimental Data

**Sources:**
- **pyRheo demos**: Public domain educational examples (Puente-Córdova et al.)
- **hermes-rheo tutorials**: Public domain educational examples
- **Miranda-Valdez et al. (2024)**: Cellulose 31:1545-1558 (academic publication)

All experimental datasets are used for educational purposes under academic fair use.

## Troubleshooting Data Loading

### FileNotFoundError

**Problem:** `File not found: 'frequency_sweep_tts.txt'`

**Solution:** Use absolute paths or relative to examples directory:
```python
from pathlib import Path
data_dir = Path('/Users/b80985/Projects/Rheo/examples/data/experimental')
data_file = data_dir / 'frequency_sweep_tts.txt'
```

### Unit Mismatch

**Problem:** Data not scaling correctly

**Solution:** Check CSV headers or TRIOS metadata:
```python
import pandas as pd
df = pd.read_csv(data_file)
print(df.columns)  # Verify column names
```

### AttributeError with RheoData

**Problem:** `AttributeError: 'numpy.ndarray' has no attribute 'x_units'`

**Solution:** Use RheoData wrapper:
```python
# Wrong:
x, y = np.loadtxt(file, unpack=True)

# Right:
from rheo.core.data import RheoData
data = RheoData.from_csv(file)
print(data.x_units)
```

---

**Last Updated:** 2025-10-31
**Total Datasets:** 8 experimental files
**Synthetic Data:** Generated in-notebook (20 notebooks)
