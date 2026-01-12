# Example Datasets

This directory contains experimental datasets used in RheoJAX example notebooks, organized by **measurement type** for intuitive access. Synthetic datasets are generated programmatically within notebooks rather than stored as files.

## Git LFS Requirement

**Large experimental files are stored using Git Large File Storage (LFS).**

The OWChirp datasets (`owchirp_tcs.txt`, `owchirp_tts.txt`) are large files (66-80MB each, 146MB total) stored with Git LFS to avoid bloating the repository.

**Installation:**
```bash
# Install Git LFS (one-time setup)
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs  # Debian/Ubuntu
sudo yum install git-lfs      # CentOS/RHEL

# Windows
# Download from https://git-lfs.github.com/

# Initialize Git LFS in your repository
git lfs install

# Clone normally - LFS files download automatically
git clone https://github.com/imewei/rheojax.git
```

**Verification:**
```bash
# Check that LFS files downloaded correctly
git lfs ls-files
# Should show: owchirp_tcs.txt, owchirp_tts.txt

# If files didn't download, fetch them manually
git lfs pull
```

## Directory Structure (Organized by Measurement Type)

```
data/
├── oscillation/              # SAOS frequency sweeps (G', G'' vs frequency)
│   ├── polystyrene/          # Polystyrene at various temperatures
│   ├── polyisoprene/         # Linear polyisoprene molecular weight series
│   ├── foods/                # Food materials (chia pudding)
│   ├── metal_networks/       # Metal-coordinating polymer networks
│   ├── hydrogels/            # (Empty - TTS data in temperature_sweep/)
│   └── foams/                # (Empty - TTS data in temperature_sweep/)
│
├── relaxation/               # Stress relaxation G(t) vs time
│   ├── polymers/             # PP, PS, PE, HDPE relaxation data
│   ├── biological/           # Fish muscle
│   ├── foams/                # Shaving foam (liquid foam)
│   └── clays/                # Laponite at various aging times
│
├── creep/                    # Creep compliance J(t) vs time
│   ├── polymers/             # Polystyrene creep data
│   ├── biological/           # Mucus, perihepatic abscess
│   └── creep_experiment.txt  # TRIOS creep measurement
│
├── flow/                     # Steady shear flow (viscosity vs shear rate)
│   ├── hydrogels/            # CNF hydrogel flow curves
│   ├── emulsions/            # Oil-water emulsions at various volume fractions
│   └── solutions/            # Ethyl cellulose solutions
│
├── temperature_sweep/        # Multi-temperature data for TTS/mastercurves
│   ├── polymers/             # frequency_sweep_tts.txt (150°C to -40°C)
│   ├── hydrogels/            # Dual network hydrogel (CNF-MC-Zn)
│   ├── foams/                # Vinyl foam DMA (-35°C to 60°C)
│   └── time_temp_water/      # Time-temperature-water superposition data
│
├── laos/                     # Large Amplitude Oscillatory Shear
│   ├── owchirp_tts.txt       # OWChirp temperature sweep (80 MB, Git LFS)
│   ├── owchirp_tcs.txt       # OWChirp during gelation (66 MB, Git LFS)
│   ├── raw_signal_0010.txt   # Low amplitude raw signal
│   ├── raw_signal_0100.txt   # Medium amplitude raw signal
│   └── raw_signal_1000.txt   # High amplitude raw signal
│
├── mastercurves/             # Pre-processed master curves and Prony data
│   ├── master_curve_ps_*.csv # Polystyrene mastercurves
│   ├── freq_master/          # Frequency-domain master curves
│   ├── freq_raw/             # Raw frequency sweep data
│   ├── time_master/          # Time-domain master curves + Prony series
│   └── examples/             # Additional example datasets
│
└── multi_technique/          # Combined measurement files
    └── multi_technique.txt   # Relaxation + oscillation + flow
```

## Privacy and Anonymization

**All experimental data files have been fully anonymized for privacy protection:**

- **Instrument identifiers**: Serial numbers, instrument names → `ANONYMIZED`
- **Personal information**: Operators, users, project names → `Anonymous`
- **Document metadata**: File paths, document IDs → `ANONYMIZED`
- **Dates**: Specific dates → `MM/DD/YYYY`
- **Sample details**: Sample names, notes → `Anonymous`

The original TRIOS file structure is preserved to ensure compatibility with `rheojax.io.readers.TRIOSReader` for educational and testing purposes.

## Dataset Catalog by Category

### Oscillation Data

| Subfolder | Files | Material | Temperature | Use Cases |
|-----------|-------|----------|-------------|-----------|
| `polystyrene/` | oscillation_ps*.csv | Polystyrene | 130-190°C | SAOS model fitting |
| `polyisoprene/` | PI_*.csv | Linear PI | -35°C | Molecular weight dependence |
| `foods/` | oscillation_chia_data.csv | Chia pudding | Ambient | Viscoelastic food materials |
| `metal_networks/` | epstein.csv | Metal-coordinated | Various | Supramolecular networks |

### Relaxation Data

| Subfolder | Files | Material | Use Cases |
|-----------|-------|----------|-----------|
| `polymers/` | stressrelaxation_*.csv | PP, PS, PE | Maxwell, Zener, fractional models |
| `biological/` | stressrelaxation_fishmuscle_data.csv | Fish muscle | Biological tissue mechanics |
| `foams/` | stressrelaxation_liquidfoam_data.csv | Shaving foam | Soft matter relaxation |
| `clays/` | rel_lapo_*.csv | Laponite | Aging, thixotropy |

### Creep Data

| Subfolder | Files | Material | Use Cases |
|-----------|-------|----------|-----------|
| `polymers/` | creep_ps*.csv | Polystyrene | Kelvin-Voigt, Burgers models |
| `biological/` | creep_mucus_data.csv | Mucus | Biopolymer creep |
| `biological/` | creep_perihepatic_data.csv | Tissue | Medical rheology |

### Flow Data

| Subfolder | Files | Material | Use Cases |
|-----------|-------|----------|-----------|
| `hydrogels/` | cellulose_hydrogel_flow.csv | CNF hydrogel | Power-law, Herschel-Bulkley |
| `emulsions/` | 0.69.csv - 0.80.csv | Emulsions | Volume fraction dependence |
| `solutions/` | ec_shear_viscosity_*.csv | EC solutions | Concentration dependence |

### Temperature Sweep (TTS) Data

| Subfolder | Files | Material | Temp Range | Use Cases |
|-----------|-------|----------|------------|-----------|
| `polymers/` | frequency_sweep_tts.txt | Polymer | 150 to -40°C | WLF, mastercurve |
| `hydrogels/` | cnf_*.csv | Dual network | 10-80°C | Hydrogel TTS |
| `foams/` | foam_dma_*.csv | Vinyl foam | -35 to 60°C | DMA analysis |
| `time_temp_water/` | ttw_*.csv | Various | 20-50°C | Combined effects |

### LAOS Data

| File | Size | Description | Use Cases |
|------|------|-------------|-----------|
| owchirp_tts.txt | 80 MB | OWChirp temp sweep | Nonlinear mastercurves |
| owchirp_tcs.txt | 66 MB | Gelation OWChirp | Mutation number, sol-gel |
| raw_signal_*.txt | ~45 KB | Raw waveforms | SPP analysis tutorial |

### Mastercurves (Pre-processed)

| Subfolder | Files | Content |
|-----------|-------|---------|
| `freq_master/` | freq_user_master.csv | Frequency-domain mastercurve |
| `time_master/` | time_user_master.csv | Time-domain mastercurve |
| `time_master/` | prony_terms_*.csv | Prony series coefficients |

## Loading Data Examples

### Loading by Measurement Type

```python
from pathlib import Path
from rheojax.io import load_csv, load_trios

# Path to data directory
data_dir = Path('/path/to/rheojax/examples/data')

# Relaxation data
polypropylene = load_csv(
    data_dir / 'relaxation' / 'polymers' / 'polypropylene_relaxation.csv',
    x_col='Time',
    y_col='Relaxation Modulus'
)

# Temperature sweep for TTS
tts_data = load_trios(data_dir / 'temperature_sweep' / 'polymers' / 'frequency_sweep_tts.txt')

# LAOS data
laos_data = load_trios(data_dir / 'laos' / 'owchirp_tts.txt')

# Or use Pipeline API with auto-detection
from rheojax.pipeline import Pipeline
pipeline = Pipeline().load(str(data_dir / 'oscillation' / 'polystyrene' / 'oscillation_ps130_data.csv'))
```

### Finding Data by Purpose

```python
# For model fitting examples:
# - oscillation/ → SAOS models (Maxwell, Zener, fractional)
# - relaxation/ → Stress relaxation models
# - creep/ → Creep models (Kelvin-Voigt, Burgers)
# - flow/ → Flow models (Power-law, Bingham, Herschel-Bulkley)

# For transform examples:
# - temperature_sweep/ → Mastercurve construction, WLF fitting
# - laos/ → SPP decomposition, OWChirp analysis

# For Bayesian examples:
# - Most use synthetic data generated in-notebook
# - verification/ notebooks use these real datasets for validation
```

## Dataset Usage by Notebook Category

| Category | Primary Data Source | Location |
|----------|--------------------|-----------|
| **Basic Model Fitting** | Synthetic (in-notebook) | N/A |
| **Transform Workflows** | Temperature sweep | `temperature_sweep/` |
| **LAOS Analysis** | OWChirp data | `laos/` |
| **Bayesian Inference** | Synthetic (in-notebook) | N/A |
| **Verification** | Real experimental | All categories |

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

## Data Quality Notes

### Experimental Data Characteristics

Real experimental files may have:
- **Truncated ranges**: Limited frequency/time coverage
- **Missing points**: Gaps in data collection
- **Outliers**: Instrument artifacts
- **Drift**: Temperature-dependent variations

Handle with RheoData operations:
```python
from rheojax.core.data import RheoData

data = RheoData(x, y)
data_clean = data.smooth(window_length=5)      # Smoothing
data_interp = data.resample(new_x=desired_x)   # Resampling
```

## Contributing Datasets

To add new experimental datasets:

1. **Identify the measurement type** (oscillation, relaxation, creep, flow, etc.)
2. **Place file in appropriate category folder**
3. **Add entry to this README** with:
   - Material description
   - Test mode and conditions
   - Data format and column names
   - Use cases and applications
   - Source/citation
4. **Create loading example** in relevant notebook
5. **Update dataset usage table** above

## Data Availability and Licensing

### Synthetic Data
All synthetic data generation code is public domain - use freely. Generation code is embedded in notebooks with documented parameters.

### Experimental Data

**Sources:**
- **pyRheo demos**: Public domain educational examples (Puente-Córdova et al.)
- **hermes-rheo tutorials**: Public domain educational examples
- **Miranda-Valdez et al. (2024)**: Cellulose 31:1545-1558 (academic publication)

All experimental datasets are used for educational purposes under academic fair use.

---

**Last Updated:** 2025-01-12
**Total Datasets:** 167 files organized by measurement type
**Categories:** oscillation, relaxation, creep, flow, temperature_sweep, laos, mastercurves, multi_technique
