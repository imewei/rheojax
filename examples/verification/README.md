# Rheological Protocol Validation Suite

This directory contains a standardized validation suite for verifying rheological experimental data across the 6 core protocols.

## Purpose

These notebooks validate **data integrity and protocol correctness**, not model fitting. They ensure that:

1. Data files contain required columns with correct types
2. Values are finite (no NaN/Inf)
3. Physical constraints are satisfied (positive time, non-negative moduli, etc.)
4. Derived quantities are computed correctly
5. Standard diagnostic plots are generated

## Notebooks

| Notebook | Protocol | Description |
|----------|----------|-------------|
| [00_verification_index.ipynb](00_verification_index.ipynb) | Index | Dashboard with protocol coverage and status |
| [01_validate_flow_curve.ipynb](01_validate_flow_curve.ipynb) | Flow Curve | σ vs γ̇, shear-thinning detection |
| [02_validate_creep.ipynb](02_validate_creep.ipynb) | Creep | J(t) = γ(t)/σ₀, monotonic compliance |
| [03_validate_stress_relaxation.ipynb](03_validate_stress_relaxation.ipynb) | Relaxation | G(t) = σ(t)/γ₀, monotonic decay |
| [04_validate_startup_shear.ipynb](04_validate_startup_shear.ipynb) | Startup | σ(t) at constant γ̇, overshoot detection |
| [05_validate_saos.ipynb](05_validate_saos.ipynb) | SAOS | G'(ω), G''(ω), tan(δ) |
| [06_validate_laos.ipynb](06_validate_laos.ipynb) | LAOS | Harmonics, Lissajous curves |

## Quick Start

### FAST Mode (2 files per protocol, ~1 min)

```bash
cd examples/verification
for nb in 0[1-6]_validate_*.ipynb; do
    uv run jupyter execute "$nb" --inplace
done
```

### FULL Mode (all files)

Edit each notebook to set `MODE = "FULL"`, then run:

```bash
cd examples/verification
for nb in 0[1-6]_validate_*.ipynb; do
    uv run jupyter execute "$nb" --inplace
done
```

### Individual Protocol

```bash
uv run jupyter execute examples/verification/01_validate_flow_curve.ipynb --inplace
```

## Configuration

Each notebook has a configuration cell at the top:

```python
MODE = "FAST"  # or "FULL"

if MODE == "FAST":
    MAX_FILES = 2
    SKIP_HEAVY_PLOTS = True
    SAVE_ARTIFACTS = False
else:
    MAX_FILES = None
    SKIP_HEAVY_PLOTS = False
    SAVE_ARTIFACTS = True
```

## Protocol Detection Logic

| Protocol | Data Directory | File Patterns | Key Columns |
|----------|----------------|---------------|-------------|
| Flow Curve | `data/flow/` | `*.csv` | Shear Rate, Stress |
| Creep | `data/creep/` | `*.csv`, `*.txt` | Time, Creep Compliance |
| Relaxation | `data/relaxation/` | `*.csv` | Time, Relaxation Modulus |
| Startup | `data/ikh/*.xlsx` | StartUp_* sheets | Step time, Stress |
| SAOS | `data/oscillation/` | `*.csv`, `*.txt` | Angular Frequency, G', G'' |
| LAOS | `data/laos/`, `data/ikh/` | `*.txt`, LAOS_* sheets | Time, Strain, Stress |

## Output Directory Structure

Validation artifacts are saved to (in FULL mode):

```
outputs/
├── flow_curve/
│   ├── cleaned_data/       # Preprocessed data files
│   ├── derived_quantities/ # Computed quantities (η, etc.)
│   ├── plots/              # Validation plots
│   └── validation_report.json
├── creep/
├── stress_relaxation/
├── startup_shear/
├── saos/
└── laos/
```

## Validation Checks by Protocol

### Common Checks (All Protocols)
- Schema validation: Required columns present
- Finite values: No NaN/Inf
- Monotonicity: Time/frequency increasing

### Flow Curve
- Shear rate > 0 (strictly positive)
- Stress ≥ 0 (non-negative)
- Shear-thinning detection (η decreases with γ̇)

### Creep
- Time > 0, monotonically increasing
- Compliance J(t) > 0
- Monotonically increasing compliance

### Stress Relaxation
- Time > 0, monotonically increasing
- Modulus G(t) > 0
- Monotonically decreasing modulus
- Equilibrium modulus G_eq/G_0 ratio

### Startup Shear
- Time ≥ 0, monotonically increasing
- Stress > 0 after initial ramp
- Overshoot detection: σ_max/σ_ss ratio

### SAOS
- Frequency ω > 0
- G' ≥ 0, G'' ≥ 0
- Derived: |G*|, tan(δ)
- Material classification (solid/liquid/viscoelastic)

### LAOS
- Uniform sampling (constant Δt)
- Fundamental frequency recovery
- Harmonic analysis: I_n amplitudes
- Nonlinearity metric: I₃/I₁
- Periodicity check

## Shared Utilities

The `utils/validation_utils.py` module provides:

- `discover_files_by_protocol()`: Find data files for each protocol
- `validate_schema()`: Check required columns
- `check_finite()`, `check_positive()`, `check_monotonic()`: Value checks
- `compute_*_derived()`: Derived quantity calculations
- `plot_*()`: Standard visualization functions
- `write_validation_report()`: JSON report generation

## References

- Ferry, J.D. "Viscoelastic Properties of Polymers" (1980)
- Macosko, C.W. "Rheology: Principles, Measurements, and Applications" (1994)
- Ewoldt et al. "New measures for characterizing nonlinear viscoelasticity" (2008) J. Rheol.
- Tschoegl, N.W. "Phenomenological Theory of Linear Viscoelastic Behavior" (1989)
