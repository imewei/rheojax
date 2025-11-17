# Test Data Fixtures for RheoJAX v0.4.0

This directory contains test data fixtures and fixture generation utilities for the RheoJAX v0.4.0 validation and performance testing suite.

## Overview

The test fixtures support validation of three major features in v0.4.0:

1. **Mode-Aware Bayesian Inference** (CORRECTNESS FIX)
   - Validates that Bayesian posterior distributions correctly respect test mode (relaxation, creep, oscillation)
   - Uses reference data generated from known models for comparison

2. **GMM Element Search Optimization** (PERFORMANCE)
   - Validates that warm-starting reduces element minimization latency 2-5x
   - Ensures optimal N selection matches v0.3.1 baseline
   - Benchmarks Prony series accuracy

3. **TRIOS Large File Auto-Chunking** (MEMORY EFFICIENCY)
   - Validates that auto-chunking reduces peak memory usage 50-70%
   - Ensures data integrity for chunked vs full-load RheoData
   - Measures latency overhead (<20%)

## Files in This Directory

### Fixture Generation Script

- **`generate_test_data.py`** - Master fixture generation utility
  - Synthetic TRIOS file generation at multiple sizes (1-100 MB)
  - Reference data generation for all three test modes
  - RheoData factory functions for pytest integration
  - Reproducible synthetic data (seeded with numpy seed=42)

### Synthetic Data Files

Generated TRIOS files (synthetic rheological data in TRIOS format):

| File | Size | Points | Type | Use Case |
|------|------|--------|------|----------|
| `trios_synthetic_1mb.txt` | 0.2 MB | 5,000 | TRIOS | Auto-chunk threshold testing |
| `trios_synthetic_5mb.txt` | 1.0 MB | 25,000 | TRIOS | Memory profiling baseline |
| `trios_synthetic_10mb.txt` | 2.1 MB | 50,000 | TRIOS | Chunked integrity validation |

Additional sizes can be generated using `generate_trios_files_batch()`.

### Documentation

- **`README.md`** - This file
- **`BASELINE_REPORT.md`** - Task Group 1 baseline test results and environment info

## Usage Examples

### Generating Test Files

```python
# Generate TRIOS files programmatically
from tests.fixtures.generate_test_data import generate_synthetic_trios_file, generate_trios_files_batch

# Single file
trios_file = generate_synthetic_trios_file(
    target_size_mb=10.0,
    output_path="tests/fixtures/trios_large.txt"
)

# Batch generation
files = generate_trios_files_batch(
    sizes_mb=[1, 5, 10, 50, 100],
    output_dir="tests/fixtures/"
)
for size_mb, path in files.items():
    print(f"Generated {size_mb} MB file: {path}")
```

### Generating Reference Data

```python
# Generate synthetic data with known ground truth
from tests.fixtures.generate_test_data import (
    generate_relaxation_reference_data,
    generate_creep_reference_data,
    generate_oscillation_reference_data,
)

# Relaxation mode (Maxwell model)
t, G_t = generate_relaxation_reference_data(
    num_points=1000,
    model_type="maxwell",
    noise_level=0.01,  # 1% Gaussian noise
    seed=42
)

# Creep mode (Maxwell model)
t, J_t = generate_creep_reference_data(
    num_points=500,
    model_type="fractional_zener",
    noise_level=0.02,
    seed=42
)

# Oscillation mode (Maxwell model)
omega, G_star = generate_oscillation_reference_data(
    num_points=800,
    model_type="maxwell",
    noise_level=0.01,
    seed=42
)
```

### Using RheoData Factory Functions

```python
# Create RheoData objects with proper metadata
from tests.fixtures.generate_test_data import (
    create_rheo_data_relaxation,
    create_rheo_data_creep,
    create_rheo_data_oscillation,
)

# Relaxation data
rheo_relax = create_rheo_data_relaxation(
    model_type="maxwell",
    num_points=1000,
    noise_level=0.01,
    seed=42
)
print(f"Domain: {rheo_relax.domain}")  # 'time'
print(f"Material: {rheo_relax.metadata.get('material_name')}")
print(f"Shape: {rheo_relax.x.shape}, {rheo_relax.y.shape}")

# Creep data
rheo_creep = create_rheo_data_creep(model_type="maxwell")

# Oscillation data
rheo_osc = create_rheo_data_oscillation(model_type="fractional_zener")
```

### In pytest Tests

```python
# tests/validation/test_bayesian_mode_aware.py
import pytest
from tests.fixtures.generate_test_data import create_rheo_data_relaxation

@pytest.mark.validation
@pytest.mark.parametrize("model_type", ["maxwell", "fractional_zener"])
def test_bayesian_relaxation_mode(model_type):
    # Create synthetic reference data
    rheo_data = create_rheo_data_relaxation(
        model_type=model_type,
        num_points=1000,
        noise_level=0.01
    )

    # Fit model and perform Bayesian inference
    model = create_model(model_type)
    result = model.fit_bayesian(rheo_data, num_samples=2000)

    # Validate MCMC diagnostics
    assert result.diagnostics['r_hat'] < 1.01
    assert result.diagnostics['ess'] > 400
    assert result.diagnostics['divergences'] < 0.01

    # Validate posterior accuracy vs pyRheo reference
    assert posterior_within_tolerance(result, reference_data=rheo_data, tol=0.05)
```

## Data Characteristics

### Synthetic TRIOS Files

- **Format**: Standard TRIOS text format with header and data columns
- **Columns**: Time(s), Stress(Pa), Strain, Note
- **Material**: Synthetic test data (not real rheological measurements)
- **Model**: Relaxation curves based on Maxwell model with single relaxation time
- **Data Quality**: Noise-free (suitable for algorithm testing)
- **Reproducibility**: All data generation uses fixed random seed (seed=42)

### Reference Data

All reference datasets are generated with:

- **Time Range** (relaxation, creep): 10⁻³ to 10⁵ seconds (log-spaced)
- **Frequency Range** (oscillation): 0.01 to 1000 rad/s (log-spaced)
- **Model Parameters**:
  - Shear modulus G₀: 1 MPa (10⁶ Pa)
  - Infinite modulus G∞: 100 kPa (10⁵ Pa) [fractional models only]
  - Relaxation time τ: 1.0 second
  - Fractional exponent α: 0.7 [fractional models only]
- **Noise**: Gaussian, 1% std by default (configurable)
- **Precision**: Float64 throughout

## Performance Characteristics

### Fixture Generation

```
Operation                    | Typical Time | Memory
-------------------------------------------------
Generate 1 MB TRIOS file    | <100 ms      | <5 MB
Generate 10 MB TRIOS file   | <500 ms      | <30 MB
Generate 100 MB TRIOS file  | ~5 sec       | <300 MB
Generate reference data     | <50 ms       | <10 MB
Create RheoData object      | <10 ms       | <5 MB
```

### File Sizes

Target vs actual sizes (due to TRIOS format overhead):

| Target | Actual | Points | Notes |
|--------|--------|--------|-------|
| 1 MB   | ~0.2 MB | 5K    | Header overhead ~80% |
| 5 MB   | ~1.0 MB | 25K   | Header overhead ~80% |
| 10 MB  | ~2.1 MB | 50K   | Header overhead ~80% |
| 50 MB  | ~10 MB  | 250K  | Scales linearly |
| 100 MB | ~21 MB  | 500K  | Scales linearly |

For memory profiling tests requiring specific file sizes, adjust `points_per_mb` parameter in `generate_synthetic_trios_file()`.

## Validation Requirements

### Bayesian Mode-Aware Tests

Required fixtures:
- Relaxation mode data (Maxwell, FZSS, FML)
- Creep mode data (Maxwell, FZSS, FZLL)
- Oscillation mode data (Maxwell, FZSS, FMG)
- Reference posteriors from pyRheo (if available)
- ANSYS APDL reference data (if available)

### GMM Element Search Tests

Required fixtures:
- Multi-decade relaxation data (10⁻³ to 10⁵ s)
- Known optimal N for various optimization factors
- Prony series parameters for comparison
- Benchmark baselines from v0.3.1

### TRIOS Auto-Chunking Tests

Required fixtures:
- TRIOS files of varying sizes (1 MB, 5 MB, 10 MB, 50 MB, 100 MB)
- Known RheoData results for comparison
- Metadata preservation validation
- Memory profiling baselines

## Adding New Fixtures

To add new test data fixtures:

1. **Add generator function** to `generate_test_data.py`
   ```python
   def generate_my_custom_data(...) -> tuple[np.ndarray, np.ndarray]:
       """Generate custom test data."""
       # Implementation
       return x, y
   ```

2. **Add factory function** for RheoData creation
   ```python
   def create_rheo_data_custom(...) -> RheoData:
       """Create RheoData object for custom mode testing."""
       x, y = generate_my_custom_data(...)
       return RheoData(x=x, y=y, domain="...", metadata={...})
   ```

3. **Document** in docstrings with examples

4. **Test** by running the generator script
   ```bash
   python tests/fixtures/generate_test_data.py
   ```

5. **Update** this README with new fixtures

## References

### Synthetic Data Models

- **Maxwell**: G(t) = G₀ × exp(-t/τ)
- **Fractional Zener Solid-Solid**: G(t) = G∞ + (G₀ - G∞) × Eα(-t^α/τ)
  - Eα: Mittag-Leffler function

### Test Modes

- **Relaxation**: Time-dependent modulus decay after step strain (single valued function)
- **Creep**: Time-dependent compliance growth under constant stress (single valued function)
- **Oscillation**: Frequency-dependent complex modulus under oscillatory strain (complex-valued, magnitude used)

### Data Formats

- **TRIOS**: Standard format for rheological data (ASCII text with header)
- **NumPy**: Binary HDF5 format via `rheojax.io.writers.h5`
- **Excel**: XLS/XLSX format via `rheojax.io.writers.excel`

## Troubleshooting

### "ModuleNotFoundError: No module named 'rheojax'"

Ensure RheoJAX is installed in development mode:
```bash
cd /path/to/rheojax
pip install -e .
```

### "ImportError: cannot import name 'mittag_leffler'"

Some older versions may not have the Mittag-Leffler function. Update RheoJAX:
```bash
pip install --upgrade rheojax
```

### "File size smaller than target"

Expected due to TRIOS format overhead. Use `points_per_mb` parameter to adjust:
```python
# Generate larger file
generate_synthetic_trios_file(
    target_size_mb=10,
    points_per_mb=10000  # Doubled from default 5000
)
```

## Related Files

- **Test specification**: `/agent-os/specs/2025-11-16-rheojax-v0.4.0-category-c/spec.md`
- **Task breakdown**: `/agent-os/specs/2025-11-16-rheojax-v0.4.0-category-c/tasks.md`
- **Baseline report**: `BASELINE_REPORT.md` (this directory)

## License

Test fixtures are part of RheoJAX and follow the same MIT license.
