# Golden Data Harness for SPP Parity Testing

This document outlines a minimal, actionable plan to create a golden-data test harness for comparing RheoJAX's SPP (Sequentially-Processed-Projections) analysis with results from MATLAB's `SPPplus_v2p1` and R's `oreo` library.

The harness will be located in `tests/validation/spp_golden_data`.

### 1. Create Directory Structure

A dedicated directory will organize all artifacts for this harness.

```bash
mkdir -p tests/validation/spp_golden_data
```

### 2. Generate Synthetic Datasets

A Python script will generate three canonical datasets for the comparison. This ensures that all frameworks operate on identical source data.

**File:** `tests/validation/spp_golden_data/generate_datasets.py`

This script will output the following files into the same directory:
- `dataset_sinusoid_3rd_harmonic.csv`: A clean sinusoidal waveform with a 3rd harmonic.
- `dataset_sinusoid_3rd_harmonic_noisy.csv`: The same waveform with Gaussian noise.
- `dataset_amplitude_sweep.csv`: A waveform with varying amplitude to test dynamic response.

### 3. Cross-Language Analysis Scripts

Scripts for MATLAB, R, and Python will run the analysis and export the results in a standardized CSV format.

- **MATLAB:** `tests/validation/spp_golden_data/run_sppplus.m`
- **R:** `tests/validation/spp_golden_data/run_oreo.R`
- **RheoJAX:** `tests/validation/spp_golden_data/run_rheojax_spp.py`

These scripts will be responsible for loading a generated dataset and producing a corresponding `_output.csv` file (e.g., `matlab_output.csv`).

**Standard Output Columns:**
Each output CSV must contain the following columns for a valid comparison:
- `Gp_t`: Elastic modulus projection
- `Gpp_t`: Viscous modulus projection
- `G_star_t`: Complex modulus projection
- `delta_t`: Phase angle projection
- `G_speed`: Modulus rate of change
- `yield_stress_g_prime`
- `yield_stress_g_double_prime`
- `frenet_T`
- `frenet_N`
- `frenet_B`
- `frenet_kappa`
- `frenet_tau`

### 4. Pytest Comparison Test

A `pytest` suite will compare the CSV outputs from each framework against the RheoJAX output.

**File:** `tests/validation/test_spp_parity.py`

**Key Features:**
- **Markers:** Tests will be marked with `@pytest.mark.slow` and `@pytest.mark.integration` to separate them from fast unit tests.
- **Fixtures:** A fixture will manage the setup, ensuring that the golden data files (from MATLAB and R) are available for comparison.
- **Tolerances:** Comparisons will use `numpy.allclose` with a relative tolerance (`rtol`) of `1e-2` to account for floating-point discrepancies between languages. The test will fail if the deviation is larger.

### 5. Execution Workflow

1.  **Data Generation:** Run `generate_datasets.py` to create the input CSVs.
2.  **External Analysis:** Manually run the MATLAB and R scripts to produce the golden output files (`matlab_output.csv`, `r_output.csv`). These are the "golden masters."
3.  **Testing:** Run `pytest`. The `test_spp_parity.py` suite will automatically execute the RheoJAX script, generate `rheojax_output.csv`, and compare it against the pre-computed golden files.
