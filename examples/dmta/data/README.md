# DMTA Example Data

Real polymer DMTA data from the [pyvisco](https://github.com/NREL/pyvisco) project (NREL, MIT License).

## Files

| File | Lines | Description |
|------|-------|-------------|
| `freq_user_master.csv` | 208 | Frequency-domain master curve at reference temperature. Columns: `f` (Hz), `E_stor` (MPa), `E_loss` (MPa). |
| `freq_user_raw.csv` | 212 | Multi-temperature frequency sweeps (21 temperatures, -50 to 100 C). Columns: `f` (Hz), `E_stor` (MPa), `E_loss` (MPa), `T` (C), `Set` (temperature index). |
| `freq_user_master__shift_factors.csv` | 24 | Frequency-domain TTS shift factors. Columns: `T` (C), `log_aT`. |
| `time_user_master.csv` | 483 | Time-domain relaxation master curve. Columns: `t` (s), `E_relax` (MPa). |
| `time_user_raw.csv` | 212 | Multi-temperature relaxation raw data. Columns: `t` (s), `E_relax` (MPa), `T` (C), `Set` (temperature index). |
| `time_user_raw__shift_factors.csv` | 24 | Time-domain TTS shift factors. Columns: `T` (C), `log_aT`. |
| `prony_terms_reference.csv` | 33 | Reference Prony series terms (30 modes). Columns: `tau_i` (s), `E_i` (MPa). Equilibrium modulus E_0 ~ 1739 MPa. |

## Attribution

Data sourced from pyvisco v1.0 examples (MIT License):
- **Repository**: https://github.com/NREL/pyvisco
- **Citation**: Beurle, D., Hartmann, S., Schuster, A. *PYVISCO: A Python library for identifying Prony series parameters of linear viscoelastic materials*. Zenodo, 2022. DOI: 10.5281/zenodo.7191238
- **License**: MIT License (Copyright NREL)

## Usage in RheoJAX

```python
import pandas as pd

# Load multi-temperature raw data
df = pd.read_csv("freq_user_raw.csv", skiprows=[1])  # Skip units row
temperatures = df["T"].unique()

# Load master curve
df_master = pd.read_csv("freq_user_master.csv", skiprows=[1])

# Load shift factors
df_shifts = pd.read_csv("freq_user_master__shift_factors.csv", skiprows=[1])
```
