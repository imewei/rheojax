# HVNM Tutorial Notebooks

Jupyter notebooks for the **Hybrid Vitrimer Nanocomposite Model (HVNM)**.

## Prerequisites

```bash
uv sync                         # Install dependencies
uv pip install openpyxl arviz   # For Excel data + ArviZ diagnostics
```

## Notebooks

### Synthetic Demos (01-07)

Protocol demonstrations with synthetic data — no fitting required.

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_hvnm_saos.ipynb` | SAOS: G'/G'', Guth-Gold, temperature sweeps |
| 02 | `02_hvnm_stress_relaxation.ipynb` | Tri-exponential relaxation G(t) |
| 03 | `03_hvnm_startup_shear.ipynb` | TST stress overshoot, phi-effect |
| 04 | `04_hvnm_creep.ipynb` | Strain amplification, creep compliance |
| 05 | `05_hvnm_flow_curve.ipynb` | Steady-state viscosity, yield stress |
| 06 | `06_hvnm_laos.ipynb` | Lissajous curves, LAOS harmonics |
| 07 | `07_hvnm_limiting_cases.ipynb` | Factory methods, phi=0 verification |

### NLSQ-to-NUTS Inference (08-15)

Bayesian inference on **real experimental data** — NLSQ warm-start then NUTS sampling.

| # | Notebook | Protocol | Data | Free Params |
|---|----------|----------|------|-------------|
| 08 | `08_data_intake_and_qc.ipynb` | All 6 | All datasets | N/A (QC only) |
| 09 | `09_flow_curve_nlsq_nuts.ipynb` | Flow curve | EC 7% solution | G_D, k_d_D |
| 10 | `10_creep_compliance_nlsq_nuts.ipynb` | Creep | PS 190 C | G_P, G_E, G_D, nu_0, k_d_D |
| 11 | `11_stress_relaxation_nlsq_nuts.ipynb` | Relaxation | Liquid foam | G_P, G_E, G_D, nu_0, k_d_D |
| 12 | `12_startup_shear_nlsq_nuts.ipynb` | Startup | PNAS DRT | G_P, G_E, G_D, nu_0, k_d_D, V_act |
| 13 | `13_saos_nlsq_nuts.ipynb` | SAOS | Epstein network | G_P, G_E, G_D, nu_0, k_d_D |
| 14 | `14_laos_nlsq_nuts.ipynb` | LAOS | PNAS DRT | G_P, G_E, G_D, nu_0, k_d_D, V_act |
| 15 | `15_global_multi_protocol.ipynb` | Multi | Flow + SAOS | G_P, G_E, G_D, nu_0, k_d_D |

## Inference Workflow (Notebooks 08-15)

1. **Load data** from `examples/data/` (real experimental datasets)
2. **QC**: NaN removal, monotonicity checks, early-time exclusion, outlier detection
3. **Configure HVNM**: Set defaults, choose fittable parameters per protocol
4. **NLSQ fit**: Fast point estimation with warm-start
5. **NUTS sampling**: Bayesian posterior via NumPyro
6. **Diagnostics**: R-hat, ESS, trace plots, forest plots (ArviZ)
7. **PPC**: Posterior predictive checks overlaid on data
8. **Save**: JSON params, NPZ posteriors, CSV summary

## FAST_MODE

Notebooks 08-15 use `FAST_MODE = True` by default (1 chain, 50 warmup, 100 samples).
For publication quality, edit `examples/utils/hvnm_tutorial_utils.py`:

```python
FAST_MODE = False  # 4 chains, 500 warmup, 1000 samples
```

## Data Sources

| Dataset | Source | Protocol |
|---------|--------|----------|
| `ec_shear_viscosity_07-00.csv` | pyRheo demos | Flow curve |
| `creep_ps190_data.csv` | pyRheo demos | Creep |
| `stressrelaxation_liquidfoam_data.csv` | pyRheo demos | Relaxation |
| `PNAS_DigitalRheometerTwin_Dataset.xlsx` | PNAS 2022 | Startup, LAOS |
| `epstein.csv` | Epstein et al. JACS 2019 | SAOS |

## HVNM Architecture

4-subnetwork constitutive model for NP-filled vitrimers:

- **P** (Permanent): Covalent crosslinks, equilibrium modulus G_P
- **E** (Exchangeable): Vitrimer BER bonds, relaxation via TST kinetics
- **D** (Dissociative): Physical bonds, Maxwell-like relaxation
- **I** (Interphase): NP-matrix interface, independent BER kinetics

Key features: Guth-Gold X(phi) reinforcement, dual TST kinetics (matrix + interphase),
phi=0 recovers HVM exactly.

## Utility Module

`examples/utils/hvnm_tutorial_utils.py` provides:

- `ProtocolData`: Standardized data container with masking and metadata
- Data loaders: `load_ec_flow_curve()`, `load_ps_creep()`, `load_foam_relaxation()`,
  `load_pnas_startup()`, `load_epstein_saos()`, `load_pnas_laos()`, `load_multi_technique()`
- `configure_hvnm_for_fit()`: Set defaults + select fittable params per protocol
- Plotting: `plot_fit_comparison()`, `plot_ppc()`, `plot_saos_components()`,
  `plot_trace_and_forest()`
- Diagnostics: `print_convergence()`, `print_parameter_table()`
- I/O: `save_results()`, `save_figure()`
