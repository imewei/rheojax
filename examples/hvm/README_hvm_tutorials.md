# HVM Tutorial Notebooks

Tutorial notebooks for the **Hybrid Vitrimer Model (HVM)**. The first notebooks introduce
forward predictions, the middle notebooks demonstrate NLSQ and NumPyro NUTS workflows on
experimental rheology data, and `14_hvm_fit_demo.ipynb` provides a positive-control fitting
case where HVM-fitted curves overlap HVM-generated rheology data.

## Notebooks

| # | Notebook | Protocol | Data Source | Key Physics |
|---|----------|----------|-------------|-------------|
| 1 | `01_hvm_saos.ipynb` | SAOS | HVM simulation | Linear viscoelastic moduli |
| 2 | `02_hvm_stress_relaxation.ipynb` | Relaxation | HVM simulation | P/E/D relaxation modes |
| 3 | `03_hvm_startup_shear.ipynb` | Startup | HVM simulation | Stress growth and rate dependence |
| 4 | `04_hvm_creep.ipynb` | Creep | HVM simulation | Compliance and applied stress |
| 5 | `05_hvm_flow_curve.ipynb` | Flow curve | HVM simulation | Steady-state subnetwork decomposition |
| 6 | `06_hvm_laos.ipynb` | LAOS | HVM simulation | Nonlinear oscillatory response |
| 7 | `07_hvm_overview.ipynb` | Overview + SAOS fit | Epstein SAOS | 3-subnetwork architecture, Arrhenius |
| 8 | `08_hvm_flow_curve.ipynb` | Flow curve fit | Oil-water emulsion φ=0.74 | Model/data suitability check |
| 9 | `09_hvm_creep.ipynb` | Creep fit | PS at 160°C | Model/data suitability check |
| 10 | `10_hvm_relaxation.ipynb` | Relaxation fit | Fish muscle | NLSQ HVM fit |
| 11 | `11_hvm_startup.ipynb` | Startup fit | PNAS Digital Twin | Model/data suitability check |
| 12 | `12_hvm_saos.ipynb` | SAOS fit | Epstein network | NLSQ and Bayesian workflow |
| 13 | `13_hvm_laos.ipynb` | LAOS comparison | PNAS Digital Twin | Lissajous and harmonic analysis |
| 14 | `14_hvm_fit_demo.ipynb` | Multi-protocol fit | HVM-generated rheology data | Positive-control fitted overlays |

## Data Sources

All data is loaded via `examples/utils/hvm_data.py`:

- **Epstein SAOS**: Metal-organic coordination network, 19 points, ω ∈ [0.1, 99] rad/s
- **PS temperature series**: Polystyrene at 130–190°C (SAOS, relaxation, creep)
- **Fish muscle relaxation**: 567 high-density points, exceptional temporal resolution
- **Oil-water emulsions**: Volume fraction series φ = 0.69–0.80, yield-stress behavior
- **PNAS Digital Rheometer Twin**: Startup (5 rates) and LAOS (3 frequencies × 12 amplitudes)

## How to Run

### Quick demo (FAST_MODE=True, ~1-2 min per notebook)

```bash
# Single notebook
cd examples/hvm
jupyter notebook 14_hvm_fit_demo.ipynb

# All notebooks (batch)
uv run python scripts/run_single_notebook_96h.py examples/hvm
```

### Production quality (FAST_MODE=False, ~10-30 min per notebook)

Edit `examples/utils/hvm_fit.py`, set `FAST_MODE = False`, then run.

## Output Structure

Results are saved to `examples/outputs/hvm/<protocol>/`:

```
examples/outputs/hvm/
├── hvm/                    # Overview outputs
│   ├── figures/
│   └── quick_saos_fit.png
├── oscillation/            # SAOS protocol
│   ├── fitted_params_nlsq.json
│   ├── posterior_samples.npz
│   ├── summary.csv
│   └── figures/
├── flow_curve/
├── relaxation/
├── creep/
├── startup/
└── laos/
```

## Utilities

- `examples/utils/hvm_data.py` — Data loaders with dataset registry (9 datasets, 6 protocols)
- `examples/utils/hvm_fit.py` — Fit pipeline: `run_nlsq_saos()`, `run_nuts()`, `posterior_predictive_saos()`
- `examples/utils/hvm_demo_fit.py` — Positive-control HVM data generation and fitted overlay helpers
- `examples/hvm/hvm_defaults.yaml` — Protocol-specific parameter hints and sampler config

## Requirements

- Python 3.12+
- JAX 0.8.0+, NLSQ 0.6.6+, NumPyro, ArviZ
- `uv sync` to install all dependencies

## Key Physics

The HVM combines three subnetworks:

1. **Permanent (P)**: Covalent crosslinks, neo-Hookean stress σ_P = G_P·(μ - I)
2. **Exchangeable (E)**: Vitrimer BER bonds, TST kinetics k_BER = ν₀·exp(-E_a/RT)·cosh(V_act·σ/RT)
3. **Dissociative (D)**: Physical bonds, Maxwell relaxation with rate k_d_D

**Vitrimer signatures**:
- Factor-of-2: τ_E_eff = 1/(2·k_BER_0) — both μ^E and μ^E_nat relax toward each other
- σ_E → 0 at steady state (natural state fully tracks deformation)
- Arrhenius temperature dependence with topology freezing T_v