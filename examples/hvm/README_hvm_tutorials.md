# HVM Tutorial Notebooks

Tutorial notebooks for the **Hybrid Vitrimer Model (HVM)** demonstrating NLSQ → NumPyro NUTS
Bayesian inference with real experimental rheology data.

## Notebooks

| # | Notebook | Protocol | Data Source | Key Physics |
|---|----------|----------|-------------|-------------|
| 0 | `hvm_00_overview.ipynb` | Overview | Epstein SAOS | 3-subnetwork architecture, factory methods, Arrhenius |
| 1 | `hvm_01_flow_curve.ipynb` | Flow curve | Oil-water emulsion φ=0.74 | σ_E→0 at steady state, yield plateau |
| 2 | `hvm_02_creep.ipynb` | Creep | PS at 160°C | Elastic jump + viscous flow, J(t) |
| 3 | `hvm_03_relaxation.ipynb` | Relaxation | Fish muscle (567 pts) | Bi-exponential + G_P plateau |
| 4 | `hvm_04_startup.ipynb` | Startup | PNAS Digital Twin | Stress overshoot, TST kinetics |
| 5 | `hvm_05_saos.ipynb` | SAOS | Epstein network | Full Bayesian, posterior predictive |
| 6 | `hvm_06_laos.ipynb` | LAOS | PNAS Digital Twin | Lissajous, harmonic analysis |

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
jupyter notebook hvm_00_overview.ipynb

# All notebooks (batch)
uv run python scripts/run_notebooks.py --suite hvm
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
