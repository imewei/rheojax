# HVM Tutorial Notebooks

Tutorial notebooks for the **Hybrid Vitrimer Model (HVM)**. Each protocol notebook is now
split into two parts: **Part A** demonstrates forward HVM prediction (no fitting), and
**Part B** runs the NLSQ + NumPyro NUTS Bayesian fitting pipeline on representative data.
`07_hvm_overview.ipynb` is the architecture overview, and `08_hvm_fit_demo.ipynb` is the
multi-protocol positive-control fit demo where HVM-fitted curves overlap HVM-generated data.

## Notebooks

| # | Notebook | Protocol | Part B Data | Key Physics |
|---|----------|----------|-------------|-------------|
| 1 | `01_hvm_saos.ipynb` | SAOS | Epstein network + HVM-synthetic multi-T | Linear viscoelastic moduli, Arrhenius validation |
| 2 | `02_hvm_stress_relaxation.ipynb` | Stress relaxation | Fish muscle | P/E/D relaxation modes |
| 3 | `03_hvm_startup_shear.ipynb` | Startup shear | HVM-synthetic | Stress growth, rate dependence, V_act sensitivity |
| 4 | `04_hvm_creep.ipynb` | Creep | HVM-synthetic | Compliance, retardation spectrum, terminal η_D |
| 5 | `05_hvm_flow_curve.ipynb` | Flow curve | Oil-water emulsion φ=0.74 | Steady-state subnetwork decomposition |
| 6 | `06_hvm_laos.ipynb` | LAOS | PNAS Digital Twin | Lissajous, harmonics, normal stress differences |
| 7 | `07_hvm_overview.ipynb` | Overview + quick SAOS fit | Epstein SAOS | 3-subnetwork architecture, factor-of-2, Arrhenius |
| 8 | `08_hvm_fit_demo.ipynb` | Multi-protocol fit | HVM-synthetic positive control | Cross-protocol fitted overlays |

Each protocol notebook (1–6) is structured as:
- **Part A — Forward Prediction**: build an `HVMLocal` model, simulate the protocol, visualise the response, and discuss the physical signature.
- **Part B — NLSQ + NUTS Bayesian Fit**: fit the model to experimental or HVM-synthetic data; show convergence diagnostics, posterior predictive checks, and parameter credible intervals.

During the 2026-05-22 consolidation the older fit-only notebooks (08–13) were merged into Part B of their respective protocol notebooks, and the multi-protocol positive-control demo was renumbered from `14_hvm_fit_demo.ipynb` to `08_hvm_fit_demo.ipynb` so the sequence is contiguous (01–08).

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
jupyter notebook 08_hvm_fit_demo.ipynb

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