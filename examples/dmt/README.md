# DMT Thixotropic Model Tutorials

Tutorial notebooks for the de Souza Mendes-Thompson (DMT) structural-kinetics thixotropic model.

## Notebooks

| # | Notebook | Protocol | Data | Parameters Probed |
|---|----------|----------|------|-------------------|
| 01 | `01_dmt_flow_curve.ipynb` | Steady-state flow curve | Real emulsion (phi=0.80) | eta_0, eta_inf, a, c |
| 02 | `02_dmt_startup_shear.ipynb` | Startup shear | Synthetic (from Nb 01) | All 7 (exp+elastic) |
| 03 | `03_dmt_stress_relaxation.ipynb` | Stress relaxation | Real laponite clay | t_eq, eta_0, G0, m_G |
| 04 | `04_dmt_creep.ipynb` | Creep | Real mucus | eta_0, eta_inf, a, c, G0 |
| 05 | `05_dmt_saos.ipynb` | SAOS | Synthetic (from Nb 01) | G0, eta_0, eta_inf |
| 06 | `06_dmt_laos.ipynb` | LAOS | Synthetic (from Nb 01) | All 7 |

## Prerequisites

- Complete `01-basic-maxwell.ipynb` and `05-bayesian-basics.ipynb` first
- Notebook 01 should be run before 02, 05, and 06 (provides calibrated parameters)
- Notebooks 03 and 04 are independent of Notebook 01

## Data Sources

**Real data:**
- Emulsion phi=0.80 flow curve: `examples/data/flow/emulsions/0.80.csv` (30 points)
- Laponite clay relaxation: `examples/data/relaxation/clays/rel_lapo_*.csv` (5 aging times)
- Mucus creep: `examples/data/creep/biological/creep_mucus_data.csv` (20 points)

**Synthetic data (Notebooks 02, 05, 06):**
Generated from parameters calibrated to the real emulsion flow curve with 3% Gaussian noise added. This is standard practice when experimental data for specific protocols is not available.

## Runtime

| Config | num_warmup | num_samples | num_chains | Typical time per notebook |
|--------|-----------|-------------|------------|--------------------------|
| Fast demo | 200 | 500 | 1 | 1-3 min |
| Production | 1000 | 2000 | 4 | 5-15 min |

## Key References

- de Souza Mendes, P.R. (2009). Modeling the thixotropic behavior of structured fluids. *J. Non-Newtonian Fluid Mech.*, 164, 66-75.
- de Souza Mendes, P.R. & Thompson, R.L. (2013). A unified approach to model elasto-viscoplastic thixotropic yield-stress materials. *Rheol. Acta*, 52, 673-694.
