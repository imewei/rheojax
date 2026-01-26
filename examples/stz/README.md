# STZ Shear Transformation Zone Tutorials

Tutorial notebooks for the Shear Transformation Zone (STZ) model — a microscopic theory for plastic flow in amorphous solids (metallic glasses, colloidal glasses, dense emulsions).

## Notebooks

| # | Notebook | Protocol | Data | Parameters Probed |
|---|----------|----------|------|-------------------|
| 01 | `01_stz_flow_curve.ipynb` | Steady-state flow curve | Synthetic (arctanh formula) | sigma_y, chi_inf, tau0, ez |
| 02 | `02_stz_startup_shear.ipynb` | Startup shear | Synthetic (from NB01) | All 8 (standard variant) |
| 03 | `03_stz_stress_relaxation.ipynb` | Stress relaxation | Real laponite clay (5 ages) | All 8, chi_inf(t_wait) aging |
| 04 | `04_stz_creep.ipynb` | Creep | Real mucus | sigma_y, chi_inf, tau0, epsilon0, c0, ez, tau_beta |
| 05 | `05_stz_saos.ipynb` | SAOS | Real polystyrene (ps145) | G0, sigma_y, chi_inf, tau0, epsilon0, ez |
| 06 | `06_stz_laos.ipynb` | LAOS | Synthetic (from NB01) | All 8 (forward sim only) |

## STZ Variants

| Variant | State Variables | Parameters | Use Case |
|---------|----------------|------------|----------|
| **minimal** | stress, chi | 7 | Fast prototyping, analytical limits |
| **standard** (default) | stress, chi, Lambda | 8 | Production use, Lambda dynamics on tau_beta |
| **full** | stress, chi, Lambda, m | 10 | Back-stress effects, kinematic hardening |

## Prerequisites

- Complete `01-basic-maxwell.ipynb` and `05-bayesian-basics.ipynb` first
- Notebook 01 should be run before 02 and 06 (provides calibrated parameters)
- Notebooks 03, 04, and 05 are independent of Notebook 01

## Dependency Graph

```
NB01 (flow curve, synthetic)
  |-- NB02 (startup, synthetic from NB01)
  |-- NB06 (LAOS, synthetic from NB01)

NB03 (relaxation, real laponite) -- independent
NB04 (creep, real mucus) -- independent
NB05 (SAOS, real polystyrene) -- independent
```

## Data Sources

**Synthetic data (Notebooks 01, 02, 06):**
- NB01: Flow curve generated from the STZ arctanh formula with known parameters and 3% log-normal noise
- NB02, NB06: Startup/LAOS data generated from NB01 calibrated parameters with 3% Gaussian noise

**Real data (Notebooks 03, 04, 05):**
- Laponite clay relaxation: `examples/data/relaxation/clays/rel_lapo_*.csv` (5 aging times, TSV)
- Mucus creep: `examples/data/creep/biological/creep_mucus_data.csv` (20 points, TSV)
- Polystyrene SAOS: `examples/data/oscillation/polystyrene/oscillation_ps145_data.csv` (32 points, TSV)

## Runtime

| Config | num_warmup | num_samples | num_chains | Typical time per notebook |
|--------|-----------|-------------|------------|--------------------------|
| Fast demo | 200 | 500 | 1 | 1-5 min |
| Production | 1000 | 2000 | 4 | 4-15 min |

**Note:** NB06 (LAOS) does not include Bayesian inference due to the high computational cost of ODE integration per MCMC sample.

## Key References

- Falk, M.L. & Langer, J.S. (1998). Dynamics of viscoplastic deformation in amorphous solids. *Phys. Rev. E*, 57, 7192-7205.
- Langer, J.S. (2008). Shear-transformation-zone theory of plastic deformation near the glass transition. *Phys. Rev. E*, 77, 021502.
- Manning, M.L., Langer, J.S. & Carlson, J.M. (2007). Strain localization in a shear transformation zone model for amorphous solids. *Phys. Rev. E*, 76, 056106.
