# EPM Mesoscopic Model Tutorials

Tutorial notebooks for the Elasto-Plastic Model (EPM) — a mesoscopic approach for amorphous solids.

## Notebooks

| # | Notebook | Protocol | Data | Key Topics |
|---|----------|----------|------|------------|
| 01 | `01_epm_flow_curve.ipynb` | Flow curve | Real emulsion (φ=0.80) | NLSQ→NUTS, avalanche physics |
| 02 | `02_epm_saos.ipynb` | Oscillation (SAOS) | Real polystyrene | G'/G'' from disorder, Cole-Cole |
| 03 | `03_epm_startup.ipynb` | Startup shear | Synthetic (from Nb 01) | Stress overshoot, N₁(t) |
| 04 | `04_epm_creep.ipynb` | Creep | Real biological (mucus) | Bounded vs unbounded creep |
| 05 | `05_epm_relaxation.ipynb` | Relaxation | Real polymer (ps145) | Disorder-induced multi-relaxation |
| 06 | `06_epm_visualization.ipynb` | All | Synthetic | Visualization gallery, tensorial fields |

## Models

- **LatticeEPM**: Scalar stress field (L×L lattice), full NLSQ + Bayesian fitting
- **TensorialEPM**: 3-component stress tensor [σ_xx, σ_yy, σ_xy], forward predictions only

## Prerequisites

- Complete `01-basic-maxwell.ipynb` and `05-bayesian-basics.ipynb` first
- Notebook 01 should be run before 03 (provides calibrated parameters for synthetic generation)
- Notebooks 02, 04, and 05 use independent real data

## Data Sources

**Real data:**
- Emulsion φ=0.80 flow curve: `examples/data/flow/emulsions/0.80.csv`
- Polystyrene SAOS: `examples/data/oscillation/polystyrene/oscillation_ps145_data.csv`
- Mucus creep: `examples/data/creep/biological/creep_mucus_data.csv`
- Polymer relaxation: `examples/data/relaxation/polymers/stressrelaxation_ps145_data.csv`

**Synthetic data (Notebook 03):**
Generated from parameters calibrated to the real emulsion flow curve with 3% Gaussian noise added.

## Runtime

| Config | num_warmup | num_samples | num_chains | Typical time per notebook |
|--------|------------|-------------|------------|---------------------------|
| Fast demo | 200 | 500 | 1 | 2-4 min |
| Production | 1000 | 2000 | 4 | 5-12 min |

**Note:** EPM simulations involve lattice dynamics with O(L²) operations per step. Use L=32 for tutorials; L=64+ for production.

## Learning Outcomes

After completing these tutorials, you will understand:

1. **EPM physics**: Eshelby propagator, plastic avalanches, disorder-induced yielding
2. **Parameter interpretation**: μ (modulus), τ_pl (plastic time), σ_c_mean/σ_c_std (yield threshold distribution)
3. **NLSQ→NUTS workflow**: Warm-start Bayesian inference with ArviZ diagnostics
4. **Normal stresses**: N₁ predictions from TensorialEPM
5. **Visualization**: Lattice fields, von Mises stress, avalanche animations

## Key References

- Hébraud, P. & Lequeux, F. (1998). Mode-coupling theory for the pasty rheology of soft glassy materials. *Phys. Rev. Lett.*, 81, 2934.
- Nicolas, A., Ferrero, E.E., Martens, K. & Barrat, J.L. (2018). Deformation and flow of amorphous solids: Insights from elastoplastic models. *Rev. Mod. Phys.*, 90, 045006.
- Picard, G., Ajdari, A., Bocquet, L. & Lequeux, F. (2002). Simple model for heterogeneous flows of yield stress fluids. *Phys. Rev. E*, 66, 051501.
