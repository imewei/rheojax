# Hébraud-Lequeux (HL) Model Tutorials

Comprehensive tutorials for the Hébraud-Lequeux (HL) mean-field elastoplastic model — a mesoscopic approach for yield-stress fluids (emulsions, foams, pastes, colloidal glasses) based on stress block dynamics with stochastic yielding.

## Model Overview

The HL model describes yield-stress fluids through a population of mesoscopic "stress blocks" that:
- Load elastically under applied shear
- Yield when stress exceeds a threshold σ_c
- Relax via mechanical noise diffusion controlled by coupling parameter α

### Key Physics

The stress distribution P(σ,t) evolves according to the HL master equation:

$$\frac{\partial P}{\partial t} + G_0 \dot{\gamma} \frac{\partial P}{\partial \sigma} = \alpha \frac{\partial^2 P}{\partial \sigma^2} + \Gamma(\sigma)[P - \delta(\sigma)]$$

where Γ(σ) is the yielding rate for blocks exceeding σ_c.

### Phase Behavior

The coupling parameter α controls material behavior:

| Regime | α Value | Behavior | Physical Signature |
|--------|---------|----------|-------------------|
| **Glass** | α < 0.5 | Solid-like | Yield stress σ_y > 0, incomplete relaxation |
| **Fluid** | α ≥ 0.5 | Liquid-like | No yield stress, complete relaxation |

**Key Result:** P(glass) = P(α < 0.5) from Bayesian posteriors quantifies phase classification uncertainty.

## Parameters

| Parameter | Symbol | Description | Typical Range | Units |
|-----------|--------|-------------|---------------|-------|
| Coupling parameter | α | Noise-to-yield ratio | 0.1–0.9 | dimensionless |
| Microscopic timescale | τ | Block relaxation time | 10⁻³–10² | s |
| Critical yield stress | σ_c | Yielding threshold | 10–10⁴ | Pa |

## Notebooks

| # | Notebook | Protocol | Data | Runtime | Key Topics |
|---|----------|----------|------|---------|------------|
| 01 | [01_hl_flow_curve.ipynb](01_hl_flow_curve.ipynb) | Flow curve | Real emulsion (6 φ) | ~3 min | Yield stress extraction, volume fraction sweep α(φ) |
| 02 | [02_hl_relaxation.ipynb](02_hl_relaxation.ipynb) | Relaxation | Real Laponite clay (5 t_w) | ~3 min | Incomplete relaxation, aging sweep α(t_w) |
| 03 | [03_hl_creep.ipynb](03_hl_creep.ipynb) | Creep | Real polystyrene (5 T) | ~3 min | Delayed yielding, temperature sweep α(T) |
| 04 | [04_hl_saos.ipynb](04_hl_saos.ipynb) | SAOS | Synthetic (calibrated) | ~3 min | G'/G'' analysis, glass vs fluid comparison |
| 05 | [05_hl_startup.ipynb](05_hl_startup.ipynb) | Startup | Synthetic (calibrated) | ~3 min | Stress overshoot, τ from transient dynamics |
| 06 | [06_hl_laos.ipynb](06_hl_laos.ipynb) | LAOS | Synthetic (calibrated) | ~4 min | Lissajous curves, Fourier harmonics I₃/I₁ |

**Runtimes:** Fast demo mode (NUM_CHAINS=1). Production mode (NUM_CHAINS=4) takes ~4x longer.

## Recommended Order

```
01_hl_flow_curve.ipynb ─────┬──> 04_hl_saos.ipynb
    (calibrates params)     ├──> 05_hl_startup.ipynb
                            └──> 06_hl_laos.ipynb

02_hl_relaxation.ipynb ──────> (independent real data)
03_hl_creep.ipynb ───────────> (independent real data)
```

1. **Start with 01-flow-curve** — establishes HL physics and calibrates parameters for synthetic data
2. **02-relaxation and 03-creep** use independent real data and can be done in any order
3. **04-saos, 05-startup, 06-laos** use synthetic data with calibrated parameters

## Prerequisites

### Knowledge
- Phase 1: Basic model fitting ([basic/](../basic/) notebooks)
- Phase 3: Bayesian inference fundamentals ([bayesian/](../bayesian/) notebooks)

### Installation
```bash
pip install rheojax arviz matplotlib numpy
# For GPU acceleration (Linux + CUDA only):
pip install jax[cuda12-local]==0.8.0   # CUDA 12.x
pip install jax[cuda13-local]==0.8.0   # CUDA 13.x
```

## Data Sources

### Real Data
- **Emulsion flow curves** (6 volume fractions φ = 0.69–0.80): `data/flow/emulsions/`
- **Laponite clay relaxation** (5 aging times t_w = 600–3600s): `data/relaxation/clays/`
- **Polystyrene creep** (5 temperatures T = 130–190°C): `data/creep/polymers/`

### Synthetic Data
- **SAOS, Startup, LAOS**: Generated from parameters calibrated to real emulsion data (3% noise)
- Uses `generate_hl_synthetic()` from `utils/hl_tutorial_utils.py`

## Key Concepts

### Glass Probability
All notebooks compute P(glass) = P(α < 0.5) from Bayesian posteriors:
```python
alpha_samples = result.posterior_samples["alpha"]
p_glass = np.mean(alpha_samples < 0.5)
```

### NLSQ → NUTS Workflow
Standard workflow for uncertainty quantification:
```python
from rheojax.models.hl import HebraudLequeux

model = HebraudLequeux()
model.fit(x, y, test_mode='flow_curve')  # NLSQ point estimate

result = model.fit_bayesian(
    x, y,
    test_mode='flow_curve',
    num_warmup=200,
    num_samples=500,
    num_chains=1,  # Fast demo
    initial_values={...},  # Warm-start from NLSQ
)
```

### Protocol-Specific Signatures

| Protocol | Glass (α < 0.5) | Fluid (α ≥ 0.5) |
|----------|----------------|-----------------|
| Flow curve | Yield stress σ_y ≈ 0.5σ_c | Newtonian at low γ̇ |
| Relaxation | G(t→∞) > 0 (incomplete) | G(t→∞) → 0 (complete) |
| Creep | Bounded strain | Unbounded flow |
| SAOS | G' plateau at low ω | G' → 0 as ω → 0 |
| Startup | Strong overshoot | Weak/no overshoot |
| LAOS | Strong nonlinearity | Weaker I₃/I₁ |

## Outputs

All notebooks save results to `outputs/hl/{protocol}/`:
- `nlsq_params_{protocol}.json` — Point estimates
- `posterior_{protocol}.json` — Posterior samples
- `*_sweep_results.json` — Parameter sweeps (where applicable)

## Comparison with Other Models

| Model | Approach | Key Parameter | Best For |
|-------|----------|---------------|----------|
| **HL** | Mean-field stress blocks | α (coupling) | Dense emulsions, soft glasses |
| **SGR** | Trap model, noise temperature | x (noise temp) | Aging, soft glassy materials |
| **EPM** | Lattice, plastic avalanches | Disorder distribution | Amorphous solids |
| **STZ** | Shear transformation zones | χ (effective temp) | Metallic glasses |
| **DMT** | Structural kinetics | λ (structure) | Thixotropic fluids |

## References

1. Hébraud, P. & Lequeux, F. (1998). Mode-coupling theory for the pasty rheology of soft glassy materials. *Physical Review Letters*, 81(14), 2934.
2. Sollich, P. et al. (1997). Rheology of soft glassy materials. *Physical Review Letters*, 78(10), 2020.
3. Fielding, S. M. et al. (2000). Aging and rheology in soft materials. *Journal of Rheology*, 44(2), 323.

## Troubleshooting

### "Float64 not enabled"
Ensure notebooks use safe JAX imports:
```python
from rheojax.core.jax_config import safe_import_jax, verify_float64
jax, jnp = safe_import_jax()
verify_float64()
```

### Low ESS or high R-hat
Increase sampling:
```python
result = model.fit_bayesian(
    x, y,
    num_warmup=1000,   # Increase from 200
    num_samples=2000,  # Increase from 500
    num_chains=4,      # Production setting
)
```

### Divergences in NUTS
Check parameter bounds and consider reparameterization. The warm-start from NLSQ typically prevents most divergence issues.
