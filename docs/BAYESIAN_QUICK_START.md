# Bayesian Workflow Quick Start Guide

**Get started with RheoJAX's Bayesian inference in 5 minutes**

This guide demonstrates the recommended NLSQ → NUTS → ArviZ diagnostic workflow for uncertainty quantification in rheological modeling.

## When to Use Bayesian Inference

### Use Bayesian When You Need:

✅ **Parameter uncertainty quantification** - "How well-constrained are my parameters?"
✅ **Credible intervals** - "95% probability the parameter is in this range"
✅ **Parameter correlations** - "Are G₀ and η identifiable with my data?"
✅ **Prediction uncertainty** - "Error bands on model predictions"
✅ **Model comparison** - "Which model best explains my data? (WAIC, LOO)"

### Skip Bayesian When:

❌ Quick screening with high signal-to-noise data
❌ Real-time analysis where speed is critical
❌ Parameters are well-constrained (uncertainty not important)

## The Recommended Workflow

RheoJAX implements a three-stage workflow combining speed and rigor:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Stage 1   │  →   │   Stage 2    │  →   │    Stage 3      │
│ NLSQ Point  │      │ NUTS Bayes   │      │ ArviZ Diagnose  │
│ (~seconds)  │      │ (~minutes)   │      │ (visual check)  │
└─────────────┘      └──────────────┘      └─────────────────┘
```

**Why this workflow?**
- **Stage 1** (NLSQ): Fast point estimates for initial analysis
- **Stage 2** (NUTS): Warm-start from NLSQ → 2-5x faster convergence
- **Stage 3** (ArviZ): Ensure results are reliable before interpretation

## Complete Example: Maxwell Model

### Step 1: Prepare Your Data

```python
import numpy as np
from rheojax.pipeline.bayesian import BayesianPipeline

# Option A: Load from file
pipeline = BayesianPipeline()
pipeline.load('relaxation_data.csv', x_col='time', y_col='modulus')

# Option B: Use synthetic data for this example
t = np.logspace(-2, 2, 50)  # 0.01 to 100 seconds
G_true = 1e5 * np.exp(-t / 0.01)  # True Maxwell relaxation
G_noisy = G_true + np.random.normal(0, 0.015 * G_true)  # 1.5% noise
```

### Step 2: Run Complete Workflow (Fluent API)

```python
# Complete NLSQ → NUTS → Diagnostics in one chain
result = (pipeline
    .load_array(t, G_noisy)
    .fit_nlsq('maxwell')  # Stage 1: Fast point estimate (~seconds)
    .fit_bayesian(        # Stage 2: Posterior sampling (~minutes)
        num_samples=2000,
        num_warmup=1000,
        num_chains=4      # IMPORTANT: Use 4 chains for robust diagnostics
    )
    .plot_posterior()     # Stage 3: Visual diagnostics
    .plot_trace()
)
```

**Expected output:**
```
Running NLSQ optimization...
  ✓ Converged in 0.4s
  G₀  = 1.005e+05 Pa
  η   = 1.004e+03 Pa·s

Running Bayesian inference with NLSQ warm-start...
  ✓ Completed in 14.2s
  4 chains × 2000 samples = 8000 posterior samples
```

### Step 3: Check Convergence (Critical!)

**Never interpret Bayesian results without checking convergence.**

```python
# Automated convergence checks
diagnostics = pipeline.get_diagnostics()

print("Convergence Diagnostics:")
print(f"  R-hat (G₀):  {diagnostics['r_hat']['G0']:.4f}  {'✓' if diagnostics['r_hat']['G0'] < 1.01 else '✗'}")
print(f"  R-hat (η):   {diagnostics['r_hat']['eta']:.4f}  {'✓' if diagnostics['r_hat']['eta'] < 1.01 else '✗'}")
print(f"  ESS (G₀):    {diagnostics['ess']['G0']:.0f}  {'✓' if diagnostics['ess']['G0'] > 400 else '✗'}")
print(f"  ESS (η):     {diagnostics['ess']['eta']:.0f}  {'✓' if diagnostics['ess']['eta'] > 400 else '✗'}")
```

**Target values:**
- ✅ R-hat < 1.01 (all parameters)
- ✅ ESS > 400 (all parameters)
- ✅ Divergences < 1%

### Step 4: Visualize Diagnostics (All 6 ArviZ Plots)

```python
# Comprehensive diagnostic suite
(pipeline
    .plot_rank()        # 1. Most sensitive convergence check
    .plot_pair(          # 2. Parameter correlations + divergences
        divergences=True,
        show=False
    )
    .plot_forest(        # 3. Credible intervals
        hdi_prob=0.95,
        show=False
    )
    .plot_autocorr(      # 4. Mixing quality
        show=False
    )
    .plot_ess(           # 5. Sampling efficiency
        kind='local',
        show=False
    )
    .plot_energy()       # 6. Posterior geometry (requires multi-chain)
)
```

**What each plot tells you:**

| Plot | Purpose | Red Flag |
|------|---------|----------|
| **Rank** | Most sensitive convergence diagnostic | Non-uniform histogram |
| **Pair** | Parameter correlations, divergence locations | Many red points (divergences) |
| **Forest** | Quick uncertainty comparison | Very wide intervals |
| **Autocorr** | How well chains are mixing | Slow decay to zero |
| **ESS** | Sampling efficiency per parameter | ESS < 400 |
| **Energy** | Posterior geometry issues | Distributions don't overlap |

### Step 5: Extract Results

```python
# Posterior summary
summary = pipeline.get_posterior_summary()
print("\nPosterior Summary:")
print(summary)

# Output:
#              mean       std    median       q05       q25       q75       q95
# G0    1.005e+05  245.3   1.004e+05  9.61e+04  9.87e+04  1.02e+05  1.05e+05
# eta   1.004e+03   24.1   1.003e+03  9.65e+02  9.88e+02  1.02e+03  1.04e+03

# Credible intervals
intervals = pipeline._last_model.get_credible_intervals(
    pipeline._bayesian_result.posterior_samples,
    credibility=0.95
)
print(f"\n95% Credible Intervals:")
print(f"  G₀:  [{intervals['G0'][0]:.3e}, {intervals['G0'][1]:.3e}] Pa")
print(f"  η:   [{intervals['eta'][0]:.3e}, {intervals['eta'][1]:.3e}] Pa·s")

# Interpretation
print("\nInterpretation:")
print("  'There is 95% probability that G₀ lies between X and Y'")
print("  This is a DIRECT probabilistic statement (Bayesian credible interval)")
```

## Troubleshooting Common Issues

### Issue 1: R-hat > 1.01 (Not Converged)

**Symptoms:**
```
R-hat (G₀): 1.0341  ✗ NOT converged
```

**Solutions:**
```python
# Increase warmup iterations
result = pipeline.fit_bayesian(
    num_warmup=2000,  # Was: 1000
    num_samples=2000,
    num_chains=4
)

# Check rank plot: Is histogram uniform?
pipeline.plot_rank()  # Non-uniform → not converged
```

### Issue 2: ESS < 400 (Poor Sampling Efficiency)

**Symptoms:**
```
ESS (G₀): 187  ✗ Low (increase samples)
```

**Solutions:**
```python
# Increase number of samples
result = pipeline.fit_bayesian(
    num_warmup=1000,
    num_samples=5000,  # Was: 2000
    num_chains=4
)

# Check autocorrelation plot: Is mixing slow?
pipeline.plot_autocorr()  # Slow decay → high autocorrelation
```

### Issue 3: High Divergence Rate (>1%)

**Symptoms:**
```
Divergences: 84 (1.05%)  ✗ High
```

**Solutions:**
```python
# Solution 1: Increase target_accept_prob
result = pipeline.fit_bayesian(
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    target_accept_prob=0.9  # Was: 0.8 (default)
)

# Solution 2: Check pair plot to see where divergences occur
pipeline.plot_pair(divergences=True)  # Red points show divergence locations

# Solution 3: Use tighter parameter bounds
model = Maxwell()
model.parameters.set_bounds('G0', (5e4, 2e5))  # Tighter than (1e3, 1e7)
model.parameters.set_bounds('eta', (5e2, 2e3))
```

### Issue 4: Strong Parameter Correlations

**Symptoms:**
```python
# Pair plot shows diagonal line between G₀ and η
correlation = np.corrcoef(G0_samples, eta_samples)[0, 1]
print(f"Correlation: {correlation:.3f}")  # |ρ| > 0.7
```

**This is often EXPECTED for rheological models:**
- Maxwell: G₀ and η both affect relaxation time τ = η/G₀
- This is a physical correlation, not a problem
- If |ρ| > 0.9, consider collecting data at different time scales

## Best Practices Checklist

### Before Running Bayesian Inference:

- ✅ Run NLSQ first (Stage 1) for warm-start
- ✅ Use meaningful parameter bounds
- ✅ Check data quality (outliers, noise level)
- ✅ Use multi-chain MCMC (`num_chains=4`)

### When Running Inference:

- ✅ Start with `num_warmup=1000`, `num_samples=2000`
- ✅ Use warm-start from NLSQ (automatically done by BayesianPipeline)
- ✅ Monitor progress (NumPyro shows progress bars)

### After Inference:

- ✅ **ALWAYS check convergence** (R-hat, ESS, divergences)
- ✅ Run all 6 ArviZ diagnostic plots
- ✅ Verify rank plot is uniform
- ✅ Check pair plot for correlations and divergences
- ✅ Document diagnostics in reports

### Red Flags (Do Not Trust Results):

- ❌ R-hat > 1.01 for any parameter
- ❌ ESS < 100 for any parameter
- ❌ Divergence rate > 5%
- ❌ Non-uniform rank plot
- ❌ Energy plot shows poor overlap

## Advanced: Model Comparison

Once you have converged posteriors for multiple models, compare using WAIC or LOO:

```python
# Fit multiple models
models = ['maxwell', 'zener', 'fractional_maxwell_liquid']
results = {}

for model_name in models:
    results[model_name] = (BayesianPipeline()
        .load_array(t, G_noisy)
        .fit_nlsq(model_name)
        .fit_bayesian(num_samples=2000, num_warmup=1000, num_chains=4)
    )

# Compare using WAIC (lower is better)
import arviz as az

for name, pipeline in results.items():
    idata = pipeline._bayesian_result.to_inference_data()
    waic = az.waic(idata)
    print(f"{name}: WAIC = {waic.elpd_waic:.1f} ± {waic.se:.1f}")

# Output:
#   maxwell: WAIC = -245.3 ± 12.1
#   zener: WAIC = -238.7 ± 11.8  ← Best (lowest WAIC)
#   fractional_maxwell_liquid: WAIC = -241.2 ± 12.5
```

**See [`examples/bayesian/04-model-comparison.ipynb`](../examples/bayesian/04-model-comparison.ipynb) for details.**

## Performance Tips

### Warm-Start is Critical

```python
# ✅ GOOD: Warm-start from NLSQ (2-5x faster)
pipeline.fit_nlsq('maxwell')  # Stage 1
pipeline.fit_bayesian(...)     # Stage 2 uses NLSQ as initial values

# ✗ BAD: Cold start (random initialization)
# Takes 5-10x longer to converge, more divergences
```

### Multi-Chain for Robustness

```python
# ✅ BEST: 4 chains for production work
pipeline.fit_bayesian(num_chains=4)  # Parallel on GPU, robust R-hat

# ⚠ OKAY: 1 chain for quick prototyping
pipeline.fit_bayesian(num_chains=1)  # Cannot compute R-hat, less reliable
```

### Typical Performance

| Stage | Duration | Speedup with Warm-Start |
|-------|----------|-------------------------|
| NLSQ optimization | 0.1-2s | N/A |
| NUTS sampling (cold start) | 1-5 min | Baseline |
| NUTS sampling (warm-start) | 0.5-1 min | **2-5x faster** |
| ArviZ diagnostics | 1-5s | N/A |

## Next Steps

### Learn More

1. **[`examples/bayesian/01-bayesian-basics.ipynb`](../examples/bayesian/01-bayesian-basics.ipynb)**
   - 40-minute tutorial on NLSQ → NUTS workflow
   - Detailed explanation of convergence diagnostics
   - Bayesian vs frequentist interpretation

2. **[`examples/bayesian/03-convergence-diagnostics.ipynb`](../examples/bayesian/03-convergence-diagnostics.ipynb)**
   - Master all 6 ArviZ diagnostic plots
   - Systematic troubleshooting guide
   - Common failure modes and solutions

3. **[`examples/bayesian/04-model-comparison.ipynb`](../examples/bayesian/04-model-comparison.ipynb)**
   - WAIC and LOO for model selection
   - Information criteria interpretation
   - Model averaging strategies

4. **[`examples/bayesian/05-uncertainty-propagation.ipynb`](../examples/bayesian/05-uncertainty-propagation.ipynb)**
   - Propagate uncertainty to predictions
   - Derived quantities (e.g., relaxation time τ = η/G₀)
   - Prediction bands for model plots

### Apply to Other Models

All 20 RheoJAX models support the same Bayesian workflow:

```python
# Classical models
models = ['maxwell', 'zener', 'springpot', 'bingham', 'power_law']

# Fractional models (11 variants)
models += ['fractional_maxwell_liquid', 'fractional_zener_sl', ...]

# Flow models
models += ['carreau', 'cross', 'herschel_bulkley', ...]

# Same workflow for all models
for model_name in models:
    (BayesianPipeline()
        .load('data.csv')
        .fit_nlsq(model_name)
        .fit_bayesian(num_samples=2000)
        .plot_pair()
    )
```

## Summary

### The 3-Step Workflow

1. **NLSQ** → Fast point estimates (~seconds)
2. **NUTS** → Posterior samples with warm-start (~minutes)
3. **ArviZ** → Diagnostic plots to verify results (~seconds)

### Critical Checkpoints

- ✅ R-hat < 1.01 (convergence)
- ✅ ESS > 400 (sufficient samples)
- ✅ Divergences < 1% (sampler quality)
- ✅ Rank plot uniform (most sensitive check)

### When in Doubt

- Increase `num_warmup` for R-hat issues
- Increase `num_samples` for ESS issues
- Increase `target_accept_prob` for divergences
- Check pair plot for correlations and divergence locations

**Ready to start? Copy the complete example above and run it on your data!**

## References

- **NumPyro Documentation**: https://num.pyro.ai/
- **ArviZ Documentation**: https://arviz-devs.github.io/arviz/
- **RheoJAX Tutorial Notebooks**: `examples/bayesian/` (5 complete tutorials)
- **Bayesian Workflow Summary**: [`docs/BAYESIAN_WORKFLOW_SUMMARY.md`](BAYESIAN_WORKFLOW_SUMMARY.md)

---

**Last Updated**: 2025-11-05
**RheoJAX Version**: 0.2.0
**Estimated Reading Time**: 10 minutes
