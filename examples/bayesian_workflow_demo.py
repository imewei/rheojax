#!/usr/bin/env python3
"""
Bayesian Workflow Demo: Complete NLSQ → NUTS → ArviZ Pipeline

This script demonstrates the recommended three-stage Bayesian workflow:
1. Stage 1: NLSQ point estimation (fast, ~seconds)
2. Stage 2: NUTS posterior sampling with warm-start (~minutes)
3. Stage 3: ArviZ diagnostic plots (visual verification)

Usage:
    python examples/bayesian_workflow_demo.py

Expected runtime: ~30 seconds (depending on hardware)

Requirements:
    - rheojax with Bayesian dependencies (numpyro, arviz)
    - matplotlib for visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from rheojax.models.maxwell import Maxwell
from rheojax.core.jax_config import safe_import_jax

# Safe JAX import (ensures float64 precision)
jax, jnp = safe_import_jax()

print("="*70)
print("BAYESIAN WORKFLOW DEMONSTRATION")
print("="*70)
print("\nThis demo shows the recommended NLSQ → NUTS → ArviZ workflow")
print("for uncertainty quantification in rheological modeling.\n")

# =============================================================================
# Step 1: Generate Synthetic Relaxation Data
# =============================================================================
print("Step 1: Generating synthetic Maxwell relaxation data...")
print("-"*70)

# True parameters
G0_true = 1e5  # Pa
eta_true = 1e3  # Pa·s
tau_true = eta_true / G0_true  # 0.01 s

print(f"  True parameters:")
print(f"    G₀  = {G0_true:.2e} Pa")
print(f"    η   = {eta_true:.2e} Pa·s")
print(f"    τ   = {tau_true:.4f} s")

# Time array (log-spaced for relaxation)
t = np.logspace(-2, 2, 50)  # 0.01 to 100 seconds

# True relaxation modulus
G_t_true = G0_true * np.exp(-t / tau_true)

# Add realistic noise (1.5% relative)
np.random.seed(42)
noise_level = 0.015
noise = np.random.normal(0, noise_level * G_t_true)
G_t_noisy = G_t_true + noise

print(f"\n  Generated {len(t)} data points from {t.min():.2e} to {t.max():.1f} s")
print(f"  Noise level: {noise_level*100:.1f}% relative")
print(f"  Signal-to-noise ratio: {np.mean(G_t_true)/np.std(noise):.1f}")

# =============================================================================
# Step 2: Stage 1 - NLSQ Point Estimation (Fast)
# =============================================================================
print("\n" + "="*70)
print("STAGE 1: NLSQ POINT ESTIMATION")
print("="*70)

model = Maxwell()
model.parameters.set_bounds('G0', (1e3, 1e7))
model.parameters.set_bounds('eta', (1e1, 1e5))

print("\nRunning NLSQ optimization...")
import time
start_nlsq = time.time()

model.fit(t, G_t_noisy, method='nlsq')

nlsq_time = time.time() - start_nlsq

# Extract NLSQ results
G0_nlsq = model.parameters.get_value('G0')
eta_nlsq = model.parameters.get_value('eta')
tau_nlsq = eta_nlsq / G0_nlsq

print(f"\n✓ NLSQ completed in {nlsq_time:.3f} seconds")
print(f"\nNLSQ Point Estimates:")
print(f"  G₀  = {G0_nlsq:.4e} Pa  (error: {abs(G0_nlsq-G0_true)/G0_true*100:.2f}%)")
print(f"  η   = {eta_nlsq:.4e} Pa·s  (error: {abs(eta_nlsq-eta_true)/eta_true*100:.2f}%)")
print(f"  τ   = {tau_nlsq:.6f} s  (error: {abs(tau_nlsq-tau_true)/tau_true*100:.2f}%)")
print(f"\n⚠  Note: NLSQ provides point estimates only (no uncertainty)")

# =============================================================================
# Step 3: Stage 2 - Bayesian Inference with Warm-Start
# =============================================================================
print("\n" + "="*70)
print("STAGE 2: BAYESIAN INFERENCE (NUTS)")
print("="*70)

print("\nRunning NUTS sampling with NLSQ warm-start...")
print("  Configuration:")
print(f"    • num_chains: 4 (for robust diagnostics)")
print(f"    • num_warmup: 1000 (burn-in iterations)")
print(f"    • num_samples: 2000 (posterior samples per chain)")
print(f"    • warm-start: Yes (from NLSQ estimates)")
print("\n  This may take 20-60 seconds depending on your hardware...")

start_bayes = time.time()

# Run Bayesian inference with warm-start
result = model.fit_bayesian(
    t, G_t_noisy,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    initial_values={  # Warm-start from NLSQ
        'G0': G0_nlsq,
        'eta': eta_nlsq
    }
)

bayes_time = time.time() - start_bayes

print(f"\n✓ Bayesian inference completed in {bayes_time:.1f} seconds")
print(f"  Total time (NLSQ + Bayes): {nlsq_time + bayes_time:.1f} seconds")
print(f"  Generated {result.num_chains * result.num_samples} posterior samples")

# =============================================================================
# Step 4: Extract Posterior Summary
# =============================================================================
print("\n" + "="*70)
print("POSTERIOR RESULTS")
print("="*70)

posterior = result.posterior_samples
summary = result.summary

print("\nPosterior Estimates (mean ± std):")
print(f"  G₀  = {summary['G0']['mean']:.4e} ± {summary['G0']['std']:.4e} Pa")
print(f"  η   = {summary['eta']['mean']:.4e} ± {summary['eta']['std']:.4e} Pa·s")

# Compute credible intervals
intervals = model.get_credible_intervals(posterior, credibility=0.95)
print("\n95% Credible Intervals:")
print(f"  G₀:  [{intervals['G0'][0]:.4e}, {intervals['G0'][1]:.4e}] Pa")
print(f"  η:   [{intervals['eta'][0]:.4e}, {intervals['eta'][1]:.4e}] Pa·s")

print("\nInterpretation:")
print("  'There is 95% probability that G₀ lies in the interval above'")
print("  This is a DIRECT probabilistic statement (Bayesian interpretation)")

# Relative uncertainties
print("\nRelative Uncertainties:")
print(f"  G₀:  {summary['G0']['std']/summary['G0']['mean']*100:.2f}%")
print(f"  η:   {summary['eta']['std']/summary['eta']['mean']*100:.2f}%")

# Check if true values are in credible intervals
G0_in_CI = intervals['G0'][0] <= G0_true <= intervals['G0'][1]
eta_in_CI = intervals['eta'][0] <= eta_true <= intervals['eta'][1]
print("\nValidation (true values in 95% CI):")
print(f"  G₀:  {'✓ Yes' if G0_in_CI else '✗ No'}")
print(f"  η:   {'✓ Yes' if eta_in_CI else '✗ No'}")

# =============================================================================
# Step 5: Stage 3 - Convergence Diagnostics (CRITICAL!)
# =============================================================================
print("\n" + "="*70)
print("STAGE 3: CONVERGENCE DIAGNOSTICS")
print("="*70)

diagnostics = result.diagnostics

print("\n⚠  ALWAYS check convergence before interpreting Bayesian results!")
print("\n1. R-hat (Gelman-Rubin Statistic):")
print(f"   Target: < 1.01 for all parameters")
for param in ['G0', 'eta']:
    rhat = diagnostics['r_hat'][param]
    status = '✓ Converged' if rhat < 1.01 else '✗ NOT converged'
    print(f"     {param:<5} R-hat = {rhat:.4f}  {status}")

print("\n2. ESS (Effective Sample Size):")
print(f"   Target: > 400 (out of {result.num_chains * result.num_samples} total)")
for param in ['G0', 'eta']:
    ess = diagnostics['ess'][param]
    efficiency = ess / (result.num_chains * result.num_samples) * 100
    status = '✓ Sufficient' if ess > 400 else '✗ Low (increase samples)'
    print(f"     {param:<5} ESS = {ess:.0f} ({efficiency:.1f}% efficient)  {status}")

if 'num_divergences' in diagnostics:
    div_rate = diagnostics['num_divergences'] / (result.num_chains * result.num_samples) * 100
    print("\n3. Divergences:")
    print(f"   Count: {diagnostics['num_divergences']} ({div_rate:.2f}%)")
    status = '✓ Good' if div_rate < 1 else '✗ High (results unreliable)'
    print(f"   Target: < 1%  {status}")

# Overall convergence assessment
all_converged = (
    all(diagnostics['r_hat'][p] < 1.01 for p in ['G0', 'eta']) and
    all(diagnostics['ess'][p] > 400 for p in ['G0', 'eta'])
)

print("\n" + "-"*70)
if all_converged:
    print("✓✓✓ EXCELLENT CONVERGENCE ✓✓✓")
    print("All diagnostic criteria met. Results are reliable.")
else:
    print("⚠⚠⚠ CONVERGENCE ISSUES ⚠⚠⚠")
    print("Increase num_warmup or num_samples and rerun.")
print("-"*70)

# =============================================================================
# Step 6: Visual Diagnostics (ArviZ Plots)
# =============================================================================
print("\n" + "="*70)
print("VISUAL DIAGNOSTICS (ArviZ Integration)")
print("="*70)

print("\nGenerating diagnostic plots...")
print("  Note: Close each plot window to continue to the next plot")

# Convert to ArviZ InferenceData
import arviz as az
idata = result.to_inference_data()

# Plot 1: Trace plot (convergence visual check)
print("\n  [1/6] Trace plot - Visual convergence check")
print("        • Left: Marginal posteriors (should overlap for all chains)")
print("        • Right: Parameter evolution ('fuzzy caterpillar', no trends)")
az.plot_trace(idata, var_names=['G0', 'eta'], figsize=(12, 6))
plt.tight_layout()
plt.savefig('/tmp/rheojax_trace.png', dpi=150, bbox_inches='tight')
print("        ✓ Saved to /tmp/rheojax_trace.png")
plt.close()

# Plot 2: Rank plot (most sensitive convergence diagnostic)
print("\n  [2/6] Rank plot - Most sensitive convergence test")
print("        • Uniform histogram → converged")
print("        • Non-uniform → NOT converged (increase warmup)")
az.plot_rank(idata, var_names=['G0', 'eta'], figsize=(12, 5))
plt.tight_layout()
plt.savefig('/tmp/rheojax_rank.png', dpi=150, bbox_inches='tight')
print("        ✓ Saved to /tmp/rheojax_rank.png")
plt.close()

# Plot 3: Pair plot (correlations and divergences)
print("\n  [3/6] Pair plot - Parameter correlations + divergences")
print("        • Diagonal line → strong correlation")
print("        • Red points → divergences (problematic regions)")
correlation = np.corrcoef(posterior['G0'], posterior['eta'])[0, 1]
az.plot_pair(
    idata,
    var_names=['G0', 'eta'],
    kind='scatter',
    divergences=True,
    figsize=(8, 8)
)
plt.tight_layout()
plt.savefig('/tmp/rheojax_pair.png', dpi=150, bbox_inches='tight')
print(f"        ✓ Saved to /tmp/rheojax_pair.png")
print(f"        ✓ Correlation(G₀, η) = {correlation:.3f}")
plt.close()

# Plot 4: Autocorrelation plot (mixing quality)
print("\n  [4/6] Autocorrelation plot - Mixing quality")
print("        • Fast decay to ~0 → good mixing")
print("        • Slow decay → high autocorrelation (increase samples)")
az.plot_autocorr(idata, var_names=['G0', 'eta'], max_lag=100, figsize=(12, 5))
plt.tight_layout()
plt.savefig('/tmp/rheojax_autocorr.png', dpi=150, bbox_inches='tight')
print("        ✓ Saved to /tmp/rheojax_autocorr.png")
plt.close()

# Plot 5: ESS plot (sampling efficiency)
print("\n  [5/6] ESS plot - Sampling efficiency per parameter")
print("        • ESS > 400 → sufficient")
print("        • ESS < 400 → increase samples")
az.plot_ess(idata, var_names=['G0', 'eta'], kind='local', figsize=(12, 5))
plt.tight_layout()
plt.savefig('/tmp/rheojax_ess.png', dpi=150, bbox_inches='tight')
print("        ✓ Saved to /tmp/rheojax_ess.png")
plt.close()

# Plot 6: Forest plot (credible intervals)
print("\n  [6/6] Forest plot - Credible interval comparison")
print("        • Quick visual of parameter magnitudes and uncertainties")
az.plot_forest(idata, var_names=['G0', 'eta'], combined=True, hdi_prob=0.95)
plt.tight_layout()
plt.savefig('/tmp/rheojax_forest.png', dpi=150, bbox_inches='tight')
print("        ✓ Saved to /tmp/rheojax_forest.png")
plt.close()

print("\n✓ All diagnostic plots saved to /tmp/rheojax_*.png")

# =============================================================================
# Step 7: Final Summary and Recommendations
# =============================================================================
print("\n" + "="*70)
print("WORKFLOW SUMMARY")
print("="*70)

print("\n✓ Completed 3-stage Bayesian workflow:")
print(f"  [1] NLSQ point estimation:      {nlsq_time:.2f}s")
print(f"  [2] NUTS posterior sampling:     {bayes_time:.1f}s (warm-start)")
print(f"  [3] ArviZ diagnostic plots:      6 plots generated")

print(f"\n✓ Convergence assessment:")
if all_converged:
    print("  • All parameters converged (R-hat < 1.01, ESS > 400)")
    print("  • Results are reliable and can be interpreted")
else:
    print("  • ⚠ Convergence issues detected")
    print("  • Increase num_warmup or num_samples and rerun")

print(f"\n✓ Uncertainty quantification:")
print(f"  • G₀ uncertainty: ±{summary['G0']['std']/summary['G0']['mean']*100:.1f}%")
print(f"  • η uncertainty:  ±{summary['eta']['std']/summary['eta']['mean']*100:.1f}%")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Apply this workflow to your own rheological data")
print("2. Try different models (20 models available)")
print("3. Explore tutorial notebooks in examples/bayesian/:")
print("   • 01-bayesian-basics.ipynb (40 min)")
print("   • 02-prior-selection.ipynb (35 min)")
print("   • 03-convergence-diagnostics.ipynb (45 min)")
print("   • 04-model-comparison.ipynb (40 min)")
print("   • 05-uncertainty-propagation.ipynb (45 min)")
print("\n4. Read documentation:")
print("   • docs/BAYESIAN_QUICK_START.md")
print("   • docs/BAYESIAN_WORKFLOW_SUMMARY.md")

print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nFor questions or issues:")
print("  • GitHub: https://github.com/imewei/rheojax")
print("  • Docs: https://rheojax.readthedocs.io")
print("="*70)
