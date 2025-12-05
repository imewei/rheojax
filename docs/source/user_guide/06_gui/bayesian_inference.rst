.. _gui-bayesian-inference:

===================
Bayesian Inference
===================

The Bayesian page provides MCMC (Markov Chain Monte Carlo) inference with
ArviZ diagnostic visualizations.

Overview
========

Bayesian inference provides:

- **Uncertainty quantification**: Credible intervals for parameters
- **Full posterior**: Complete probability distributions
- **Diagnostics**: Convergence metrics (R-hat, ESS)
- **Model comparison**: Evidence-based model selection

When to Use Bayesian
--------------------

Use Bayesian inference when you need:

- Parameter uncertainties
- Confidence intervals for predictions
- Robust handling of measurement noise
- Publication-quality error bars

Configuration
=============

MCMC Settings
-------------

**Warmup Samples**
   Number of initial samples for adaptation (discarded).
   Recommended: 1000-2000

**Posterior Samples**
   Number of samples for inference.
   Recommended: 2000-4000

**Chains**
   Number of independent MCMC chains.
   Recommended: 4 (for R-hat calculation)

**Random Seed**
   For reproducibility.
   Default: 42

Prior Configuration
-------------------

Access via **Prior Settings** panel:

**Default Priors**
   Automatic priors based on parameter bounds:

   - Uniform for bounded parameters
   - Log-uniform for scale parameters

**Custom Priors**
   Specify distribution for each parameter:

   - Normal(mean, std)
   - LogNormal(mean, std)
   - Uniform(low, high)
   - HalfNormal(std)

Running Inference
=================

Prerequisites
-------------

1. Load data and select model
2. Run NLSQ fit first (provides warm start)
3. Configure MCMC settings

Starting MCMC
-------------

1. Click **"Run Bayesian Inference"**
2. Progress shows:

   - Warmup phase progress
   - Sampling phase progress
   - Current acceptance rate

3. Wait for completion (typically 5-60 seconds)

Warm Start
----------

**Always run NLSQ first!**

Warm start from NLSQ:

- Initializes chains near optimum
- Reduces warmup time
- Improves convergence
- Avoids poor initial samples

Diagnostics
===========

ArviZ Plot Types
----------------

The **ArviZ Canvas** provides multiple diagnostic views:

**Trace Plot** (Default)
   - Left: Posterior density
   - Right: MCMC chain trace
   - Check: Chains should mix well

**Pair Plot**
   - Parameter correlations
   - Divergence markers
   - Check: No extreme correlations

**Forest Plot**
   - Credible intervals comparison
   - Point estimates with HDI
   - Check: Intervals don't overlap zero (if significant)

**Posterior Plot**
   - Marginal distributions
   - HDI intervals
   - Check: Unimodal, well-defined peaks

**Energy Plot**
   - NUTS energy diagnostics
   - Check: Marginal and transition should overlap

**Rank Plot**
   - Chain rank statistics
   - Check: Uniform distribution across chains

**ESS Plot**
   - Effective sample size
   - Check: ESS > 400 for reliable estimates

**Autocorrelation**
   - Chain autocorrelation
   - Check: Quick decay to zero

Convergence Metrics
-------------------

**R-hat (Gelman-Rubin)**
   Potential scale reduction factor.

   - Good: R-hat < 1.01
   - Acceptable: R-hat < 1.1
   - Problematic: R-hat > 1.1 (more samples needed)

**ESS (Effective Sample Size)**
   Independent samples equivalent.

   - Good: ESS > 400 for all parameters
   - Low ESS indicates autocorrelation

**Divergences**
   Numerical integration issues.

   - Good: 0 divergences
   - Some: May indicate model issues
   - Many: Investigate model/data

Results
=======

Posterior Summary
-----------------

After inference completes:

- **Mean**: Posterior mean
- **Std**: Posterior standard deviation
- **HDI 3%/97%**: 94% credible interval
- **MCSE**: Monte Carlo standard error

Credible Intervals
------------------

Different credibility levels:

- 50% HDI: Core distribution
- 94% HDI: Standard reporting
- 99% HDI: Conservative bounds

Prediction Intervals
--------------------

Generate prediction uncertainty:

1. Click **"Plot Predictions"**
2. Shows fit with credible bands
3. Inner band: Parameter uncertainty
4. Outer band: Plus observation noise

Exporting Results
=================

Posterior Samples
-----------------

Export raw samples:

1. Go to **Export** page
2. Select **Posterior Samples**
3. Choose format (CSV, HDF5)

ArviZ InferenceData
-------------------

Export full ArviZ object:

1. Select **ArviZ InferenceData**
2. Save as NetCDF or pickle
3. Load later for further analysis

Diagnostic Plots
----------------

Export diagnostic figures:

1. Right-click any plot
2. Select **Export Plot**
3. Choose format (PNG, PDF, SVG)

Troubleshooting
===============

High R-hat
----------

R-hat > 1.1 indicates poor convergence:

1. Increase warmup samples
2. Increase total samples
3. Check for multimodality
4. Simplify model

Low ESS
-------

Low effective samples:

1. Increase total samples
2. Check autocorrelation
3. Tune step size (advanced)

Divergences
-----------

Many divergences indicate problems:

1. Reparameterize model
2. Adjust priors
3. Check data quality
4. Try non-centered parameterization

Slow Sampling
-------------

MCMC is slow:

1. Ensure NLSQ warm start
2. Reduce model complexity
3. Use GPU if available
4. Reduce number of chains

Best Practices
==============

1. **Always warm start from NLSQ**
2. **Use 4+ chains for diagnostics**
3. **Check all diagnostic plots**
4. **Report R-hat and ESS with results**
5. **Use HDI, not mean±std for reporting**
6. **Save InferenceData for reproducibility**
