==================
Bayesian Inference
==================

This guide covers RheoJAX's comprehensive Bayesian inference capabilities, including NLSQ -> NUTS workflow,
ArviZ diagnostic visualization, and best practices for MCMC analysis.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

RheoJAX implements a state-of-the-art Bayesian workflow that combines:

1. **NLSQ Optimization**: Fast GPU-accelerated point estimation (5-270x speedup)
2. **NumPyro NUTS**: Bayesian inference with automatic warm-start from NLSQ
3. **ArviZ Diagnostics**: Comprehensive MCMC quality assessment tools (all 6 diagnostic plots)

This workflow provides 2-5x faster convergence compared to cold-start MCMC while maintaining
full Bayesian uncertainty quantification.

**Implementation Status**: [done] **FULLY IMPLEMENTED** (v0.2.0, enhanced v0.4.0)

All 21 rheological models support the complete Bayesian workflow through ``BayesianMixin`` inheritance.

**v0.4.0 Enhancement**: Mode-aware Bayesian inference ensures correct posteriors for all test modes (relaxation, creep, oscillation).
Previously, test_mode was captured as class state, causing all Bayesian fits to use the last-fitted mode.
Now, test_mode is captured as a closure parameter, guaranteeing physically correct posteriors regardless of prior `.fit()` calls.

When to Use Bayesian Inference
==============================

Use Bayesian When You Need
--------------------------

[done] **Parameter uncertainty quantification** - "How well-constrained are my parameters?"

[done] **Credible intervals** - "95% probability the parameter is in this range"

[done] **Parameter correlations** - "Are G_0 and eta identifiable with my data?"

[done] **Prediction uncertainty** - "Error bands on model predictions"

[done] **Model comparison** - "Which model best explains my data? (WAIC, LOO)"

Skip Bayesian When
------------------

- Avoid Bayesian inference for quick screening with high signal-to-noise data.

- Avoid when real-time analysis speed is critical.

- Avoid when parameters are already well-constrained and uncertainty is not important.

Quick Start
===========

Basic Bayesian Workflow
-----------------------

.. code-block:: python

   from rheojax.models.maxwell import Maxwell
   import numpy as np

   # 1. Prepare data
   t = np.linspace(0.1, 10, 50)
   G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, size=t.shape)

   # 2. NLSQ point estimation (fast)
   model = Maxwell()
   model.fit(t, G_data)
   print(f"NLSQ: G0={model.parameters.get_value('G0'):.3e}")

   # 3. Bayesian inference (warm-start from NLSQ)
   result = model.fit_bayesian(
       t, G_data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=1
   )

   # 4. Check convergence
   print(f"R-hat: {result.diagnostics['r_hat']}")
   print(f"ESS: {result.diagnostics['ess']}")

Using BayesianPipeline
----------------------

For complex workflows, use the fluent API:

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   pipeline = (BayesianPipeline()
       .load('data.csv', x_col='time', y_col='stress')
       .fit_nlsq('maxwell')
       .fit_bayesian(num_samples=2000, num_warmup=1000)
       .plot_posterior()
       .plot_trace()
       .save('results.hdf5'))

   # Access results
   summary = pipeline.get_posterior_summary()
   diagnostics = pipeline.get_diagnostics()

NLSQ + NUTS Workflow
====================

Step 1: NLSQ Point Estimation
-----------------------------

NLSQ provides fast, GPU-accelerated optimization:

.. code-block:: python

   from rheojax.models.maxwell import Maxwell

   model = Maxwell()
   model.fit(t, G_data, method='nlsq')

   # Access fitted parameters
   G0 = model.parameters.get_value('G0')
   eta = model.parameters.get_value('eta')

**Advantages:**
- 5-270x faster than scipy optimization
- GPU acceleration available
- Robust convergence with JAX gradients
- Perfect for warm-starting MCMC

Step 2: Bayesian Inference with Warm-Start
------------------------------------------

Use NLSQ point estimates to initialize NUTS:

.. code-block:: python

   # Extract initial values from NLSQ fit
   initial_values = {
       'G0': model.parameters.get_value('G0'),
       'eta': model.parameters.get_value('eta')
   }

   # Run NUTS with warm-start
   result = model.fit_bayesian(
       t, G_data,
       num_warmup=1000,
       num_samples=2000,
       initial_values=initial_values  # Warm-start
   )

**Benefits of Warm-Start:**
- 2-5x faster convergence
- Fewer divergent transitions
- Better exploration of posterior
- Reduced warmup time needed

Understanding Results
=====================

Posterior Samples
-----------------

Access posterior samples for each parameter:

.. code-block:: python

   posterior_G0 = result.posterior_samples['G0']
   posterior_eta = result.posterior_samples['eta']

   # Compute statistics
   G0_mean = np.mean(posterior_G0)
   G0_std = np.std(posterior_G0)

Summary Statistics
------------------

Get comprehensive summary for all parameters:

.. code-block:: python

   summary = result.summary
   print(summary['G0'])
   # {'mean': 1.02e5, 'std': 2.1e3, 'median': 1.01e5,
   #  'q05': 9.8e4, 'q25': 1.00e5, 'q75': 1.03e5, 'q95': 1.05e5}

Credible Intervals
------------------

Compute highest density intervals (HDI):

.. code-block:: python

   # 95% credible intervals
   intervals_95 = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )
   print(f"G0 95% CI: {intervals_95['G0']}")

   # 68% credible intervals (1 sigma)
   intervals_68 = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.68
   )

ArviZ Diagnostic Plots
======================

Rheo integrates ArviZ for comprehensive MCMC diagnostics. All plotting methods support
the fluent API pattern with ``show`` parameter and ``.save_figure()`` chaining.

1. Pair Plot (Parameter Correlations)
-------------------------------------

Visualize pairwise parameter relationships to identify correlations and non-identifiability:

.. code-block:: python

   pipeline.plot_pair(
       var_names=['G0', 'eta'],    # Specific parameters (or None for all)
       kind='scatter',              # 'scatter', 'kde', or 'hexbin'
       divergences=True             # Highlight problematic regions
   )

   # Save without displaying
   pipeline.plot_pair(show=False).save_figure('pair.pdf')

**Use Cases:**
- Detect strong parameter correlations (indicates non-identifiability)
- Identify funnel geometry (divergences concentrated in narrow regions)
- Diagnose multimodal posteriors (multiple clusters visible)

**Interpretation:**
- Diagonal pattern: Strong correlation (consider reparameterization)
- Scattered cloud: Good, independent parameters
- Red points (divergences): Problematic posterior geometry

2. Forest Plot (Credible Intervals)
-----------------------------------

Compare parameter estimates with uncertainty visualization:

.. code-block:: python

   pipeline.plot_forest(
       hdi_prob=0.95,              # 0.68 (1 sigma), 0.95 (2 sigma), 0.997 (3 sigma)
       combined=True                # Combine multiple chains
   )

   # Different credibility levels
   pipeline.plot_forest(hdi_prob=0.68)  # 68% CI (1 sigma)

**Use Cases:**
- Quick comparison of parameter magnitudes
- Assess parameter uncertainty at a glance
- Identify poorly estimated parameters (wide intervals)

**Interpretation:**
- Point: Posterior mean or median
- Error bars: Credible interval (HDI)
- Narrow bars: Well-constrained parameter
- Wide bars: High uncertainty

3. Energy Plot (NUTS Diagnostic)
--------------------------------

NUTS-specific diagnostic for posterior geometry:

.. code-block:: python

   pipeline.plot_energy()

**Use Cases:**
- Detect heavy-tailed posteriors
- Identify funnel geometry
- Diagnose problematic parameterizations

**Interpretation:**
- Similar distributions (marginal vs. transition): Good sampling
- Different distributions: Problematic posterior geometry
- Requires multi-chain MCMC (not available for single-chain)

**Note**: Energy plot requires ``num_chains > 1`` in ``fit_bayesian()``.

4. Autocorrelation Plot (Mixing Diagnostic)
-------------------------------------------

Check MCMC chain mixing quality:

.. code-block:: python

   pipeline.plot_autocorr(
       max_lag=100,                # Lag length to display
       combined=False               # Per-chain or combined
   )

**Use Cases:**
- Assess mixing efficiency
- Determine if more samples needed
- Identify poor parameter exploration

**Interpretation:**
- **Goal**: Autocorrelation drops to ~0 within 10-20 lags
- High persistent autocorrelation: Poor mixing, need more samples
- Quick decay: Good mixing, efficient sampling

5. Rank Plot (Convergence Diagnostic)
-------------------------------------

Modern convergence diagnostic (alternative to trace plots):

.. code-block:: python

   pipeline.plot_rank()

**Use Cases:**
- Detect non-convergence between chains
- Identify chain sticking
- Assess mixing uniformity

**Interpretation:**
- **Goal**: Uniform histogram across all bins
- Non-uniform distribution: Poor convergence
- Vertical bands: Chain sticking to specific values
- Patterns in ranks: Insufficient mixing

6. ESS Plot (Effective Sample Size)
-----------------------------------

Quantify sampling efficiency:

.. code-block:: python

   pipeline.plot_ess(
       kind='local'                # 'local', 'quantile', or 'evolution'
   )

   # Quantile ESS for tail behavior
   pipeline.plot_ess(kind='quantile')

**Use Cases:**
- Assess sampling efficiency per parameter
- Identify parameters needing more samples
- Evaluate overall chain quality

**Interpretation:**
- **Goal**: ESS > 400 for bulk and tail estimates
- ESS < 400: Need more samples or better mixing
- ESS / total_samples: Sampling efficiency ratio
- Low ESS: High autocorrelation, poor exploration

Complete Diagnostic Workflow
----------------------------

Run all diagnostics in sequence:

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   pipeline = (BayesianPipeline()
       .load('data.csv', x_col='time', y_col='stress')
       .fit_nlsq('maxwell')
       .fit_bayesian(num_samples=2000, num_warmup=1000))

   # Run all ArviZ diagnostics
   (pipeline
       .plot_pair(divergences=True, show=False).save_figure('pair.pdf')
       .plot_forest(hdi_prob=0.95, show=False).save_figure('forest.pdf')
       .plot_autocorr(show=False).save_figure('autocorr.pdf')
       .plot_rank(show=False).save_figure('rank.pdf')
       .plot_ess(kind='local', show=False).save_figure('ess.pdf'))

Advanced ArviZ Integration
==========================

Converting to InferenceData
---------------------------

Access ArviZ InferenceData for advanced analysis:

.. code-block:: python

   # Get InferenceData from BayesianResult
   idata = result.to_inference_data()

   # Use any ArviZ function
   import arviz as az
   az.plot_trace(idata)
   az.summary(idata)
   az.plot_posterior(idata)

   # Custom analysis
   az.loo(idata)  # Leave-one-out cross-validation
   az.waic(idata)  # Widely applicable information criterion

InferenceData Structure
-----------------------

The InferenceData object contains:

- **posterior**: Posterior samples for all parameters
- **sample_stats**: NUTS diagnostics (energy, divergences, tree depth)
- **observed_data**: Original observed data
- **posterior_predictive**: Predictions from posterior (if available)

.. code-block:: python

   # Explore InferenceData structure
   print(idata.posterior)
   print(idata.sample_stats)

   # Access specific diagnostic
   divergences = idata.sample_stats.diverging
   print(f"Divergent transitions: {divergences.sum().item()}")

Convergence Diagnostics
=======================

R-hat (Gelman-Rubin Statistic)
------------------------------

Measures convergence across multiple chains:

.. code-block:: python

   r_hat = result.diagnostics['r_hat']
   for param, value in r_hat.items():
       print(f"{param}: R-hat = {value:.4f}")

**Interpretation:**
- **R-hat < 1.01**: Excellent convergence
- **1.01 < R-hat < 1.05**: Acceptable
- **R-hat > 1.05**: Poor convergence, increase warmup

**Troubleshooting High R-hat:**
1. Increase ``num_warmup`` (try 2000-5000)
2. Increase ``num_samples`` (try 5000+)
3. Use warm-start from NLSQ
4. Check for multimodal posterior

Effective Sample Size (ESS)
---------------------------

Quantifies independent samples:

.. code-block:: python

   ess = result.diagnostics['ess']
   for param, value in ess.items():
       print(f"{param}: ESS = {value:.0f}")

**Interpretation:**
- **ESS > 400**: Good
- **200 < ESS < 400**: Acceptable but consider more samples
- **ESS < 200**: Insufficient, increase ``num_samples``

**Improving ESS:**
1. Increase ``num_samples``
2. Use warm-start initialization
3. Check for high autocorrelation
4. Consider reparameterization if correlations high

Divergent Transitions
---------------------

Indicates problematic posterior geometry:

.. code-block:: python

   div_count = result.diagnostics['divergences']
   print(f"Divergent transitions: {div_count}")

**Interpretation:**
- **0 divergences**: Excellent
- **< 1% of samples**: Acceptable
- **> 1% of samples**: Problematic, investigate

**Troubleshooting Divergences:**
1. **Use NLSQ warm-start** (most effective)
2. Increase ``adapt_step_size`` parameter
3. Check parameter bounds are reasonable
4. Verify model is appropriate for data
5. Look at pair plot to identify problematic regions

Best Practices
==============

Workflow Recommendations
------------------------

1. **Always use NLSQ warm-start**

   .. code-block:: python

      # GOOD: Warm-start workflow
      model.fit(t, G_data)  # NLSQ first
      result = model.fit_bayesian(t, G_data)  # Auto warm-start

      # AVOID: Cold start
      result = model.fit_bayesian(t, G_data, initial_values=None)

2. **Check convergence diagnostics**

   .. code-block:: python

      # Always verify R-hat < 1.01 and ESS > 400
      assert all(r < 1.01 for r in result.diagnostics['r_hat'].values())
      assert all(e > 400 for e in result.diagnostics['ess'].values())

3. **Use sufficient samples**

   .. code-block:: python

      # Minimum recommended
      result = model.fit_bayesian(
          t, G_data,
          num_warmup=1000,
          num_samples=2000
      )

      # For production / publication
      result = model.fit_bayesian(
          t, G_data,
          num_warmup=2000,
          num_samples=5000
      )

4. **Run diagnostic plots**

   .. code-block:: python

      # Minimal diagnostics
      pipeline.plot_pair().plot_forest()

      # Comprehensive diagnostics
      pipeline.plot_pair().plot_forest().plot_autocorr().plot_rank().plot_ess()

Parameter Settings
------------------

Recommended settings by use case:

**Quick Exploration** (fast iteration):

.. code-block:: python

   result = model.fit_bayesian(
       t, G_data,
       num_warmup=500,
       num_samples=1000
   )

**Standard Analysis** (recommended default):

.. code-block:: python

   result = model.fit_bayesian(
       t, G_data,
       num_warmup=1000,
       num_samples=2000
   )

**Production / Publication** (high quality):

.. code-block:: python

   result = model.fit_bayesian(
       t, G_data,
       num_warmup=2000,
       num_samples=5000,
       num_chains=4  # For parallel sampling
   )

Common Pitfalls
===============

1. Skipping NLSQ Warm-Start
---------------------------

**Problem**: Cold-start MCMC converges slowly with many divergences.

**Solution**: Always fit with NLSQ first:

.. code-block:: python

   # WRONG
   result = model.fit_bayesian(t, G_data, initial_values=None)

   # RIGHT
   model.fit(t, G_data)  # NLSQ first
   result = model.fit_bayesian(t, G_data)  # Auto warm-start

2. Ignoring Convergence Diagnostics
-----------------------------------

**Problem**: Using results without checking convergence.

**Solution**: Always verify R-hat and ESS:

.. code-block:: python

   result = model.fit_bayesian(t, G_data)

   # Check diagnostics
   r_hat = result.diagnostics['r_hat']
   ess = result.diagnostics['ess']

   if any(r > 1.01 for r in r_hat.values()):
       print("WARNING: Poor convergence, increase num_warmup")

   if any(e < 400 for e in ess.values()):
       print("WARNING: Low ESS, increase num_samples")

3. Insufficient Samples
-----------------------

**Problem**: Using too few samples leads to poor posterior approximation.

**Solution**: Use at least 2000 samples:

.. code-block:: python

   # WRONG
   result = model.fit_bayesian(t, G_data, num_samples=100)

   # RIGHT
   result = model.fit_bayesian(t, G_data, num_samples=2000)

4. Not Checking Pair Plots
--------------------------

**Problem**: Missing parameter correlations and non-identifiability.

**Solution**: Always check pair plot for correlations:

.. code-block:: python

   pipeline.plot_pair(divergences=True)

   # If strong correlations visible, consider:
   # 1. Reparameterization
   # 2. More informative priors
   # 3. More data or different experimental conditions

Examples
========

Example 1: Maxwell Model with Full Diagnostics
----------------------------------------------

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline
   import numpy as np
   import pandas as pd

   # Generate synthetic data
   t = np.linspace(0.1, 10, 50)
   G0_true, eta_true = 1e5, 1e5
   G_true = G0_true * np.exp(-t * G0_true / eta_true)
   G_data = G_true + np.random.normal(0, 1e3, size=t.shape)

   # Save to CSV
   pd.DataFrame({'time': t, 'stress': G_data}).to_csv('maxwell_data.csv', index=False)

   # Complete workflow
   pipeline = (BayesianPipeline()
       .load('maxwell_data.csv', x_col='time', y_col='stress')
       .fit_nlsq('maxwell')
       .fit_bayesian(num_samples=2000, num_warmup=1000))

   # Check convergence
   diagnostics = pipeline.get_diagnostics()
   print(f"R-hat: {diagnostics['r_hat']}")
   print(f"ESS: {diagnostics['ess']}")
   print(f"Divergences: {diagnostics['divergences']}")

   # Generate all diagnostic plots
   (pipeline
       .plot_posterior(show=False).save_figure('posterior.pdf')
       .plot_trace(show=False).save_figure('trace.pdf')
       .plot_pair(show=False).save_figure('pair.pdf')
       .plot_forest(show=False).save_figure('forest.pdf')
       .plot_autocorr(show=False).save_figure('autocorr.pdf')
       .plot_rank(show=False).save_figure('rank.pdf')
       .plot_ess(show=False).save_figure('ess.pdf'))

   # Get summary
   summary = pipeline.get_posterior_summary()
   print(summary)

Example 2: Comparing Credibility Levels
---------------------------------------

.. code-block:: python

   from rheojax.models.maxwell import Maxwell
   import numpy as np

   model = Maxwell()
   t = np.linspace(0.1, 10, 50)
   G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, size=t.shape)

   # Fit
   model.fit(t, G_data)
   result = model.fit_bayesian(t, G_data)

   # Compare different credibility levels
   ci_68 = model.get_credible_intervals(result.posterior_samples, credibility=0.68)
   ci_95 = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   ci_997 = model.get_credible_intervals(result.posterior_samples, credibility=0.997)

   print("G0 Credible Intervals:")
   print(f"  68% (1 sigma): {ci_68['G0']}")
   print(f"  95% (2 sigma): {ci_95['G0']}")
   print(f"  99.7% (3 sigma): {ci_997['G0']}")

Example 3: Advanced ArviZ Analysis
----------------------------------

.. code-block:: python

   from rheojax.models.maxwell import Maxwell
   import arviz as az
   import numpy as np

   # Fit model
   model = Maxwell()
   t = np.linspace(0.1, 10, 50)
   G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, size=t.shape)

   model.fit(t, G_data)
   result = model.fit_bayesian(t, G_data, num_samples=2000)

   # Convert to InferenceData
   idata = result.to_inference_data()

   # Use ArviZ functions
   summary = az.summary(idata, hdi_prob=0.95)
   print(summary)

   # Plot posterior with HDI
   az.plot_posterior(idata, hdi_prob=0.95)

   # Analyze effective sample size
   ess_bulk = az.ess(idata, var_names=['G0', 'eta'])
   print(f"ESS (bulk): {ess_bulk}")

   # Check Monte Carlo standard error
   mcse = az.mcse(idata, var_names=['G0', 'eta'])
   print(f"MCSE: {mcse}")

Troubleshooting Common Issues
=============================

Issue 1: R-hat > 1.01 (Not Converged)
-------------------------------------

**Symptoms:**

.. code-block:: text

   R-hat (G_0): 1.0341  [X] NOT converged

**Solutions:**

.. code-block:: python

   # Increase warmup iterations
   result = pipeline.fit_bayesian(
       num_warmup=2000,  # Was: 1000
       num_samples=2000,
       num_chains=4
   )

   # Check rank plot: Is histogram uniform?
   pipeline.plot_rank()  # Non-uniform -> not converged

Issue 2: ESS < 400 (Poor Sampling Efficiency)
---------------------------------------------

**Symptoms:**

.. code-block:: text

   ESS (G_0): 187  [X] Low (increase samples)

**Solutions:**

.. code-block:: python

   # Increase number of samples
   result = pipeline.fit_bayesian(
       num_warmup=1000,
       num_samples=5000,  # Was: 2000
       num_chains=4
   )

   # Check autocorrelation plot: Is mixing slow?
   pipeline.plot_autocorr()  # Slow decay -> high autocorrelation

Issue 3: High Divergence Rate (>1%)
-----------------------------------

**Symptoms:**

.. code-block:: text

   Divergences: 84 (1.05%)  [X] High

**Solutions:**

.. code-block:: python

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

Issue 4: Strong Parameter Correlations
--------------------------------------

**Symptoms:**

.. code-block:: python

   # Pair plot shows diagonal line between G_0 and eta
   import numpy as np
   correlation = np.corrcoef(G0_samples, eta_samples)[0, 1]
   print(f"Correlation: {correlation:.3f}")  # |rho| > 0.7

**This is often EXPECTED for rheological models:**

- Maxwell: G_0 and eta both affect relaxation time tau = eta/G_0
- This is a physical correlation, not a problem
- If correlation > 0.9, consider collecting data at different time scales

Best Practices Checklist
========================

Before Running Bayesian Inference
---------------------------------

[done] Run NLSQ first (Stage 1) for warm-start

[done] Use meaningful parameter bounds

[done] Check data quality (outliers, noise level)

[done] Use multi-chain MCMC (``num_chains=4``)

When Running Inference
----------------------

[done] Start with ``num_warmup=1000``, ``num_samples=2000``

[done] Use warm-start from NLSQ (automatically done by BayesianPipeline)

[done] Monitor progress (NumPyro shows progress bars)

After Inference
---------------

[done] **ALWAYS check convergence** (R-hat, ESS, divergences)

[done] Run all 6 ArviZ diagnostic plots

[done] Verify rank plot is uniform

[done] Check pair plot for correlations and divergences

[done] Document diagnostics in reports

Red Flags (Do Not Trust Results)
--------------------------------

- R-hat > 1.01 for any parameter

- Effective sample size (ESS) < 100 for any parameter

- Divergence rate > 5%

- Non-uniform rank plot

- Energy plot shows poor overlap

Model Support
=============

All 20 rheological models inherit ``BayesianMixin`` through ``BaseModel`` and support complete Bayesian workflows:

Classical Viscoelastic (3 models)
---------------------------------

- Maxwell
- Zener
- SpringPot

Flow Models (6 models)
----------------------

- Bingham
- PowerLaw
- Herschel-Bulkley
- Carreau
- Carreau-Yasuda
- Cross

Fractional Models (11 models)
-----------------------------

- fractional_burgers
- fractional_jeffreys
- fractional_kelvin_voigt
- fractional_kv_zener
- fractional_maxwell_gel
- fractional_maxwell_liquid
- fractional_maxwell_model
- fractional_poynting_thomson
- fractional_zener_ll
- fractional_zener_sl
- fractional_zener_ss

Each model provides:

- ``.fit()`` -> NLSQ point estimation
- ``.fit_bayesian()`` -> NUTS posterior sampling
- ``.sample_prior()`` -> Prior predictive sampling
- ``.get_credible_intervals()`` -> Posterior uncertainty quantification

Performance Characteristics
===========================

NLSQ Optimization
-----------------

- **5-270x speedup** over scipy on CPU
- **Additional GPU acceleration** on CUDA systems
- Warm-start from good initial guesses improves convergence
- Typical: 50-500 iterations for rheological models

Bayesian Inference (NUTS)
-------------------------

- **2-5x faster convergence** with NLSQ warm-start vs cold start
- Typical settings: ``num_warmup=1000``, ``num_samples=2000``
- Good convergence: R-hat < 1.01, ESS > 400
- Warm-start reduces divergences significantly (10-100x fewer)

Typical Performance
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 30

   * - Stage
     - Duration
     - Speedup with Warm-Start
   * - NLSQ optimization
     - 0.1-2s
     - N/A
   * - NUTS sampling (cold start)
     - 1-5 min
     - Baseline
   * - NUTS sampling (warm-start)
     - 0.5-1 min
     - **2-5x faster**
   * - ArviZ diagnostics
     - 1-5s
     - N/A

Further Reading
===============

**NumPyro Documentation**
   https://num.pyro.ai/en/stable/

**ArviZ Documentation**
   https://arviz-devs.github.io/arviz/

**NLSQ Repository**
   https://github.com/imewei/NLSQ

**MCMC Best Practices**
   - Vehtari et al. (2021). "Rank-Normalization, Folding, and Localization"
   - Gelman et al. (2013). "Bayesian Data Analysis"
   - Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo"

**RheoJAX Examples**
   See :doc:`../examples/index` for 5 comprehensive Bayesian inference tutorial notebooks:

   1. ``examples/bayesian/01-bayesian-basics.ipynb`` - Complete NLSQ -> NUTS workflow
   2. ``examples/bayesian/02-prior-selection.ipynb`` - Prior elicitation and sensitivity
   3. ``examples/bayesian/03-convergence-diagnostics.ipynb`` - Master all 6 ArviZ diagnostic plots
   4. ``examples/bayesian/04-model-comparison.ipynb`` - WAIC, LOO, and model selection
   5. ``examples/bayesian/05-uncertainty-propagation.ipynb`` - Derived quantities and prediction bands
