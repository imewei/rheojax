.. _gui-diagnostics:

===========
Diagnostics
===========

The Diagnostics page provides comprehensive MCMC diagnostics and posterior analysis
with ArviZ integration for Bayesian inference results.

Overview
========

The Diagnostics page offers:

- **ArviZ Plots**: Seven diagnostic plot types for MCMC quality assessment
- **Goodness-of-Fit Metrics**: R-squared, R-hat, ESS, divergences with color coding
- **Model Comparison**: Side-by-side comparison of multiple Bayesian results
- **Export**: Save diagnostic plots in publication-quality formats

Accessing Diagnostics
=====================

From the main window:

1. Run Bayesian inference on the **Bayesian** tab
2. Click **Show Diagnostics** or navigate to the **Diagnostics** tab
3. Select a model from the dropdown if multiple results exist

ArviZ Plot Types
================

The Diagnostics page provides seven ArviZ diagnostic plot types via tabbed interface:

Trace Plot
----------

.. image:: /_static/gui/trace_plot.png
   :alt: Trace Plot Example
   :width: 600px
   :align: center

**Purpose**: Visualize MCMC chain behavior and posterior distributions.

- **Left panel**: Posterior density (KDE)
- **Right panel**: Chain trace over iterations

**What to look for**:

- Chains should overlap (good mixing)
- No trends or drift
- Stationary behavior after warmup

Forest Plot
-----------

**Purpose**: Compare credible intervals across parameters.

- Point estimates with HDI (Highest Density Interval)
- Default 95% credibility level
- Customize HDI via ``hdi_prob`` parameter

**What to look for**:

- Intervals that don't overlap zero (significant parameters)
- Relative uncertainty between parameters
- Compare credible widths

Pair Plot
---------

**Purpose**: Visualize parameter correlations and divergences.

- Scatter plots of parameter pairs
- Marginal distributions on diagonal
- Divergence markers (red points)

**What to look for**:

- High correlations indicate reparameterization may help
- Divergences clustered in regions indicate model issues
- Funnel shapes suggest non-centered parameterization needed

Energy Plot
-----------

**Purpose**: NUTS-specific energy diagnostics.

- Marginal energy distribution
- Energy transition distribution

**What to look for**:

- Distributions should overlap significantly
- Large separation indicates exploration problems

Autocorrelation Plot
--------------------

**Purpose**: Assess chain mixing via autocorrelation.

- Autocorrelation function (ACF) per parameter
- Decay pattern across lags

**What to look for**:

- Quick decay to zero (good mixing)
- Slow decay indicates low ESS

Rank Plot
---------

**Purpose**: Rank statistics for convergence assessment.

- Rank histogram per chain
- Should be approximately uniform

**What to look for**:

- Uniform distribution across chains
- Non-uniform indicates chain issues

ESS Plot
--------

**Purpose**: Effective Sample Size visualization.

- ESS per parameter
- Comparison across parameters

**What to look for**:

- ESS > 400 for reliable inference
- Low ESS indicates need for more samples

Goodness-of-Fit Metrics
=======================

The metrics panel displays:

.. list-table:: Diagnostic Metrics
   :header-rows: 1
   :widths: 25 50 25

   * - Metric
     - Description
     - Good Values
   * - R-squared
     - Coefficient of determination
     - > 0.95
   * - Chi-squared
     - Sum of squared residuals
     - Lower is better
   * - MPE (%)
     - Mean Percentage Error
     - < 5%
   * - WAIC
     - Widely Applicable Information Criterion
     - Lower is better
   * - LOO
     - Leave-One-Out cross-validation
     - Lower is better
   * - ESS (min)
     - Minimum Effective Sample Size
     - > 400
   * - R-hat (max)
     - Maximum Gelman-Rubin statistic
     - < 1.01
   * - Divergences
     - Number of divergent transitions
     - 0

Color Coding
------------

Metrics are color-coded for quick assessment:

- **Green**: Good (R-hat < 1.01, ESS > 400, Divergences = 0)
- **Yellow**: Warning (R-hat < 1.1, ESS > 100)
- **Red**: Problem (R-hat > 1.1, ESS < 100, Divergences > 0)

Model Comparison
================

Compare multiple Bayesian results:

1. Run Bayesian inference on multiple models
2. Go to **Diagnostics** page
3. View **Model Comparison** panel
4. Click **Refresh Comparison**

Comparison metrics include:

- WAIC (Widely Applicable Information Criterion)
- LOO (Leave-One-Out cross-validation)
- ELPD (Expected Log Pointwise Predictive Density)
- Stacking weights

Exporting Plots
===============

Export diagnostic plots:

1. Select the plot tab you want to export
2. Click **Export [Plot Type] Plot**
3. Choose format:

   - **PNG**: Raster format (150-300 DPI)
   - **PDF**: Vector format (publication quality)
   - **SVG**: Vector format (editable)

4. Select save location
5. Click **Save**

Troubleshooting
===============

High R-hat (> 1.1)
------------------

Chains haven't converged:

1. Increase ``num_warmup`` (try 2000-4000)
2. Increase ``num_samples`` (try 4000-8000)
3. Check for multimodality in trace plot
4. Try different initial values

Low ESS (< 400)
---------------

Insufficient effective samples:

1. Increase ``num_samples``
2. Check autocorrelation plot
3. Consider reparameterization
4. Increase ``num_chains``

Divergences (> 0)
-----------------

Numerical integration issues:

1. Check pair plot for divergence patterns
2. Try ``target_accept_prob=0.95`` (stricter)
3. Reparameterize model (non-centered)
4. Adjust priors to avoid extreme regions

No Results Available
--------------------

If "No Bayesian Results" message appears:

1. Run Bayesian inference on **Bayesian** tab first
2. Ensure inference completed successfully
3. Check that model name matches

Best Practices
==============

1. **Always check all diagnostic plots** before trusting results
2. **Run multiple chains** (4+) for reliable R-hat
3. **Use NLSQ warm-start** for faster convergence
4. **Report R-hat and ESS** with published results
5. **Export ArviZ InferenceData** for reproducibility
6. **Compare models** using WAIC/LOO, not R-squared alone

See Also
========

- :ref:`gui-bayesian-inference` - Bayesian inference configuration
- :ref:`gui-exporting` - Exporting results and plots
- :doc:`/user_guide/03_advanced_topics/bayesian_inference` - Advanced Bayesian topics
