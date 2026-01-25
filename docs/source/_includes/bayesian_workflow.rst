Standard Bayesian Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^

RheoJAX uses a two-stage NLSQ → NUTS workflow for robust Bayesian inference:

**Stage 1: NLSQ Point Estimation (Warm-Start)**

.. code-block:: python

   from rheojax.models import {ModelClass}

   # Fast point estimation using GPU-accelerated NLSQ
   model = {ModelClass}()
   model.fit(x, y, test_mode='{mode}')

   # This provides initial parameter estimates for Bayesian sampling

**Stage 2: Bayesian Inference with NUTS**

.. code-block:: python

   # Bayesian with warm-start from NLSQ fit
   result = model.fit_bayesian(
       x, y,
       test_mode='{mode}',
       num_warmup=1000,     # MCMC warmup iterations
       num_samples=2000,    # Posterior samples
       num_chains=4         # Parallel chains (default)
   )

**Stage 3: Diagnostics with ArviZ**

.. code-block:: python

   import arviz as az

   # Check chain convergence
   print(f"R-hat: {az.rhat(result.inference_data)}")
   print(f"ESS: {az.ess(result.inference_data)}")

   # Visualize diagnostics
   az.plot_trace(result.inference_data)       # Trace plots
   az.plot_pair(result.inference_data)        # Pair plots with correlations
   az.plot_forest(result.inference_data)      # Credible intervals
   az.plot_energy(result.inference_data)      # NUTS energy diagnostics

**Stage 4: Extract Results**

.. code-block:: python

   # Credible intervals (95% by default)
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   for param, (low, high) in intervals.items():
       print(f"{param}: [{low:.3g}, {high:.3g}]")

   # Point estimates (posterior mean)
   means = {k: v.mean() for k, v in result.posterior_samples.items()}

Recommended Settings
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Bayesian inference settings by use case
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Use Case
     - num_warmup
     - num_samples
     - num_chains
     - Notes
   * - Quick exploration
     - 500
     - 1000
     - 1
     - Fast but diagnostics unreliable
   * - Standard analysis
     - 1000
     - 2000
     - 4
     - **Default**, good balance
   * - Publication quality
     - 2000
     - 4000
     - 4
     - For uncertainty reporting
   * - Difficult posteriors
     - 4000
     - 8000
     - 8
     - Multi-modal or correlated

Diagnostic Targets
~~~~~~~~~~~~~~~~~~

- **R-hat < 1.01**: Chains have converged (< 1.05 acceptable)
- **ESS > 400**: Sufficient effective samples (per parameter)
- **BFMI > 0.3**: No energy divergence issues
- **Divergences = 0**: No numerical issues during sampling

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Issue
     - Symptom
     - Solution
   * - High R-hat (> 1.1)
     - Chains haven't converged
     - Increase num_warmup, check priors
   * - Low ESS
     - Autocorrelated chains
     - Increase num_samples, check correlations
   * - Divergences
     - Numerical instability
     - Check data scaling, tighten priors
   * - Slow sampling
     - > 10s per sample
     - Use NLSQ warm-start, simplify model

BayesianPipeline (Fluent API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For streamlined workflows, use the ``BayesianPipeline``:

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   (BayesianPipeline()
       .load('data.csv', x_col='omega', y_col='G_star')
       .fit_nlsq('{model}')              # Warm-start
       .fit_bayesian(num_samples=2000)   # NUTS sampling
       .plot_trace()                      # Diagnostics
       .plot_pair(divergences=True)
       .plot_forest(hdi_prob=0.95)
       .save('results.hdf5'))
