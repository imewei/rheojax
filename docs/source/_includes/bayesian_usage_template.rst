Bayesian Inference
~~~~~~~~~~~~~~~~~~

RheoJAX supports full Bayesian parameter estimation using NumPyro's NUTS sampler.
The recommended workflow uses NLSQ warm-start for faster convergence:

.. code-block:: python

   # Stage 1: NLSQ point estimation (fast warm-start)
   model.fit(x, y, test_mode='{test_mode}')

   # Stage 2: Bayesian inference with NUTS
   result = model.fit_bayesian(
       x, y,
       test_mode='{test_mode}',
       num_warmup=1000,      # MCMC warmup iterations
       num_samples=2000,     # Posterior samples
       num_chains=4,         # Parallel chains (default since v0.6.0)
       seed=42               # For reproducibility
   )

   # Stage 3: Diagnostics with ArviZ
   import arviz as az

   # Check convergence (R-hat < 1.01, ESS > 400)
   print(f"R-hat: {az.rhat(result.inference_data)}")
   print(f"ESS: {az.ess(result.inference_data)}")

   # Visualize diagnostics
   az.plot_trace(result.inference_data)
   az.plot_pair(result.inference_data, divergences=True)
   az.plot_forest(result.inference_data, hdi_prob=0.95)

   # Stage 4: Extract credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   for param, (low, high) in intervals.items():
       print(f"{param}: [{low:.3g}, {high:.3g}]")

For detailed Bayesian workflow guidance, see :doc:`/user_guide/bayesian_inference`.
