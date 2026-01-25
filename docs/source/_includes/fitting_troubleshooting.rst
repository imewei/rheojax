.. _fitting-troubleshooting:

Common Fitting Issues
^^^^^^^^^^^^^^^^^^^^^

This section covers typical problems encountered when fitting rheological models
and their solutions.

General Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Universal fitting diagnostics
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - Optimizer doesn't converge
     - ``max_iter`` reached without tolerance
     - Increase ``max_iter`` to 5000+, check initialization
   * - Converged but poor fit
     - Low R² despite convergence
     - Wrong model choice, check residual pattern
   * - Parameter hits bounds
     - Fitted value at upper/lower limit
     - Widen bounds or reconsider model
   * - Numerical overflow/NaN
     - ``nan`` in predictions or gradients
     - Check data for zeros/infinities, scale inputs
   * - Very slow fitting
     - Minutes per optimization
     - Enable JIT compilation, check data size

Data Quality Issues
~~~~~~~~~~~~~~~~~~~

.. list-table:: Data-related problems
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - Noisy low-frequency data
     - Large scatter at low :math:`\omega`
     - Use log-weighted residuals, remove outliers
   * - Insufficient frequency range
     - Can't resolve plateaus
     - Extend frequency sweep, use TTS
   * - Phase angle artifacts
     - :math:`\tan\delta > 1` when expecting solid
     - Check inertia correction, reduce :math:`\omega`
   * - Non-monotonic viscosity
     - :math:`\eta` increases then decreases
     - Wall slip, shear banding - check with different gaps
   * - Temperature drift
     - Irreproducible :math:`G'`, :math:`G''`
     - Improve thermal equilibration, reduce sweep time

Model Selection Issues
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Choosing the right model
   :header-rows: 1
   :widths: 30 35 35

   * - Symptom
     - Likely Cause
     - Recommended Action
   * - Systematic residual curvature
     - Model too simple
     - Add more relaxation times (multi-mode)
   * - Fitted :math:`\tau` outside data range
     - Relaxation not captured
     - Extend frequency range or use fractional model
   * - Can't fit low and high :math:`\omega` simultaneously
     - Multiple mechanisms
     - Generalized Maxwell or bi-modal model
   * - Power-law exponent > 1 or < 0
     - Unphysical result
     - Check units, constrain exponent bounds
   * - Modulus trends in wrong direction
     - Test mode mismatch
     - Verify ``test_mode`` parameter matches data

Bayesian-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: MCMC sampling problems
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - R-hat > 1.1
     - Chains haven't mixed
     - Increase warmup, improve initialization
   * - ESS < 100
     - High autocorrelation
     - Increase samples, check parameter correlations
   * - Many divergences
     - Posterior geometry issues
     - Rescale data, adjust step size
   * - Posterior hits prior bounds
     - Prior too narrow
     - Use wider, less informative priors
   * - Bimodal posterior
     - Multiple solutions exist
     - Physics indicates which mode is correct

Parameter Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For exponential relaxation models (Maxwell, Zener)**:

1. Locate :math:`G''(\omega)` maximum → :math:`\tau \approx 1/\omega_{\max}`
2. High-frequency :math:`G'` plateau → :math:`G`
3. Compute :math:`\eta = G\tau`

**For power-law models (Power Law, Carreau, Cross)**:

1. Log-log slope in transition region → :math:`n - 1`
2. Low-rate plateau → :math:`\eta_0`
3. High-rate plateau → :math:`\eta_\infty`

**For fractional models (FMG, FML, SpringPot)**:

1. :math:`\tan\delta` slope vs :math:`\log\omega` → :math:`\alpha`
2. Use auto-initialization (RheoJAX v0.2.0+)

**For yield stress models (Herschel-Bulkley, Bingham)**:

1. Intercept of :math:`\sigma` vs :math:`\dot{\gamma}` at :math:`\dot{\gamma} \to 0` → :math:`\sigma_y`
2. High-rate slope → :math:`K` (consistency)
3. Log-log slope at high rates → :math:`n`

Validation Checklist
~~~~~~~~~~~~~~~~~~~~

After fitting, verify:

- [ ] R² > 0.95 (oscillatory) or R² > 0.99 (relaxation/flow)
- [ ] Residuals randomly distributed (no trends)
- [ ] Parameters physically reasonable (check against literature)
- [ ] Derived quantities consistent (e.g., :math:`\tau = \eta/G`)
- [ ] Model predicts other test modes correctly (cross-validation)
- [ ] Uncertainties don't span orders of magnitude (if Bayesian)
