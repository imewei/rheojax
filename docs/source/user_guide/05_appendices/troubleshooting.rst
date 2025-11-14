.. _troubleshooting:

Troubleshooting Guide
=====================

This guide provides solutions to 15+ common problems encountered when fitting rheological models.

Fitting Problems
----------------

Problem: Optimization Does Not Converge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: "Optimization did not converge" error, max iterations reached

**Solutions**:

1. Increase max iterations: ``model.fit(x, y, max_iter=5000)``
2. Relax tolerances: ``model.fit(x, y, ftol=1e-6, xtol=1e-6)``
3. Use smart initialization (fractional models, oscillation mode)
4. Try simpler model (Maxwell before FractionalMaxwellLiquid)
5. Check data quality (remove outliers, smooth noise)

Problem: Unphysical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Negative moduli, tau < 10⁻⁹ s, alpha > 1

**Solutions**:

1. Tighten parameter bounds: ``model.parameters.set_bounds('alpha', (0.1, 0.95))``
2. Enable compatibility checking: ``model.fit(x, y, check_compatibility=True)``
3. Verify correct test_mode: 'oscillation' not 'relaxation' for SAOS
4. Try different model family

Problem: Poor Fit Quality (Error > 10%)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Systematic deviations, curvature mismatch

**Solutions**:

1. Try more complex model (Classical → Fractional)
2. Check if model is appropriate using compatibility checking
3. Verify data preprocessing (units, scaling)
4. Consider multi-mode model

Data Problems
-------------

Problem: File Not Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions**:

1. Use auto_read(): ``data = auto_read('file.txt')``
2. Check file encoding (UTF-8 vs. Latin-1)
3. Try specific reader: ``read_trios()``, ``read_csv()``
4. Verify file path (absolute vs. relative)

Problem: Test Mode Not Detected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions**:

1. Manually set: ``data.metadata['test_mode'] = 'relaxation'``
2. Check data characteristics (monotonic, domain)
3. Verify column names match expected patterns

Model Selection Problems
------------------------

Problem: Don't Know Which Model to Use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions**:

1. Use decision flowchart: :doc:`../02_model_usage/model_selection`
2. Enable compatibility checking for suggestions
3. Start with classical (Maxwell, Zener), upgrade if needed
4. Check material classification: :doc:`../01_fundamentals/material_classification`

Problem: Model Compatibility Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example**: "FZSS expects power-law decay but detected exponential"

**Solutions**:

1. Follow suggested alternatives in warning message
2. Use compatibility checking before fitting
3. Verify test mode is correct

Bayesian Inference Problems
----------------------------

Problem: Low ESS (Effective Sample Size < 400)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions**:

1. Increase num_samples: ``fit_bayesian(num_samples=5000)``
2. Use NLSQ warm-start (done automatically)
3. Check for multimodal posterior
4. Increase num_warmup

Problem: High R-hat (> 1.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions**:

1. Increase num_warmup: ``fit_bayesian(num_warmup=2000)``
2. Use warm-start from NLSQ fit
3. Run multiple chains: ``num_chains=4``

Quick Reference Flowchart
-------------------------

.. code-block:: text

   [Problem Category]
      │
      ├─→ Convergence failure
      │      └─→ Increase max_iter, relax tol, smart init
      │
      ├─→ Unphysical parameters
      │      └─→ Tighten bounds, compatibility check
      │
      ├─→ Poor fit (error > 10%)
      │      └─→ Try complex model, check compatibility
      │
      ├─→ File loading issues
      │      └─→ Use auto_read(), check encoding
      │
      └─→ Model selection
             └─→ Use flowchart, compatibility checking

Getting More Help
-----------------

1. Check :doc:`../02_model_usage/fitting_strategies` for advanced techniques
2. Review example notebooks in ``examples/``
3. Consult model-specific pages in :doc:`/models/index`
4. Open GitHub issue with minimal reproducible example
