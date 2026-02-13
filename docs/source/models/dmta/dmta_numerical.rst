Numerical Implementation
========================

This page covers computational strategies for DMTA data fitting in RheoJAX,
including JAX/JIT considerations, parameter bounds, convergence criteria,
and memory management.

JIT Compilation Strategy
-------------------------

All RheoJAX models use JAX's ``@jit`` decorator for GPU-accelerated fitting.
DMTA-specific considerations:

- **First fit is slow** (~5--30 s): JIT compiles the model function, residual
  computation, and Jacobian.  Subsequent fits with the same model and data
  shape reuse the compiled code (< 1 s).

- **Shape changes trigger recompilation**: Fitting 206-point and 481-point
  datasets back-to-back compiles twice.  Group datasets by size when possible.

- **Float64 is mandatory**: Use ``safe_import_jax()`` (not ``import jax``)
  to ensure 64-bit precision.  32-bit arithmetic causes convergence failures
  with the wide dynamic range of DMTA data (0.1--10 000 MPa).

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

.. _dmta-bounds-handling:

Parameter Bounds for Tensile Data
----------------------------------

DMTA data is in tensile modulus space (Pa), which is typically 2--3 :math:`\times`
larger than shear modulus space.  Real polymer DMTA data spans:

.. list-table::
   :header-rows: 1
   :widths: 35 25 20 20

   * - Material State
     - :math:`E'` Range
     - :math:`G'` Range
     - Scale Factor
   * - Glassy plateau
     - 1--10 GPa
     - 0.4--3.7 GPa
     - :math:`2(1+\nu) \approx 2.7`
   * - Rubbery plateau
     - 0.1--10 MPa
     - 0.03--3.3 MPa
     - :math:`2(1+\nu) = 3.0`
   * - Glass transition
     - 1 MPa -- 5 GPa
     - 0.3 MPa -- 1.9 GPa
     - varies with :math:`\nu(\omega)`

**Default bounds handling:**

- ``GeneralizedMaxwell(modulus_type='tensile')`` automatically uses wider
  bounds: :math:`E_i \in [10^{-3}, 10^{12}]` Pa (vs :math:`G_i \in [10^{-3}, 10^{9}]`
  for shear).

- Other models (Zener, FZSS, etc.) with ``deformation_mode='tension'``
  convert E* |rarr| G* at the ``fit()`` boundary, so their internal bounds
  (in G-space) are sufficient.

.. |rarr| unicode:: U+2192

.. _dmta-bounds-widening:

**When bounds errors occur:**

If a model raises ``ValueError: Value ... violates constraints``, the fitted
value exceeds the parameter bounds.  Fix by widening:

.. code-block:: python

   # Widen bounds for a specific parameter
   param = model.parameters["G0"]
   param.bounds = (param.bounds[0], 1e12)
   for c in param.constraints:
       if c.type == "bounds":
           c.min_value, c.max_value = param.bounds

.. _dmta-element-minimisation:

Element Minimisation and Mode Reduction
-----------------------------------------

The ``GeneralizedMaxwell`` model supports automatic mode reduction via
``optimization_factor``.  This creates internal sub-models with default
bounds for each candidate mode count.

.. warning::

   When fitting real DMTA data with ``modulus_type='tensile'``, set
   ``optimization_factor=None`` to avoid element minimisation (which
   uses default bounds internally):

   .. code-block:: python

      gmm = GeneralizedMaxwell(n_modes=10, modulus_type='tensile')
      gmm.fit(omega, E_star, test_mode='oscillation',
              optimization_factor=None)

   Alternatively, use ``modulus_type='shear'`` with
   ``deformation_mode='tension'`` to fit in G-space where default bounds
   and element minimisation work correctly.

Convergence Criteria
---------------------

NLSQ convergence for DMTA data:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - DMTA Recommendation
   * - ``max_iter``
     - 200
     - 500--1000 (broad master curves need more iterations)
   * - ``ftol``
     - :math:`10^{-8}`
     - :math:`10^{-8}` (sufficient)
   * - ``xtol``
     - :math:`10^{-8}`
     - :math:`10^{-8}` (sufficient)
   * - ``n_modes`` (GMM)
     - 3
     - 10--30 (match decades of data)

**Rule of thumb**: Use approximately 1 Prony mode per 3 decades of
frequency data.  For a master curve spanning 20 decades, ``n_modes=7`` is
a minimum; ``n_modes=15--20`` gives excellent fits.

Bayesian Inference (NUTS) Settings
------------------------------------

NUTS sampling for DMTA data follows the standard NLSQ |rarr| NUTS pipeline
(see :doc:`dmta_workflows` Workflow 3 for a complete example):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Setting
     - FAST_MODE
     - Production
     - Notes
   * - ``num_warmup``
     - 50
     - 200--1000
     - More warmup for multi-modal posteriors
   * - ``num_samples``
     - 100
     - 500--2000
     - Check ESS > 400
   * - ``num_chains``
     - 1
     - 4
     - Multi-chain for R-hat diagnostics
   * - ``target_accept_prob``
     - 0.8
     - 0.8--0.95
     - Increase if divergences > 0

Memory Management
------------------

Sequential DMTA model fits can exhaust memory (especially on 16 GB machines).
Follow this pattern between fits:

.. code-block:: python

   import gc
   import jax

   # Fit model 1
   model1.fit(omega, E_star, test_mode='oscillation',
              deformation_mode='tension')
   E_pred1 = model1.predict(omega, test_mode='oscillation')

   # Clean up before next fit
   del model1
   gc.collect()
   jax.clear_caches()

   # Fit model 2
   model2.fit(omega, E_star, ...)

For notebooks, also use ``plt.close('all')`` instead of ``plt.show()`` to
prevent figure accumulation in headless (CI) environments.

FAST_MODE Guidelines
---------------------

All DMTA example notebooks support ``FAST_MODE`` (default ``True`` in CI):

.. code-block:: python

   import os
   FAST_MODE = os.environ.get('FAST_MODE', '1') == '1'

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - FAST_MODE
     - Full Mode
   * - GMM modes
     - ``n_modes=5``
     - ``n_modes=10--15``
   * - NUTS samples
     - 50 warmup + 100 samples
     - 200--1000 warmup + 500--2000 samples
   * - FZSS/extra models
     - Skip
     - Include
   * - Cross-domain validation
     - Skip or reduced
     - Full (requires :math:`n \geq 15`)
   * - Data subsampling
     - 200 points max
     - Full dataset

Set ``FAST_MODE=0`` for publication-quality results.

.. seealso::

   - :doc:`dmta_workflows` --- complete examples using the settings above
   - :doc:`dmta_models` --- model selection guide (complexity vs. expressiveness)
   - :doc:`dmta_theory` --- E* |leftrightarrow| G* conversion and bounds rationale

.. |leftrightarrow| unicode:: U+2194
