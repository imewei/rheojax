Core Module (rheojax.core)
===========================

The core module provides fundamental data structures and abstractions for rheological analysis.

Data Container
--------------

RheoData
~~~~~~~~

.. autoclass:: rheojax.core.data.RheoData
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __getitem__, __add__, __sub__, __mul__
   :exclude-members: x, y, x_units, y_units, domain, metadata, validate

Base Classes
------------

BaseModel
~~~~~~~~~

.. autoclass:: rheojax.core.base.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
   :special-members: __init__, __repr__

   Abstract base class for all rheological models. Provides a consistent interface
   with support for scikit-learn style API and JAX arrays.

BaseTransform
~~~~~~~~~~~~~

.. autoclass:: rheojax.core.base.BaseTransform
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
   :special-members: __init__, __add__, __repr__

   Abstract base class for data transforms. Supports fit, transform, and
   inverse_transform operations with pipeline composition.

TransformPipeline
~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.base.TransformPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Pipeline for composing multiple transforms that are applied sequentially.

Parameters
----------

Parameter
~~~~~~~~~

.. autoclass:: rheojax.core.parameters.Parameter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
   :special-members: __init__

   Single parameter with value, bounds, units, and constraints.

   **Attributes:**

   - **name** (*str*) -- Parameter name
   - **value** (*float | None*) -- Current value
   - **bounds** (*tuple[float, float] | None*) -- (min, max) bounds
   - **units** (*str | None*) -- Physical units
   - **description** (*str | None*) -- Parameter description
   - **constraints** (*list[ParameterConstraint]*) -- List of constraints

ParameterSet
~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterSet
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
   :special-members: __init__, __len__, __contains__, __iter__

   Collection of parameters for a model or transform.

ParameterConstraint
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterConstraint
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Constraint on a parameter value.

   **Types:**

   - ``"bounds"``: Min/max value bounds
   - ``"positive"``: Must be > 0
   - ``"integer"``: Must be an integer
   - ``"fixed"``: Fixed to specific value
   - ``"relative"``: Relative to another parameter
   - ``"custom"``: Custom validator function

SharedParameterSet
~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.SharedParameterSet
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Manages parameters shared across multiple models.

ParameterOptimizer
~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Optimizer for parameter fitting with JAX gradient support.

Test Modes
----------

.. automodule:: rheojax.core.test_modes
   :members:
   :undoc-members:
   :show-inheritance:

TestMode
~~~~~~~~

.. autoclass:: rheojax.core.test_modes.TestMode
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Enumeration of rheological test modes.

   **Values:**

   - ``RELAXATION``: Stress relaxation test
   - ``CREEP``: Creep compliance test
   - ``OSCILLATION``: Oscillatory (SAOS/LAOS) test
   - ``ROTATION``: Steady shear (flow curve) test
   - ``UNKNOWN``: Unknown or ambiguous test type

Functions
~~~~~~~~~

.. autofunction:: rheojax.core.test_modes.detect_test_mode
   :noindex:

.. autofunction:: rheojax.core.test_modes.validate_test_mode
   :noindex:

.. autofunction:: rheojax.core.test_modes.is_monotonic_increasing
   :noindex:

.. autofunction:: rheojax.core.test_modes.is_monotonic_decreasing
   :noindex:

.. autofunction:: rheojax.core.test_modes.get_compatible_test_modes
   :noindex:

.. autofunction:: rheojax.core.test_modes.suggest_models_for_test_mode
   :noindex:

Bayesian Inference
------------------

The Bayesian inference module provides NumPyro NUTS sampling capabilities with NLSQ warm-start
for all rheological models through the BayesianMixin class.

BayesianMixin
~~~~~~~~~~~~~

.. autoclass:: rheojax.core.bayesian.BayesianMixin
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
   :special-members: __init__

   Mixin class that adds Bayesian inference capabilities to models. Provides:

   - NLSQ → NUTS warm-start workflow (2-5x faster convergence)
   - Automatic prior specification from parameter bounds
   - Credible interval calculation
   - Model function for NumPyro NUTS sampling

BayesianResult
~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.bayesian.BayesianResult
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Dataclass storing complete Bayesian inference results:

   **Attributes:**

   - ``posterior_samples``: Dict mapping parameter names to posterior samples (float64 arrays)
   - ``summary``: Dict with summary statistics (mean, std, quantiles) for each parameter
   - ``diagnostics``: Convergence diagnostics including R-hat, ESS, divergences
   - ``waic``: WAIC model comparison metric (if computed)
   - ``loo``: LOO cross-validation metric (if computed)
   - ``inference_data``: ArviZ InferenceData object for advanced diagnostics

JAX Configuration
-----------------

.. automodule:: rheojax.core.jax_config
   :members:
   :undoc-members:
   :show-inheritance:

The JAX configuration module ensures float64 precision throughout the JAX stack by enforcing
proper import order (NLSQ must be imported before JAX).

.. autofunction:: rheojax.core.jax_config.safe_import_jax
   :noindex:

   Safe JAX import that verifies NLSQ was imported first for float64 precision.

   **Usage:**

   .. code-block:: python

      # CORRECT - Always use in RheoJAX modules
      from rheojax.core.jax_config import safe_import_jax
      jax, jnp = safe_import_jax()

      # INCORRECT - Never import JAX directly
      import jax  # Will raise ImportError if NLSQ not imported first

.. autofunction:: rheojax.core.jax_config.verify_float64
   :noindex:

   Verify JAX is operating in float64 mode. Raises exception if not.

Registry
--------

.. automodule:: rheojax.core.registry
   :members:
   :undoc-members:
   :show-inheritance:

The registry system provides a centralized way to discover and instantiate models and transforms.
This will be fully implemented in Phase 2.

.. note::
   Full registry functionality (model and transform registration) will be available in Phase 2.

Examples
--------

Creating RheoData
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from rheojax.core import RheoData

    # Simple time-domain data
    time = np.array([0.1, 1.0, 10.0])
    stress = np.array([1000, 800, 600])
    data = RheoData(
        x=time,
        y=stress,
        x_units="s",
        y_units="Pa",
        domain="time"
    )

    # Complex frequency-domain data
    omega = np.logspace(-2, 2, 50)
    Gp = 1000 * omega**0.5
    Gpp = 500 * omega**0.3
    G_star = Gp + 1j * Gpp

    freq_data = RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency"
    )

Working with Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.core import Parameter, ParameterSet

    # Create parameter set
    params = ParameterSet()
    params.add(
        name="E",
        value=1000.0,
        bounds=(100, 10000),
        units="Pa",
        description="Elastic modulus"
    )
    params.add(
        name="tau",
        value=1.0,
        bounds=(0.01, 100),
        units="s",
        description="Relaxation time"
    )

    # Get/set values
    E_value = params.get_value("E")
    params.set_value("tau", 2.5)

    # Array interface
    values = params.get_values()  # [1000.0, 2.5]
    params.set_values([2000, 1.5])

Test Mode Detection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.core.test_modes import detect_test_mode, TestMode

    # Automatic detection
    mode = detect_test_mode(data)
    print(mode)  # TestMode.RELAXATION

    # Check test mode
    if data.test_mode == TestMode.RELAXATION:
        print("This is a stress relaxation test")

Using Base Classes (Phase 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.core import BaseModel, ParameterSet
    import jax.numpy as jnp

    class MaxwellModel(BaseModel):
        def __init__(self, E=1000.0, tau=1.0):
            super().__init__()
            self.parameters.add("E", value=E, bounds=(1, 1e6), units="Pa")
            self.parameters.add("tau", value=tau, bounds=(0.01, 1000), units="s")

        def _fit(self, X, y, **kwargs):
            # Fitting implementation
            return self

        def _predict(self, X):
            E = self.parameters.get_value("E")
            tau = self.parameters.get_value("tau")
            return E * jnp.exp(-X / tau)

    # Use model
    model = MaxwellModel()
    model.fit(time, stress)
    predictions = model.predict(time)

Bayesian Inference (Phase 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.maxwell import Maxwell
    import numpy as np

    # Generate data
    t = np.linspace(0.1, 10, 50)
    G_data = 1e5 * np.exp(-t / 0.01) + np.random.normal(0, 1e3, size=t.shape)

    # 1. NLSQ optimization (fast point estimate)
    model = Maxwell()
    model.fit(t, G_data)
    print(f"NLSQ: G0={model.parameters.get_value('G0'):.3e}")

    # 2. Bayesian inference with warm-start
    result = model.fit_bayesian(
        t, G_data,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1
    )

    # 3. Analyze results
    print(f"Posterior mean: G0={result.summary['G0']['mean']:.3e} ± {result.summary['G0']['std']:.3e}")
    print(f"Convergence: R-hat={result.diagnostics['r_hat']['G0']:.4f}")

    # 4. Get credible intervals
    intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
    print(f"G0 95% CI: [{intervals['G0'][0]:.3e}, {intervals['G0'][1]:.3e}]")
