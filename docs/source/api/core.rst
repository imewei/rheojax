Core Module (rheojax.core)
=======================

The core module provides fundamental data structures and abstractions for rheological analysis.

Data Container
--------------

.. automodule:: rheojax.core.data
   :members:
   :undoc-members:
   :show-inheritance:

RheoData
~~~~~~~~

.. autoclass:: rheojax.core.data.RheoData
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __getitem__, __add__, __sub__, __mul__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~RheoData.from_piblin
      ~RheoData.to_piblin
      ~RheoData.to_jax
      ~RheoData.to_numpy
      ~RheoData.copy
      ~RheoData.update_metadata
      ~RheoData.to_dict
      ~RheoData.from_dict
      ~RheoData.interpolate
      ~RheoData.resample
      ~RheoData.smooth
      ~RheoData.derivative
      ~RheoData.integral
      ~RheoData.to_frequency_domain
      ~RheoData.to_time_domain
      ~RheoData.slice

Base Classes
------------

.. automodule:: rheojax.core.base
   :members:
   :undoc-members:
   :show-inheritance:

BaseModel
~~~~~~~~~

.. autoclass:: rheojax.core.base.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __repr__

   Abstract base class for all rheological models. Provides a consistent interface
   with support for scikit-learn style API and JAX arrays.

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~BaseModel.fit
      ~BaseModel.predict
      ~BaseModel.fit_predict
      ~BaseModel.score
      ~BaseModel.get_params
      ~BaseModel.set_params
      ~BaseModel.to_dict
      ~BaseModel.from_dict

BaseTransform
~~~~~~~~~~~~~

.. autoclass:: rheojax.core.base.BaseTransform
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __add__, __repr__

   Abstract base class for data transforms. Supports fit, transform, and
   inverse_transform operations with pipeline composition.

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~BaseTransform.transform
      ~BaseTransform.inverse_transform
      ~BaseTransform.fit
      ~BaseTransform.fit_transform

TransformPipeline
~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.base.TransformPipeline
   :members:
   :undoc-members:
   :show-inheritance:

   Pipeline for composing multiple transforms that are applied sequentially.

Parameters
----------

.. automodule:: rheojax.core.parameters
   :members:
   :undoc-members:
   :show-inheritance:

Parameter
~~~~~~~~~

.. autoclass:: rheojax.core.parameters.Parameter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Single parameter with value, bounds, units, and constraints.

   **Attributes:**

   - **name** (*str*) -- Parameter name
   - **value** (*float | None*) -- Current value
   - **bounds** (*tuple[float, float] | None*) -- (min, max) bounds
   - **units** (*str | None*) -- Physical units
   - **description** (*str | None*) -- Parameter description
   - **constraints** (*list[ParameterConstraint]*) -- List of constraints

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~Parameter.validate
      ~Parameter.to_dict
      ~Parameter.from_dict

ParameterSet
~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterSet
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __len__, __contains__, __iter__

   Collection of parameters for a model or transform.

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~ParameterSet.add
      ~ParameterSet.get
      ~ParameterSet.set_value
      ~ParameterSet.get_value
      ~ParameterSet.get_values
      ~ParameterSet.set_values
      ~ParameterSet.get_bounds
      ~ParameterSet.to_dict
      ~ParameterSet.from_dict

ParameterConstraint
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterConstraint
   :members:
   :undoc-members:
   :show-inheritance:

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

   Manages parameters shared across multiple models.

ParameterOptimizer
~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.core.parameters.ParameterOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

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

.. autofunction:: rheojax.core.test_modes.validate_test_mode

.. autofunction:: rheojax.core.test_modes.is_monotonic_increasing

.. autofunction:: rheojax.core.test_modes.is_monotonic_decreasing

.. autofunction:: rheojax.core.test_modes.get_compatible_test_modes

.. autofunction:: rheojax.core.test_modes.suggest_models_for_test_mode

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
