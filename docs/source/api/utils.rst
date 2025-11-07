Utilities (rheojax.utils)
======================

The utils module provides numerical utilities for rheological analysis, including special functions and optimization tools.

Mittag-Leffler Functions
-------------------------

.. automodule:: rheojax.utils.mittag_leffler
   :members:
   :undoc-members:
   :show-inheritance:

The Mittag-Leffler function is essential for fractional calculus in rheology.
This module provides JAX-compatible implementations with high accuracy.

Functions
~~~~~~~~~

.. autofunction:: rheojax.utils.mittag_leffler.mittag_leffler_e

.. autofunction:: rheojax.utils.mittag_leffler.mittag_leffler_e2

Aliases
~~~~~~~

.. data:: rheojax.utils.mittag_leffler.ml_e

   Alias for :func:`mittag_leffler_e`

.. data:: rheojax.utils.mittag_leffler.ml_e2

   Alias for :func:`mittag_leffler_e2`

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

One-Parameter Function
^^^^^^^^^^^^^^^^^^^^^^

The one-parameter Mittag-Leffler function is defined as:

.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

where :math:`\Gamma` is the gamma function and :math:`0 < \alpha \leq 2`.

**Special cases:**

- :math:`\alpha = 1`: :math:`E_1(z) = e^z` (exponential function)
- :math:`\alpha = 2`: :math:`E_2(z^2) = \cosh(z)` (hyperbolic cosine)

Two-Parameter Function
^^^^^^^^^^^^^^^^^^^^^^^

The two-parameter generalization:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

**Special cases:**

- :math:`\beta = 1`: :math:`E_{\alpha,1}(z) = E_\alpha(z)` (one-parameter form)
- :math:`\alpha = \beta = 1`: :math:`E_{1,1}(z) = e^z` (exponential)

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The implementation uses Padé approximations for optimal performance:

- **Method**: Padé(6,3) approximation
- **Accuracy**: < 1e-6 relative error for :math:`|z| < 10`
- **Performance**: JIT-compiled with JAX for speed
- **Range**: Optimized for rheological applications (:math:`|z| < 10`)

Examples
~~~~~~~~

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    import jax.numpy as jnp
    from rheojax.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2

    # Single value
    result = mittag_leffler_e(0.5, alpha=0.5)
    print(result)  # ~1.6487...

    # Array of values
    z = jnp.linspace(0, 2, 10)
    results = mittag_leffler_e(z, alpha=0.8)

    # Two-parameter form
    result2 = mittag_leffler_e2(0.5, alpha=0.5, beta=1.0)

Fractional Relaxation Modulus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import jax.numpy as jnp
    from rheojax.utils.mittag_leffler import mittag_leffler_e

    def fractional_maxwell_relaxation(t, E, tau, alpha):
        """Relaxation modulus for fractional Maxwell model.

        Parameters
        ----------
        t : array
            Time values
        E : float
            Elastic modulus (Pa)
        tau : float
            Relaxation time (s)
        alpha : float
            Fractional order (0 < alpha < 1)

        Returns
        -------
        G : array
            Relaxation modulus G(t)
        """
        return E * mittag_leffler_e(-(t / tau)**alpha, alpha)

    # Compute relaxation modulus
    time = jnp.logspace(-2, 2, 100)
    G = fractional_maxwell_relaxation(time, E=1000, tau=1.0, alpha=0.5)

JIT Compilation
^^^^^^^^^^^^^^^

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from rheojax.utils.mittag_leffler import mittag_leffler_e

    # JIT compile function (alpha must be static)
    @jax.jit
    def compute_ml(z):
        return mittag_leffler_e(z, alpha=0.5)

    # Use compiled function
    z = jnp.linspace(0, 5, 1000)
    result = compute_ml(z)  # Fast computation

Optimization
------------

.. automodule:: rheojax.utils.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Functions and Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.utils.optimization.OptimizationResult
   :members:
   :undoc-members:
   :show-inheritance:

   Result container for optimization.

.. autofunction:: rheojax.utils.optimization.nlsq_optimize

.. autofunction:: rheojax.utils.optimization.optimize_with_bounds

.. autofunction:: rheojax.utils.optimization.residual_sum_of_squares

.. autofunction:: rheojax.utils.optimization.create_least_squares_objective

Aliases
~~~~~~~

.. data:: rheojax.utils.optimization.optimize

   Alias for :func:`nlsq_optimize`

.. data:: rheojax.utils.optimization.fit_parameters

   Alias for :func:`nlsq_optimize`

Optimization Methods
~~~~~~~~~~~~~~~~~~~~

The following scipy.optimize methods are supported:

- **"L-BFGS-B"**: L-BFGS algorithm with bounds (default for bounded problems)
- **"TNC"**: Truncated Newton with bounds
- **"SLSQP"**: Sequential Least Squares Programming
- **"trust-constr"**: Trust-region constrained optimization
- **"BFGS"**: Broyden-Fletcher-Goldfarb-Shanno (default for unbounded)

JAX Gradient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

When ``use_jax=True``, gradients are computed using JAX automatic differentiation:

.. math::

   \nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]

This provides exact gradients (up to floating-point precision) compared to numerical
finite-difference approximations, leading to faster and more robust optimization.

Examples
~~~~~~~~

Basic Optimization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core.parameters import ParameterSet
    from rheojax.utils.optimization import nlsq_optimize
    import jax.numpy as jnp

    # Define objective function
    def objective(params):
        x, y = params
        return (x - 5.0)**2 + (y - 3.0)**2

    # Set up parameters
    params = ParameterSet()
    params.add("x", value=0.0, bounds=(-10, 10))
    params.add("y", value=0.0, bounds=(-10, 10))

    # Optimize
    result = nlsq_optimize(objective, params, use_jax=True)
    print(f"Optimal: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
    print(f"Function value: {result.fun:.6f}")
    print(f"Success: {result.success}")

Model Fitting
^^^^^^^^^^^^^

.. code-block:: python

    import jax.numpy as jnp
    from rheojax.core.parameters import ParameterSet
    from rheojax.utils.optimization import nlsq_optimize

    # Experimental data
    t_exp = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    stress_exp = jnp.array([1000, 800, 650, 500, 320, 200])

    # Maxwell model
    def maxwell_model(t, params):
        E, tau = params
        return E * jnp.exp(-t / tau)

    # Objective: minimize residuals
    def objective(params):
        predictions = maxwell_model(t_exp, params)
        residuals = predictions - stress_exp
        return jnp.sum(residuals**2)

    # Set up parameters
    params = ParameterSet()
    params.add("E", value=1000, bounds=(100, 5000))
    params.add("tau", value=1.0, bounds=(0.1, 100))

    # Fit model
    result = nlsq_optimize(
        objective,
        params,
        use_jax=True,
        method="L-BFGS-B"
    )

    # Extract fitted parameters
    E_fit = params.get_value("E")
    tau_fit = params.get_value("tau")
    print(f"Fitted: E={E_fit:.1f} Pa, tau={tau_fit:.2f} s")

Custom Objective with Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.utils.optimization import create_least_squares_objective

    # Define model function
    def power_law(shear_rate, params):
        K, n = params
        return K * shear_rate**n

    # Data
    shear_rate = jnp.logspace(-2, 2, 50)
    viscosity = 100 * shear_rate**(-0.7)

    # Create objective
    objective = create_least_squares_objective(
        power_law,
        shear_rate,
        viscosity,
        normalize=True  # Use relative error
    )

    # Set up parameters
    params = ParameterSet()
    params.add("K", value=100, bounds=(1, 1000))
    params.add("n", value=-0.5, bounds=(-2, 0))

    # Optimize
    result = nlsq_optimize(objective, params, use_jax=True)

Optimization with Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core.parameters import ParameterConstraint

    # Add relative constraint: tau1 < tau2
    params = ParameterSet()
    params.add("tau1", value=1.0, bounds=(0.1, 100))

    tau2_constraint = ParameterConstraint(
        type="relative",
        relation="greater_than",
        other_param="tau1"
    )
    params.add(
        "tau2",
        value=10.0,
        bounds=(0.1, 100),
        constraints=[tau2_constraint]
    )

Monitoring Optimization
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core.parameters import ParameterOptimizer

    # Create optimizer with tracking
    optimizer = ParameterOptimizer(
        parameters=params,
        use_jax=True,
        track_history=True
    )

    optimizer.set_objective(objective)

    # Define callback
    def callback(iteration, values, obj_value):
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: f={obj_value:.6f}")

    optimizer.set_callback(callback)

    # Run optimization (integrate with scipy)
    # result = optimizer.optimize()

    # Get history
    history = optimizer.get_history()
    for entry in history:
        print(f"Iter {entry['iteration']}: {entry['objective']}")

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use JAX gradients**: Set ``use_jax=True`` for faster optimization
2. **Choose appropriate method**: L-BFGS-B for bounds, BFGS for unbounded
3. **Scale parameters**: Normalize to similar magnitudes (0.1-10 range)
4. **Provide good initial guess**: Closer to optimum = faster convergence
5. **Set tolerances**: Adjust ``ftol``, ``xtol``, ``gtol`` for speed vs accuracy

.. code-block:: python

    # Good parameter scaling
    params.add("E", value=1.0, bounds=(0.1, 10))  # Scaled from Pa
    params.add("tau", value=1.0, bounds=(0.1, 10))  # Scaled from s

    # In objective, unscale:
    def objective(params_scaled):
        E = params_scaled[0] * 1000  # Convert back to Pa
        tau = params_scaled[1]        # Already in seconds
        # ... compute objective

See Also
--------

- :doc:`core` - Parameter system
- :doc:`../user_guide/getting_started` - Basic usage examples
- `SciPy optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ - Optimization algorithms
- `JAX autodiff <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_ - Automatic differentiation

Model-Data Compatibility
-------------------------

.. automodule:: rheojax.utils.compatibility
   :members:
   :undoc-members:
   :show-inheritance:

The compatibility module provides intelligent detection of when rheological models
are inappropriate for experimental data based on underlying physics. This helps users
understand when model failures are expected due to physics mismatch rather than
optimization issues.

Enums
~~~~~

.. autoclass:: rheojax.utils.compatibility.DecayType
   :members:
   :undoc-members:
   :show-inheritance:

   Types of relaxation decay behavior:

   - **EXPONENTIAL**: Simple Maxwell-like exp(-t/tau)
   - **POWER_LAW**: Power-law t^(-alpha) (gel-like)
   - **STRETCHED**: Stretched exponential exp(-(t/tau)^beta)
   - **MITTAG_LEFFLER**: Mittag-Leffler E_alpha(-(t/tau)^alpha) (fractional)
   - **MULTI_MODE**: Multiple relaxation modes
   - **UNKNOWN**: Cannot determine

.. autoclass:: rheojax.utils.compatibility.MaterialType
   :members:
   :undoc-members:
   :show-inheritance:

   Types of material behavior:

   - **SOLID**: Solid-like (finite equilibrium modulus)
   - **LIQUID**: Liquid-like (zero equilibrium modulus, flows)
   - **GEL**: Gel-like (power-law relaxation)
   - **VISCOELASTIC_SOLID**: Viscoelastic solid
   - **VISCOELASTIC_LIQUID**: Viscoelastic liquid
   - **UNKNOWN**: Cannot determine

Functions
~~~~~~~~~

.. autofunction:: rheojax.utils.compatibility.detect_decay_type

   Analyzes relaxation modulus data to determine the type of decay pattern.
   Uses linear regression on log-transformed data to identify exponential,
   power-law, stretched exponential, or Mittag-Leffler behavior.

.. autofunction:: rheojax.utils.compatibility.detect_material_type

   Classifies material behavior from relaxation or oscillation data.
   Detects solid-like, liquid-like, gel-like, or viscoelastic behavior based on
   equilibrium modulus or low-frequency response.

.. autofunction:: rheojax.utils.compatibility.check_model_compatibility

   Comprehensive compatibility check comparing model physics with data characteristics.
   Returns detailed compatibility information including warnings and model recommendations.

.. autofunction:: rheojax.utils.compatibility.format_compatibility_message

   Formats compatibility check results as a human-readable message with warnings,
   detected characteristics, and alternative model recommendations.

Decay Detection Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~

The decay type detection uses statistical analysis on log-transformed data:

**Exponential Decay Detection**

Linear regression on log(G) vs t:

.. math::

   \log G(t) = \log G_0 - \frac{t}{\tau}

High R² (> 0.90) indicates exponential decay (Maxwell-like behavior).

**Power-Law Decay Detection**

Linear regression on log(G) vs log(t):

.. math::

   \log G(t) = \log G_0 - \alpha \log t

High R² (> 0.90) indicates power-law decay (gel-like behavior).

**Stretched Exponential Detection**

Linear regression on log(-log(G/G₀)) vs log(t):

.. math::

   \log\left(-\log\frac{G(t)}{G_0}\right) = \beta \log t + \text{const}

High R² (> 0.90) indicates stretched exponential behavior.

Material Type Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From Relaxation Data**

Material type is determined by the decay ratio:

.. math::

   \text{decay ratio} = \frac{\text{mean}(G_{\text{final}})}{\text{mean}(G_{\text{initial}})}

- **Solid-like**: decay ratio > 0.5 (significant equilibrium modulus)
- **Liquid-like**: decay ratio < 0.1 (nearly complete relaxation)
- **Power-law materials**: Classified based on decay type regardless of ratio

**From Oscillation Data**

Material type is determined by low-frequency behavior:

- **Solid**: G' > G" at lowest frequency (elastic dominant)
- **Liquid**: G" > G' at lowest frequency (viscous dominant)

Examples
~~~~~~~~

Basic Compatibility Check
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
    from rheojax.utils.compatibility import (
        check_model_compatibility,
        format_compatibility_message
    )
    import numpy as np

    # Generate exponential decay data (Maxwell-like)
    t = np.logspace(-2, 2, 50)
    G_t = 1e5 * np.exp(-t / 1.0)

    # Check if FZSS is appropriate
    model = FractionalZenerSolidSolid()
    compatibility = check_model_compatibility(
        model, t=t, G_t=G_t, test_mode='relaxation'
    )

    # Print human-readable report
    print(format_compatibility_message(compatibility))
    # Output:
    # ⚠ Model may not be appropriate for this data
    #   Confidence: 90%
    #   Detected decay: exponential
    #   Material type: viscoelastic_liquid
    #
    # Warnings:
    #   • FZSS model expects Mittag-Leffler (power-law) relaxation,
    #     but data shows exponential decay.
    #
    # Recommended alternative models:
    #   • Maxwell
    #   • Zener

Automatic Checking During Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.models.maxwell import Maxwell
    import numpy as np

    # Enable automatic compatibility checking
    model = Maxwell()
    model.fit(
        t, G_data,
        test_mode='relaxation',
        check_compatibility=True  # Warns if model-data mismatch
    )

    # If incompatible, warning is logged and enhanced error
    # messages provide physics-based explanations

Detecting Decay Type
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.utils.compatibility import detect_decay_type, DecayType
    import numpy as np

    # Power-law decay (gel-like)
    t = np.logspace(-2, 2, 100)
    G_gel = 1e5 * t**(-0.5)

    decay_type = detect_decay_type(t, G_gel)
    print(decay_type)  # DecayType.POWER_LAW

    # Exponential decay (Maxwell-like)
    G_maxwell = 1e5 * np.exp(-t / 1.0)
    decay_type = detect_decay_type(t, G_maxwell)
    print(decay_type)  # DecayType.EXPONENTIAL

Classifying Material Type
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.utils.compatibility import detect_material_type
    import numpy as np

    # Solid-like material (finite equilibrium modulus)
    t = np.logspace(-2, 2, 50)
    G_solid = 5e4 + 5e4 * np.exp(-t / 1.0)  # Ge + Gm*exp(-t/tau)

    material_type = detect_material_type(t=t, G_t=G_solid)
    print(material_type)  # MaterialType.VISCOELASTIC_SOLID

    # Liquid-like material (no equilibrium modulus)
    G_liquid = 1e5 * np.exp(-t / 1.0)
    material_type = detect_material_type(t=t, G_t=G_liquid)
    print(material_type)  # MaterialType.VISCOELASTIC_LIQUID

Oscillation Data Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.utils.compatibility import (
        check_model_compatibility,
        detect_material_type
    )
    from rheojax.models.fractional_maxwell_liquid import FractionalMaxwellLiquid
    import numpy as np

    # Oscillation data (G', G")
    omega = np.logspace(-2, 2, 50)
    G_prime = 1e5 * np.ones(50)  # Constant storage modulus
    G_double_prime = 1e3 * omega**0.5  # Frequency-dependent loss

    G_star = np.column_stack([G_prime, G_double_prime])

    # Detect material type
    material_type = detect_material_type(omega=omega, G_star=G_star)
    print(material_type)  # MaterialType.SOLID (G' > G" at low freq)

    # Check model compatibility
    model = FractionalMaxwellLiquid()
    compatibility = check_model_compatibility(
        model,
        omega=omega,
        G_star=G_star,
        test_mode='oscillation'
    )

    if not compatibility['compatible']:
        print(f"Confidence: {compatibility['confidence']}")
        print(f"Warnings: {compatibility['warnings']}")
        print(f"Try instead: {compatibility['recommendations']}")

Enhanced Error Messages
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

    # Generate exponential data (incompatible with FZSS)
    np.random.seed(42)
    t = np.logspace(-2, 2, 50)
    G_t = 1e5 * np.exp(-t / 1.0) + np.random.normal(0, 1000, size=len(t))

    model = FractionalZenerSolidSolid()

    try:
        # Fit will fail with enhanced error message
        model.fit(t, G_t, test_mode='relaxation', max_iter=100)
    except RuntimeError as e:
        print(e)
        # Output includes:
        # - Original optimization error
        # - Detected decay type and material type
        # - Physics-based explanation of mismatch
        # - Recommended alternative models
        # - Guidance that failures are normal in model comparison

Model Compatibility Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional Zener Solid-Solid (FZSS)**

- Expects: Mittag-Leffler or power-law relaxation with finite equilibrium modulus
- Incompatible with: Exponential decay (use Maxwell/Zener instead)
- Incompatible with: Liquid-like behavior (use FractionalMaxwellLiquid)

**Fractional Maxwell Liquid (FML)**

- Expects: Liquid-like behavior (no equilibrium modulus)
- Incompatible with: Solid-like materials (use FZSS or FractionalKelvinVoigt)

**Fractional Maxwell Gel (FMG)**

- Expects: Power-law relaxation (gel-like)
- Incompatible with: Exponential decay (use Maxwell instead)

**Maxwell Model**

- Expects: Exponential decay
- Incompatible with: Power-law decay (use FMG or FZSS)

**Zener Model**

- Expects: Exponential decay with equilibrium modulus
- Incompatible with: Power-law decay (use FZSS)

**Fractional Kelvin-Voigt**

- Expects: Solid-like behavior
- Incompatible with: Liquid-like behavior (use FractionalMaxwellLiquid)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Fast detection**: < 1 ms for typical datasets (50-100 points)
- **Minimal overhead**: Can be enabled during fitting without performance impact
- **Robust to noise**: Uses statistical regression with confidence thresholds
- **Automatic test mode detection**: Works with relaxation, creep, and oscillation data

Use Cases
~~~~~~~~~

1. **Model Selection**: Identify appropriate models before fitting
2. **Error Diagnosis**: Understand why optimization failed
3. **Automated Pipelines**: Filter incompatible model-data combinations
4. **Model Comparison**: Expect some models to fail (this is normal!)
5. **Educational**: Learn about rheological model physics

See Also
--------

- :doc:`core` - BaseModel integration with check_compatibility parameter
- :doc:`../user_guide/model_selection` - Comprehensive model selection guide
- :doc:`../user_guide/getting_started` - Basic usage examples
