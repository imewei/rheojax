Utilities (rheo.utils)
======================

The utils module provides numerical utilities for rheological analysis, including special functions and optimization tools.

Mittag-Leffler Functions
-------------------------

.. automodule:: rheo.utils.mittag_leffler
   :members:
   :undoc-members:
   :show-inheritance:

The Mittag-Leffler function is essential for fractional calculus in rheology.
This module provides JAX-compatible implementations with high accuracy.

Functions
~~~~~~~~~

.. autofunction:: rheo.utils.mittag_leffler.mittag_leffler_e

.. autofunction:: rheo.utils.mittag_leffler.mittag_leffler_e2

Aliases
~~~~~~~

.. data:: rheo.utils.mittag_leffler.ml_e

   Alias for :func:`mittag_leffler_e`

.. data:: rheo.utils.mittag_leffler.ml_e2

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
    from rheo.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2

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
    from rheo.utils.mittag_leffler import mittag_leffler_e

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
    from rheo.utils.mittag_leffler import mittag_leffler_e

    # JIT compile function (alpha must be static)
    @jax.jit
    def compute_ml(z):
        return mittag_leffler_e(z, alpha=0.5)

    # Use compiled function
    z = jnp.linspace(0, 5, 1000)
    result = compute_ml(z)  # Fast computation

Optimization
------------

.. automodule:: rheo.utils.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Functions and Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.utils.optimization.OptimizationResult
   :members:
   :undoc-members:
   :show-inheritance:

   Result container for optimization.

.. autofunction:: rheo.utils.optimization.nlsq_optimize

.. autofunction:: rheo.utils.optimization.optimize_with_bounds

.. autofunction:: rheo.utils.optimization.residual_sum_of_squares

.. autofunction:: rheo.utils.optimization.create_least_squares_objective

Aliases
~~~~~~~

.. data:: rheo.utils.optimization.optimize

   Alias for :func:`nlsq_optimize`

.. data:: rheo.utils.optimization.fit_parameters

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

    from rheo.core.parameters import ParameterSet
    from rheo.utils.optimization import nlsq_optimize
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
    from rheo.core.parameters import ParameterSet
    from rheo.utils.optimization import nlsq_optimize

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

    from rheo.utils.optimization import create_least_squares_objective

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

    from rheo.core.parameters import ParameterConstraint

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

    from rheo.core.parameters import ParameterOptimizer

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
