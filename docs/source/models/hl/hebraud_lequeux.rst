.. _hebraud_lequeux:

Hébraud–Lequeux (HL) Model
==========================

The **Hébraud–Lequeux (HL) model** is a seminal mean-field elastoplastic model for soft glassy materials (SGMs), introduced by Hébraud and Lequeux in 1998. It describes the rheology of yield-stress fluids, foams, emulsions, and pastes by considering the statistical evolution of local stresses.

.. note::
   This implementation uses high-performance JAX kernels with a Finite Volume Method (FVM) solver, enabling efficient fitting to flow curves, creep, relaxation, and LAOS data.

Physical Basis
--------------

The model considers a material composed of mesoscopic elastoplastic blocks. Each block carries a local shear stress :math:`\sigma` which evolves through three processes:

1.  **Elastic Loading**: Under macroscopic shear rate :math:`\dot{\gamma}`, blocks accumulate stress elastically (:math:`\dot{\sigma} = G_0 \dot{\gamma}`).
2.  **Plastic Yielding**: If the local stress exceeds a threshold :math:`\sigma_c`, the block yields (relaxes stress to zero) at a rate :math:`1/\tau`.
3.  **Mechanical Noise**: Yielding events redistribute stress to neighbors. In a mean-field approximation, this is modeled as a stress diffusion process with diffusivity :math:`D(t)`.

Key Distinction from SGR/STZ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **SGR (Soft Glassy Rheology)**: Disorder is in the *energy barrier heights* (trap depths). Noise is thermal-like effective temperature :math:`x`.
*   **STZ (Shear Transformation Zone)**: Dynamics are controlled by creation/annihilation of *structural defects* (STZs).
*   **HL (Hébraud–Lequeux)**: Disorder is in the *local stress state*. Noise is self-generated *mechanical diffusion*.

Mathematical Formulation
------------------------

The probability density function :math:`P(\sigma, t)` of local stresses evolves according to the Fokker-Planck equation:

.. math::

   \partial_t P(\sigma, t) = \underbrace{-\dot{\gamma}(t) \partial_\sigma P}_{\text{Advection}} + \underbrace{D(t) \partial^2_\sigma P}_{\text{Diffusion}} - \underbrace{\frac{1}{\tau} \Theta(|\sigma|-\sigma_c) P}_{\text{Yielding}} + \underbrace{\Gamma(t) \delta(\sigma)}_{\text{Reinjection}}

Self-Consistency
~~~~~~~~~~~~~~~~

The model is closed by coupling the noise strength :math:`D(t)` to the plastic activity rate :math:`\Gamma(t)`:

.. math::

   \Gamma(t) = \frac{1}{\tau} \int_{|\sigma| > \sigma_c} P(\sigma, t) \, d\sigma

   D(t) = \alpha \Gamma(t)

where :math:`\alpha` is the dimensionless coupling parameter.

Phase Behavior
--------------

The parameter :math:`\alpha` controls the phase state of the material:

*   **Glassy Phase** (:math:`\alpha < 0.5`): The material exhibits a finite yield stress :math:`\sigma_y`. Below this stress, the material is solid-like (creep arrest).
*   **Fluid Phase** (:math:`\alpha \ge 0.5`): The material flows at any non-zero stress.

Predictions and Protocols
-------------------------

Flow Curve (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

In steady shear (:math:`\dot{\gamma} = \text{const}`), the HL model predicts a **Herschel-Bulkley** behavior near yield:

.. math::

   \Sigma(\dot{\gamma}) \approx \Sigma_y + A \dot{\gamma}^{1/2}

The exponent 0.5 is a universal prediction of the HL model for the glassy phase.

Creep (Step Stress)
~~~~~~~~~~~~~~~~~~~

Under constant stress :math:`\Sigma_0`:
*   If :math:`\Sigma_0 < \Sigma_y`: The shear rate decays to zero (arrest).
*   If :math:`\Sigma_0 > \Sigma_y`: The shear rate reaches a steady finite value.
*   The model captures **delayed yielding** (creep rupture) dynamics near the yield point.

Stress Relaxation (Step Strain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following a step strain :math:`\gamma_0`, the stress relaxes via:
1.  Fast relaxation (yielding of highly stressed elements).
2.  Slow relaxation (diffusion).
For :math:`\alpha < 0.5`, residual stress may persist (finite elastic modulus).

LAOS (Large Amplitude Oscillatory Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In LAOS, the PDF :math:`P(\sigma, t)` oscillates. Large amplitudes drive the distribution past :math:`\sigma_c`, causing periodic fluidization and non-linear Lissajous figures.

Implementation Details
----------------------

The RheoJAX implementation uses an explicit Finite Volume Method (FVM) solver written in JAX.

*   **Advection**: First-order upwind scheme (stable for hyperbolic transport).
*   **Diffusion**: Central difference scheme.
*   **Time-stepping**: Operator splitting with Forward Euler.
*   **JIT Compilation**: The entire solver is JIT-compiled for GPU acceleration, typically running :math:`100\times` faster than pure Python implementations.

Numerical Parameters
~~~~~~~~~~~~~~~~~~~~

The solver uses a discretized stress grid:
*   `sigma_max`: Default 5.0 (normalized units).
*   `n_bins`: Default 501.
*   `dt`: Default 0.005.

These can be adjusted if necessary, but defaults work for most experimental ranges.

Usage Example
-------------

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   # Initialize model
   model = HebraudLequeux()

   # Set to glassy state
   model.parameters.set_value("alpha", 0.3)
   model.parameters.set_value("sigma_c", 10.0)  # Pa
   model.parameters.set_value("tau", 0.1)       # s

   # 1. Fit Flow Curve
   gdot = np.logspace(-2, 1, 20)
   stress_data = ... # Experimental data
   model.fit(gdot, stress_data, test_mode="steady_shear")

   # 2. Predict Creep
   time = np.linspace(0, 100, 1000)
   J_pred = model.predict(time, test_mode="creep", stress_target=12.0)

   # 3. Bayesian Inference
   # Run NUTS to get uncertainty on alpha and yield stress
   result = model.fit_bayesian(gdot, stress_data, test_mode="steady_shear", num_samples=1000)
   model.plot_pair(result)

References
----------

1.  Hébraud, P., & Lequeux, F. (1998). Mode-coupling theory for the pasty rheology of soft glassy materials. *Physical Review Letters*, 81(14), 2934.
2.  Fielding, S. M., et al. (2000). Aging and rheology in soft materials. *Journal of Rheology*.
