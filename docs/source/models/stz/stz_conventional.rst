Shear Transformation Zone (STZ)
===============================

Overview
--------

The Shear Transformation Zone (STZ) theory provides a physical description of plastic deformation in amorphous materials such as metallic glasses, colloidal suspensions, emulsions, and granular matter. Unlike crystalline materials where plasticity is mediated by dislocations, amorphous solids deform through localized rearrangements of particle clusters known as Shear Transformation Zones.

The **STZ Conventional** model (:class:`rheojax.models.stz.conventional.STZConventional`) implements the effective temperature formulation developed by Langer, Falk, and Bouchbinder (Langer 2008). It captures key nonlinear rheological phenomena including:

*   **Yield Stress**: Emergence of a dynamic yield stress from structural disorder.
*   **Aging & Rejuvenation**: Time-dependent evolution of the structural state (effective temperature).
*   **Transient Overshoot**: Stress peaks during startup flow.
*   **Shear Banding**: (In spatial implementations) Instabilities arising from effective temperature gradients.

Variants
--------

The implementation supports three complexity levels suitable for different applications:

.. list-table:: Model Variants
   :widths: 20 25 15 40
   :header-rows: 1

   * - Variant
     - State Variables
     - Complexity
     - Best For
   * - **Minimal**
     - :math:`s, \chi`
     - Low
     - Steady-state flow curves, simple yield stress fluids.
   * - **Standard**
     - :math:`s, \chi, \Lambda`
     - Medium
     - **Default**. Aging, thixotropy, stress overshoot, transients.
   * - **Full**
     - :math:`s, \chi, \Lambda, m`
     - High
     - LAOS, back-stress, Bauschinger effect, strong anisotropy.

Theoretical Background
----------------------

Physical Basis
~~~~~~~~~~~~~~
The central concept of STZ theory is the **Effective Temperature** (:math:`\chi`), which characterizes the configurational disorder of the material's inherent structure.

*   **Low** :math:`\chi`: Deeply annealed, jammed state (solid-like).
*   **High** :math:`\chi`: Rejuvenated, disordered state (liquid-like).

Plastic flow is produced by STZs flipping between two stable configurations (aligned "+" or anti-aligned "-") under the bias of applied stress.

Mathematical Formulation
------------------------

Core Kinetics
~~~~~~~~~~~~~
The plastic strain rate :math:`\dot{\varepsilon}^{pl}` is governed by the density of STZs and the rate of their transitions:

.. math::

   \dot{\varepsilon}^{pl} = \frac{\varepsilon_0}{\tau_0} \Lambda(\chi) \mathcal{C}(s) \mathcal{T}(s)

where:

*   :math:`\Lambda(\chi) = e^{-1/\chi}` is the **STZ Density**.
*   :math:`\mathcal{C}(s) = \cosh(s/\sigma_y)^q` is the **Rate Factor** (activation).
*   :math:`\mathcal{T}(s) = \tanh(s/\sigma_y)` is the **Transition Bias**.
*   :math:`s` is the deviatoric stress.
*   :math:`\sigma_y` is the yield stress scale.

State Evolution Equations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Effective Temperature Dynamics** (:math:`\chi`)
   Driven by plastic work (rejuvenation) and thermal relaxation (aging):

   .. math::

      \dot{\chi} = \frac{s \dot{\varepsilon}^{pl}}{c_0 \sigma_y} (\chi_\infty - \chi) + \text{Aging}(\chi)

   The term :math:`s \dot{\varepsilon}^{pl}` represents the rate of energy dissipation. :math:`\chi_\infty` is the steady-state effective temperature at high drive.

2. **STZ Density Dynamics** (:math:`\Lambda`)
   (Standard/Full variants) Relaxes toward the equilibrium value :math:`e^{-1/\chi}`:

   .. math::

      \dot{\Lambda} = -\frac{\Lambda - e^{-1/\chi}}{\tau_\beta}

3. **Orientation Dynamics** (:math:`m`)
   (Full variant) Describes the kinematic hardening or back-stress due to STZ alignment:

   .. math::

      \dot{m} = \frac{2 \mathcal{C}(s)}{\tau_0} (\mathcal{T}(s) - m) - \Gamma m

Numerical Implementation
------------------------

This implementation leverages **JAX** and **Diffrax** for high-performance simulation:

*   **JIT Compilation**: All physics kernels are JIT-compiled for speed.
*   **Stiff Solvers**: Uses implicit ODE solvers (e.g., Kvaerno5, Tsit5) to handle the fast timescales of STZ flips vs. slow aging.
*   **Protocol Support**:
    *   **Steady Shear**: Algebraic solution (instantaneous).
    *   **Transient**: ODE integration for startup, relaxation, and creep.
    *   **LAOS**: Full cycle integration + FFT for harmonic analysis.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 60

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``G0``
     - :math:`G_0`
     - Pa
     - High-frequency elastic shear modulus.
   * - ``sigma_y``
     - :math:`\sigma_y`
     - Pa
     - Yield stress scale (activation barrier).
   * - ``chi_inf``
     - :math:`\chi_\infty`
     - -
     - Steady-state effective temperature limit.
   * - ``tau0``
     - :math:`\tau_0`
     - s
     - Molecular vibration timescale (attempt time).
   * - ``epsilon0``
     - :math:`\varepsilon_0`
     - -
     - Strain increment per STZ rearrangement (typically 0.1-0.3).
   * - ``c0``
     - :math:`c_0`
     - -
     - Effective specific heat (controls rate of :math:`\chi` evolution).
   * - ``ez``
     - :math:`e_Z`
     - -
     - STZ formation energy (normalized by :math:`k_B T_g`).
   * - ``tau_beta``
     - :math:`\tau_\beta`
     - s
     - Relaxation timescale for STZ density :math:`\Lambda`.
   * - ``m_inf``
     - :math:`m_\infty`
     - -
     - Saturation value for orientational bias (Full variant).
   * - ``rate_m``
     - :math:`\Gamma_m`
     - -
     - Rate coefficient for orientational bias evolution (Full variant).

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from rheojax.models import STZConventional

   # Initialize model (Standard variant includes Lambda dynamics)
   model = STZConventional(variant="standard")

   # --- 1. Steady State Flow Curve Fitting ---
   # Fit to shear rate vs stress data
   gamma_dot = np.logspace(-3, 1, 20)
   stress_data = ... # Experimental data

   model.fit(gamma_dot, stress_data, test_mode="steady_shear")

   print(model.parameters.get_value("sigma_y"))

   # --- 2. Transient Startup Simulation ---
   # Simulate stress overshoot at constant shear rate
   t = np.linspace(0, 10, 1000)
   stress_overshoot = model.predict(t, test_mode="startup", gamma_dot=1.0)

   # --- 3. LAOS Simulation ---
   # Large Amplitude Oscillatory Shear
   strain, stress = model.simulate_laos(gamma_0=1.0, omega=5.0)

API Reference
-------------

.. autoclass:: rheojax.models.stz.conventional.STZConventional
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: rheojax.models.stz._base.STZBase
   :members: get_initial_state
   :undoc-members:
