Giesekus Nonlinear Viscoelastic Models
======================================

This section documents the Giesekus family of models for polymer melts and
solutions exhibiting shear-thinning, normal stress differences, and
stress overshoot behavior.


Overview
--------

The Giesekus model (1982) extends the Upper-Convected Maxwell (UCM) framework
with a quadratic stress term representing anisotropic molecular mobility:

.. math::

   \boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2 \eta_p \mathbf{D}

Where:

- :math:`\boldsymbol{\tau}` is the polymer extra stress tensor
- :math:`\lambda` is the relaxation time
- :math:`\alpha` is the mobility factor (0 ≤ α ≤ 0.5)
- :math:`\eta_p` is the polymer viscosity
- :math:`\overset{\nabla}{\boldsymbol{\tau}}` is the upper-convected derivative
- :math:`\mathbf{D}` is the rate-of-deformation tensor

The mobility factor α controls shear-thinning:

- **α = 0**: Recovers UCM (no shear-thinning)
- **α = 0.5**: Maximum anisotropy/shear-thinning


Model Variants
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model
     - Description
   * - :class:`~rheojax.models.giesekus.GiesekusSingleMode`
     - Single relaxation time with all 6 protocols
   * - :class:`~rheojax.models.giesekus.GiesekusMultiMode`
     - N parallel modes for broad relaxation spectra


Key Features
------------

**Shear-Thinning Viscosity:**

.. math::

   \eta(\dot{\gamma}) = \eta_s + \eta_p \cdot f(\text{Wi})

where f(Wi) decreases with Weissenberg number Wi = λγ̇.

**Normal Stress Differences:**

.. math::

   N_1 = \tau_{xx} - \tau_{yy} > 0 \quad \text{(first normal stress)}

   N_2 = \tau_{yy} - \tau_{zz} < 0 \quad \text{(second normal stress)}

**Diagnostic Ratio:**

.. math::

   \frac{N_2}{N_1} = -\frac{\alpha}{2}

This provides a direct experimental route to determine α.


Supported Protocols
-------------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Protocol
     - Method
     - Notes
   * - FLOW_CURVE
     - Analytical
     - Steady shear σ(γ̇), η(γ̇)
   * - OSCILLATION
     - Analytical
     - SAOS G'(ω), G''(ω) (α-independent)
   * - STARTUP
     - ODE (diffrax)
     - Stress overshoot at constant rate
   * - RELAXATION
     - ODE (diffrax)
     - Faster-than-exponential decay
   * - CREEP
     - ODE (diffrax)
     - Strain evolution under constant stress
   * - LAOS
     - ODE + FFT
     - Nonlinear harmonics I₃, I₅, ...


Quick Start
-----------

**Basic Usage:**

.. code-block:: python

   from rheojax.models.giesekus import GiesekusSingleMode
   import numpy as np

   # Create model
   model = GiesekusSingleMode()
   model.parameters.set_value("eta_p", 100.0)
   model.parameters.set_value("lambda_1", 1.0)
   model.parameters.set_value("alpha", 0.3)

   # Predict flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   # Predict SAOS
   omega = np.logspace(-1, 3, 50)
   G_prime, G_double_prime = model.predict_saos(omega)

   # Simulate startup with overshoot
   t = np.linspace(0, 10, 500)
   sigma_t = model.simulate_startup(t, gamma_dot=10.0)

**Multi-Mode:**

.. code-block:: python

   from rheojax.models.giesekus import GiesekusMultiMode

   # Create 3-mode model
   model = GiesekusMultiMode(n_modes=3)

   # Set mode parameters
   model.set_mode_params(0, eta_p=100.0, lambda_1=10.0, alpha=0.3)
   model.set_mode_params(1, eta_p=50.0, lambda_1=1.0, alpha=0.2)
   model.set_mode_params(2, eta_p=20.0, lambda_1=0.1, alpha=0.1)

   # Predict SAOS
   G_prime, G_double_prime = model.predict_saos(omega)


When to Use Giesekus
--------------------

**Use Giesekus when you observe:**

1. Shear-thinning viscosity
2. Non-zero first and second normal stress differences
3. Stress overshoot in startup flow
4. Linear SAOS that can be fit by Maxwell modes

**Decision Tree:**

::

   Is N₂ measurable (negative)?
   ├── YES → Giesekus captures N₂/N₁ = -α/2
   │
   └── NO → Is only shear-thinning needed?
       ├── YES → Consider simpler Carreau/Cross
       └── NO → Consider PTT or FENE-P for extensional

**Material-Specific Recommendations:**

.. list-table::
   :widths: 25 20 20 35
   :header-rows: 1

   * - Material
     - Typical α
     - n_modes
     - Key Protocol
   * - Polymer melts
     - 0.1-0.3
     - 3-5
     - Flow curve + SAOS
   * - Polymer solutions
     - 0.2-0.4
     - 1-3
     - Startup + SAOS
   * - Wormlike micelles
     - 0.3-0.5
     - 1
     - Startup overshoot


References
----------

1. Giesekus, H. (1982). "A simple constitutive equation for polymer fluids
   based on the concept of deformation-dependent tensorial mobility."
   *J. Non-Newtonian Fluid Mech.*, 11, 69-109.

2. Bird, R.B., Armstrong, R.C., & Hassager, O. (1987).
   *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics.* 2nd ed. Wiley.

3. Larson, R.G. (1988). *Constitutive Equations for Polymer Melts and Solutions.*
   Butterworths.


.. toctree::
   :maxdepth: 1
   :caption: Detailed Documentation

   giesekus
