.. _model-giesekus:

Giesekus Model
==============

Quick Reference
---------------

- **Use when:** Polymer melts/solutions with shear-thinning, normal stress differences, stress overshoot
- **Parameters:** 4 (:math:`\eta_p`, :math:`\lambda`, :math:`\alpha`, :math:`\eta_s`)
- **Key equation:** :math:`\boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2 \eta_p \mathbf{D}`
- **Diagnostic:** :math:`N_2/N_1 = -\alpha/2` (direct experimental route to :math:`\alpha`)
- **Test modes:** Flow curve, oscillation, startup, relaxation, creep, LAOS
- **Material examples:** Polymer melts, concentrated solutions, wormlike micelles

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\boldsymbol{\tau}`
     - Polymer extra stress tensor (Pa)
   * - :math:`\eta_p`
     - Polymer viscosity (Pa·s). Zero-shear polymer contribution.
   * - :math:`\lambda`
     - Relaxation time (s). Characteristic stress decay time.
   * - :math:`\alpha`
     - Mobility factor (dimensionless, 0 ≤ :math:`\alpha` ≤ 0.5). Controls shear-thinning.
   * - :math:`\eta_s`
     - Solvent viscosity (Pa·s). Newtonian background contribution.
   * - :math:`\eta_0`
     - Zero-shear viscosity, :math:`\eta_0 = \eta_p + \eta_s`
   * - :math:`G`
     - Elastic modulus, :math:`G = \eta_p / \lambda`
   * - :math:`\text{Wi}`
     - Weissenberg number, :math:`\text{Wi} = \lambda \dot{\gamma}`
   * - :math:`N_1`
     - First normal stress difference, :math:`N_1 = \tau_{xx} - \tau_{yy}`
   * - :math:`N_2`
     - Second normal stress difference, :math:`N_2 = \tau_{yy} - \tau_{zz}`
   * - :math:`\overset{\nabla}{\boldsymbol{\tau}}`
     - Upper-convected derivative (frame-invariant time derivative)

Overview
--------

The Giesekus model (1982) is a nonlinear differential constitutive equation that
extends the Upper-Convected Maxwell (UCM) model with a quadratic stress term
representing anisotropic molecular mobility. It provides a physically motivated
description of:

1. **Shear-thinning viscosity**: Viscosity decreases with increasing shear rate
2. **Normal stress differences**: Both :math:`N_1 > 0` and :math:`N_2 < 0`
3. **Stress overshoot**: Peak stress in startup flow at constant rate
4. **Faster-than-exponential relaxation**: Due to the quadratic stress term

The model is particularly valuable because it predicts both first and second
normal stress differences with a fixed ratio :math:`N_2/N_1 = -\alpha/2`, providing
a direct experimental route to determine the mobility parameter :math:`\alpha`.

Historical Context
~~~~~~~~~~~~~~~~~~

Hanswalter Giesekus introduced this model in 1982 as a "simple constitutive equation
based on the concept of deformation-dependent tensorial mobility." The key insight
was that molecular mobility in polymer melts is not isotropic—molecules aligned by
flow experience different friction in different directions.

The model became widely adopted because:

- It uses only one additional parameter (:math:`\alpha`) beyond the Maxwell model
- It captures essential nonlinear features with simple mathematics
- The parameter :math:`\alpha` has clear physical interpretation
- Predictions agree well with experimental data for many polymeric systems

Physical Foundations
--------------------

Molecular Picture: Anisotropic Drag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Giesekus model arises from considering how polymer chains experience drag
in a flowing medium. When chains are stretched and aligned by flow:

**Isotropic drag (UCM model)**:
   Chains experience the same friction regardless of orientation.
   Result: No shear-thinning, :math:`N_2 = 0`

**Anisotropic drag (Giesekus model)**:
   Aligned chains slip more easily along their backbone than perpendicular to it.
   Result: Shear-thinning, :math:`N_2 < 0`

The mobility parameter :math:`\alpha` quantifies this anisotropy:

- :math:`\alpha = 0`: Isotropic drag → recovers UCM model
- :math:`\alpha = 0.5`: Maximum anisotropy → strongest thinning
- Typical values: 0.1-0.4 for most polymer melts and solutions

Network Interpretation
~~~~~~~~~~~~~~~~~~~~~~

Alternatively, the Giesekus model can be derived from a temporary network theory
where:

- Polymer chains form a transient network of entanglements
- Network junctions break and reform with rate dependent on local stress
- Higher stress → faster junction breakage → lower effective viscosity

The quadratic :math:`\boldsymbol{\tau} \cdot \boldsymbol{\tau}` term represents the
stress-induced acceleration of network relaxation.

Derivation from Configuration Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most rigorous derivation uses a configuration tensor :math:`\mathbf{c}` representing
the average molecular conformation:

.. math::

   \mathbf{c} + \lambda \overset{\nabla}{\mathbf{c}} + \frac{\alpha}{\eta_p}(\mathbf{c} \cdot \boldsymbol{\tau}) = \mathbf{I}

Combined with the stress-configuration relation :math:`\boldsymbol{\tau} = G(\mathbf{c} - \mathbf{I})`,
this yields the Giesekus equation. The anisotropic term :math:`\mathbf{c} \cdot \boldsymbol{\tau}`
couples stress relaxation to molecular orientation.

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The complete stress for the Giesekus model is:

.. math::

   \boldsymbol{\sigma} = -p\mathbf{I} + 2\eta_s \mathbf{D} + \boldsymbol{\tau}

where :math:`\boldsymbol{\tau}` satisfies the Giesekus constitutive equation:

.. math::

   \boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2 \eta_p \mathbf{D}

The upper-convected derivative is:

.. math::

   \overset{\nabla}{\boldsymbol{\tau}} = \frac{\partial \boldsymbol{\tau}}{\partial t} + \mathbf{v} \cdot \nabla \boldsymbol{\tau} - (\nabla \mathbf{v})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot (\nabla \mathbf{v})

Simple Shear Flow
~~~~~~~~~~~~~~~~~

For steady simple shear with velocity :math:`v_x = \dot{\gamma} y`, the stress
components satisfy:

.. math::

   \tau_{xx} + \frac{\alpha \lambda}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2) &= 2\lambda \dot{\gamma} \tau_{xy}

   \tau_{yy} + \frac{\alpha \lambda}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2) &= 0

   \tau_{xy} + \frac{\alpha \lambda}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) &= \eta_p \dot{\gamma}

These coupled nonlinear equations admit analytical solutions in terms of an
auxiliary function :math:`f(\text{Wi})`.

Steady-State Viscosity
~~~~~~~~~~~~~~~~~~~~~~

The steady shear viscosity is:

.. math::

   \eta(\dot{\gamma}) = \eta_s + \eta_p \cdot f(\text{Wi})

where :math:`f(\text{Wi})` is the polymeric viscosity reduction factor satisfying
an implicit quartic equation. Key limits:

- **Low Wi** (:math:`\text{Wi} \ll 1`): :math:`f \to 1`, so :math:`\eta \to \eta_0 = \eta_p + \eta_s`
- **High Wi** (:math:`\text{Wi} \gg 1`): :math:`f \sim \text{Wi}^{-1}` for :math:`\alpha > 0`

Normal Stress Differences
~~~~~~~~~~~~~~~~~~~~~~~~~

The Giesekus model predicts:

.. math::

   N_1 = \tau_{xx} - \tau_{yy} > 0 \quad \text{(rod climbing, die swell)}

   N_2 = \tau_{yy} - \tau_{zz} < 0 \quad \text{(secondary flows)}

With the fundamental diagnostic ratio:

.. math::

   \frac{N_2}{N_1} = -\frac{\alpha}{2}

This ratio is **independent of shear rate**, making it an excellent experimental
route to determine :math:`\alpha`.

SAOS Response
~~~~~~~~~~~~~

Small-amplitude oscillatory shear (SAOS) is independent of :math:`\alpha` and matches the
Maxwell model:

.. math::

   G'(\omega) = G \frac{(\omega\lambda)^2}{1 + (\omega\lambda)^2}

   G''(\omega) = G \frac{\omega\lambda}{1 + (\omega\lambda)^2} + \eta_s \omega

where :math:`G = \eta_p/\lambda` is the elastic modulus.

Transient Flows
~~~~~~~~~~~~~~~

**Startup flow** (constant :math:`\dot{\gamma}` from rest):

The Giesekus model predicts stress overshoot—a peak stress higher than
the steady-state value. The overshoot:

- Occurs at strain :math:`\gamma \sim O(1)`
- Increases with Weissenberg number
- Is more pronounced for larger :math:`\alpha`

**Stress relaxation** (cessation of flow):

Stress decays faster than pure exponential due to the quadratic term:

.. math::

   \sigma(t) < \sigma_0 \exp(-t/\lambda)

The quadratic :math:`\boldsymbol{\tau} \cdot \boldsymbol{\tau}` term accelerates
relaxation when stress is high.

Parameters
----------

.. list-table:: Giesekus Model Parameters
   :widths: 15 15 15 20 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Bounds
     - Physical Meaning
   * - eta_p
     - :math:`\eta_p`
     - Pa·s
     - (1e-3, 1e6)
     - Polymer zero-shear viscosity
   * - lambda_1
     - :math:`\lambda`
     - s
     - (1e-6, 1e4)
     - Characteristic relaxation time
   * - alpha
     - :math:`\alpha`
     - —
     - [0, 0.5]
     - Mobility anisotropy factor
   * - eta_s
     - :math:`\eta_s`
     - Pa·s
     - [0, 1e4)
     - Solvent/Newtonian viscosity

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Polymer viscosity** :math:`\eta_p`:
   - Dominant contribution to zero-shear viscosity
   - Scales with molecular weight: :math:`\eta_p \sim M_w^{3.4}` above entanglement
   - Temperature dependent via Arrhenius/WLF

**Relaxation time** :math:`\lambda`:
   - Time for stress to decay to 1/e of initial value
   - Scales with molecular weight: :math:`\lambda \sim M_w^{3.4}`
   - Defines crossover frequency: :math:`\omega_c = 1/\lambda`

**Mobility factor** :math:`\alpha`:
   - :math:`\alpha = 0`: Isotropic mobility (UCM limit)
   - :math:`\alpha = 0.5`: Maximum anisotropy
   - Directly measurable: :math:`\alpha = -2 N_2/N_1`
   - Typical values:
     - Polymer melts: 0.1–0.3
     - Concentrated solutions: 0.2–0.4
     - Wormlike micelles: 0.3–0.5

**Solvent viscosity** :math:`\eta_s`:
   - Newtonian background contribution
   - Important for dilute/semi-dilute solutions
   - Often negligible for melts (:math:`\eta_s \ll \eta_p`)

Derived Quantities
~~~~~~~~~~~~~~~~~~

- **Zero-shear viscosity**: :math:`\eta_0 = \eta_p + \eta_s`
- **Elastic modulus**: :math:`G = \eta_p/\lambda`
- **Weissenberg number**: :math:`\text{Wi} = \lambda \dot{\gamma}`
- **Deborah number**: :math:`\text{De} = \lambda/t_{\text{obs}}`

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Incompressibility**: Constant density during deformation
2. **Homogeneous deformation**: No spatial gradients in material properties
3. **Isothermal conditions**: Temperature held constant
4. **Upper-convected derivative**: Frame-invariant stress transport
5. **Single relaxation time**: Monodisperse or narrow distribution

Validity Range
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Condition
     - Range
     - Notes
   * - Weissenberg number
     - Wi ≲ 100
     - Numerical stability limit
   * - Shear rate
     - :math:`\dot{\gamma} < 1/\lambda` to :math:`100/\lambda`
     - Power-law region
   * - Strain (startup)
     - :math:`\gamma` ≲ 10
     - Overshoot captured
   * - Temperature
     - Near reference T
     - Use TTS for other temperatures

Limitations
~~~~~~~~~~~

1. **Single relaxation time**: Real polymers have spectra (use multi-mode)
2. **No extensional hardening**: Underpredicts extensional viscosity
3. **Fixed** :math:`N_2/N_1` **ratio**: Cannot vary independently
4. **Numerical stiffness**: High Wi may require adaptive solvers

When NOT to Use
~~~~~~~~~~~~~~~

- **Extensional flows**: Use FENE-P or PTT for extensional hardening
- **Broad relaxation spectra**: Use multi-mode Giesekus
- **Thixotropic materials**: Use fluidity models
- **Yield stress fluids**: Use EVP models (Saramito)

Regimes and Behavior
--------------------

Weissenberg Number Regimes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 20 30 35
   :header-rows: 1

   * - Regime
     - Wi Range
     - Viscosity
     - Physics
   * - Newtonian
     - Wi ≪ 1
     - :math:`\eta \approx \eta_0`
     - Linear response, no thinning
   * - Transition
     - Wi ~ 1
     - Onset of thinning
     - Nonlinear effects begin
   * - Power-law
     - Wi ≫ 1
     - :math:`\eta \sim \text{Wi}^{n-1}`
     - Strong shear-thinning

Effect of :math:`\alpha` on Behavior
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 25 30 30
   :header-rows: 1

   * - :math:`\alpha` value
     - Shear-thinning
     - :math:`N_2/N_1`
     - Example materials
   * - 0
     - None (UCM)
     - 0
     - Ideal elastic liquid
   * - 0.1
     - Weak
     - −0.05
     - Some polymer melts
   * - 0.3
     - Moderate
     - −0.15
     - Typical polymers
   * - 0.5
     - Maximum
     - −0.25
     - Wormlike micelles

What You Can Learn
------------------

From Flow Curve Fitting
~~~~~~~~~~~~~~~~~~~~~~~

**Primary outputs:**

- :math:`\eta_0 = \eta_p + \eta_s`: Zero-shear viscosity (plateau value)
- :math:`\lambda` **from onset**: Shear rate where thinning begins ≈ :math:`1/\lambda`
- **Power-law index**: High-Wi slope gives effective n

**What this reveals:**

- Molecular weight (via :math:`\eta_0 and \lambda` scaling)
- Entanglement density
- Solution concentration effects

From SAOS Fitting
~~~~~~~~~~~~~~~~~

**Primary outputs:**

- **G =** :math:`\eta_p/\lambda`: From high-frequency G' plateau
- :math:`\lambda` **from crossover**: Where G' = G''
- :math:`\eta_s` **from G''**: High-frequency slope

**What this reveals:**

- Elastic modulus (network strength)
- Relaxation spectrum width (single mode = narrow peak)
- Solvent contribution

From Normal Stress Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Primary output:**

- :math:`\alpha = -2N_2/N_1`: Direct measurement of mobility factor

**What this reveals:**

- Degree of molecular anisotropy
- Network structure
- Material classification (polymer vs. micelle)

From Startup Flow
~~~~~~~~~~~~~~~~~

**Primary outputs:**

- **Overshoot ratio** :math:`\sigma_{\text{max}}/\sigma_{ss}`: Increases with Wi
- **Strain at peak**: :math:`\gamma_{\text{peak}}` ~ 1-3 for Giesekus
- **Transient timescale**: Approach to steady state

**What this reveals:**

- Nonlinear viscoelastic character
- Network reformation dynamics
- Stress relaxation mechanisms

Experimental Design
-------------------

When to Use Giesekus
~~~~~~~~~~~~~~~~~~~~

Use the Giesekus model when your material exhibits:

1. Shear-thinning viscosity
2. Measurable :math:`N_2` (negative second normal stress difference)
3. Stress overshoot in startup flow
4. SAOS that fits Maxwell/Generalized Maxwell
5. Single or narrow relaxation time distribution

Decision Tree
~~~~~~~~~~~~~

::

   Is N_2 measurable (negative)?
   ├── YES → Giesekus captures N_2/N_1 = -α/2
   │
   └── NO → Is only shear-thinning needed?
       ├── YES → Consider simpler Carreau/Cross
       └── NO → Consider PTT or FENE-P for extensional

Recommended Protocol Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **SAOS first**: Determine :math:`\eta_p`, :math:`\lambda`, :math:`\eta_s` from linear regime
2. **Flow curve**: Confirm thinning, refine parameters
3. **Normal stresses**: Measure :math:`N_2/N_1` to determine :math:`\alpha`
4. **Startup flow**: Validate overshoot predictions
5. **Relaxation**: Confirm faster-than-exponential decay

Material-Specific Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Material
     - Typical :math:`\alpha`
     - n_modes
     - Key protocols
   * - Polymer melts
     - 0.1–0.3
     - 3–5
     - Flow curve + SAOS + :math:`N_2`
   * - Polymer solutions
     - 0.2–0.4
     - 1–3
     - Startup + SAOS
   * - Wormlike micelles
     - 0.3–0.5
     - 1
     - Startup overshoot + relaxation
   * - Biological fluids
     - 0.2–0.4
     - 2–3
     - SAOS + low-Wi flow curve

Computational Implementation
----------------------------

RheoJAX Implementation
~~~~~~~~~~~~~~~~~~~~~~

The Giesekus model in RheoJAX uses:

- **JAX acceleration**: JIT-compiled kernels for fast predictions
- **diffrax integration**: Adaptive ODE solvers (Tsit5) for transients
- **Analytical solutions**: Where available (steady shear, SAOS)
- **Float64 precision**: Essential for accurate stress calculations

Architecture
~~~~~~~~~~~~

::

   GiesekusBase (ABC)
   ├── GiesekusSingleMode
   │   ├── Analytical: flow_curve, SAOS
   │   └── ODE: startup, relaxation, creep, LAOS
   │
   └── GiesekusMultiMode
       ├── SAOS superposition (analytical)
       └── Extended state vector ODE

Numerical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

**Steady-state solver:**

- Newton iteration for auxiliary function f(Wi)
- Converges in 5-10 iterations typically
- May need damping at very high Wi

**ODE integration:**

- Tsit5 (Runge-Kutta 5(4)) for accuracy
- Adaptive step size with PIDController
- rtol=1e-6, atol=1e-8 default tolerances

**Numerical stability:**

- High Wi (>100) may require reduced tolerances
- Very small :math:`\alpha` (<0.01) approaches UCM singularities
- Use log-residuals for fitting flow curves

Fitting Guidance
----------------

Initial Parameter Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From SAOS data:**

.. code-block:: python

   # At crossover (G' = G'')
   lambda_1 = 1 / omega_crossover
   G = G_prime_at_crossover * 2  # G' = G'' = G/2 at crossover
   eta_p = G * lambda_1

**From flow curve:**

.. code-block:: python

   # Zero-shear plateau
   eta_0 = stress[0] / gamma_dot[0]  # At lowest rate

   # Onset of thinning
   lambda_1 = 1 / gamma_dot_onset  # Where η starts dropping

:math:`\alpha` **estimation:**

.. code-block:: python

   # From normal stresses (if available)
   alpha = -2 * N2 / N1

   # From thinning slope (rough estimate)
   # High-Wi slope of η vs γ̇ in log-log ≈ (n-1)
   # For Giesekus: n ≈ 0.5 at alpha = 0.5

Fitting Strategy
~~~~~~~~~~~~~~~~

1. **Fix** :math:`\eta_s` **if known** (pure solvent viscosity)
2. **Fit SAOS first** for :math:`\eta_p`, :math:`\lambda` (:math:`\alpha`-independent)
3. **Fit flow curve** to refine and get :math:`\alpha`
4. **Validate with startup** for dynamic behavior

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Problem
     - Likely Cause
     - Solution
   * - Poor flow curve fit
     - Wrong :math:`\alpha`
     - Use :math:`N_2/N_1` to fix :math:`\alpha`, then fit others
   * - Overshoot too small
     - :math:`\alpha` too low
     - Increase :math:`\alpha` toward 0.5
   * - No convergence at high Wi
     - Numerical stiffness
     - Reduce max Wi, use adaptive solver
   * - Relaxation too slow
     - :math:`\lambda` too long
     - Fit SAOS crossover more carefully
   * - SAOS mismatch
     - Single mode inadequate
     - Use multi-mode Giesekus

Usage Examples
--------------

Basic Single-Mode
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.giesekus import GiesekusSingleMode
   import numpy as np

   # Create model with parameters
   model = GiesekusSingleMode()
   model.parameters.set_value("eta_p", 100.0)  # Pa·s
   model.parameters.set_value("lambda_1", 1.0)  # s
   model.parameters.set_value("alpha", 0.3)     # dimensionless
   model.parameters.set_value("eta_s", 10.0)    # Pa·s

   # Predict flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   # Get viscosity
   _, eta, _ = model.predict_flow_curve(gamma_dot, return_components=True)

Predict SAOS
~~~~~~~~~~~~

.. code-block:: python

   # SAOS is alpha-independent (linear regime)
   omega = np.logspace(-2, 3, 50)
   G_prime, G_double_prime = model.predict_saos(omega)

   # Complex modulus
   G_star = np.sqrt(G_prime**2 + G_double_prime**2)

Normal Stress Prediction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Normal stress differences
   gamma_dot = np.logspace(-1, 2, 30)
   N1, N2 = model.predict_normal_stresses(gamma_dot)

   # Verify diagnostic ratio
   ratio = N2 / N1  # Should equal -alpha/2 = -0.15 (for alpha=0.3)

Startup with Overshoot
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Startup flow at constant rate
   t = np.linspace(0, 10, 500)
   sigma_t = model.simulate_startup(t, gamma_dot=10.0)

   # Find overshoot
   sigma_max = np.max(sigma_t)
   sigma_ss = sigma_t[-1]
   overshoot_ratio = sigma_max / sigma_ss  # > 1 indicates overshoot

   # Get full stress tensor evolution
   result = model.simulate_startup(t, gamma_dot=10.0, return_full=True)

Multi-Mode Giesekus
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.giesekus import GiesekusMultiMode

   # Create 3-mode model
   model = GiesekusMultiMode(n_modes=3)

   # Set per-mode parameters
   model.set_mode_params(0, eta_p=100.0, lambda_1=10.0, alpha=0.3)
   model.set_mode_params(1, eta_p=50.0, lambda_1=1.0, alpha=0.25)
   model.set_mode_params(2, eta_p=20.0, lambda_1=0.1, alpha=0.2)
   model.parameters.set_value("eta_s", 5.0)

   # SAOS captures broad spectrum
   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega)

Bayesian Fitting
~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.data import RheoData

   # Create data object
   data = RheoData(x=omega, y=G_star, test_mode='oscillation')

   # NLSQ warm-start
   model.fit(data)

   # Bayesian inference
   result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)

Model Comparison
----------------

vs. Upper-Convected Maxwell (UCM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - UCM (:math:`\alpha = 0`)
     - Giesekus (:math:`\alpha > 0`)
   * - Viscosity
     - Constant
     - Shear-thinning
   * - :math:`N_1`
     - Positive
     - Positive
   * - :math:`N_2`
     - Zero
     - Negative
   * - Startup
     - Overshoot (weak)
     - Overshoot (strong)
   * - Relaxation
     - Exponential
     - Faster than exponential

vs. Phan-Thien–Tanner (PTT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - Giesekus
     - PTT
   * - Thinning mechanism
     - Anisotropic drag
     - Network destruction
   * - :math:`N_2/N_1`
     - Fixed = :math:`-\alpha/2`
     - Adjustable
   * - Extensional
     - Bounded
     - Bounded (stronger)
   * - Parameters
     - 4
     - 4-5
   * - Best for
     - Shear flows
     - Mixed flows

vs. FENE-P
~~~~~~~~~~

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - Giesekus
     - FENE-P
   * - Mechanism
     - Anisotropic drag
     - Finite extensibility
   * - Extensional
     - Moderate
     - Strong hardening
   * - Shear thinning
     - Strong
     - Moderate
   * - :math:`N_2`
     - Nonzero
     - Zero
   * - Best for
     - Shear + :math:`N_2`
     - Extensional flows

When to Choose Each Model
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Giesekus**: Need :math:`N_2` prediction, shear-dominated flows
- **PTT**: Mixed shear-extension, adjustable :math:`N_2/N_1`
- **FENE-P**: Extension-dominated, fiber spinning
- **Oldroyd-B/UCM**: Simple validation, teaching

See Also
--------

Related Models
~~~~~~~~~~~~~~

- :ref:`model-maxwell` — Linear viscoelastic foundation
- :ref:`model-generalized-maxwell` — Multi-mode linear model
- PTT — Alternative nonlinear model
- FENE-P — Finite extensibility model

Related Topics
~~~~~~~~~~~~~~

- :ref:`transform-mastercurve` — Time-temperature superposition
- :ref:`protocol-saos` — Small-amplitude oscillatory shear
- :ref:`protocol-startup` — Startup flow experiments
- Bayesian inference — Parameter uncertainty quantification

References
----------

Primary Sources
~~~~~~~~~~~~~~~

1. Giesekus, H. (1982). "A simple constitutive equation for polymer fluids
   based on the concept of deformation-dependent tensorial mobility."
   *J. Non-Newtonian Fluid Mech.*, 11, 69-109.
   https://doi.org/10.1016/0377-0257(82)85016-7

2. Giesekus, H. (1983). "Stressing behaviour in simple shear flow as
   predicted by a new constitutive model for polymer fluids."
   *J. Non-Newtonian Fluid Mech.*, 12, 367-374.

Textbooks
~~~~~~~~~

3. Bird, R.B., Armstrong, R.C., & Hassager, O. (1987).
   *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics.* 2nd ed.
   Wiley-Interscience. Chapter 4.

4. Larson, R.G. (1988). *Constitutive Equations for Polymer Melts and Solutions.*
   Butterworths. Chapter 4.

5. Morrison, F.A. (2001). *Understanding Rheology.*
   Oxford University Press. Chapter 9.

6. Macosko, C.W. (1994). *Rheology: Principles, Measurements, and Applications.*
   Wiley-VCH. Chapter 3.

Applications
~~~~~~~~~~~~

7. Yoo, J.Y., & Choi, H.C. (1989). "On the steady simple shear flows of the
   one-mode Giesekus fluid." *Rheol. Acta*, 28, 13-24.

8. Schleiniger, G., & Weinacht, R.J. (1991). "Steady Poiseuille flows for a
   Giesekus fluid." *J. Non-Newtonian Fluid Mech.*, 40, 79-102.

9. Quinzani, L.M., Armstrong, R.C., & Brown, R.A. (1994). "Birefringence and
   laser-Doppler velocimetry (LDV) studies of viscoelastic flow through a
   planar contraction." *J. Non-Newtonian Fluid Mech.*, 52, 1-36.

Normal Stress Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~

10. Magda, J.J., & Baek, S.G. (1994). "Concentrated entangled and semidilute
    entangled polystyrene solutions and the second normal stress difference."
    *Polymer*, 35, 1187-1194.

11. Lee, C.S., Tripp, B.C., & Magda, J.J. (1992). "Does :math:`N_2` depend on the shear
    rate in polymer melts?" *Rheol. Acta*, 31, 306-314.

Multi-Mode Extensions
~~~~~~~~~~~~~~~~~~~~~

12. Quinzani, L.M., McKinley, G.H., Brown, R.A., & Armstrong, R.C. (1990).
    "Modeling the rheology of polyisobutylene solutions."
    *J. Rheol.*, 34, 705-748.

13. Debbaut, B., & Crochet, M.J. (1988). "Extensional effects in complex flows."
    *J. Non-Newtonian Fluid Mech.*, 30, 169-184.

Numerical Methods
~~~~~~~~~~~~~~~~~

14. Hulsen, M.A., Fattal, R., & Kupferman, R. (2005). "Flow of viscoelastic
    fluids past a cylinder at high Weissenberg number: Stabilized simulations
    using matrix logarithms." *J. Non-Newtonian Fluid Mech.*, 127, 27-39.

15. Guénette, R., & Fortin, M. (1995). "A new mixed finite element method for
    computing viscoelastic flows." *J. Non-Newtonian Fluid Mech.*, 60, 27-52.

API Reference
-------------

.. autoclass:: rheojax.models.giesekus.GiesekusSingleMode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rheojax.models.giesekus.GiesekusMultiMode
   :members:
   :undoc-members:
   :show-inheritance:
