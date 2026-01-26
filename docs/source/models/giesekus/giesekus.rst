.. _model-giesekus:

===========================
Giesekus Model — Handbook
===========================

Quick Reference
---------------

- **Use when:** Polymer melts/solutions with shear-thinning, normal stress differences, stress overshoot
- **Parameters:** 4 (:math:`\eta_p`, :math:`\lambda`, :math:`\alpha`, :math:`\eta_s`)
- **Key equation:** :math:`\boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2 \eta_p \mathbf{D}`
- **Diagnostic:** :math:`N_2/N_1 = -\alpha/2` (direct experimental route to :math:`\alpha`)
- **Test modes:** Flow curve, oscillation, startup, relaxation, creep, LAOS
- **Material examples:** Polymer melts, concentrated solutions, wormlike micelles

----

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
   * - :math:`\text{De}`
     - Deborah number, :math:`\text{De} = \lambda / t_{\text{obs}}`
   * - :math:`N_1`
     - First normal stress difference, :math:`N_1 = \tau_{xx} - \tau_{yy}`
   * - :math:`N_2`
     - Second normal stress difference, :math:`N_2 = \tau_{yy} - \tau_{zz}`
   * - :math:`\Psi_1`
     - First normal stress coefficient, :math:`\Psi_1 = N_1 / \dot{\gamma}^2`
   * - :math:`\Psi_2`
     - Second normal stress coefficient, :math:`\Psi_2 = N_2 / \dot{\gamma}^2`
   * - :math:`\eta^*`
     - Complex viscosity, :math:`\eta^* = \eta' - i\eta''`
   * - :math:`J(t)`
     - Creep compliance, :math:`J(t) = \gamma(t) / \sigma_0`
   * - :math:`\overset{\nabla}{\boldsymbol{\tau}}`
     - Upper-convected derivative (frame-invariant time derivative)
   * - :math:`\mathbf{c}`
     - Conformation tensor (average molecular conformation)

----

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

Hanswalter Giesekus introduced this model in 1982 [1]_ as a "simple constitutive
equation based on the concept of deformation-dependent tensorial mobility." The key
insight was that molecular mobility in polymer melts is not isotropic—molecules
aligned by flow experience different friction in different directions.

The model became widely adopted because:

- It uses only one additional parameter (:math:`\alpha`) beyond the Maxwell model
- It captures essential nonlinear features with simple mathematics
- The parameter :math:`\alpha` has clear physical interpretation
- Predictions agree well with experimental data for many polymeric systems

----

Physical Foundations
--------------------

Molecular Picture: Anisotropic Drag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
- Typical values: 0.1–0.4 for most polymer melts and solutions

Network Interpretation
~~~~~~~~~~~~~~~~~~~~~~

Alternatively, the Giesekus model can be derived from a temporary network theory
where:

- Polymer chains form a transient network of entanglements
- Network junctions break and reform with rate dependent on local stress
- Higher stress → faster junction breakage → lower effective viscosity

The quadratic :math:`\boldsymbol{\tau} \cdot \boldsymbol{\tau}` term represents the
stress-induced acceleration of network relaxation.

Stress Decomposition
~~~~~~~~~~~~~~~~~~~~

The total Cauchy stress for an incompressible Giesekus fluid is split into solvent
and polymeric contributions:

.. math::

   \boldsymbol{\sigma} = -p\mathbf{I} + \boldsymbol{\sigma}_s + \boldsymbol{\tau}

where:

- :math:`\boldsymbol{\sigma}_s = 2\eta_s \mathbf{D}` is the Newtonian solvent stress
- :math:`\boldsymbol{\tau}` is the polymer extra stress evolving via the Giesekus law
- :math:`p` is the isotropic pressure

Some texts use :math:`\boldsymbol{\sigma}_p` in place of :math:`\boldsymbol{\tau}` for
the polymeric stress. Throughout this handbook we use :math:`\boldsymbol{\tau}` to denote
the polymer contribution, consistent with the rest of RheoJAX documentation.

Conformation Tensor Form
~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative and often numerically preferred formulation uses the conformation
tensor :math:`\mathbf{c}` representing the average molecular conformation. The
stress–configuration relation is:

.. math::

   \boldsymbol{\tau} = \frac{\eta_p}{\lambda}(\mathbf{c} - \mathbf{I})

The evolution of :math:`\mathbf{c}` follows:

.. math::

   \lambda \overset{\nabla}{\mathbf{c}} + (\mathbf{c} - \mathbf{I}) + \alpha (\mathbf{c} - \mathbf{I})^2 = 0

This form is preferred in CFD applications because it guarantees positive-definiteness
of :math:`\mathbf{c}` when combined with appropriate numerical methods [14]_.

Material Functions
~~~~~~~~~~~~~~~~~~

The Giesekus model defines the following material functions, which are measurable
experimentally:

**Shear viscosity** (from steady shear):

.. math::

   \eta(\dot{\gamma}) = \frac{\sigma_{xy}}{\dot{\gamma}} = \frac{\tau_{xy}}{\dot{\gamma}} + \eta_s

**Complex viscosity** (from oscillatory shear):

.. math::

   \eta^*(\omega) = \eta'(\omega) - i\eta''(\omega)

**Normal stress coefficients** (from steady shear):

.. math::

   \Psi_1(\dot{\gamma}) = \frac{N_1}{\dot{\gamma}^2} = \frac{\tau_{xx} - \tau_{yy}}{\dot{\gamma}^2}

   \Psi_2(\dot{\gamma}) = \frac{N_2}{\dot{\gamma}^2} = \frac{\tau_{yy} - \tau_{zz}}{\dot{\gamma}^2}

**Crossover frequency** (from SAOS):

.. math::

   \omega_c = \frac{1}{\lambda} \quad \text{where } G'(\omega_c) = G''(\omega_c) - \eta_s \omega_c

----

Governing Equations
-------------------

Kinematics and Notation
~~~~~~~~~~~~~~~~~~~~~~~~

The velocity field :math:`\mathbf{v}(\mathbf{x}, t)` defines the velocity gradient
tensor :math:`\nabla\mathbf{v}` and the rate-of-deformation tensor:

.. math::

   \mathbf{D} = \frac{1}{2}\bigl(\nabla\mathbf{v} + (\nabla\mathbf{v})^T\bigr)

The **upper-convected derivative** of a tensor :math:`\mathbf{A}` is the
frame-invariant time derivative:

.. math::

   \overset{\nabla}{\mathbf{A}} = \frac{D\mathbf{A}}{Dt} - (\nabla\mathbf{v})^T \cdot \mathbf{A} - \mathbf{A} \cdot (\nabla\mathbf{v})

where :math:`D/Dt = \partial_t + \mathbf{v} \cdot \nabla` is the material derivative.
For homogeneous flows (spatially uniform stress), the convective term
:math:`\mathbf{v} \cdot \nabla\boldsymbol{\tau}` vanishes and the material derivative
reduces to the ordinary time derivative.

**Simple shear geometry:**

The velocity field :math:`\mathbf{v} = (\dot{\gamma} y, 0, 0)` defines:

.. math::

   \nabla\mathbf{v} = \begin{pmatrix} 0 & \dot{\gamma} & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \qquad
   \mathbf{D} = \frac{1}{2}\begin{pmatrix} 0 & \dot{\gamma} & 0 \\ \dot{\gamma} & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

The polymer stress tensor in simple shear has the structure:

.. math::

   \boldsymbol{\tau} = \begin{pmatrix} \tau_{xx} & \tau_{xy} & 0 \\ \tau_{xy} & \tau_{yy} & 0 \\ 0 & 0 & \tau_{zz} \end{pmatrix}

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The polymer stress :math:`\boldsymbol{\tau}` satisfies the Giesekus constitutive
equation [1]_:

.. math::

   \boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2 \eta_p \mathbf{D}

The measurable total shear stress is:

.. math::

   \sigma_{xy} = \eta_s \dot{\gamma} + \tau_{xy}

The normal stress differences are:

.. math::

   N_1 = \tau_{xx} - \tau_{yy}, \qquad N_2 = \tau_{yy} - \tau_{zz}

Component-wise ODE System
~~~~~~~~~~~~~~~~~~~~~~~~~

In simple shear, the constitutive equation reduces to four coupled ODEs for
the stress components [3]_:

.. math::

   \frac{d\tau_{xx}}{dt} = -\frac{\tau_{xx}}{\lambda} + 2\dot{\gamma}\,\tau_{xy} - \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2)

.. math::

   \frac{d\tau_{yy}}{dt} = -\frac{\tau_{yy}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2)

.. math::

   \frac{d\tau_{xy}}{dt} = -\frac{\tau_{xy}}{\lambda} + \dot{\gamma}\,\tau_{yy} - \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) + \frac{\eta_p}{\lambda}\dot{\gamma}

.. math::

   \frac{d\tau_{zz}}{dt} = -\frac{\tau_{zz}}{\lambda} - \frac{\alpha}{\eta_p}\tau_{zz}^2

Each equation has three contributions:

1. **Linear relaxation**: :math:`-\tau_{ij}/\lambda` (exponential decay toward equilibrium)
2. **Convective coupling**: terms involving :math:`\dot{\gamma}\tau_{ij}` (flow-induced stress transfer)
3. **Quadratic nonlinearity**: terms involving :math:`\alpha \tau^2/\eta_p` (anisotropic drag)

The :math:`\tau_{zz}` component decouples from the other three and relaxes to zero
from any initial condition.

Dimensionless Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Define dimensionless variables:

- **Weissenberg number**: :math:`\text{Wi} = \lambda \dot{\gamma}`
- **Dimensionless stress**: :math:`\tau_{ij}^* = \tau_{ij} \lambda / \eta_p`
- **Dimensionless time**: :math:`t^* = t / \lambda`

The ODEs become:

.. math::

   \frac{d\tau_{xx}^*}{dt^*} = -\tau_{xx}^* + 2\,\text{Wi}\;\tau_{xy}^* - \alpha(\tau_{xx}^{*2} + \tau_{xy}^{*2})

.. math::

   \frac{d\tau_{yy}^*}{dt^*} = -\tau_{yy}^* - \alpha(\tau_{xy}^{*2} + \tau_{yy}^{*2})

.. math::

   \frac{d\tau_{xy}^*}{dt^*} = -\tau_{xy}^* + \text{Wi}\;\tau_{yy}^* - \alpha\,\tau_{xy}^*(\tau_{xx}^* + \tau_{yy}^*) + \text{Wi}

This formulation is useful because:

- All behavior is parameterized by just **two numbers**: :math:`\text{Wi}` and :math:`\alpha`
- Universal behavior curves collapse data at different rates and relaxation times
- Improved numerical conditioning when :math:`\eta_p/\lambda` spans many orders of magnitude

Analytical Steady-State Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At steady state (:math:`d/dt = 0`), the ODE system reduces to a nonlinear algebraic
system that admits closed-form solutions [1]_ [7]_.

Define the auxiliary discriminant:

.. math::

   \Lambda = \sqrt{1 + 16\,\alpha(1-\alpha)\,\text{Wi}^2}

and the auxiliary function:

.. math::

   f(\text{Wi}) = \frac{1 - \Lambda}{8\,\alpha(1-\alpha)\,\text{Wi}^2}\Bigl[1 + \Lambda + 2(1-2\alpha)\,\text{Wi}^2\Bigr]

The steady-state polymer stress components are:

.. math::

   \tau_{xy,\text{ss}} = \frac{\eta_p}{\lambda} \cdot \frac{(1 - f)\,\text{Wi}}{1 + (1-2\alpha)\,f}

.. math::

   \tau_{xx,\text{ss}} = \frac{\eta_p}{\lambda} \cdot \frac{2\,(1-f)^2\,\text{Wi}^2}{[1 + (1-2\alpha)\,f]\,[1 - \alpha\,f]}

.. math::

   \tau_{yy,\text{ss}} = \frac{\eta_p}{\lambda} \cdot \frac{-2\,\alpha\,f\,(1-f)\,\text{Wi}^2}{[1 + (1-2\alpha)\,f]\,[1 - \alpha\,f]}

.. math::

   \tau_{zz,\text{ss}} = 0

The **steady-state shear viscosity** is:

.. math::

   \eta(\dot{\gamma}) = \eta_s + \eta_p \cdot \frac{1 - f}{1 + (1 - 2\alpha)\,f}

where the term :math:`(1-f)/[1 + (1-2\alpha)\,f]` is the polymeric viscosity reduction
factor.

**Normal stress coefficients** at steady state:

.. math::

   \Psi_1(\dot{\gamma}) = \frac{\tau_{xx,\text{ss}} - \tau_{yy,\text{ss}}}{\dot{\gamma}^2}, \qquad
   \Psi_2(\dot{\gamma}) = \frac{\tau_{yy,\text{ss}}}{\dot{\gamma}^2}

Limiting Behaviors
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 30 30 20
   :header-rows: 1

   * - Quantity
     - Low Wi (:math:`\text{Wi} \ll 1`)
     - High Wi (:math:`\text{Wi} \gg 1`)
     - Notes
   * - :math:`\eta`
     - :math:`\eta_0 = \eta_p + \eta_s`
     - :math:`\sim \text{Wi}^{-1}`
     - Shear-thinning
   * - :math:`\Psi_1`
     - :math:`\Psi_{1,0} = 2\eta_p\lambda`
     - :math:`\sim \text{Wi}^{-2}`
     - Decreases
   * - :math:`\Psi_2`
     - :math:`\Psi_{2,0} = -\alpha\,\eta_p\lambda`
     - :math:`\sim \text{Wi}^{-2}`
     - Negative
   * - :math:`N_2/N_1`
     - :math:`-\alpha`
     - :math:`-\alpha/2`
     - Rate-independent at high Wi

----

Protocol-Specific Equations
----------------------------

This section presents the complete equations for each experimental protocol
supported by the Giesekus model. Each protocol specifies the imposed
kinematic or stress condition, the resulting ODE system (or algebraic system),
initial conditions, and characteristic output observables.

Steady Shear (Flow Curve)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** Constant shear rate :math:`\dot{\gamma} = \text{const}`, solve at
steady state (:math:`\partial_t \boldsymbol{\tau} = 0`).

**Governing system:** Setting all time derivatives to zero in the component
ODEs yields the nonlinear algebraic system:

.. math::

   \frac{\tau_{xx}}{\lambda} - 2\dot{\gamma}\,\tau_{xy} + \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2) = 0

.. math::

   \frac{\tau_{yy}}{\lambda} + \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2) = 0

.. math::

   \frac{\tau_{xy}}{\lambda} - \dot{\gamma}\,\tau_{yy} + \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) = \frac{\eta_p}{\lambda}\dot{\gamma}

.. math::

   \frac{\tau_{zz}}{\lambda} + \frac{\alpha}{\eta_p}\tau_{zz}^2 = 0 \quad \Rightarrow \quad \tau_{zz} = 0

**Solution method:** Use the analytical formulas from the previous section or
Newton–Raphson iteration.

**Output observables:**

- **Flow curve**: :math:`\sigma_{xy}(\dot{\gamma}) = \eta_s \dot{\gamma} + \tau_{xy,\text{ss}}(\dot{\gamma})`
- **Viscosity**: :math:`\eta(\dot{\gamma}) = \sigma_{xy}/\dot{\gamma}`
- **Normal stresses**: :math:`N_1 = \tau_{xx} - \tau_{yy} > 0`, :math:`N_2 = \tau_{yy} < 0`

**Shear-thinning behavior:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Wi range
     - :math:`\eta` behavior
     - Physics
   * - :math:`\text{Wi} \ll 1`
     - :math:`\eta \approx \eta_0`
     - Newtonian plateau
   * - :math:`\text{Wi} \sim 1`
     - Onset of thinning
     - Nonlinear drag effects begin
   * - :math:`\text{Wi} \gg 1`
     - :math:`\eta \sim \text{Wi}^{-1}`
     - Power-law region

**Normal stress ratio:**

.. math::

   \frac{N_2}{N_1} \approx -\frac{\alpha}{2} \quad (\text{at high Wi})

This ratio is approximately independent of shear rate, making it the primary
experimental route to determine :math:`\alpha` [10]_ [11]_.

Startup of Steady Shear
~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** Apply constant shear rate from rest: :math:`\dot{\gamma}(t) = \dot{\gamma}_0\,H(t)`,
where :math:`H(t)` is the Heaviside step function.

**Initial conditions:** :math:`\tau_{xx}(0) = \tau_{yy}(0) = \tau_{xy}(0) = \tau_{zz}(0) = 0`

**ODE system:**

.. math::

   \frac{d\tau_{xx}}{dt} = -\frac{\tau_{xx}}{\lambda} + 2\dot{\gamma}_0\,\tau_{xy} - \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2)

.. math::

   \frac{d\tau_{yy}}{dt} = -\frac{\tau_{yy}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2)

.. math::

   \frac{d\tau_{xy}}{dt} = -\frac{\tau_{xy}}{\lambda} + \dot{\gamma}_0\,\tau_{yy} - \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) + \frac{\eta_p}{\lambda}\dot{\gamma}_0

**Output:** :math:`\sigma_{xy}(t) = \eta_s \dot{\gamma}_0 + \tau_{xy}(t)` and
:math:`N_1(t) = \tau_{xx}(t) - \tau_{yy}(t)`.

**Characteristic features:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Time/strain regime
     - Behavior
   * - :math:`t \ll \lambda` (linear elastic)
     - :math:`\tau_{xy} \approx G\,\dot{\gamma}_0\,t` (affine, slope = :math:`G`)
   * - :math:`\gamma \sim O(1)` (overshoot)
     - Stress peaks above steady state, :math:`N_1` also overshoots
   * - :math:`t \gg \lambda` (steady state)
     - Stress relaxes to :math:`\tau_{xy,\text{ss}}`

**Overshoot characteristics:**

- **Peak strain**: :math:`\gamma_{\text{peak}} \sim 2\text{–}3` strain units (depends on Wi and :math:`\alpha`)
- **Overshoot ratio**: :math:`\sigma_{\text{peak}}/\sigma_{\text{ss}}` increases with Wi
- Higher :math:`\alpha` → smaller overshoot (stronger nonlinear damping)
- **High-Wi scaling**: :math:`\gamma_{\text{peak}} \sim \text{const}` (2–3), :math:`\sigma_{\text{peak}}/\sigma_{\text{ss}} \sim \text{Wi}^{1/2}`

Stress Relaxation
~~~~~~~~~~~~~~~~~

**Protocol:** Apply instantaneous step strain :math:`\gamma_0` at :math:`t = 0`, then
:math:`\dot{\gamma}(t > 0) = 0`.

**Initial conditions** (from instantaneous elastic response):

.. math::

   \tau_{xy}(0^+) = G\,\gamma_0 = \frac{\eta_p}{\lambda}\,\gamma_0

.. math::

   \tau_{xx}(0^+) = 2\,G\,\gamma_0^2

.. math::

   \tau_{yy}(0^+) = 0, \qquad \tau_{zz}(0^+) = 0

**Relaxation ODEs** (with :math:`\dot{\gamma} = 0`):

.. math::

   \frac{d\tau_{xx}}{dt} = -\frac{\tau_{xx}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2)

.. math::

   \frac{d\tau_{yy}}{dt} = -\frac{\tau_{yy}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2)

.. math::

   \frac{d\tau_{xy}}{dt} = -\frac{\tau_{xy}}{\lambda} - \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy})

**Linear regime** (small :math:`\gamma_0`, quadratic terms negligible):

.. math::

   G(t) = \frac{\tau_{xy}(t)}{\gamma_0} = G\,e^{-t/\lambda}

**Nonlinear regime** (finite :math:`\gamma_0`):

The quadratic :math:`\alpha`-terms accelerate relaxation when stress is high,
giving **faster-than-exponential** initial decay:

.. math::

   \sigma(t) < \sigma_0 \exp(-t/\lambda)

**Damping function** (quantifies strain-dependent relaxation):

.. math::

   h(\gamma_0) = \frac{G(t, \gamma_0)}{G(t)} \quad \text{at early times}

For the Giesekus model, the instantaneous response obeys the Lodge–Meissner
rule (:math:`h(\gamma) = 1` at :math:`t = 0^+`), but nonlinear effects emerge
during the relaxation process.

**Time-strain separability** (approximate):

.. math::

   G(t, \gamma_0) \approx G(t) \cdot h(\gamma_0)

where :math:`G(t) = G\,e^{-t/\lambda}` and :math:`h(\gamma_0)` is the damping function.

Creep (Step Stress)
~~~~~~~~~~~~~~~~~~~

**Protocol:** Apply constant total shear stress :math:`\sigma_{xy}(t) = \sigma_0\,H(t)`.

**Stress-control closure:** The applied stress constraint gives:

.. math::

   \dot{\gamma}(t) = \frac{\sigma_0 - \tau_{xy}(t)}{\eta_s}

This makes the shear rate a dependent variable computed from the evolving polymer
stress.

**Coupled ODE system** (5 equations: 4 stress + strain):

.. math::

   \frac{d\tau_{xx}}{dt} = -\frac{\tau_{xx}}{\lambda} + 2\dot{\gamma}\,\tau_{xy} - \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2)

.. math::

   \frac{d\tau_{yy}}{dt} = -\frac{\tau_{yy}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2)

.. math::

   \frac{d\tau_{xy}}{dt} = -\frac{\tau_{xy}}{\lambda} + \dot{\gamma}\,\tau_{yy} - \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) + \frac{\eta_p}{\lambda}\dot{\gamma}

.. math::

   \frac{d\gamma}{dt} = \dot{\gamma}(t) = \frac{\sigma_0 - \tau_{xy}(t)}{\eta_s}

where :math:`\dot{\gamma}` in the stress equations is evaluated from the closure at
each time step.

**Initial conditions:** :math:`\tau_{xx} = \tau_{yy} = \tau_{xy} = \tau_{zz} = \gamma = 0`

**Creep compliance:**

.. math::

   J(t) = \frac{\gamma(t)}{\sigma_0}

**Limiting behaviors:**

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Time
     - :math:`J(t)`
     - Physics
   * - :math:`t \to 0^+`
     - :math:`J_0 = 1/G = \lambda/\eta_p`
     - Instantaneous elastic compliance
   * - :math:`t \to \infty`
     - :math:`J(t) \sim t/\eta_0`
     - Steady-state viscous flow

**Recovery after unloading** (stress removed at :math:`t = t_1`):

- Elastic strain recovered: :math:`\Delta\gamma_{\text{rec}} \approx \sigma_0/G`
- Permanent (viscous) strain: :math:`\gamma_{\text{perm}} = \gamma(t_1) - \Delta\gamma_{\text{rec}}`

.. note::

   When :math:`\eta_s = 0` (no solvent), the stress-control closure becomes
   singular. This case requires a DAE (differential-algebraic equation) solver
   or reformulation.

Small-Amplitude Oscillatory Shear (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0 \ll 1`.

In the linear limit, the quadratic :math:`\alpha`-term is negligible and the Giesekus
model reduces to the Oldroyd-B/Maxwell response. The SAOS moduli are therefore
**independent of** :math:`\alpha`:

**Storage modulus:**

.. math::

   G'(\omega) = G \frac{(\omega\lambda)^2}{1 + (\omega\lambda)^2} = \frac{\eta_p \omega^2 \lambda}{1 + (\omega\lambda)^2}

**Loss modulus:**

.. math::

   G''(\omega) = G \frac{\omega\lambda}{1 + (\omega\lambda)^2} + \eta_s \omega = \frac{\eta_p \omega}{1 + (\omega\lambda)^2} + \eta_s \omega

where :math:`G = \eta_p/\lambda` is the elastic modulus.

**Complex viscosity:**

.. math::

   \eta'(\omega) = \frac{\eta_p}{1 + (\omega\lambda)^2} + \eta_s

.. math::

   \eta''(\omega) = \frac{\eta_p \omega\lambda}{1 + (\omega\lambda)^2}

**Limiting behaviors:**

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Frequency
     - :math:`G'`
     - :math:`G''`
   * - :math:`\omega \to 0`
     - :math:`G' \sim \omega^2`
     - :math:`G'' \sim \omega`
   * - :math:`\omega \to \infty`
     - :math:`G' \to G = \eta_p/\lambda`
     - :math:`G'' \sim \eta_s \omega` (solvent)

**Crossover frequency:**

.. math::

   \omega_c = 1/\lambda \quad \text{where } G'(\omega_c) = G''(\omega_c) - \eta_s\omega_c

Large-Amplitude Oscillatory Shear (LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0` finite.

The shear rate is :math:`\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)`.

**Full ODE system:** The component equations are the same as the general simple
shear system with the time-dependent :math:`\dot{\gamma}(t)` inserted:

.. math::

   \frac{d\tau_{xx}}{dt} = -\frac{\tau_{xx}}{\lambda} + 2\dot{\gamma}(t)\,\tau_{xy} - \frac{\alpha}{\eta_p}(\tau_{xx}^2 + \tau_{xy}^2)

.. math::

   \frac{d\tau_{yy}}{dt} = -\frac{\tau_{yy}}{\lambda} - \frac{\alpha}{\eta_p}(\tau_{xy}^2 + \tau_{yy}^2)

.. math::

   \frac{d\tau_{xy}}{dt} = -\frac{\tau_{xy}}{\lambda} + \dot{\gamma}(t)\,\tau_{yy} - \frac{\alpha}{\eta_p}\tau_{xy}(\tau_{xx} + \tau_{yy}) + \frac{\eta_p}{\lambda}\dot{\gamma}(t)

The total shear stress is :math:`\sigma(t) = \tau_{xy}(t) + \eta_s \dot{\gamma}(t)`.

**Fourier decomposition** of the periodic stress response [16]_:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots} \bigl[\sigma_n' \sin(n\omega t) + \sigma_n'' \cos(n\omega t)\bigr]

Only odd harmonics appear due to the symmetry of shear flow.

**First harmonic moduli** (strain-amplitude dependent):

.. math::

   G_1'(\omega, \gamma_0) = \frac{\sigma_1'}{\gamma_0}, \qquad G_1''(\omega, \gamma_0) = \frac{\sigma_1''}{\gamma_0}

**Third harmonic ratio** (primary nonlinearity measure):

.. math::

   I_{3/1} = \frac{\sqrt{\sigma_3'^2 + \sigma_3''^2}}{\sqrt{\sigma_1'^2 + \sigma_1''^2}}

**MAOS scaling** (medium-amplitude regime):

.. math::

   I_{3/1} \sim \gamma_0^2 \quad \text{as } \gamma_0 \to 0

**Chebyshev decomposition:**

.. math::

   \sigma(\gamma, \dot{\gamma}) = \gamma_0 \sum_{n \text{ odd}} \bigl[e_n\,T_n(x) + v_n\,T_n(y)\bigr]

where :math:`x = \gamma/\gamma_0`, :math:`y = \dot{\gamma}/(\gamma_0\omega)`, and
:math:`T_n` are Chebyshev polynomials.

**Pipkin diagram regimes:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - :math:`\text{De} = \omega\lambda`
     - :math:`\text{Wi} = \gamma_0 \omega \lambda`
     - Regime
   * - Any
     - :math:`\ll 1`
     - Linear viscoelastic (SAOS)
   * - :math:`\ll 1`
     - Any
     - Quasi-steady nonlinear
   * - :math:`\gg 1`
     - :math:`\gg 1`
     - Highly nonlinear viscoelastic

**Giesekus LAOS signatures:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Giesekus behavior
   * - Strain softening
     - :math:`G_1'` decreases with :math:`\gamma_0` (from :math:`\alpha > 0`)
   * - Higher harmonics
     - Present due to quadratic stress term
   * - Lissajous shape
     - Ellipse → tilted/distorted at high :math:`\gamma_0`
   * - :math:`I_{3/1}` scaling
     - :math:`I_{3/1} \sim \gamma_0^2` in MAOS regime

----

Multi-Mode Giesekus
--------------------

Motivation
~~~~~~~~~~

Real polymer systems have a broad spectrum of relaxation times arising from
polydispersity and the range of molecular conformations. A single-mode Giesekus
model cannot capture the broad frequency dependence typically observed in
:math:`G'(\omega)` and :math:`G''(\omega)` data. The multi-mode extension
addresses this by superposing :math:`N` independent Giesekus modes.

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The total stress is:

.. math::

   \boldsymbol{\sigma} = -p\mathbf{I} + 2\eta_s\mathbf{D} + \sum_{k=1}^{N} \boldsymbol{\tau}_k

where each mode :math:`k` evolves independently:

.. math::

   \boldsymbol{\tau}_k + \lambda_k \overset{\nabla}{\boldsymbol{\tau}_k} + \frac{\alpha_k \lambda_k}{\eta_{p,k}} \boldsymbol{\tau}_k \cdot \boldsymbol{\tau}_k = 2\eta_{p,k}\,\mathbf{D}

Each mode has its own relaxation time :math:`\lambda_k`, polymer viscosity
:math:`\eta_{p,k}`, and mobility factor :math:`\alpha_k`.

Linear Viscoelastic Spectra (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-mode SAOS, the moduli superpose linearly:

.. math::

   G'(\omega) = \sum_{k=1}^{N} \frac{\eta_{p,k}\,\omega^2\,\lambda_k}{1 + (\omega\lambda_k)^2}

.. math::

   G''(\omega) = \sum_{k=1}^{N} \frac{\eta_{p,k}\,\omega}{1 + (\omega\lambda_k)^2} + \eta_s\,\omega

Zero-Shear Properties
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \eta_0 = \sum_{k=1}^{N} \eta_{p,k} + \eta_s

.. math::

   \Psi_{1,0} = 2 \sum_{k=1}^{N} \eta_{p,k}\,\lambda_k

.. math::

   \Psi_{2,0} = -\sum_{k=1}^{N} \alpha_k\,\eta_{p,k}\,\lambda_k

Multi-Mode ODE State Vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For transient simulations, the state vector has :math:`4N` components (4 stress
components per mode). Each mode evolves independently with its own parameters
but shares the same velocity field :math:`\dot{\gamma}(t)`:

.. math::

   \mathbf{y} = [\tau_{xx}^{(1)}, \tau_{yy}^{(1)}, \tau_{xy}^{(1)}, \tau_{zz}^{(1)}, \ldots, \tau_{xx}^{(N)}, \tau_{yy}^{(N)}, \tau_{xy}^{(N)}, \tau_{zz}^{(N)}]

Fitting Strategy
~~~~~~~~~~~~~~~~

1. **Discrete spectrum from SAOS**: Fit :math:`G_k = \eta_{p,k}/\lambda_k` and :math:`\lambda_k` to SAOS data
2. **Logarithmic spacing**: Place :math:`\lambda_k` at logarithmically spaced points across the frequency window
3. **Regularization**: Use non-negative least squares (NNLS) or Tikhonov regularization to avoid overfitting
4. **Typical mode count**: 5–10 modes cover 4–6 decades in frequency
5. **Fix** :math:`\alpha_k` **from nonlinear data**: The linear spectrum determines :math:`\eta_{p,k}` and :math:`\lambda_k`; fit :math:`\alpha_k` to flow curve or normal stress data

----

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

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

- :math:`0 \leq \alpha \leq 0.5` for most physical systems
- :math:`\alpha > 0.5` can produce unphysical behavior at high Wi (non-monotonic flow curves)
- :math:`\eta_s \geq 0`, :math:`\lambda > 0`, :math:`\eta_p > 0`

Typical Parameter Ranges by Material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 20 20 15 20
   :header-rows: 1

   * - Material
     - :math:`\eta_p` (Pa·s)
     - :math:`\lambda` (s)
     - :math:`\alpha`
     - :math:`\eta_s` (Pa·s)
   * - Polymer solutions
     - 0.1–1000
     - 0.001–10
     - 0.1–0.5
     - 0.001–1
   * - Polymer melts
     - 100–10\ :sup:`6`
     - 0.1–1000
     - 0.1–0.5
     - ~0
   * - Wormlike micelles
     - 1–100
     - 0.1–10
     - 0.3–0.5
     - 0.001–0.1

Derived Quantities
~~~~~~~~~~~~~~~~~~

- **Zero-shear viscosity**: :math:`\eta_0 = \eta_p + \eta_s`
- **Elastic modulus**: :math:`G = \eta_p/\lambda`
- **Weissenberg number**: :math:`\text{Wi} = \lambda \dot{\gamma}`
- **Deborah number**: :math:`\text{De} = \lambda/t_{\text{obs}}`

----

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

----

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

----

What You Can Learn
------------------

From SAOS Data
~~~~~~~~~~~~~~

**Extractable parameters:** :math:`\eta_p`, :math:`\lambda`, :math:`\eta_s`, :math:`G`

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Observable
     - Extracted quantity
   * - :math:`G''/\omega` as :math:`\omega \to 0`
     - :math:`\eta_0 = \eta_p + \eta_s`
   * - :math:`G''/\omega` as :math:`\omega \to \infty`
     - :math:`\eta_s` (high-frequency limit)
   * - Crossover :math:`G' = G'' - \eta_s\omega`
     - :math:`\lambda = 1/\omega_c`
   * - :math:`G'` plateau (:math:`\omega \to \infty`)
     - :math:`G = \eta_p/\lambda`

**What this reveals:**

- **Elastic modulus** :math:`G`: Network strength / entanglement density
- **Relaxation time** :math:`\lambda`: Molecular weight, longest relaxation mode
- **Relaxation spectrum width**: Single-mode fit quality indicates how narrow/broad the spectrum is

.. note::

   The mobility parameter :math:`\alpha` is **not** determinable from SAOS data because
   SAOS is :math:`\alpha`-independent (linear regime).

From Steady Shear (Flow Curve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Extractable parameters:** :math:`\eta_0`, :math:`\lambda` (onset), :math:`\alpha` (shape)

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Observable
     - Extracted quantity
   * - Low-rate plateau :math:`\eta(\dot{\gamma} \to 0)`
     - :math:`\eta_0 = \eta_p + \eta_s`
   * - Onset shear rate for thinning
     - :math:`\lambda \approx 1/\dot{\gamma}_{\text{onset}}`
   * - Shape of thinning curve
     - :math:`\alpha` (controls power-law slope)

**What this reveals:**

- **Molecular weight** (via :math:`\eta_0` and :math:`\lambda` scaling laws)
- **Entanglement density**: :math:`\eta_0 \sim c^{3.4}` for entangled systems
- **Cross-validation**: Compare :math:`\eta_0` and :math:`\lambda` from SAOS

From Normal Stress Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Primary output:** Direct :math:`\alpha` determination.

.. math::

   \alpha = -\frac{2\,N_2}{N_1}

**What this reveals:**

- **Degree of molecular anisotropy**: Higher :math:`\alpha` → more anisotropic drag
- **Material classification**: Polymer melts (:math:`\alpha \sim 0.1\text{–}0.3`), wormlike micelles (:math:`\alpha \sim 0.3\text{–}0.5`)
- **Experimental techniques**: Cone-and-plate (for :math:`N_1`), parallel-plate edge measurements or cone-partitioned plate (for :math:`N_2`)

From Startup Flow
~~~~~~~~~~~~~~~~~

**Primary outputs:**

- **Overshoot ratio** :math:`\sigma_{\text{max}}/\sigma_{\text{ss}}`: Increases with Wi, quantifies nonlinear viscoelastic character
- **Strain at peak**: :math:`\gamma_{\text{peak}} \sim 2\text{–}3` — network deformation scale
- **Time to steady state**: :math:`\sim 3\text{–}5\lambda` — validates relaxation time
- **Initial slope**: :math:`d\sigma/d\gamma|_{t \to 0} = G` — instantaneous elastic modulus

From Stress Relaxation
~~~~~~~~~~~~~~~~~~~~~~

**Primary outputs:**

- **Exponential vs. faster-than-exponential decay**: Faster initial decay confirms :math:`\alpha > 0`
- **Relaxation modulus**: :math:`G(t) = \tau_{xy}(t)/\gamma_0` — full time-dependent response
- **Damping function** :math:`h(\gamma_0)`: Quantifies nonlinear strain effects (strain thinning)
- **Time-strain separability**: Whether :math:`G(t, \gamma_0) \approx G(t) \cdot h(\gamma_0)` holds

From Creep
~~~~~~~~~~

**Primary outputs:**

- **Instantaneous compliance**: :math:`J_0 = 1/G = \lambda/\eta_p`
- **Steady-state viscosity**: Long-time slope :math:`dJ/dt \to 1/\eta_0`
- **Elastic recovery** (after unloading): Recoverable strain :math:`\approx \sigma_0/G`
- **Retardation spectrum**: Transition from elastic to viscous response

From LAOS
~~~~~~~~~

**Primary outputs** [16]_:

- **Strain softening onset**: Critical :math:`\gamma_0` where :math:`G_1'` begins decreasing — identifies linear-to-nonlinear transition
- **Third harmonic ratio** :math:`I_{3/1}`: Quantifies strength of nonlinearity
- **MAOS scaling** :math:`I_{3/1} \sim \gamma_0^2`: Material time exponent from intrinsic nonlinearity
- **Lissajous shapes**: Visual nonlinear fingerprint — ellipse distortion at high amplitude
- **Chebyshev coefficients** :math:`e_n, v_n`: Decompose intracycle elastic and viscous nonlinearity

Combined Multi-Protocol Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended fitting sequence:**

1. **SAOS** → :math:`\eta_p, \lambda, \eta_s` (linear parameters, :math:`\alpha`-independent)
2. **Flow curve** → refine :math:`\eta_p, \lambda`; determine :math:`\alpha` from thinning shape
3. **Normal stresses** → fix :math:`\alpha = -2N_2/N_1` (most direct route)
4. **Startup** → validate overshoot predictions, refine :math:`\alpha`
5. **Relaxation/Creep** → confirm time constants, validate nonlinear response

**Parameter-to-data mapping:**

.. list-table::
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Data type
     - :math:`\eta_p`
     - :math:`\lambda`
     - :math:`\alpha`
     - :math:`\eta_s`
     - Strength
   * - SAOS
     - ✓
     - ✓
     - —
     - ✓
     - Best for linear params
   * - Flow curve
     - ✓
     - ✓
     - ✓
     - ✓
     - Thinning shape → :math:`\alpha`
   * - :math:`N_1, N_2`
     - —
     - —
     - ✓✓
     - —
     - Most direct :math:`\alpha`
   * - Startup
     - ✓
     - ✓
     - ✓
     - —
     - Overshoot validates model
   * - Relaxation
     - ✓
     - ✓
     - (✓)
     - —
     - Decay rate confirms :math:`\lambda`
   * - Creep
     - ✓
     - ✓
     - (✓)
     - ✓
     - Compliance confirms :math:`G`

(✓✓ = primary route; ✓ = determinable; (✓) = weakly sensitive; — = not accessible)

----

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

----

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
- Converges in 5–10 iterations typically
- May need damping at very high Wi

**ODE integration:**

- Tsit5 (Runge-Kutta 5(4)) for accuracy
- Adaptive step size with PIDController
- rtol=1e-6, atol=1e-8 default tolerances

**Numerical stability:**

- High Wi (>100) may require reduced tolerances
- Very small :math:`\alpha` (<0.01) approaches UCM singularities
- Use log-residuals for fitting flow curves

----

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

**From transient data:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Observable
     - Estimated parameter
   * - Time to steady state
     - :math:`\lambda \approx t_{\text{ss}} / (3\text{–}5)`
   * - Overshoot magnitude
     - Higher :math:`\alpha` → smaller overshoot
   * - Initial slope in startup
     - :math:`G = \eta_p/\lambda` from :math:`d\sigma/d\gamma|_0`

Parameter Estimation Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 25 25 35
   :header-rows: 1

   * - Parameter
     - From SAOS
     - From steady shear
     - From transient
   * - :math:`\eta_p`
     - :math:`\eta_0 - \eta_s`
     - Low-rate plateau − :math:`\eta_s`
     - Initial slope / :math:`\lambda`
   * - :math:`\lambda`
     - :math:`1/\omega_c`
     - :math:`1/\dot{\gamma}_{\text{onset}}`
     - :math:`t_{\text{ss}}/(3\text{–}5)`
   * - :math:`\alpha`
     - — (not accessible)
     - Thinning shape; :math:`-2N_2/N_1`
     - Overshoot ratio
   * - :math:`\eta_s`
     - High-:math:`\omega` :math:`G''/\omega`
     - High-rate plateau
     - —

Fitting Strategy
~~~~~~~~~~~~~~~~

1. **Fix** :math:`\eta_s` **if known** (pure solvent viscosity)
2. **Fit SAOS first** for :math:`\eta_p`, :math:`\lambda` (:math:`\alpha`-independent)
3. **Fit flow curve** to refine and get :math:`\alpha`
4. **Validate with startup** for dynamic behavior

Multi-Mode Fitting
~~~~~~~~~~~~~~~~~~

1. **Discrete spectrum from SAOS**: Fit :math:`G_k, \lambda_k` pairs to :math:`G'(\omega), G''(\omega)` using logarithmically spaced relaxation times
2. **Non-negative least squares (NNLS)**: Ensures :math:`\eta_{p,k} \geq 0`
3. **Tikhonov regularization**: Prevents overfitting when the number of modes exceeds data quality
4. **Fix** :math:`\alpha_k` **after linear fit**: Determine mobility factors from nonlinear data (flow curve, normal stresses)
5. **Typical**: 5–10 modes for 4–6 decades in frequency

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

----

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

----

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

----

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

----

References
----------

.. [1] Giesekus, H. (1982). "A simple constitutive equation for polymer fluids
   based on the concept of deformation-dependent tensorial mobility."
   *J. Non-Newtonian Fluid Mech.*, **11**, 69-109.
   https://doi.org/10.1016/0377-0257(82)85016-7

.. [2] Giesekus, H. (1983). "Stressing behaviour in simple shear flow as
   predicted by a new constitutive model for polymer fluids."
   *J. Non-Newtonian Fluid Mech.*, **12**, 367-374.

.. [3] Bird, R.B., Armstrong, R.C., & Hassager, O. (1987).
   *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics.* 2nd ed.
   Wiley-Interscience. Chapter 4.

.. [4] Larson, R.G. (1988). *Constitutive Equations for Polymer Melts and Solutions.*
   Butterworths. Chapter 4.

.. [5] Morrison, F.A. (2001). *Understanding Rheology.*
   Oxford University Press. Chapter 9.

.. [6] Macosko, C.W. (1994). *Rheology: Principles, Measurements, and Applications.*
   Wiley-VCH. Chapter 3.

.. [7] Yoo, J.Y., & Choi, H.C. (1989). "On the steady simple shear flows of the
   one-mode Giesekus fluid." *Rheol. Acta*, **28**, 13-24.

.. [8] Schleiniger, G., & Weinacht, R.J. (1991). "Steady Poiseuille flows for a
   Giesekus fluid." *J. Non-Newtonian Fluid Mech.*, **40**, 79-102.

.. [9] Quinzani, L.M., Armstrong, R.C., & Brown, R.A. (1994). "Birefringence and
   laser-Doppler velocimetry (LDV) studies of viscoelastic flow through a
   planar contraction." *J. Non-Newtonian Fluid Mech.*, **52**, 1-36.

.. [10] Magda, J.J., & Baek, S.G. (1994). "Concentrated entangled and semidilute
   entangled polystyrene solutions and the second normal stress difference."
   *Polymer*, **35**, 1187-1194.

.. [11] Lee, C.S., Tripp, B.C., & Magda, J.J. (1992). "Does :math:`N_2` depend on
   the shear rate in polymer melts?" *Rheol. Acta*, **31**, 306-314.

.. [12] Quinzani, L.M., McKinley, G.H., Brown, R.A., & Armstrong, R.C. (1990).
   "Modeling the rheology of polyisobutylene solutions."
   *J. Rheol.*, **34**, 705-748.

.. [13] Debbaut, B., & Crochet, M.J. (1988). "Extensional effects in complex flows."
   *J. Non-Newtonian Fluid Mech.*, **30**, 169-184.

.. [14] Hulsen, M.A., Fattal, R., & Kupferman, R. (2005). "Flow of viscoelastic
   fluids past a cylinder at high Weissenberg number: Stabilized simulations
   using matrix logarithms." *J. Non-Newtonian Fluid Mech.*, **127**, 27-39.

.. [15] Guénette, R., & Fortin, M. (1995). "A new mixed finite element method for
   computing viscoelastic flows." *J. Non-Newtonian Fluid Mech.*, **60**, 27-52.

.. [16] Hyun, K., Wilhelm, M., Klein, C.O., Cho, K.S., Nam, J.G., Ahn, K.H.,
   Lee, S.J., Ewoldt, R.H., & McKinley, G.H. (2011). "A review of nonlinear
   oscillatory shear tests: Analysis and application of large amplitude
   oscillatory shear (LAOS)." *Prog. Polym. Sci.*, **36**, 1697-1753.

.. [17] Dealy, J.M., & Wissbrun, K.F. (1990). *Melt Rheology and its Role in
   Plastics Processing.* Van Nostrand Reinhold.

Further Reading
~~~~~~~~~~~~~~~

- Giesekus, H. (1985). "Constitutive equations for polymer fluids based on the
  concept of configuration-dependent molecular mobility: a generalized
  mean-configuration model." *J. Non-Newtonian Fluid Mech.*, **17**, 349-372.

- Bird, R.B., & Wiest, J.M. (1995). "Constitutive equations for polymeric liquids."
  *Annual Review of Fluid Mechanics*, **27**, 169-193.

- Owens, R.G., & Phillips, T.N. (2002). *Computational Rheology.*
  Imperial College Press. Chapter 3.

- Ewoldt, R.H., & McKinley, G.H. (2010). "On secondary loops in LAOS via
  self-intersection of Lissajous-Bowditch curves."
  *Rheol. Acta*, **49**, 213-219.

----

API References
--------------

- Module: :mod:`rheojax.models.giesekus`
- Class: :class:`rheojax.models.giesekus.GiesekusSingleMode`
- Class: :class:`rheojax.models.giesekus.GiesekusMultiMode`
