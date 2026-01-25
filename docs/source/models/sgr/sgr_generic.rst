.. _model-sgr-generic:

SGR GENERIC (Thermodynamically Consistent)
==========================================

Quick Reference
---------------

**Use when:** Thermodynamic consistency required, entropy production analysis, nonequilibrium thermodynamics research
**Parameters:** 4+ (x, G0, tau0, dissipation parameters)
**Key equation:** GENERIC framework with Poisson bracket + friction matrix
**Test modes:** Oscillation, relaxation, nonequilibrium thermodynamics validation
**Material examples:** Same as SGR Conventional, plus thermodynamic model validation

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{z}`
     - State variables (probability distribution :math:`P(E, l, t)`)
   * - :math:`E(\mathbf{z})`
     - Total energy functional (Hamiltonian)
   * - :math:`S(\mathbf{z})`
     - Entropy functional (Boltzmann form)
   * - :math:`F(\mathbf{z})`
     - Free energy :math:`F = E - xS`
   * - :math:`\mathbf{L}`
     - Poisson bracket matrix (antisymmetric, reversible dynamics)
   * - :math:`\mathbf{M}`
     - Friction matrix (symmetric positive semidefinite, dissipative dynamics)
   * - :math:`\dot{S}_{\text{prod}}`
     - Entropy production rate (non-negative by construction)
   * - :math:`P(E, l, t)`
     - Distribution of trap depth :math:`E` and local strain :math:`l`
   * - :math:`\rho(E)`
     - Trap density of states :math:`e^{-E}`
   * - :math:`x`
     - Effective noise temperature (control parameter)
   * - :math:`Z`
     - Partition function

Overview
--------

The SGR GENERIC model extends the conventional Soft Glassy Rheology framework to satisfy
the GENERIC (General Equation for Nonequilibrium Reversible-Irreversible Coupling)
structure—a mathematically rigorous framework for nonequilibrium thermodynamics [1]_.

The GENERIC formulation, developed by Grmela and Öttinger [2]_, guarantees:

1. **First law**: Total energy conservation
2. **Second law**: Non-negative entropy production
3. **Onsager symmetry**: Dissipative couplings satisfy reciprocal relations
4. **Clear separation**: Reversible (Hamiltonian) and irreversible (dissipative) dynamics

This implementation follows Fuereder and Ilg's thermodynamically consistent reformulation
of the SGR model [1]_, enabling entropy production calculations and thermodynamic validation.

Physical Foundations
--------------------

The GENERIC framework extends the SGR mesoscopic trap model with rigorous
nonequilibrium thermodynamic structure. All physical foundations from the
conventional SGR model apply (see :doc:`sgr_conventional`), with the addition
of thermodynamically consistent separation of reversible and irreversible
dynamics.

**Key additions beyond conventional SGR:**

1. **Energy-Entropy Split**: Dynamics are decomposed into Hamiltonian (energy-conserving)
   and dissipative (entropy-producing) contributions
2. **Degeneracy Conditions**: Mathematical constraints that enforce the first and
   second laws of thermodynamics
3. **Fluctuation-Dissipation**: Automatic consistency with equilibrium statistical
   mechanics when :math:`x = k_B T`

The physical interpretation of traps, yielding, and effective temperature remains
identical to conventional SGR. The GENERIC formulation adds mathematical rigor
and enables calculation of thermodynamic quantities (entropy production, free energy).

Theoretical Foundation
----------------------

GENERIC Structure
~~~~~~~~~~~~~~~~~

The GENERIC framework describes the time evolution of state variables :math:`\mathbf{z}` as:

.. math::

   \frac{d\mathbf{z}}{dt} = \mathbf{L}(\mathbf{z}) \cdot \frac{\partial E}{\partial \mathbf{z}}
                         + \mathbf{M}(\mathbf{z}) \cdot \frac{\partial S}{\partial \mathbf{z}}

where:
   - :math:`E(\mathbf{z})` is the total energy (Hamiltonian)
   - :math:`S(\mathbf{z})` is the entropy
   - :math:`\mathbf{L}` is the **Poisson bracket matrix** (antisymmetric, reversible dynamics)
   - :math:`\mathbf{M}` is the **friction matrix** (symmetric positive semidefinite, dissipative dynamics)

The two generators must satisfy **degeneracy conditions**:

.. math::

   \mathbf{L} \cdot \frac{\partial S}{\partial \mathbf{z}} = 0 \quad \text{(entropy unchanged by reversible dynamics)}

   \mathbf{M} \cdot \frac{\partial E}{\partial \mathbf{z}} = 0 \quad \text{(energy unchanged by dissipation)}

These conditions ensure thermodynamic consistency: reversible processes conserve entropy,
while irreversible processes conserve energy.

State Variables for SGR
~~~~~~~~~~~~~~~~~~~~~~~

For the SGR model, the state variable is the probability distribution :math:`P(E, l, t)`
of finding mesoscopic elements with trap depth :math:`E` and local strain :math:`l`.

The macroscopic observables are moments of this distribution:

- **Stress**: :math:`\sigma = k \int dE \, dl \, l \, P(E, l, t)`
- **Energy**: :math:`U = \frac{1}{2}k \int dE \, dl \, l^2 \, P(E, l, t)`

Energy Functional
~~~~~~~~~~~~~~~~~

The internal energy of the SGR system is:

.. math::

   E[P] = \int_0^\infty dE \int_{-\infty}^{\infty} dl \, P(E, l) \left[ E + \frac{1}{2}kl^2 \right]

The first term represents trap potential energy, and the second is elastic strain energy.

Entropy Functional
~~~~~~~~~~~~~~~~~~

The entropy is given by the Boltzmann form:

.. math::

   S[P] = -\int_0^\infty dE \int_{-\infty}^{\infty} dl \, P(E, l) \ln\frac{P(E, l)}{\rho(E)}

where :math:`\rho(E) = e^{-E}` is the trap density of states.

The equilibrium distribution maximizing :math:`S` at fixed :math:`E` is:

.. math::

   P_{\text{eq}}(E, l) = \frac{1}{Z} \rho(E) \exp\left(\frac{E - \frac{1}{2}kl^2}{x}\right) \delta(l)

where :math:`Z` is the partition function.

Free Energy
~~~~~~~~~~~

The nonequilibrium free energy is:

.. math::

   F[P] = E[P] - x \, S[P] = x \int dE \, dl \, P \ln\frac{P}{\rho} + \frac{1}{2}k \int dE \, dl \, l^2 P

This Helmholtz-like free energy uses the effective noise temperature :math:`x` instead of
thermal temperature.

Poisson Bracket (Reversible Dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Poisson bracket for SGR describes convective transport under affine deformation:

.. math::

   \{A, B\} = \int dE \, dl \, P(E, l) \left[
       \frac{\delta A}{\delta P} \frac{\partial}{\partial l}\frac{\delta B}{\delta P}
     - \frac{\delta B}{\delta P} \frac{\partial}{\partial l}\frac{\delta A}{\delta P}
   \right] \dot{\gamma}

This generates the affine strain rate contribution :math:`\dot{l} = \dot{\gamma}` for all elements.

**Key properties**:
   - Antisymmetry: :math:`\{A, B\} = -\{B, A\}`
   - Jacobi identity satisfied
   - Energy conserved: :math:`\{S, E\} = 0`
   - Generates Hamiltonian (time-reversible) dynamics

Friction Matrix (Irreversible Dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The friction matrix describes activated hopping (yielding) processes:

.. math::

   \mathbf{M} \cdot \frac{\delta S}{\delta P} = \int dE' \, M(E, E', l) \left[
       \frac{\delta S}{\delta P(E', 0)} - \frac{\delta S}{\delta P(E, l)}
   \right]

where the transition kernel :math:`M(E, E', l)` satisfies:

.. math::

   M(E, E', l) = \rho(E') \Gamma_0 \exp\left(-\frac{E - \frac{1}{2}kl^2}{x}\right) P(E, l)

This describes elements yielding from state :math:`(E, l)` to a new trap with depth :math:`E'`
drawn from :math:`\rho(E')`, resetting their strain to zero.

**Key properties**:
   - Symmetric positive semidefinite: :math:`\mathbf{M} \geq 0`
   - Satisfies detailed balance with :math:`\rho(E)`
   - Generates entropy production
   - Energy conserved: :math:`\mathbf{M} \cdot \delta E / \delta P = 0`

Entropy Production
~~~~~~~~~~~~~~~~~~

The entropy production rate is always non-negative:

.. math::

   \dot{S}_{\text{prod}} = \frac{\delta S}{\delta P}^\top \cdot \mathbf{M} \cdot \frac{\delta S}{\delta P} \geq 0

This is guaranteed by the positive semidefiniteness of :math:`\mathbf{M}`.

Explicitly:

.. math::

   \dot{S}_{\text{prod}} = \int dE \, dl \, \Gamma(E, l) P(E, l) \left[
       \frac{E - \frac{1}{2}kl^2}{x} + \ln\frac{P(E, l)}{P_{\text{eq}}(E)}
   \right]

At equilibrium, :math:`P = P_{\text{eq}}` and :math:`\dot{S}_{\text{prod}} = 0`.

Governing Equations
-------------------

The GENERIC time evolution equation for the state variable :math:`\mathbf{z}` is:

.. math::

   \frac{d\mathbf{z}}{dt} = \mathbf{L}(\mathbf{z}) \cdot \frac{\partial E}{\partial \mathbf{z}}
                         + \mathbf{M}(\mathbf{z}) \cdot \frac{\partial S}{\partial \mathbf{z}}

This structure guarantees:

- **First law**: :math:`\dot{E} = \mathbf{L} \cdot \partial E/\partial \mathbf{z} \cdot \partial E/\partial \mathbf{z} + \mathbf{M} \cdot \partial E/\partial \mathbf{z} \cdot \partial S/\partial \mathbf{z} = 0`
  (by degeneracy condition :math:`\mathbf{M} \cdot \partial E/\partial \mathbf{z} = 0`)

- **Second law**: :math:`\dot{S} = \mathbf{M} \cdot \partial S/\partial \mathbf{z} \cdot \partial S/\partial \mathbf{z} \geq 0`
  (by positive semidefiniteness of :math:`\mathbf{M}`)

For the SGR system, :math:`\mathbf{z} = P(E, l, t)` is the distribution function, and the
energy and entropy functionals are given in the theoretical foundation section.

Constitutive Equations
----------------------

The GENERIC formulation reproduces all the linear and nonlinear constitutive equations
of conventional SGR (see :doc:`sgr_conventional`), with the additional guarantee of
thermodynamic consistency.

Linear Response (Oscillatory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as conventional SGR:

.. math::

   G^*(\omega) = G_0 \frac{\Gamma(1-x)(i\omega\tau_0)^{x-1}}{1 + \Gamma(1-x)(i\omega\tau_0)^{x-1}}

The GENERIC formulation allows verification that this response satisfies:
   - Kramers-Kronig relations (causality)
   - Fluctuation-dissipation theorem
   - Non-negative dissipation at all frequencies

Nonequilibrium Steady States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under constant shear rate :math:`\dot{\gamma}`, the system reaches a nonequilibrium steady
state (NESS) with constant entropy production:

.. math::

   \dot{S}_{\text{prod}}^{\text{NESS}} = \frac{\sigma \dot{\gamma}}{x}

where :math:`\sigma` is the steady-state stress. This connects mechanical dissipation
(:math:`\sigma \dot{\gamma}`) to thermodynamic entropy production.

----

Rheological Analysis Equations
------------------------------

The GENERIC formulation validates the following constitutive laws for analyzing
rheometer data. The control parameter is the noise temperature :math:`x`
(:math:`x=1` is the glass transition).

Measurement Protocols
~~~~~~~~~~~~~~~~~~~~~

**Steady Rotation (Flow Curve)**:

.. math::
   \dot{\gamma}(t) = \dot{\gamma} = \text{constant}

**Stress Relaxation (Step Strain)**:

.. math::
   \gamma(t) = \gamma_0 H(t) \quad (\text{small step strain } \gamma_0)

**Creep (Step Stress)**:

.. math::
   \sigma(t) = \sigma_0 H(t)

**Oscillatory Shear (SAOS)**:

.. math::
   \gamma(t) = \gamma_0 e^{i\omega t}

Flow Curve (Rotation)
~~~~~~~~~~~~~~~~~~~~~

**Fluid Regime** (:math:`1 < x < 2`):

.. math::
   \sigma = B \cdot \dot{\gamma}^{x-1}

Power-law shear thinning behavior.

**Glass Regime** (:math:`x < 1`):

.. math::
   \sigma = \sigma_y + A \cdot \dot{\gamma}^{1-x}

Herschel-Bulkley yield stress fluid behavior.

Stress Relaxation
~~~~~~~~~~~~~~~~~

**Fluid Regime** (:math:`1 < x < 2`):

.. math::
   G(t) \sim t^{-(x-1)}

Power-law decay with exponent :math:`x-1`.

**Glass Regime** (:math:`x < 1`):

.. math::
   G(t) \approx G_{\text{plateau}}

Effectively permanent elasticity due to ergodicity breaking.

Creep Compliance
~~~~~~~~~~~~~~~~

**Fluid Regime** (:math:`1 < x < 2`):

.. math::
   J(t) \sim t^{x-1}

Power-law growth with exponent :math:`x-1`.

**Glass Regime** (:math:`x < 1`, :math:`\sigma < \sigma_y`):

.. math::
   J(t) \to \text{constant}

Solid-like response below yield stress.

Oscillatory Shear (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~

**Fluid Regime** (:math:`1 < x < 2`):

.. math::
   G'(\omega) &\propto \omega^{x-1} \\
   G''(\omega) &\propto \omega^{x-1} \\
   \delta &= (x-1)\frac{\pi}{2}

Constant loss angle (phase angle) across frequency.

**Glass Regime** (:math:`x < 1`):

.. math::
   G'(\omega) &\approx \text{constant} \\
   G''(\omega) &\ll G'(\omega)

Solid-like dominance with weak dissipation.

Scaling Summary Table
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: SGR GENERIC Scaling Predictions
   :header-rows: 1
   :widths: 20 20 60

   * - Measurement
     - Regime
     - Scaling Prediction
   * - **Flow Curve**
     - Fluid (:math:`x > 1`)
     - :math:`\sigma \sim \dot{\gamma}^{x-1}`
   * -
     - Glass (:math:`x < 1`)
     - :math:`\sigma = \sigma_y + A\dot{\gamma}^{1-x}`
   * - **Relaxation**
     - Fluid (:math:`x > 1`)
     - :math:`G(t) \sim t^{-(x-1)}`
   * -
     - Glass (:math:`x < 1`)
     - :math:`G(t) \approx G_{\text{plateau}}`
   * - **Creep**
     - Fluid (:math:`x > 1`)
     - :math:`J(t) \sim t^{x-1}`
   * -
     - Glass (:math:`\sigma < \sigma_y`)
     - :math:`J(t) \to \text{const}`
   * - **Oscillation**
     - Fluid (:math:`x > 1`)
     - :math:`G', G'' \sim \omega^{x-1}`, :math:`\tan\delta = \tan((x-1)\pi/2)`

----

LAOS Extensions
---------------

To capture Large Amplitude Oscillatory Shear (LAOS) nonlinearities, the standard
yielding rate :math:`\Gamma` is modified with strain-dependent extensions.

Extended Master Equation
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \frac{\partial P}{\partial t} + \dot{\gamma}\frac{\partial P}{\partial \ell}
   = -\Gamma_{\text{LAOS}}(E,\ell) P + Y(t)\rho(E)\delta(\ell)

The LAOS yield rate includes a strain-enhancement factor :math:`h(\ell)`:

.. math::
   \Gamma_{\text{LAOS}}(E, \ell) = \Gamma_0 \, h(\ell) \, \exp\left[ -\frac{E - \frac{1}{2}k\ell^2}{x} \right]

Model A: Strain-Activated Hopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enhances yielding at large absolute strains. Useful for fitting **stress overshoots**
in start-up flow:

.. math::
   h(\ell) = 1 + \left( \frac{|\ell|}{\gamma_c} \right)^\nu

Parameters:
   - :math:`\gamma_c`: Critical strain (typically 0.1–1.0)
   - :math:`\nu`: Power-law exponent (typically 1 or 2)

Model B: Mechanical Fluidization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The noise temperature :math:`x` becomes a dynamic variable driven by power input:

.. math::
   x(t) = x_{\text{thermal}} + \mu \left| \sigma(t) \dot{\gamma}(t) \right|

Parameters:
   - :math:`\mu`: Fluidization susceptibility

This captures **shear banding** and **viscosity bifurcations**.

LAOS Observables
~~~~~~~~~~~~~~~~

**Fourier-Chebyshev Coefficients**:

.. math::
   e_3 \approx \frac{G'_3}{G'_1} \quad \text{(elastic nonlinearity)}

.. math::
   v_3 \approx \frac{G''_3}{G''_1} \quad \text{(viscous nonlinearity)}

**Lissajous-Bowditch Figures**:
   - Elastic projection (:math:`\sigma` vs :math:`\gamma`): Ellipse → Parallelogram
   - Viscous projection (:math:`\sigma` vs :math:`\dot{\gamma}`): Ellipse → Sigmoidal

.. note::
   For LAOS simulations with strain-dependent extensions, **use the PDE solver**
   (Population Balance) rather than analytic solutions, which no longer exist in
   closed form.

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 15 12 12 18 43

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``x``
     - :math:`x`
     - —
     - :math:`0 < x < 3`
     - Effective noise temperature
   * - ``G0``
     - :math:`G_0`
     - Pa
     - :math:`G_0 > 0`
     - Elastic modulus scale
   * - ``tau0``
     - :math:`\tau_0`
     - s
     - :math:`\tau_0 > 0`
     - Attempt time (inverse of :math:`\Gamma_0`)
   * - ``k``
     - :math:`k`
     - Pa
     - :math:`k > 0`
     - Local elastic constant (often set equal to :math:`G_0`)

What You Can Learn
------------------

The GENERIC formulation of SGR extends all insights from the conventional model with rigorous thermodynamic validation capabilities. The same parameters (x, G₀, τ₀) appear, but with additional thermodynamic interpretation.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**x (Effective Noise Temperature)**:
   In the GENERIC framework, x is the configurational temperature that couples energy and entropy evolution.

   *For graduate students*: x appears in the free energy F = E - xS as the Lagrange multiplier enforcing the constraint that reversible dynamics (Poisson bracket) conserve entropy while irreversible dynamics (friction matrix) conserve energy. The degeneracy conditions L·∂S/∂z = 0 and M·∂E/∂z = 0 ensure thermodynamic consistency. The glass transition at x = 1 is where the entropy functional S[P] becomes unbounded, making equilibrium impossible.

   *For practitioners*: Same interpretation as conventional SGR (x < 1 is glass, 1 < x < 2 is power-law fluid, x ≥ 2 is Newtonian), but GENERIC provides additional validation: you can check that entropy production Ṡ_prod ≥ 0 for all deformation histories. If this fails, your fitted parameters are thermodynamically inconsistent.

**G₀ (Plateau Modulus)**:
   The elastic modulus scale, appearing in the energy functional E[P].

   *For graduate students*: G₀ sets the elastic strain energy contribution ½k∫l²P(E,l)dEdl to the total energy. In GENERIC, the Poisson bracket generates affine deformation (∂_t l = γ̇), which is purely reversible (energy-conserving). The friction matrix generates yielding (l → 0), which is dissipative (entropy-producing).

   *For practitioners*: Same as conventional SGR. GENERIC adds the guarantee that the predicted G₀ is consistent with thermodynamic stability (positive definite friction matrix M ≥ 0).

**τ₀ (Attempt Time)**:
   The microscopic timescale for dissipative transitions.

   *For graduate students*: In GENERIC, 1/τ₀ = Γ₀ appears in the friction matrix M as the rate coefficient for yield transitions. The detailed balance condition ensures that M satisfies Onsager reciprocity, connecting forward and reverse transition rates via the equilibrium distribution ρ(E).

   *For practitioners*: Same as conventional SGR. GENERIC provides validation that τ₀ is consistent with equilibrium fluctuation-dissipation (FDT) relations if x equals thermal temperature.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SGR GENERIC Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - x Range
     - Thermodynamic State
     - Typical Materials
     - GENERIC Diagnostics
   * - **x < 0.5**
     - Deep non-equilibrium glass
     - Highly aged colloidal gels, arrested emulsions, structural glasses
     - Large Ṡ_prod during flow, multiple F[P] minima, no normalizable P_eq(E)
   * - **0.5 < x < 1**
     - Marginal non-equilibrium glass
     - Fresh colloidal suspensions, carbopol gels, foams
     - Moderate Ṡ_prod, metastable F[P], equilibrium exists but unreachable
   * - **1 < x < 1.5**
     - Near-equilibrium fluid
     - Dilute emulsions, soft foams, near-critical suspensions
     - Low Ṡ_prod, single F[P] minimum, FDT approximately satisfied
   * - **1.5 < x < 2**
     - Equilibrium fluid
     - Surfactant solutions, polymer-colloid mixtures
     - Ṡ_prod → 0 as γ̇ → 0, FDT satisfied, unique equilibrium state
   * - **x ≥ 2**
     - Thermalized fluid
     - Dilute suspensions, simple liquids
     - Exponential relaxation to equilibrium, full FDT, thermal noise dominates

Thermodynamic Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

**Entropy Production Rate**: Quantifies irreversibility during flow

.. math::

   \dot{S}_{\text{prod}} = \frac{\sigma \dot{\gamma}}{x} \geq 0

- **Zero for quiescent aging**: :math:`\dot{S}_{\text{prod}} = 0` when :math:`\dot{\gamma} = 0`
- **Positive during flow**: Mechanical dissipation drives entropy production
- **Connection to Rayleighian**: :math:`\dot{S}_{\text{prod}}` can be derived from a variational principle

**Free Energy Landscape**: Compute the nonequilibrium free energy :math:`F = E - xS`:

- **Glass phase (x < 1)**: Multiple metastable minima (aging attracts system to deeper states)
- **Fluid phase (x > 1)**: Single global minimum (equilibrium state)
- **Barrier heights**: Quantify activation energies for structural rearrangements

**Fluctuation-Dissipation Verification**: Check consistency between:

- Storage modulus :math:`G'(\omega)` (energy storage)
- Loss modulus :math:`G''(\omega)` (dissipation)
- Thermal fluctuation spectrum (from :math:`S`)

Nonequilibrium Steady States (NESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under constant shear rate, the system reaches a NESS with:

.. math::

   \dot{S}_{\text{prod}}^{\text{NESS}} = \frac{\sigma_{ss} \dot{\gamma}}{x}

This connects the mechanical stress to the rate of configurational entropy production.

**Interpretation**: The effective temperature :math:`x` acts as a "configurational
temperature" that converts mechanical dissipation into entropy. Higher :math:`x`
means the system is more disordered, so the same dissipation produces less entropy.

Model Validation Beyond Rheology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GENERIC formulation allows validation against:

1. **Calorimetry**: Specific heat related to :math:`c_0` parameter
2. **Structural probes**: Trap distribution :math:`\rho(E)` vs X-ray PDF analysis
3. **Thermodynamic inequalities**: Clausius-Duhem inequality satisfaction
4. **Onsager reciprocity**: Symmetry of dissipative couplings

**When to use GENERIC**: If you need to verify that your constitutive model
satisfies fundamental thermodynamic laws, or if you want to compute entropy
production and free energy evolution during complex deformation histories.

Validity and Assumptions
------------------------

Same as conventional SGR, plus:

- **GENERIC structure**: Satisfies all degeneracy conditions
- **Thermodynamic consistency**: Guaranteed non-negative entropy production
- **Detailed balance**: Transition rates satisfy equilibrium condition

Applications
------------

Entropy Production Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRGeneric

   model = SGRGeneric(x=1.3, G0=100.0, tau0=0.01)

   # Compute entropy production under steady shear
   gamma_dot = 1.0  # s^-1
   S_dot = model.entropy_production(gamma_dot)

   print(f"Entropy production rate: {S_dot:.4f} J/(K·m³·s)")

   # Verify second law
   assert S_dot >= 0, "Second law violated!"

Free Energy Landscape
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SGRGeneric

   model = SGRGeneric(x=0.8, G0=100.0, tau0=0.01)

   # Compute free energy as function of mean strain
   strains = np.linspace(-1, 1, 100)
   F = model.free_energy(strains)

   # For x < 1, free energy has multiple minima (glassy metastability)

Fluctuation-Dissipation Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRGeneric
   import numpy as np

   model = SGRGeneric(x=1.5, G0=100.0, tau0=0.01)
   omega = np.logspace(-2, 2, 50)

   # Compute complex modulus
   G_star = model.predict(omega, test_mode='oscillation')
   G_pp = np.imag(G_star)

   # Fluctuation-dissipation: G'' relates to thermal fluctuations
   # For SGR: G''(omega) ~ omega * S_omega / (x * omega)
   # where S_omega is the strain fluctuation spectrum
   fdt_ratio = model.verify_fluctuation_dissipation(omega)

   print(f"FDT ratio (should be ~1): {fdt_ratio:.3f}")

Fitting Guidance
----------------

Same strategy as conventional SGR (see :doc:`sgr_conventional`), with additional
thermodynamic validation steps:

1. **Initial fit**: Use NLSQ with oscillatory or flow curve data to estimate :math:`x, G_0, \tau_0`
2. **Verify GENERIC structure**: Call ``validate_generic_structure()`` to check all degeneracy conditions
3. **Check entropy production**: Ensure :math:`\dot{S}_{\text{prod}} \geq 0` for all flow rates
4. **Bayesian with thermodynamic priors**: Use informative priors based on calorimetric data if available

**Troubleshooting**: If GENERIC validation fails, the parameter values may be
unphysical (e.g., negative :math:`\tau_0` or :math:`G_0`). Re-fit with tighter bounds.

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SGRGeneric

   # Create GENERIC-consistent SGR model
   model = SGRGeneric()

   # Fit to oscillatory data
   omega = np.logspace(-2, 2, 50)
   model.fit(omega, G_star_data, test_mode='oscillation')

   # Access thermodynamic functions
   E = model.internal_energy()
   S = model.entropy()
   F = model.free_energy()

   print(f"Internal energy: {E:.2e} J/m³")
   print(f"Entropy: {S:.2e} J/(K·m³)")
   print(f"Free energy: {F:.2e} J/m³")

Thermodynamic Validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRGeneric

   model = SGRGeneric(x=1.3, G0=100.0, tau0=0.01)

   # Verify GENERIC structure
   validation = model.validate_generic_structure()

   print(f"Poisson bracket antisymmetry: {validation['poisson_antisymmetric']}")
   print(f"Friction matrix positive: {validation['friction_positive']}")
   print(f"Energy degeneracy: {validation['energy_degeneracy']}")
   print(f"Entropy degeneracy: {validation['entropy_degeneracy']}")

   # All should be True for thermodynamic consistency

Comparison with Conventional SGR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRConventional, SGRGeneric
   import numpy as np

   # Both models give identical rheological predictions
   omega = np.logspace(-2, 2, 50)

   conv = SGRConventional(x=1.3, G0=100.0, tau0=0.01)
   generic = SGRGeneric(x=1.3, G0=100.0, tau0=0.01)

   G_conv = conv.predict(omega, test_mode='oscillation')
   G_generic = generic.predict(omega, test_mode='oscillation')

   # Predictions match
   np.testing.assert_allclose(G_conv, G_generic, rtol=1e-10)

   # But SGRGeneric provides additional thermodynamic information
   S_prod = generic.entropy_production(gamma_dot=1.0)
   print(f"Entropy production: {S_prod:.4f}")

See Also
--------

- :doc:`sgr_conventional` — Standard SGR model (simpler, same rheological predictions; start here)
- :doc:`../../transforms/srfs` — Strain-Rate Frequency Superposition transform
- :doc:`../multi_mode/generalized_maxwell` — Alternative multi-mode approach for viscoelastic spectra
- :doc:`../fractional/fractional_maxwell_gel` — Fractional models for power-law gels (alternative to SGR for x ~ 1.5)

**Related advanced models:**

- :doc:`../stz/stz_conventional` — STZ theory (effective temperature formulation, similar aging physics)
- :doc:`../fluidity/fluidity_saramito_local` — Fluidity models (phenomenological thixotropy)

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.SGRGeneric`

References
----------

.. [1] Fuereder, I. & Ilg, P. "GENERIC treatment of soft glassy rheology."
   *Physical Review E*, **88**, 042134 (2013).
   https://doi.org/10.1103/PhysRevE.88.042134

.. [2] Grmela, M. & Öttinger, H. C. "Dynamics and thermodynamics of complex fluids. I.
   Development of a general formalism." *Physical Review E*, **56**, 6620 (1997).
   https://doi.org/10.1103/PhysRevE.56.6620

.. [3] Öttinger, H. C. *Beyond Equilibrium Thermodynamics*. Wiley (2005).
   ISBN: 978-0471666585

.. [4] Sollich, P. & Cates, M. E. "Thermodynamic interpretation of soft glassy rheology models."
   *Physical Review E*, **85**, 031127 (2012).
   https://doi.org/10.1103/PhysRevE.85.031127

.. [5] Sollich, P. "Rheological constitutive equation for a model of soft glassy materials."
   *Physical Review E*, **58**, 738 (1998).
   https://doi.org/10.1103/PhysRevE.58.738
.. [6] Grmela, M. "Bracket formulation of dissipative fluid mechanics equations."
   *Physics Letters A*, **102**, 355-358 (1984).
   https://doi.org/10.1016/0375-9601(84)90297-4

.. [7] Morrison, P. J. "Bracket formulation for irreversible classical fields."
   *Physics Letters A*, **100**, 423-427 (1984).
   https://doi.org/10.1016/0375-9601(84)90635-2

.. [8] Beris, A. N. & Edwards, B. J. *Thermodynamics of Flowing Systems with Internal Microstructure*.
   Oxford University Press (1994). ISBN: 978-0195076943

.. [9] Fielding, S. M., Sollich, P., & Cates, M. E. "Aging and rheology in soft materials."
   *Journal of Rheology*, **44**, 323-369 (2000).
   https://doi.org/10.1122/1.551088

.. [10] Nicolas, A., Ferrero, E. E., Martens, K., & Barrat, J.-L. "Deformation and flow of amorphous solids: Insights from elastoplastic models."
   *Reviews of Modern Physics*, **90**, 045006 (2018).
   https://doi.org/10.1103/RevModPhys.90.045006

