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

See also
--------

- :doc:`sgr_conventional` — Standard SGR model (simpler, same rheological predictions)
- :doc:`../../transforms/srfs` — Strain-Rate Frequency Superposition transform
- :doc:`../multi_mode/generalized_maxwell` — Alternative multi-mode approach

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

.. [3] Öttinger, H. C. *Beyond Equilibrium Thermodynamics*. Wiley, 2005.
   Comprehensive treatment of GENERIC and nonequilibrium thermodynamics.

.. [4] Sollich, P. & Cates, M. E. "Thermodynamic interpretation of soft glassy rheology models."
   *Physical Review E*, **85**, 031127 (2012).
   https://doi.org/10.1103/PhysRevE.85.031127
