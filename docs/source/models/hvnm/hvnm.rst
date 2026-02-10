HVNMLocal — Full Model Reference
=================================

.. note::

   HVNM extends HVM (:doc:`/models/hvm/hvm`), which builds on VLB
   (:doc:`/models/vlb/vlb`).  The E-network and D-network equations are
   identical to HVM.  This page focuses on the **I-network** (interphase)
   additions and HVNM-specific behavior.

Quick Reference
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - Class
     - ``HVNMLocal``
   * - Registry names
     - ``"hvnm_local"``, ``"hvnm"``
   * - Parameters
     - 15 (base) to 25 (all flags on)
   * - Parent class
     - ``HVNMBase(HVMBase(VLBBase(BaseModel)))``
   * - Feature flags
     - ``include_dissociative``, ``include_damage``, ``include_interfacial_damage``, ``include_diffusion``
   * - ODE state
     - 17 or 18 components (simple shear)


Notation Guide
--------------

.. list-table::
   :widths: 15 20 10 55
   :header-rows: 1

   * - Symbol
     - Parameter
     - Units
     - Description
   * - :math:`G_P`
     - ``G_P``
     - Pa
     - Permanent (covalent) network modulus
   * - :math:`G_E`
     - ``G_E``
     - Pa
     - Exchangeable (vitrimer) network modulus
   * - :math:`G_D`
     - ``G_D``
     - Pa
     - Dissociative (physical) network modulus
   * - :math:`\nu_0`
     - ``nu_0``
     - 1/s
     - Matrix TST attempt frequency
   * - :math:`E_a`
     - ``E_a``
     - J/mol
     - Matrix activation energy
   * - :math:`V_{act}`
     - ``V_act``
     - m³/mol
     - Matrix activation volume
   * - :math:`T`
     - ``T``
     - K
     - Temperature
   * - :math:`k_{d,D}`
     - ``k_d_D``
     - 1/s
     - Dissociative rate constant
   * - :math:`\beta_I`
     - ``beta_I``
     - —
     - Interphase reinforcement ratio :math:`G_I / G_E`
   * - :math:`\nu_0^{int}`
     - ``nu_0_int``
     - 1/s
     - Interfacial TST attempt frequency
   * - :math:`E_a^{int}`
     - ``E_a_int``
     - J/mol
     - Interfacial activation energy
   * - :math:`V_{act}^{int}`
     - ``V_act_int``
     - m³/mol
     - Interfacial activation volume
   * - :math:`\phi`
     - ``phi``
     - —
     - NP volume fraction
   * - :math:`R_{NP}`
     - ``R_NP``
     - m
     - NP radius
   * - :math:`\delta_m`
     - ``delta_m``
     - m
     - Mobile interphase thickness
   * - :math:`D`
     - ``D``
     - —
     - Permanent-network damage variable :math:`\in [0,1]`
   * - :math:`D_{int}`
     - ``D_int``
     - —
     - Interfacial damage variable :math:`\in [0,1]`
   * - :math:`\phi_I`
     - (derived)
     - —
     - Interphase volume fraction from NP geometry
   * - :math:`X(\phi)`
     - (derived)
     - —
     - Guth-Gold strain amplification factor
   * - :math:`G_{I,eff}`
     - (derived)
     - Pa
     - Effective interphase modulus :math:`\beta_I G_E \phi_I`
   * - :math:`k_{diff}`
     - ``k_diff``
     - 1/s
     - Diffusion-limited slow mode rate (``include_diffusion=True``)
   * - :math:`h_{int}`
     - (derived)
     - 1/s
     - Interfacial self-healing rate (Arrhenius)
   * - :math:`\Gamma_0`
     - ``Gamma_0``
     - 1/s
     - Damage rate coefficient (``include_damage=True``)


Physical Foundations
--------------------

**4-Subnetwork Architecture**

The HVNM Cauchy stress in simple shear is:

.. math::

   \sigma_{tot} = \underbrace{(1-D) G_P X(\phi) \gamma}_{\text{permanent}}
   + \underbrace{G_E (\mu^E_{xy} - \mu^{E,nat}_{xy})}_{\text{exchangeable}}
   + \underbrace{G_D (\mu^D_{xy} - \delta_{xy})}_{\text{dissociative}}
   + \underbrace{(1-D_{int}) G_{I,eff} X_I (\mu^I_{xy} - \mu^{I,nat}_{xy})}_{\text{interphase}}

where:

- :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2` is the Guth-Gold strain amplification
- :math:`G_{I,eff} = \beta_I G_E \phi_I` is the effective interphase modulus
- :math:`\phi_I` is the interphase volume fraction from NP geometry
- :math:`X_I = X(\phi_I)` is the interphase amplification factor

**Dual TST Kinetics**

Matrix and interfacial BER rates are independent:

.. math::

   k_{BER}^{mat} &= \nu_0 \exp\!\left(-\frac{E_a}{RT}\right) \cosh\!\left(\frac{V_{act} \sigma_{VM}^E}{RT}\right) \\
   k_{BER}^{int} &= \nu_0^{int} \exp\!\left(-\frac{E_a^{int}}{RT}\right) \cosh\!\left(\frac{V_{act}^{int} \sigma_{VM}^I}{RT}\right)

**I-Network Evolution**

The interphase distribution tensor evolves with amplified affine deformation:

.. math::

   \dot{\mu}^I_{xy} = X_I \dot{\gamma} (\mu^I_{xx} + 1)/2 - k_{BER}^{int}(\mu^I_{xy} - \mu^{I,nat}_{xy})

The I-network natural-state tensor evolves symmetrically with the E-network:

.. math::

   \dot{\mu}^{I,nat}_{ij} = k_{BER}^{int}(\mu^I_{ij} - \mu^{I,nat}_{ij})

.. _hvnm-dual-factor-of-2:

Dual Factor-of-2
^^^^^^^^^^^^^^^^^

This coupled evolution gives the same **factor-of-2** as the E-network
(:ref:`hvm-factor-of-2`):
the I-network stress relaxes with :math:`\tau_I = 1/(2k_{BER,0}^{int})`.

**How HVNM Differs from HVM:**

- **P-network**: modulus amplified by :math:`X(\phi)` — rigid inclusions
  increase effective strain
- **I-network**: entirely new fourth subnetwork with independent TST kinetics
- **Steady state**: both :math:`\sigma_E = 0` and :math:`\sigma_I = 0`
  (all natural states track deformation)
- **SAOS**: three Maxwell modes instead of two (E, D, I) plus amplified plateau
- **Parameter count**: 15-25 vs HVM's 6-10
- **Damage**: optional interfacial damage :math:`D_{int}` with self-healing
  (see :ref:`hvnm-damage-mechanics`)
- **Diffusion**: optional slow mode :math:`k_{diff}` for long-time relaxation
  tail (see :ref:`hvnm-diffusion-mode`)


Interphase Volume Fraction
--------------------------

The interphase volume fraction is computed from NP geometry:

.. math::

   \phi_I = \phi \left[\left(\frac{R_{NP} + \delta_m}{R_{NP}}\right)^3 - 1\right]

For dilute suspensions (:math:`\phi < 0.2`), the interphase shells do not
overlap.  At higher :math:`\phi`, percolation occurs when :math:`\phi_I`
exceeds a critical threshold.  See :ref:`hvnm-interphase-model` for the
full three-layer interphase model and percolation analysis.

**Guth-Gold Strain Amplification:**

.. math::

   X(\phi) = 1 + 2.5\phi + 14.1\phi^2

This applies to the P-network modulus (:math:`G_P X(\phi)`) and to the
interphase amplification (:math:`X_I = X(\phi_I)`).  The quadratic term
captures hydrodynamic interactions between NPs.


Parameter Table
---------------

.. list-table::
   :widths: 12 12 15 10 51
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``G_P``
     - 1e4
     - (0, 1e9)
     - Pa
     - Permanent network modulus (covalent crosslinks)
   * - ``G_E``
     - 1e4
     - (0, 1e9)
     - Pa
     - Exchangeable network modulus (matrix vitrimer bonds)
   * - ``nu_0``
     - 1e10
     - (1e6, 1e14)
     - 1/s
     - Matrix TST attempt frequency
   * - ``E_a``
     - 80e3
     - (20e3, 200e3)
     - J/mol
     - Matrix activation energy for BER
   * - ``V_act``
     - 1e-5
     - (1e-8, 1e-2)
     - m³/mol
     - Matrix activation volume
   * - ``T``
     - 300
     - (200, 500)
     - K
     - Temperature
   * - ``phi``
     - 0.05
     - (0.0, 0.5)
     - --
     - NP volume fraction
   * - ``R_NP``
     - 20e-9
     - (1e-9, 1e-6)
     - m
     - NP radius
   * - ``delta_m``
     - 10e-9
     - (1e-9, 1e-7)
     - m
     - Mobile interphase thickness
   * - ``beta_I``
     - 3.0
     - (1.0, 10.0)
     - --
     - Interphase reinforcement ratio :math:`G_I/G_E`
   * - ``nu_0_int``
     - 1e10
     - (1e6, 1e14)
     - 1/s
     - Interfacial TST attempt frequency
   * - ``E_a_int``
     - 90e3
     - (30e3, 250e3)
     - J/mol
     - Interfacial activation energy (typically > :math:`E_a`)
   * - ``V_act_int``
     - 5e-6
     - (1e-8, 1e-2)
     - m³/mol
     - Interfacial activation volume
   * - ``G_D``
     - 1e3
     - (0, 1e8)
     - Pa
     - Dissociative network modulus (``include_dissociative=True``)
   * - ``k_d_D``
     - 1.0
     - (1e-6, 1e6)
     - 1/s
     - Dissociative bond rate (``include_dissociative=True``)
   * - ``Gamma_0``
     - 1e-4
     - (0, 0.1)
     - 1/s
     - Damage rate coefficient (``include_damage=True``)
   * - ``lambda_crit``
     - 2.0
     - (1.001, 10)
     - --
     - Critical stretch for damage onset (``include_damage=True``)
   * - ``Gamma_0_int``
     - 1e-3
     - (0, 1.0)
     - 1/s
     - Interfacial damage rate (``include_interfacial_damage=True``)
   * - ``lambda_crit_int``
     - 1.5
     - (1.001, 5.0)
     - --
     - Interfacial critical stretch (``include_interfacial_damage=True``)
   * - ``h_0``
     - 1e-4
     - (0.0, 1.0)
     - 1/s
     - Interfacial healing prefactor (``include_interfacial_damage=True``)
   * - ``E_a_heal``
     - 100e3
     - (30e3, 300e3)
     - J/mol
     - Healing activation energy (``include_interfacial_damage=True``)
   * - ``k_diff_0_mat``
     - 1e-4
     - (0.0, 1.0)
     - 1/s
     - Matrix diffusion rate constant (``include_diffusion=True``)
   * - ``k_diff_0_int``
     - 1e-6
     - (0.0, 0.1)
     - 1/s
     - Interphase diffusion rate constant (``include_diffusion=True``)
   * - ``E_a_diff``
     - 120e3
     - (50e3, 400e3)
     - J/mol
     - Diffusion activation energy (``include_diffusion=True``)


.. _hvnm-protocol-summary:

Protocol Summary
----------------

For complete derivations and closed-form solutions, see :doc:`hvnm_protocols`.

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Protocol
     - Method
     - Key Result
   * - :ref:`Flow Curve <hvnm-flow-curve>`
     - Analytical
     - :math:`\sigma_E = \sigma_I = 0` at steady state; :math:`\sigma^{ss} = (1-D) G_P X \gamma + \eta_D \dot{\gamma}`
   * - :ref:`SAOS <hvnm-saos>`
     - Analytical
     - Three Maxwell modes + :math:`G_P X` plateau; dual factor-of-2
   * - :ref:`Startup <hvnm-startup>`
     - ODE
     - Dual TST overshoot; amplified initial slope :math:`G_{tot}^{NC}`
   * - :ref:`Relaxation <hvnm-relaxation>`
     - ODE
     - Quad-exponential + :math:`G_P X` plateau; optional :math:`k_{diff}` tail
   * - :ref:`Creep <hvnm-creep>`
     - ODE
     - Three retardation modes; NP reduces compliance
   * - :ref:`LAOS <hvnm-laos>`
     - ODE
     - Payne onset at :math:`\gamma_c / X_I`; Lissajous + harmonic extraction


Limiting Cases
--------------

**Factory Methods:**

.. list-table::
   :widths: 25 35 25 15
   :header-rows: 1

   * - Limiting Case
     - Conditions
     - Factory Method
     - Behavior
   * - HVM (unfilled)
     - :math:`\phi = 0`
     - ``unfilled_vitrimer()``
     - Exact HVM
   * - Filled elastomer
     - :math:`G_E = G_D = 0`
     - ``filled_elastomer()``
     - Neo-Hookean + NP
   * - Partial vitrimer NC
     - :math:`G_D = 0`
     - ``partial_vitrimer_nc()``
     - P + E + I
   * - Conventional filled rubber
     - :math:`G_E = 0`, frozen I
     - ``conventional_filled_rubber()``
     - P + D + elastic I
   * - Matrix-only exchange
     - Frozen interphase
     - ``matrix_only_exchange()``
     - P + E + D

**Additional Limiting Regimes:**

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Regime
     - Conditions
     - Physical Interpretation
   * - Low-:math:`T` (glassy)
     - :math:`T < T_v^{mat}`
     - All exchange frozen; elastic solid
   * - Intermediate-:math:`T`
     - :math:`T_v^{mat} < T < T_v^{int}`
     - Matrix relaxes; interphase frozen
   * - High-:math:`T`
     - :math:`T > T_v^{int}`
     - Both networks relax; :math:`G_P X` plateau only
   * - Dilute filler
     - :math:`\phi \ll 0.05`
     - :math:`X \approx 1`, negligible interphase
   * - Percolation
     - :math:`\phi_I > \phi_I^{perc}`
     - Interphase shells overlap, enhanced modulus
   * - Strong confinement
     - :math:`\beta_I \gg 1`
     - I-network dominates at high :math:`\phi`
   * - No damage
     - :math:`D = D_{int} = 0`
     - Conservative system (default)


Advanced Theory
---------------

For thermodynamic foundations (Helmholtz energy with 4 networks + 2 damage
variables), the three-layer interphase model, enhanced damage mechanics with
self-healing, diffusion-limited slow modes, and numerical implementation
details, see :doc:`hvnm_advanced`.

For troubleshooting, cross-protocol validation, knowledge extraction workflows,
and Payne effect interpretation, see :doc:`hvnm_knowledge`.
