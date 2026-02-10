HVM Model Reference
===================

This page provides the complete mathematical formulation and implementation
details for the Hybrid Vitrimer Model (HVM).

.. note::

   HVM builds on VLB transient network theory.  For foundational distribution
   tensor derivations and the governing ODE, see :doc:`/models/vlb/vlb`.  For
   shared protocol methodology (flow curve, SAOS, startup, etc.), see
   :doc:`/models/vlb/vlb_protocols`.  The D-network in HVM follows the same
   evolution equation as a single VLB transient network.


Quick Reference
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Full name**
     - Hybrid Vitrimer Model
   * - **Class**
     - :class:`~rheojax.models.hvm.HVMLocal`
   * - **Base class**
     - :class:`~rheojax.models.hvm._base.HVMBase` (extends VLBBase)
   * - **Registry names**
     - ``"hvm_local"``, ``"hvm"``
   * - **Parameters**
     - 6 (base) + 2 (dissociative) + 2 (damage) = 6-10
   * - **TST coupling**
     - ``kinetics="stress"`` (default) or ``"stretch"``
   * - **ODE solver**
     - diffrax Tsit5 with adaptive stepping
   * - **Bayesian**
     - Full NLSQ -> NUTS pipeline


Notation Guide
--------------

.. list-table::
   :widths: 15 50 15 20
   :header-rows: 1

   * - Symbol
     - Meaning
     - Units
     - Typical Range
   * - :math:`G_P`
     - Permanent network modulus
     - Pa
     - :math:`10^2` -- :math:`10^6`
   * - :math:`G_E`
     - Exchangeable network modulus
     - Pa
     - :math:`10^2` -- :math:`10^6`
   * - :math:`G_D`
     - Dissociative network modulus
     - Pa
     - :math:`10^1` -- :math:`10^5`
   * - :math:`\nu_0`
     - TST attempt frequency
     - 1/s
     - :math:`10^8` -- :math:`10^{12}`
   * - :math:`E_a`
     - Activation energy for BER
     - J/mol
     - 40k -- 150k
   * - :math:`V_{act}`
     - Activation volume
     - m\ :sup:`3`/mol
     - :math:`10^{-7}` -- :math:`10^{-4}`
   * - :math:`T`
     - Temperature
     - K
     - 250 -- 450
   * - :math:`k_d^D`
     - Dissociative bond rate
     - 1/s
     - :math:`10^{-3}` -- :math:`10^3`
   * - :math:`k_{BER}`
     - Bond exchange rate
     - 1/s
     - (computed from TST)
   * - :math:`\boldsymbol{\mu}^E`
     - E-network distribution tensor
     - --
     - (state variable)
   * - :math:`\boldsymbol{\mu}^E_{nat}`
     - E-network natural-state tensor
     - --
     - (state variable)
   * - :math:`\boldsymbol{\mu}^D`
     - D-network distribution tensor
     - --
     - (state variable)
   * - :math:`D`
     - Damage variable
     - --
     - [0, 1]
   * - :math:`R`
     - Universal gas constant
     - J/(mol K)
     - 8.314


Three-Subnetwork Architecture
-----------------------------

The HVM decomposes the total stress as:

.. math::

   \boldsymbol{\sigma} = \boldsymbol{\sigma}_P + \boldsymbol{\sigma}_E + \boldsymbol{\sigma}_D

**Permanent (P) network:**

Neo-Hookean elastic response from covalent crosslinks:

.. math::

   \boldsymbol{\sigma}_P = (1 - D) \, G_P (\mathbf{B} - \mathbf{I})

where :math:`\mathbf{B}` is the left Cauchy-Green tensor and :math:`D` is the
damage variable. In simple shear at strain :math:`\gamma`:

.. math::

   \sigma_{P,xy} = (1 - D) \, G_P \, \gamma

**Exchangeable (E) network:**

Stress arises from the deviation of :math:`\boldsymbol{\mu}^E` from its
evolving natural state :math:`\boldsymbol{\mu}^E_{nat}`:

.. math::

   \boldsymbol{\sigma}_E = G_E (\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})

This is the vitrimer hallmark: the natural state is **not** the identity tensor
but evolves to track the current deformation via BER.

**Dissociative (D) network:**

Standard upper-convected Maxwell:

.. math::

   \boldsymbol{\sigma}_D = G_D (\boldsymbol{\mu}^D - \mathbf{I})


Evolution Equations (Simple Shear)
----------------------------------

**E-network distribution tensor** (:math:`\boldsymbol{\mu}^E`):

.. math::

   \dot{\mu}^E_{xx} &= 2 \dot{\gamma} \, \mu^E_{xy} + k_{BER}(\mu^E_{nat,xx} - \mu^E_{xx}) \\
   \dot{\mu}^E_{yy} &= k_{BER}(\mu^E_{nat,yy} - \mu^E_{yy}) \\
   \dot{\mu}^E_{xy} &= \dot{\gamma} \, \mu^E_{yy} + k_{BER}(\mu^E_{nat,xy} - \mu^E_{xy})

**E-network natural-state tensor** (:math:`\boldsymbol{\mu}^E_{nat}`):

.. math::

   \dot{\mu}^E_{nat,ij} = k_{BER}(\mu^E_{ij} - \mu^E_{nat,ij})

This coupled evolution is the vitrimer hallmark: the natural state continuously
drifts toward the current state at rate :math:`k_{BER}`.

**D-network distribution tensor** (:math:`\boldsymbol{\mu}^D`):

.. math::

   \dot{\mu}^D_{xx} &= 2 \dot{\gamma} \, \mu^D_{xy} - k_d^D(\mu^D_{xx} - 1) \\
   \dot{\mu}^D_{yy} &= -k_d^D(\mu^D_{yy} - 1) \\
   \dot{\mu}^D_{xy} &= \dot{\gamma} \, \mu^D_{yy} - k_d^D \, \mu^D_{xy}

This is identical to the VLB single-network evolution.


TST Kinetics
------------

**Thermal BER rate** (zero stress):

.. math::

   k_{BER,0} = \nu_0 \exp\!\left(-\frac{E_a}{R T}\right)

**Stress-coupled BER rate** (``kinetics="stress"``):

.. math::

   k_{BER} = k_{BER,0} \cosh\!\left(\frac{V_{act} \, \sigma_{VM}^E}{R T}\right)

where :math:`\sigma_{VM}^E` is the von Mises stress of the E-network:

.. math::

   \sigma_{VM}^E = G_E \sqrt{(\sigma^E_{xx})^2 + (\sigma^E_{yy})^2
   - \sigma^E_{xx} \sigma^E_{yy} + 3(\sigma^E_{xy})^2}

with :math:`\sigma^E_{ij} = \mu^E_{ij} - \mu^E_{nat,ij}`.

**Stretch-coupled BER rate** (``kinetics="stretch"``):

.. math::

   \lambda_E = \sqrt{\frac{\text{tr}(\boldsymbol{\mu}^E)}{3}}
   \qquad
   k_{BER} = k_{BER,0} \cosh\!\left(\frac{V_{act} \, G_E(\lambda_E - 1)}{R T}\right)


Cooperative Damage Shielding
-----------------------------

When ``include_damage=True``, the damage variable :math:`D \in [0, 1]` evolves
as:

.. math::

   \dot{D} = \Gamma_0 (1 - D) \max(0, \lambda_{chain} - \lambda_{crit})

where:

- :math:`\Gamma_0` is the damage rate coefficient
- :math:`\lambda_{crit}` is the critical stretch for damage onset
- :math:`\lambda_{chain} = \sqrt{\text{tr}(\mathbf{B})/3}` is the effective chain stretch

The :math:`(1 - D)` factor provides cooperative shielding: damage slows as the
network degrades, preventing unphysical complete fracture.


.. _hvm-factor-of-2:

Factor-of-2 in Relaxation
--------------------------

For the E-network under constant :math:`k_{BER}`, the stress difference
:math:`\Delta\mu_{ij} = \mu^E_{ij} - \mu^E_{nat,ij}` satisfies:

.. math::

   \dot{\Delta\mu}_{ij} = -2 k_{BER} \, \Delta\mu_{ij}

Therefore the E-network contribution to stress relaxes with effective time
constant:

.. math::

   \tau_{E,eff} = \frac{1}{2 k_{BER,0}}

A naive Maxwell fit yields :math:`\tau_{fit} = \tau_{E,eff} = \tau_E / 2`.
This is a fundamental vitrimer signature.


Protocol Summary
-----------------

The HVM supports six rheological protocols.  For complete derivations and
closed-form solutions, see :doc:`hvm_protocols`.

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Protocol
     - Method
     - Key Result
   * - :ref:`Flow Curve <hvm-flow-curve>`
     - Analytical
     - :math:`\sigma_E = 0` at steady state; :math:`\sigma^{ss} = G_P\gamma + \eta_D\dot{\gamma}`
   * - :ref:`SAOS <hvm-saos>`
     - Analytical
     - Two Maxwell modes + :math:`G_P` plateau; :math:`\hat{\tau}_E = 1/(2k_{BER,0})`
   * - :ref:`Startup <hvm-startup>`
     - ODE
     - TST creates stress overshoot; analytical constant-rate solution
   * - :ref:`Relaxation <hvm-relaxation>`
     - ODE
     - Bi-exponential + :math:`G_P` plateau; TST gives KWW-like decay
   * - :ref:`Creep <hvm-creep>`
     - ODE
     - Two retardation times + vitrimer plastic creep :math:`\delta J(t)`
   * - :ref:`LAOS <hvm-laos>`
     - ODE
     - TST generates odd harmonics; :math:`N_1` at :math:`2\omega`

For the HVM vs VLB comparison and the thermodynamic framework, see
:doc:`hvm_advanced`.


Limiting Cases
--------------

.. list-table::
   :widths: 25 20 30 25
   :header-rows: 1

   * - Case
     - Parameters
     - Equivalent Model
     - Factory Method
   * - Neo-Hookean
     - :math:`G_E=0, G_D=0`
     - Pure elastic
     - ``HVMLocal.neo_hookean(G_P)``
   * - Maxwell / VLBLocal
     - :math:`G_P=0, G_E=0`
     - Single Maxwell element (equivalent to :class:`~rheojax.models.vlb.VLBLocal`)
     - ``HVMLocal.maxwell(G_D, k_d_D)``
   * - Zener (SLS)
     - :math:`G_E=0`
     - Standard Linear Solid
     - ``HVMLocal.zener(G_P, G_D, k_d_D)``
   * - Pure vitrimer
     - :math:`G_P=0, G_D=0`
     - E-network only
     - ``HVMLocal.pure_vitrimer(G_E, ...)``
   * - Partial vitrimer
     - :math:`G_D=0`
     - Meng et al. (2019)
     - ``HVMLocal.partial_vitrimer(G_P, G_E, ...)``
   * - Full HVM
     - All :math:`> 0`
     - Full 3-network
     - ``HVMLocal()``


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
     - Exchangeable network modulus (vitrimer bonds)
   * - ``nu_0``
     - 1e10
     - (1e6, 1e14)
     - 1/s
     - TST attempt frequency
   * - ``E_a``
     - 80e3
     - (20e3, 200e3)
     - J/mol
     - Activation energy for bond exchange
   * - ``V_act``
     - 1e-5
     - (1e-8, 1e-2)
     - m\ :sup:`3`/mol
     - Activation volume (mechanochemical coupling)
   * - ``T``
     - 300
     - (200, 500)
     - K
     - Temperature (typically fixed)
   * - ``G_D``
     - 1e3
     - (0, 1e8)
     - Pa
     - Dissociative network modulus (when ``include_dissociative=True``)
   * - ``k_d_D``
     - 1.0
     - (1e-6, 1e6)
     - 1/s
     - Dissociative bond rate (when ``include_dissociative=True``)
   * - ``Gamma_0``
     - 1e-4
     - (0, 0.1)
     - 1/s
     - Damage rate coefficient (when ``include_damage=True``)
   * - ``lambda_crit``
     - 2.0
     - (1.001, 10)
     - --
     - Critical stretch for damage onset (when ``include_damage=True``)


Advanced Theory
----------------

For thermodynamic foundations (Helmholtz energy, Clausius-Duhem derivation),
upper-convected kinematics, topological freezing, and numerical implementation
details, see :doc:`hvm_advanced`.

For troubleshooting, cross-protocol validation, and knowledge extraction
workflows, see :doc:`hvm_knowledge`.
