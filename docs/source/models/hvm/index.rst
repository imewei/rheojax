HVM (Hybrid Vitrimer Model)
===========================

This section documents the Hybrid Vitrimer Model (HVM) for polymers with
permanent covalent crosslinks, associative (vitrimer-type) exchangeable bonds,
and optionally dissociative (physical) reversible bonds.


Overview
--------

The HVM extends the VLB transient network framework (Vernerey, Long & Brighenti,
2017) with an **evolving natural-state tensor** that captures plastic flow
through topology rearrangement without network breakdown.  The key physics is
**bond exchange reactions (BER)** accelerated by Transition State Theory (TST)
kinetics, where mechanical stress lowers the activation barrier for bond
exchange.

The model employs three subnetworks:

1. **Permanent (P)**: Covalent crosslinks, neo-Hookean elastic (:math:`G_P`)
2. **Exchangeable (E)**: Associative vitrimer bonds with BER kinetics (:math:`G_E`)
3. **Dissociative (D)**: Physical reversible bonds, standard Maxwell (:math:`G_D`)

These models are particularly well-suited for:

- Covalent adaptable networks (CANs) and vitrimers
- Self-healing polymers with dynamic bonds
- Shape-memory polymers with topology rearrangement
- Multi-mechanism polymer networks
- Temperature-dependent viscoelastic materials


Model Hierarchy
---------------

::

   HVM Family
   |
   +-- HVMLocal (Homogeneous, simple shear)
   |   |
   |   +-- Full HVM: G_P + G_E + G_D (3-network)
   |   |   +-- TST kinetics: stress or stretch coupling
   |   |   +-- Optional cooperative damage shielding
   |   |
   |   +-- Limiting Cases (via factory methods):
   |       +-- neo_hookean(G_P)       -> G_E=0, G_D=0
   |       +-- maxwell(G_D, k_d_D)   -> G_P=0, G_E=0
   |       +-- zener(G_P, G_D, ...)  -> G_E=0 (SLS)
   |       +-- pure_vitrimer(G_E, ...)    -> G_P=0, G_D=0
   |       +-- partial_vitrimer(G_P, G_E, ...) -> G_D=0 (Meng 2019)


Quick Reference
---------------

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Class**
     - :class:`~rheojax.models.hvm.HVMLocal`
   * - **Registry**
     - ``"hvm_local"``, ``"hvm"``
   * - **Parameters**
     - 6-10 (depending on options)
   * - **Protocols**
     - Flow curve, SAOS, Startup, Relaxation, Creep, LAOS
   * - **Inheritance**
     - ``BaseModel -> VLBBase -> HVMBase -> HVMLocal``
   * - **Solver**
     - Analytical (SAOS, flow curve) + diffrax ODE (startup, relaxation, creep, LAOS)


When to Use This Model
----------------------

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Behavior
     - HVM Appropriate?
     - Alternative
   * - Vitrimer with BER kinetics
     - Yes (primary use case)
     - N/A
   * - Permanent + exchangeable network
     - Yes (partial vitrimer)
     - VLBMultiNetwork (if no BER)
   * - Single Maxwell relaxation
     - Use limiting case
     - VLBLocal (simpler)
   * - Associating polymer (no BER)
     - Use D-network only
     - TNT or VLB models
   * - TST stress-enhanced exchange
     - Yes
     - N/A
   * - Temperature-dependent relaxation
     - Yes (Arrhenius BER rate)
     - TTS transforms


Supported Protocols
-------------------

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Protocol
     - Method
     - Notes
   * - FLOW_CURVE
     - Analytical
     - :math:`\sigma_E = 0` at steady state; :math:`\sigma = G_P \gamma + \eta_D \dot{\gamma}`
   * - OSCILLATION
     - Analytical
     - Two Maxwell modes + :math:`G_P` plateau; :math:`\tau_{E,eff} = 1/(2k_{BER,0})`
   * - STARTUP
     - ODE (diffrax)
     - TST creates stress overshoot; analytical for constant-rate
   * - RELAXATION
     - ODE (diffrax)
     - Bi-exponential + :math:`G_P` plateau; TST gives non-exponential decay
   * - CREEP
     - ODE (diffrax)
     - Two retardation modes; vitrimer plastic creep at intermediate times
   * - LAOS
     - ODE (diffrax)
     - TST generates odd harmonics; Lissajous curves + harmonic extraction


Quick Start
-----------

**Full HVM (3 subnetworks):**

.. code-block:: python

   from rheojax.models import HVMLocal

   model = HVMLocal(kinetics="stress", include_dissociative=True)
   model.parameters.set_value("G_P", 5000.0)
   model.parameters.set_value("G_E", 3000.0)
   model.parameters.set_value("G_D", 1000.0)

   # SAOS
   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega, return_components=False)

   # Startup with TST feedback
   t = np.linspace(0.01, 50, 200)
   result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)

**Partial vitrimer (Meng 2019):**

.. code-block:: python

   model = HVMLocal.partial_vitrimer(G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3)

**Bayesian inference:**

.. code-block:: python

   model = HVMLocal()
   model.fit(omega, G_star, test_mode="oscillation")     # NLSQ warm start
   result = model.fit_bayesian(
       omega, G_star, test_mode="oscillation",
       num_warmup=1000, num_samples=2000,
   )


Key Physics: Factor-of-2
-------------------------

The exchangeable network effective relaxation time is:

.. math::

   \tau_{E,eff} = \frac{1}{2 k_{BER,0}}

The factor-of-2 arises because both the distribution tensor :math:`\boldsymbol{\mu}^E`
and the natural-state tensor :math:`\boldsymbol{\mu}^{E}_{nat}` relax toward each
other at rate :math:`k_{BER}`, so their **difference** (which determines stress)
decays at :math:`2 k_{BER}`.  This is the vitrimer hallmark: a standard Maxwell
fit to relaxation data will yield :math:`\tau_{fit} = \tau_E / 2`, not :math:`\tau_E`.


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   hvm
   hvm_knowledge


References
----------

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1-20.

2. Meng, F., Simon, S.L., Niu, R., McKenna, G.B. & Hallinan, D.T. (2019).
   "Poly(ethylene oxide)/Poly(ether-thiourea) vitrimers from a single monomer."
   *Macromolecules*, 52(8), 3065-3072.

3. Denissen, W., Winne, J.M. & Du Prez, F.E. (2016). "Vitrimers: permanent
   organic networks with glass-like fluidity." *Chem. Sci.*, 7, 30-38.

4. Dyre, J.C. (2006). "Colloquium: The glass transition and relaxation in
   supercooled liquids and glasses." *Rev. Mod. Phys.*, 78, 953.

5. Montarnal, D., Capelot, M., Tournilhac, F. & Leibler, L. (2011).
   "Silica-like malleable materials from permanent organic networks."
   *Science*, 334, 965-968.
