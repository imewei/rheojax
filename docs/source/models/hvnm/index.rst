HVNM (Hybrid Vitrimer Nanocomposite Model)
==========================================

This section documents the Hybrid Vitrimer Nanocomposite Model (HVNM) for
nanoparticle-filled vitrimers — polymer networks containing rigid NP fillers
that create an interphase subnetwork with distinct kinetics.


Overview
--------

The HVNM extends the Hybrid Vitrimer Model (HVM) with a **fourth interphase (I)
subnetwork** that forms around nanoparticle surfaces.  The key new physics are:

- **Guth-Gold strain amplification**: :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2`
- **Dual TST kinetics**: independent matrix and interfacial bond exchange rates
- **Interphase volume fraction**: computed from NP geometry (:math:`\phi`, :math:`R_{NP}`, :math:`\delta_m`)
- **Optional interfacial damage with self-healing**: reversible above :math:`T_v^{int}`

The model employs four subnetworks:

1. **Permanent (P)**: Covalent crosslinks with amplified modulus :math:`G_P X(\phi)`
2. **Exchangeable (E)**: Matrix vitrimer bonds with BER/TST kinetics (:math:`G_E`)
3. **Dissociative (D)**: Physical reversible bonds, standard Maxwell (:math:`G_D`)
4. **Interphase (I)**: NP-bound confined polymer with amplified affine deformation (:math:`G_{I,eff} X_I`)

These models are particularly well-suited for:

- Nanoparticle-filled vitrimers and covalent adaptable networks
- Silica/carbon-black reinforced polymer networks
- Materials exhibiting Payne effect (strain-dependent modulus)
- Systems with dual topological freezing temperatures
- Multi-timescale relaxation from matrix vs interfacial exchange


Model Hierarchy
---------------

::

   HVNM Family (extends HVM)
   |
   +-- HVNMLocal (Homogeneous, simple shear)
   |   |
   |   +-- Full HVNM: G_P + G_E + G_D + G_I (4-network)
   |   |   +-- Dual TST kinetics: matrix + interphase
   |   |   +-- Guth-Gold strain amplification
   |   |   +-- Optional interfacial damage with self-healing
   |   |   +-- Optional diffusion modes
   |   |
   |   +-- Limiting Cases (via factory methods):
   |       +-- unfilled_vitrimer(...)         -> phi=0 (recovers HVM)
   |       +-- filled_elastomer(G_P, phi)     -> G_E=0, G_D=0
   |       +-- partial_vitrimer_nc(...)       -> G_D=0
   |       +-- conventional_filled_rubber(...) -> G_E=0, frozen I
   |       +-- matrix_only_exchange(...)      -> frozen interphase


Quick Reference
---------------

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Class**
     - :class:`~rheojax.models.hvnm.HVNMLocal`
   * - **Registry**
     - ``"hvnm_local"``, ``"hvnm"``
   * - **Parameters**
     - 13-25 (depending on feature flags)
   * - **Protocols**
     - Flow curve, SAOS, Startup, Relaxation, Creep, LAOS
   * - **Inheritance**
     - ``BaseModel -> VLBBase -> HVMBase -> HVNMBase -> HVNMLocal``
   * - **Solver**
     - Analytical (SAOS, flow curve) + diffrax ODE (startup, relaxation, creep, LAOS)


When to Use This Model
----------------------

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Behavior
     - HVNM Appropriate?
     - Alternative
   * - NP-filled vitrimer
     - Yes (primary use case)
     - N/A
   * - Unfilled vitrimer
     - Use phi=0 factory
     - HVMLocal (simpler)
   * - Payne effect observed
     - Yes
     - N/A
   * - Multi-timescale relaxation with phi dependence
     - Yes
     - N/A
   * - Filled elastomer (no exchange)
     - Use limiting case
     - VLBMultiNetwork
   * - Single relaxation mode
     - Overkill
     - VLBLocal or Maxwell


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
     - :math:`\sigma_E = \sigma_I = 0` at steady state; :math:`\sigma = \eta_D \dot{\gamma}`
   * - OSCILLATION
     - Analytical
     - Three Maxwell modes + :math:`G_P X` plateau; dual factor-of-2
   * - STARTUP
     - ODE (diffrax)
     - Dual TST overshoot; amplified initial slope
   * - RELAXATION
     - ODE (diffrax)
     - Quad-exponential + :math:`G_P X` plateau
   * - CREEP
     - ODE (diffrax)
     - Three retardation modes; NP reduces compliance
   * - LAOS
     - ODE (diffrax)
     - Payne onset at lower :math:`\gamma_0`; Lissajous + harmonic extraction


Quick Start
-----------

**Full HVNM (4 subnetworks):**

.. code-block:: python

   from rheojax.models import HVNMLocal

   model = HVNMLocal(kinetics="stress", include_dissociative=True)
   model.parameters.set_value("G_P", 5000.0)
   model.parameters.set_value("G_E", 3000.0)
   model.parameters.set_value("G_D", 1000.0)
   model.parameters.set_value("phi", 0.1)
   model.parameters.set_value("beta_I", 3.0)

   # SAOS: three Maxwell modes + amplified plateau
   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega)

   # Startup with dual TST feedback
   t = np.linspace(0.01, 50, 200)
   result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)

**Unfilled vitrimer (recovers HVM):**

.. code-block:: python

   model = HVNMLocal.unfilled_vitrimer(G_P=5000, G_E=3000, G_D=1000)

**Bayesian inference:**

.. code-block:: python

   model = HVNMLocal()
   model.fit(omega, G_star, test_mode="oscillation")
   result = model.fit_bayesian(
       omega, G_star, test_mode="oscillation",
       num_warmup=1000, num_samples=2000,
   )


Key Physics
-----------

**Dual Factor-of-2:**

Both the matrix and interphase effective relaxation times are:

.. math::

   \tau_{E,eff} = \frac{1}{2 k_{BER,0}^{mat}}, \quad
   \tau_{I,eff} = \frac{1}{2 k_{BER,0}^{int}}

This arises because both the distribution tensor and natural-state tensor relax
toward each other at rate :math:`k_{BER}`, so their difference decays at :math:`2k_{BER}`.

**Guth-Gold Strain Amplification:**

Rigid NP inclusions amplify the local strain field:

.. math::

   X(\phi) = 1 + 2.5\phi + 14.1\phi^2

The permanent network experiences amplified stress :math:`\sigma_P = (1-D) G_P X \gamma`,
and the interphase affine deformation is :math:`X_I \dot{\gamma}`.


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   hvnm
   hvnm_knowledge


References
----------

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1-20.

2. Guth, E. (1945). "Theory of filler reinforcement." *J. Appl. Phys.*, 16, 20-25.

3. Smallwood, H.M. (1944). "Limiting law of the reinforcement of rubber."
   *J. Appl. Phys.*, 15(11), 758-766.

4. Denissen, W., Winne, J.M. & Du Prez, F.E. (2016). "Vitrimers: permanent
   organic networks with glass-like fluidity." *Chem. Sci.*, 7, 30-38.

5. Payne, A.R. (1962). "The dynamic properties of carbon black-loaded natural
   rubber vulcanizates." *J. Appl. Polym. Sci.*, 6(19), 57-63.
