VLB Transient Network Models
============================

This section documents the Vernerey-Long-Brighenti (VLB) family of models
for polymers with dynamic (reversible) cross-links.


.. admonition:: VLB Transient Network Family

   The VLB framework serves as the foundation for a hierarchy of models:

   - **VLB** — Base transient network theory (this section)
   - **HVM** — Hybrid Vitrimer Model: extends VLB with evolving natural-state tensor
     and TST kinetics for vitrimers (:doc:`/models/hvm/index`)
   - **HVNM** — Hybrid Vitrimer Nanocomposite Model: extends HVM with a 4th
     interphase subnetwork for NP-filled vitrimers (:doc:`/models/hvnm/index`)

   Inheritance: ``BaseModel → VLBBase → HVMBase → HVNMBase``


Overview
--------

The VLB framework (Vernerey, Long & Brighenti, 2017) provides a statistically
grounded continuum theory for transient polymer networks.  Starting from the
chain end-to-end vector distribution :math:`\varphi(\mathbf{r},t)`, one derives
a second-moment **distribution tensor** :math:`\boldsymbol{\mu}` whose
evolution is governed by:

.. math::

   \dot{\boldsymbol{\mu}} = k_d(\mathbf{I} - \boldsymbol{\mu})
   + \mathbf{L} \cdot \boldsymbol{\mu}
   + \boldsymbol{\mu} \cdot \mathbf{L}^T

where :math:`k_d` is the bond dissociation rate and :math:`\mathbf{L}` is the
velocity gradient.  The Cauchy stress is:

.. math::

   \boldsymbol{\sigma} = G_0 (\boldsymbol{\mu} - \mathbf{I}) + p\mathbf{I}

With constant :math:`k_d` the single-network model is **exactly Maxwell**:
relaxation time :math:`t_R = 1/k_d`, zero-shear viscosity
:math:`\eta_0 = G_0 / k_d`.  All six standard protocols admit closed-form
solutions.  The multi-network variant extends this to a generalized Maxwell
spectrum with M transient networks, an optional permanent network, and solvent
viscosity.


Model Variants
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model
     - Description
   * - :class:`~rheojax.models.vlb.VLBLocal`
     - Single transient network (2 params: :math:`G_0, k_d`). All protocols analytical.
   * - :class:`~rheojax.models.vlb.VLBMultiNetwork`
     - M transient + optional permanent network + solvent viscosity (:math:`2M+1` or :math:`2M+2` params).
   * - :class:`~rheojax.models.vlb.VLBVariant`
     - Bell breakage + FENE-P + temperature flags (2-6 params). Shear thinning, bounded extension.
   * - :class:`~rheojax.models.vlb.VLBNonlocal`
     - Spatially-resolved PDE with tensor diffusion (4-6 params). Shear banding detection.


Model Hierarchy
---------------

::

   VLB Family (4 Classes)
   │
   ├── VLBLocal (Single transient network)
   │   ├── Parameters: G₀, k_d
   │   ├── Relaxation time: t_R = 1/k_d
   │   ├── Viscosity: η₀ = G₀/k_d
   │   └── All 6 protocols: analytical (LAOS via ODE)
   │
   ├── VLBMultiNetwork (Generalized Maxwell)
   │   ├── N transient networks: {G_I, k_d_I} for I = 0..N-1
   │   ├── Optional permanent network: G_e (include_permanent=True)
   │   ├── Solvent viscosity: η_s
   │   ├── Relaxation spectrum: G(t) = G_e + Σ G_I e^{-k_d_I·t}
   │   └── Creep: ODE-based (analytical for 1 transient + permanent)
   │
   ├── VLBVariant (Bell + FENE-P + Temperature)
   │   ├── Bell breakage: k_d(μ) = k_d₀·exp(ν·(λ_c - 1))
   │   ├── FENE-P stress: σ = G₀·f(tr(μ))·(μ - I)
   │   ├── Temperature: Arrhenius k_d(T), G_0(T) = G_0_ref·T/T_ref
   │   └── All 6 protocols via ODE (SAOS analytical)
   │
   └── VLBNonlocal (Spatial PDE)
       ├── 1D gap-resolved PDE with tensor diffusion D_μ∇²μ
       ├── Shear banding detection and band width analysis
       ├── Cooperativity length: ξ = √(D_μ/k_d₀)
       └── Protocols: steady shear, startup, creep


When to Use Which Model
-----------------------

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Behavior
     - VLBLocal
     - VLBMultiNetwork
     - VLBVariant
     - VLBNonlocal
   * - Single relaxation time
     - Use this
     - Overkill
     - Use for nonlinear
     - N/A
   * - Shear thinning
     - No
     - No
     - Yes (Bell)
     - Yes (Bell)
   * - Bounded extension
     - No
     - No
     - Yes (FENE)
     - Yes (FENE)
   * - Shear banding
     - No
     - No
     - No
     - Use this
   * - Temperature effects
     - No
     - No
     - Yes (Arrhenius)
     - No
   * - Broad spectrum
     - Cannot capture
     - Use this
     - No
     - No
   * - Fewest parameters
     - 2 params
     - :math:`2N+1` or :math:`2N+2`
     - 2-6 params
     - 4-6 params


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
     - Newtonian: :math:`\sigma = (G_0/k_d) \dot{\gamma}`
   * - OSCILLATION
     - Analytical
     - Maxwell SAOS: :math:`G'(\omega), G''(\omega)`
   * - STARTUP
     - Analytical
     - :math:`\sigma(t) = (G_0 \dot{\gamma}/k_d)(1 - e^{-k_d t})`
   * - RELAXATION
     - Analytical
     - :math:`G(t) = G_0 e^{-k_d t}`
   * - CREEP
     - Analytical / ODE
     - Single: :math:`J(t) = (1 + k_d t)/G_0`; multi: ODE
   * - LAOS
     - ODE (diffrax)
     - Linear :math:`\sigma_{12}`, :math:`N_1` has :math:`2\omega` harmonics


Quick Start
-----------

**Single network:**

.. code-block:: python

   from rheojax.models import VLBLocal

   model = VLBLocal()
   model.fit(omega, G_star, test_mode='oscillation')

   # Properties
   print(f"G₀ = {model.G0:.1f} Pa")
   print(f"k_d = {model.k_d:.3f} 1/s")
   print(f"t_R = {model.relaxation_time:.3f} s")
   print(f"η₀ = {model.viscosity:.1f} Pa·s")

**Multi-network:**

.. code-block:: python

   from rheojax.models import VLBMultiNetwork

   model = VLBMultiNetwork(n_modes=3, include_permanent=True)
   model.fit(omega, G_star, test_mode='oscillation')

   # Relaxation spectrum
   spectrum = model.get_relaxation_spectrum()
   for G_i, t_R_i in spectrum:
       print(f"G = {G_i:.1f} Pa, t_R = {t_R_i:.3f} s")

**Bayesian inference:**

.. code-block:: python

   model = VLBLocal()
   model.fit(omega, G_star, test_mode='oscillation')  # NLSQ warm start
   result = model.fit_bayesian(
       omega, G_star, test_mode='oscillation',
       num_warmup=1000, num_samples=2000,
   )


Relation to TNT Models
----------------------

VLB and TNT both describe transient polymer networks but differ in their
tensorial variables and derivation:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - VLB
     - TNT
   * - State variable
     - Distribution tensor :math:`\boldsymbol{\mu} = \langle \mathbf{r r} \rangle / \langle r_0^2 \rangle`
     - Conformation tensor :math:`\mathbf{S} = \langle \mathbf{Q Q} \rangle`
   * - Derivation
     - Statistical distribution :math:`\varphi(\mathbf{r},t)`
     - Network theory (Green-Tobolsky)
   * - Equilibrium
     - :math:`\boldsymbol{\mu}_{eq} = \mathbf{I}`
     - :math:`\mathbf{S}_{eq} = \mathbf{I}`
   * - Stress
     - :math:`\boldsymbol{\sigma} = G_0(\boldsymbol{\mu} - \mathbf{I})`
     - :math:`\boldsymbol{\sigma} = G(\mathbf{S} - \mathbf{I})`
   * - With constant :math:`k_d`
     - Maxwell (identical predictions)
     - Maxwell (identical predictions)
   * - Extensions
     - Langevin chains, Bell :math:`k_d(\mu)`
     - Bell, FENE-P, non-affine, loop-bridge

At the constant-:math:`k_d` level the models are **mathematically equivalent**.
The VLB formulation provides a clearer path to molecular extensions (Langevin
chains, entropy-based :math:`k_d`).


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   vlb
   vlb_variant
   vlb_nonlocal
   vlb_protocols
   vlb_knowledge
   vlb_advanced


References
----------

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1-20.

2. Green, M.S. & Tobolsky, A.V. (1946). "A New Approach to the Theory of
   Relaxing Polymeric Media." *J. Chem. Phys.*, 14(2), 80-92.

3. Tanaka, F. & Edwards, S.F. (1992). "Viscoelastic properties of physically
   crosslinked networks." *J. Non-Newtonian Fluid Mech.*, 43(2-3), 247-271.
