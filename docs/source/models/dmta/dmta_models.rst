DMTA Model Selection & Applicability
=====================================

Every oscillation-capable model in RheoJAX works with DMTA data through the
automatic :math:`E^* \leftrightarrow G^*` conversion in ``BaseModel``.
This page provides a complete inventory of compatible models and transforms,
a tiered recommendation guide, and the conversion mechanism.

.. contents:: On This Page
   :local:
   :depth: 2

Recommended Starting Points
----------------------------

If you are new to DMTA analysis in RheoJAX, start here:

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - #
     - Model / Workflow
     - When to Use
   * - 1
     - :class:`~rheojax.models.fractional.FractionalZenerSolidSolid`
     - **Broad glass transitions** in amorphous polymers.  3--5 parameters
       capture the full :math:`T_g` transition with physically meaningful
       fractional order :math:`\alpha`.  Best first choice for single-sweep
       DMTA data.
   * - 2
     - :class:`~rheojax.models.multi_mode.GeneralizedMaxwell`
     - **Any spectrum shape** + FEM export.  N-mode Prony series fits arbitrary
       relaxation spectra.  Use ``modulus_type='tensile'`` for direct
       :math:`E_i` parameters, or default shear with ``deformation_mode='tension'``.
   * - 3
     - :class:`~rheojax.models.hvm.HVMLocal`
     - **Vitrimers** showing topology-freezing transition :math:`T_v` in DMTA.
       Built-in Arrhenius kinetics for bond-exchange, permanent + exchangeable
       crosslink moduli.
   * - 4
     - :class:`~rheojax.transforms.Mastercurve` |rarr|
       :class:`~rheojax.models.fractional.FractionalZenerSolidSolid`
     - **Multi-temperature TTS workflow**.  Collapse multi-T frequency sweeps
       into a master curve, then fit with FZSS for compact parameterisation
       or GMM for FEM export.  Extracts WLF :math:`C_1, C_2` and activation
       energy :math:`E_a`.

.. |rarr| unicode:: U+2192

.. code-block:: python

   # Recommended starting point: FZSS for single-sweep DMTA
   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,   # rubber (0.35 glassy, 0.40 semicrystalline)
   )
   E_pred = model.predict(omega, test_mode='oscillation')  # returns E*

How DMTA Support Works
------------------------

The E* |leftrightarrow| G* conversion is handled by ``BaseModel``, not by
individual models.  This means **all 45 oscillation-capable model registrations
(39 unique classes)** work with DMTA data without any model-level changes.

.. |leftrightarrow| unicode:: U+2194

.. code-block:: text

   fit(ω, E*, test_mode='oscillation', deformation_mode='tension', ν=0.5)
     │
     ├─ 1. Parse deformation_mode → DeformationMode.TENSION
     ├─ 2. Store: self._deformation_mode, self._poisson_ratio
     ├─ 3. is_tensile() → True → convert_modulus(E*, "tension", "shear", ν)
     │     E* ÷ 2(1+ν) → G*
     ├─ 4. Call self._fit(ω, G*, ...)  ← model sees G* only
     └─ 5. Parameters stored in G-space (G₀, not E₀)

   predict(ω, test_mode='oscillation')
     │
     ├─ 1. Call self._predict(ω) → G*
     ├─ 2. Check self._deformation_mode
     ├─ 3. is_tensile() → True → G* × 2(1+ν) → E*
     └─ 4. Return E*

``fit_bayesian()`` follows the same conversion pattern.

.. note::

   Some models are registered under multiple names (aliases).  For example,
   ``hvm`` and ``hvm_local`` both instantiate :class:`~rheojax.models.hvm.HVMLocal`.
   The 45 registrations correspond to 39 unique model classes.

Complete Model Inventory
--------------------------

The following tables list **every** model in RheoJAX and its DMTA compatibility.

Tier 1 — Primary DMTA Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models are the first choice for standard DMTA data analysis.

.. list-table::
   :header-rows: 1
   :widths: 30 10 25 35

   * - Model
     - Params
     - Registry Name(s)
     - Best For
   * - :class:`~rheojax.models.fractional.FractionalZenerSolidSolid`
     - 3--5
     - ``fractional_zener_ss``
     - Amorphous polymers through :math:`T_g`, broad :math:`\tan\delta` peak
   * - :class:`~rheojax.models.multi_mode.GeneralizedMaxwell`
     - 2N+1
     - ``generalized_maxwell``
     - Any polymer, master curves, FEM export (Prony series)
   * - :class:`~rheojax.models.classical.Zener`
     - 3
     - ``zener``
     - Quick fits, narrow relaxation (< 1 decade)
   * - :class:`~rheojax.models.classical.Maxwell`
     - 2
     - ``maxwell``
     - Single-mode baseline comparison
   * - :class:`~rheojax.models.classical.SpringPot`
     - 2
     - ``springpot``
     - Power-law relaxation element

Tier 2 — Temperature-Dependent Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models with built-in Arrhenius or WLF temperature dependence, ideal for
multi-temperature DMTA datasets.

.. list-table::
   :header-rows: 1
   :widths: 30 10 25 35

   * - Model
     - Params
     - Registry Name(s)
     - Best For
   * - :class:`~rheojax.models.vlb.VLBLocal`
     - 2+
     - ``vlb``, ``vlb_local``
     - Crosslinked polymers, Arrhenius :math:`k_d(T)`
   * - :class:`~rheojax.models.vlb.VLBVariant`
     - 4+
     - ``vlb_variant``
     - Advanced VLB with :math:`G(T)`, first-principles TTS
   * - :class:`~rheojax.models.vlb.VLBMultiNetwork`
     - 6+
     - ``vlb_multi_network``
     - Multi-network VLB with independent dissociation
   * - :class:`~rheojax.models.hvm.HVMLocal`
     - 10
     - ``hvm``, ``hvm_local``
     - Vitrimers / CANs showing :math:`T_v` transition
   * - :class:`~rheojax.models.hvnm.HVNMLocal`
     - 15
     - ``hvnm``, ``hvnm_local``
     - NP-filled vitrimers with Payne effect + :math:`T_v`
   * - :class:`~rheojax.models.fikh.FIKH`
     - 8+
     - ``fikh``
     - Thixotropic polymers with T-dependent yield
   * - :class:`~rheojax.models.fikh.FMLIKH`
     - 8+
     - ``fmlikh``
     - Fractional modified Lenoir-IKH

Tier 3 — Specialist Viscoelastic Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models target specific material classes or physics.

.. list-table::
   :header-rows: 1
   :widths: 30 10 25 35

   * - Model
     - Params
     - Registry Name(s)
     - Best For
   * - :class:`~rheojax.models.giesekus.GiesekusSingleMode`
     - 4
     - ``giesekus``, ``giesekus_single``
     - Polymer melts above :math:`T_g`
   * - :class:`~rheojax.models.giesekus.GiesekusMultiMode`
     - 4N
     - ``giesekus_multi``, ``giesekus_multimode``
     - Multi-mode polymer melts
   * - :class:`~rheojax.models.tnt.TNTSingleMode`
     - 5+
     - ``tnt``, ``tnt_single_mode``
     - Telechelic polymers, physically crosslinked hydrogels
   * - :class:`~rheojax.models.sgr.SGRConventional`
     - 3+
     - ``sgr_conventional``
     - Soft glassy materials (foams, pastes, colloidal glasses)
   * - :class:`~rheojax.models.sgr.SGRGeneric`
     - 4+
     - ``sgr_generic``
     - Generalized SGR with GENERIC structure
   * - :class:`~rheojax.models.itt_mct.ITTMCTSchematic`
     - 6+
     - ``itt_mct_schematic``
     - Dense colloids near glass transition
   * - :class:`~rheojax.models.itt_mct.ITTMCTIsotropic`
     - 6+
     - ``itt_mct_isotropic``
     - Isotropic MCT variant
   * - :class:`~rheojax.models.stz.STZConventional`
     - 6+
     - ``stz_conventional``
     - Shear transformation zones (metallic glasses)
   * - :class:`~rheojax.models.hl.HebraudLequeux`
     - 4+
     - ``hebraud_lequeux``
     - Hébraud-Lequeux glassy dynamics
   * - :class:`~rheojax.models.dmt.DMTLocal`
     - 4+
     - ``dmt_local``
     - de Souza Mendes-Thompson thixotropy

Tier 4 — Extended Fractional & Kinematic Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional fractional viscoelastic and kinematic hardening models.

.. list-table::
   :header-rows: 1
   :widths: 30 10 25 35

   * - Model
     - Params
     - Registry Name(s)
     - Best For
   * - :class:`~rheojax.models.fractional.FractionalZenerSolidLiquid`
     - 3--5
     - ``fractional_zener_sl``
     - Solid-to-liquid fractional transition
   * - :class:`~rheojax.models.fractional.FractionalZenerLiquidLiquid`
     - 3--5
     - ``fractional_zener_ll``
     - Liquid-liquid fractional regime
   * - :class:`~rheojax.models.fractional.FractionalMaxwellModel`
     - 3
     - ``fractional_maxwell_model``
     - Simple fractional dashpot
   * - :class:`~rheojax.models.fractional.FractionalMaxwellGel`
     - 3
     - ``fractional_maxwell_gel``
     - Gel-like power-law response
   * - :class:`~rheojax.models.fractional.FractionalMaxwellLiquid`
     - 3
     - ``fractional_maxwell_liquid``
     - Fractional liquid behaviour
   * - :class:`~rheojax.models.fractional.FractionalKelvinVoigt`
     - 3
     - ``fractional_kelvin_voigt``
     - Solid-dominant fractional
   * - :class:`~rheojax.models.fractional.FractionalKelvinVoigtZener`
     - 4--5
     - ``fractional_kv_zener``
     - Kelvin-Voigt + Zener combination
   * - :class:`~rheojax.models.fractional.FractionalBurgersModel`
     - 5--7
     - ``fractional_burgers``
     - Two fractional elements in series
   * - :class:`~rheojax.models.fractional.FractionalJeffreysModel`
     - 5--6
     - ``fractional_jeffreys``
     - Jeffreys model with fractional order
   * - :class:`~rheojax.models.fractional.FractionalPoyntingThomson`
     - 5
     - ``fractional_poynting_thomson``
     - Poynting-Thomson with springpot
   * - :class:`~rheojax.models.ikh.MIKH`
     - 6+
     - ``mikh``
     - Modified isotropic kinematic hardening
   * - :class:`~rheojax.models.ikh.MLIKH`
     - 6+
     - ``ml_ikh``
     - Multi-layer IKH
   * - :class:`~rheojax.models.epm.LatticeEPM`
     - 3+
     - ``lattice_epm``
     - Elastoplastic lattice model
   * - :class:`~rheojax.models.epm.TensorialEPM`
     - 4+
     - ``tensorial_epm``
     - Tensorial elastoplastic model
   * - :class:`~rheojax.models.fluidity.FluidityLocal`
     - 3+
     - ``fluidity_local``
     - Fluidity-based model
   * - :class:`~rheojax.models.fluidity.FluidityNonlocal`
     - 4+
     - ``fluidity_nonlocal``
     - Nonlocal fluidity
   * - :class:`~rheojax.models.fluidity.FluiditySaramitoLocal`
     - 5+
     - ``fluidity_saramito_local``
     - Saramito EVP fluidity

Models NOT Compatible with DMTA
---------------------------------

The following 14 models **cannot** be used with DMTA data because they lack
``Protocol.OSCILLATION`` support.

**Flow-curve models** (steady shear :math:`\eta(\dot\gamma)` only):

- ``power_law`` — Power law
- ``carreau`` — Carreau
- ``carreau_yasuda`` — Carreau-Yasuda
- ``cross`` — Cross
- ``bingham`` — Bingham
- ``herschel_bulkley`` — Herschel-Bulkley

**TNT advanced variants** (startup/flow protocols only):

- ``tnt_cates`` — Cates living polymer model
- ``tnt_sticky_rouse`` — Sticky Rouse
- ``tnt_loop_bridge`` — Loop-bridge topology
- ``tnt_multi_species`` — Multi-species TNT

**Nonlocal PDE models** (spatially resolved, no oscillation):

- ``vlb_nonlocal`` — VLB nonlocal
- ``dmt_nonlocal`` — DMT nonlocal
- ``fluidity_saramito_nonlocal`` — Saramito EVP nonlocal

**Other**:

- ``spp_yield_stress`` — SPP yield stress decomposition (LAOS shear analysis)

.. tip::

   To check DMTA compatibility programmatically:

   .. code-block:: python

      from rheojax.core.registry import ModelRegistry
      from rheojax.core.test_modes import DeformationMode
      from rheojax.core.inventory import Protocol

      # All models supporting tension + oscillation
      models = ModelRegistry.find(
          protocol=Protocol.OSCILLATION,
          deformation_mode=DeformationMode.TENSION,
      )

Transform Applicability for DMTA
-----------------------------------

RheoJAX transforms vary in their DMTA compatibility.  Unlike models, transforms
do **not** have built-in ``deformation_mode`` handling — the conversion must be
managed at the model or utility level.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Transform
     - DMTA Status
     - Notes
   * - :class:`~rheojax.transforms.Mastercurve`
     - **Direct**
     - Modulus-agnostic TTS; works on E* or G* without conversion
   * - ``prony_conversion``
     - **Direct**
     - :math:`E(t) \leftrightarrow E^*(\omega)` via Prony series; amplitude-agnostic
   * - ``spectrum_inversion``
     - **Direct**
     - Recover :math:`H(\tau)` from :math:`E'(\omega)`, :math:`E''(\omega)` directly
   * - ``fft_analysis``
     - After E* |rarr| G*
     - Operates on time-domain signals; convert first
   * - ``lve_envelope``
     - After E* |rarr| G*
     - LVE envelope from :math:`G', G''`
   * - ``mutation_number``
     - After E* |rarr| G*
     - Mutation number from :math:`G'(t)` evolution
   * - ``smooth_derivative``
     - After E* |rarr| G*
     - Numerical derivative (generic utility)
   * - ``cox_merz``
     - **Not applicable**
     - Shear-specific (:math:`\eta^*` vs :math:`\eta`)
   * - ``owchirp``
     - **Not applicable**
     - Shear chirp protocol
   * - ``spp_decomposer``
     - **Not applicable**
     - LAOS shear decomposition
   * - ``srfs``
     - **Not applicable**
     - Shear rate frequency superposition

**Using transforms that require conversion:**

.. code-block:: python

   from rheojax.utils.modulus_conversion import convert_modulus

   # Convert E* to G* before applying the transform
   G_star = convert_modulus(E_star, "tension", "shear", poisson_ratio=0.5)

   # Apply the transform on G*
   from rheojax.transforms import LVEEnvelope
   envelope = LVEEnvelope()
   result = envelope.transform(omega, G_star)

**Modulus conversion presets:**

.. code-block:: python

   from rheojax.utils.modulus_conversion import POISSON_PRESETS

   # Available presets:
   # rubber / elastomer:        0.50
   # glassy_polymer:            0.35
   # semicrystalline:           0.40
   # hydrogel:                  0.50
   # foam:                      0.30
   # metal:                     0.30
   # ceramic:                   0.25
   # biological_tissue:         0.45

Model Complexity Ladder
------------------------

Models are listed from simplest to most expressive.  Start with the simplest
model that captures your data and move up only if residuals are systematic
(see :doc:`dmta_numerical` for convergence criteria):

1. **Maxwell** (2 params) --- Single relaxation time.  Baseline; fails for
   broad transitions.

2. **Zener** (3 params) --- Adds equilibrium modulus :math:`G_e`.
   Plateau--transition--plateau.  Fails if the transition spans > 1 decade.

3. **Fractional Zener** (3--5 params) --- Power-law springpot replaces the
   dashpot.  Captures broad :math:`T_g` transitions with
   :math:`\alpha \in (0, 1)`.

4. **Generalized Maxwell** (2N + 1 params) --- N-mode Prony series.  Fits
   any smooth spectrum.  Use ~1 mode per 3 decades of data.

5. **VLB / HVM / HVNM** (5--15 params) --- Molecular-level models with
   bond-exchange kinetics, temperature dependence, and network topology.
   Use when physical insight (activation energy, crosslink density,
   topology freezing) is the primary goal.

.. tip::

   For a master curve spanning 15--20 decades, the **GMM** with
   ``n_modes=10--20`` is almost always the best practical choice.  Fractional
   models are better when you need compact parameterisations or physical
   interpretation of the relaxation breadth.

Experimental Situation |rarr| Model
--------------------------------------

Choose your model based on the data you have:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Your Situation
     - Recommended Model(s)
   * - Single isothermal frequency sweep
     - :class:`~rheojax.models.classical.Zener` (quick),
       :class:`~rheojax.models.fractional.FractionalZenerSolidSolid` (broad peak),
       :class:`~rheojax.models.multi_mode.GeneralizedMaxwell` (arbitrary)
   * - Multi-temperature frequency sweeps
     - :class:`~rheojax.transforms.Mastercurve` |rarr| GMM or FZSS
   * - Temperature sweep with first-principles TTS
     - :class:`~rheojax.models.vlb.VLBVariant` (Arrhenius :math:`k_d(T)`)
   * - Vitrimer :math:`T_v` transition in DMTA
     - :class:`~rheojax.models.hvm.HVMLocal` or
       :class:`~rheojax.models.hvnm.HVNMLocal` (nanocomposite)
   * - Stress relaxation :math:`E(t)`
     - :class:`~rheojax.models.multi_mode.GeneralizedMaxwell`
       (``test_mode='relaxation'``)
   * - Need FEM Prony series export
     - :class:`~rheojax.models.multi_mode.GeneralizedMaxwell` exclusively
   * - Uncertainty quantification on any model
     - Any model + ``fit_bayesian()`` (NUTS)
   * - Thixotropic yield + temperature
     - :class:`~rheojax.models.fikh.FIKH` with ``thermal=True``
   * - Physically crosslinked hydrogel
     - :class:`~rheojax.models.tnt.TNTSingleMode`
   * - Soft glassy material (foam, paste)
     - :class:`~rheojax.models.sgr.SGRConventional`
   * - Dense colloids near glass transition
     - :class:`~rheojax.models.itt_mct.ITTMCTSchematic`

.. seealso::

   - :doc:`dmta_workflows` --- end-to-end fitting examples for each model tier
   - :doc:`dmta_theory` --- E* expressions and conversion details per model
   - :doc:`dmta_knowledge` --- physical insight extractable from each model family
