DMTA Model Selection
====================

Every oscillation-capable model in RheoJAX works with DMTA data through the
automatic :math:`E^* \leftrightarrow G^*` conversion.  This page groups models
by their suitability for common DMTA use cases.

Tier 1 --- Primary DMTA Models
-------------------------------

These models are the first choice for standard DMTA data analysis.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - When to Use
     - Key Advantage
   * - :class:`~rheojax.models.multi_mode.GeneralizedMaxwell`
     - Any polymer, master curves, FEM export
     - Native ``modulus_type='tensile'`` support, Prony series
   * - :class:`~rheojax.models.fractional.FractionalZenerSolidSolid`
     - Amorphous polymers through :math:`T_g`, broad :math:`\tan\delta` peak
     - 3--4 parameters capture broad glass transition
   * - :class:`~rheojax.models.classical.Zener`
     - Quick fits, narrow relaxation
     - Simplest solid model with plateau--transition--plateau
   * - :class:`~rheojax.models.classical.Maxwell`
     - Single-mode viscoelastic liquid
     - Baseline comparison

**Example: Fractional Zener for DMTA**

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.35,  # glassy polymer
   )
   # alpha parameter = breadth of glass transition

Tier 2 --- Temperature-Dependent Models
-----------------------------------------

Models with built-in Arrhenius or WLF temperature dependence are ideal for
multi-temperature DMTA datasets.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Temperature Support
     - Best For
   * - :class:`~rheojax.models.vlb.VLBVariant`
     - Arrhenius :math:`k_d(T)`, :math:`G(T)`
     - Crosslinked polymers, TTS from first principles
   * - :class:`~rheojax.models.hvm.HVMLocal`
     - TST/Arrhenius :math:`E_a`, topology freezing :math:`T_v`
     - Vitrimers / CANs showing :math:`T_v` transition
   * - :class:`~rheojax.models.hvnm.HVNMLocal`
     - Dual Arrhenius (matrix + interphase)
     - NP-filled vitrimers with Payne effect + :math:`T_v`
   * - :class:`~rheojax.models.fikh.FIKH`
     - Arrhenius :math:`\eta(T)`, :math:`G(T)`, :math:`\sigma_y(T)`
     - Thixotropic polymers with T-dependent yield

Tier 3 --- Specialist Models
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Material Class
   * - :class:`~rheojax.models.sgr.SGRConventional`
     - Soft glassy materials (foams, pastes, colloidal glasses)
   * - :class:`~rheojax.models.itt_mct.ITTMCTSchematic`
     - Dense colloids near glass transition
   * - :class:`~rheojax.models.giesekus.Giesekus`
     - Polymer melts above :math:`T_g`
   * - :class:`~rheojax.models.tnt.TNTSingleMode`
     - Physically crosslinked hydrogels, telechelic polymers

Experimental Situation |rarr| Model
--------------------------------------

.. |rarr| unicode:: U+2192

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
     - :class:`~rheojax.models.tnt.TNTStickyRouse`

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

Shear-Only Models (Not DMTA-Compatible)
-----------------------------------------

The following models are **shear-only** and cannot be used with DMTA data:

- Flow curve models: ``bingham``, ``carreau``, ``cross``, ``herschel_bulkley``,
  ``power_law``, ``carreau_yasuda``
- SPP yield stress decomposition: ``spp_yield_stress``
- Nonlocal PDE models: ``dmt_nonlocal``, ``vlb_nonlocal``,
  ``fluidity_saramito_nonlocal``

Use :func:`~rheojax.core.registry.ModelRegistry.find` to query compatible models:

.. code-block:: python

   from rheojax.core.registry import ModelRegistry
   from rheojax.core.test_modes import DeformationMode
   from rheojax.core.inventory import Protocol

   # All models supporting tension + oscillation
   models = ModelRegistry.find(
       protocol=Protocol.OSCILLATION,
       deformation_mode=DeformationMode.TENSION,
   )

.. seealso::

   - :doc:`dmta_workflows` --- end-to-end fitting examples for each model tier
   - :doc:`dmta_theory` --- E* expressions and conversion details per model
   - :doc:`dmta_knowledge` --- physical insight extractable from each model family
