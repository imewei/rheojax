Fractional IKH (FIKH) Models
============================

This section documents the Fractional Isotropic-Kinematic Hardening (FIKH) family
of models for thixotropic elasto-viscoplastic (TEvp) materials with power-law memory.


Overview
--------

The FIKH family extends the classical :doc:`../ikh/index` framework by replacing the integer-order
structure kinetics with a **Caputo fractional derivative**. This captures the
**power-law memory** observed in many complex fluids:

- **Standard IKH (α = 1)**: Exponential recovery :math:`\lambda \sim \exp(-t/\tau)`
- **Fractional FIKH (α < 1)**: Power-law recovery :math:`\lambda \sim t^{\alpha-1}` at long times

Fractional derivatives introduce a fading memory where recent deformation history
affects the current structure more than distant past. This single parameter α captures
a broad distribution of restructuring timescales without requiring multiple modes.

**Thermokinematic coupling** adds:

- Temperature-dependent yield stress: :math:`\sigma_y(\lambda, T)`
- Arrhenius viscosity: :math:`\eta(T) = \eta_0 \cdot \exp(E_a/RT)`
- Thermal evolution from plastic dissipation

These models are particularly suited for:

- Waxy crude oils (cold restart, pipeline flow assurance)
- Colloidal gels with hierarchical structure
- Materials exhibiting stretched-exponential recovery
- Systems with thermal feedback (shear heating)


Model Hierarchy
---------------

::

   FIKH Family
   │
   ├── FIKH (Single Mode)
   │   ├── 12 parameters (base)
   │   ├── 20 parameters (with thermal coupling)
   │   ├── 22 parameters (full: thermal + isotropic hardening)
   │   └── Single fractional structure variable
   │
   └── FMLIKH (Multi-Mode)
       ├── Per-mode: G_i, η_i, C_i, γ_dyn_i, τ_thix_i, Γ_i
       ├── Shared or per-mode fractional order α
       └── Global yield with distributed kinetics


When to Use FIKH
----------------

Experimental Signatures
^^^^^^^^^^^^^^^^^^^^^^^

**Use FIKH when you observe**:

1. **Power-law stress relaxation** at long times: G(t) ~ t^(-α), not exp(-t/τ)
2. **Stretched exponential recovery** after shear cessation
3. **Broad relaxation spectrum** in frequency sweep (Cole-Cole depression)
4. **Delayed yielding** in creep tests below apparent yield stress
5. **Temperature-dependent flow** with Arrhenius-like behavior
6. **Stress overshoot with long tail** in startup (not sharp exponential decay)

Decision Tree
^^^^^^^^^^^^^

::

   Is recovery exponential (single timescale)?
   ├── YES → Use MIKH (simpler, faster)
   └── NO → Is recovery power-law?
       ├── YES → Use FIKH (single α captures spectrum)
       └── NO → Is there hierarchical structure?
           ├── YES → Use FMLIKH (multiple modes)
           └── NO → Consider SGR or DMT models


Model Comparison
----------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Behavior
     - Single Mode (FIKH)
     - Multi-Mode (FMLIKH)
   * - Power-law recovery
     - ✓ Use this
     - Also works
   * - Single structural population
     - ✓ Use this
     - Overkill
   * - Broad relaxation spectrum
     - Limited
     - ✓ Use this
   * - Few parameters needed
     - ✓ Use this
     - More params
   * - Hierarchical structure
     - Limited
     - ✓ Use this
   * - When α → 1 (exponential)
     - Consider MIKH
     - Consider ML-IKH


Material-Specific Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 15 45
   :header-rows: 1

   * - Material
     - Recommended Model
     - Typical α
     - Key Protocol
   * - Waxy crude oils
     - FIKH (thermal)
     - 0.5-0.7
     - Startup at different T
   * - Colloidal gels
     - FMLIKH
     - 0.3-0.6
     - Frequency sweep
   * - Food gels
     - FIKH
     - 0.6-0.8
     - Creep recovery
   * - Drilling muds
     - FIKH (thermal)
     - 0.4-0.6
     - Flow curve + relaxation
   * - Greases
     - FIKH
     - 0.5-0.7
     - LAOS + startup
   * - Cement pastes
     - FMLIKH
     - 0.4-0.6
     - Multiple rest times


Key Features
------------

**Fractional Structure Evolution:**

- Caputo derivative captures power-law fading memory
- Single α parameter spans exponential (α=1) to strong memory (α→0)
- Mittag-Leffler relaxation generalizes the exponential

**Armstrong-Frederick Kinematic Hardening:**

- Back-stress A tracks deformation history
- Captures Bauschinger effect in cyclic loading
- Dynamic recovery prevents unbounded hardening

**Full Thermokinematic Coupling:**

- Arrhenius temperature dependence for viscosity
- Structure-temperature yield stress coupling
- Plastic dissipation heating with heat loss

**Supported Protocols:**

- Flow curve (steady state)
- Startup shear (stress overshoot)
- Stress relaxation (Mittag-Leffler decay)
- Creep (delayed yielding, thermal runaway)
- SAOS (fractional Maxwell moduli)
- LAOS (harmonic generation, Lissajous figures)


Quick Start
-----------

**Basic FIKH with thermal coupling:**

.. code-block:: python

   from rheojax.models.fikh import FIKH

   # Create model with fractional order α = 0.7
   model = FIKH(include_thermal=True, alpha_structure=0.7)

   # Set key parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 10.0)
   model.parameters.set_value("delta_sigma_y", 50.0)
   model.parameters.set_value("tau_thix", 100.0)

   # Fit to startup data
   model.fit(t, stress, test_mode='startup', strain=strain)

   # Predict flow curve
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

**Multi-mode FMLIKH:**

.. code-block:: python

   from rheojax.models.fikh import FMLIKH

   # Create 3-mode model with shared fractional order
   model = FMLIKH(n_modes=3, include_thermal=False, shared_alpha=True)

   # Set per-mode parameters
   for i, tau in enumerate([1.0, 10.0, 100.0], 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)

   # Fit to oscillation data
   model.fit(omega, G_star, test_mode='oscillation')


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   fikh
   fmlikh


References
----------

**Fractional Calculus:**

1. Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.

2. Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press.

3. Diethelm, K. (2010). *The Analysis of Fractional Differential Equations*. Springer.

**Fractional Rheology:**

4. Jaishankar, A. & McKinley, G.H. (2014). "A fractional K-BKZ constitutive formulation
   for describing the nonlinear rheology of multiscale complex fluids."
   *J. Rheol.*, 58, 1751-1788.

**IKH Foundation:**

For foundational IKH references (Dimitriou 2014, Geri 2017, Wei 2018), see :doc:`../ikh/index`.
