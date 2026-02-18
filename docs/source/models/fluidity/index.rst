Fluidity Models
===============

This section documents the Fluidity family of models for thixotropic
and elastoviscoplastic materials.


Overview
--------

The Fluidity family provides constitutive equations for complex fluids
where the relaxation time (or its inverse, fluidity) evolves dynamically
due to competing aging and shear-rejuvenation processes. These models capture:

- **Thixotropy**: Time-dependent viscosity from structural buildup/breakdown
- **Yield stress behavior**: Solid-like response at rest, liquid-like under flow
- **Stress overshoot**: Transient peak during startup after aging
- **Shear banding**: Spatial flow heterogeneity (nonlocal variants)
- **Normal stresses**: First normal stress difference :math:`N_1` (Saramito EVP)

These models are well-suited for:

- Colloidal gels and pastes
- Concentrated emulsions (mayonnaise, cosmetics)
- Polymer gels (Carbopol, hydrogels)
- Drilling fluids and muds
- Waxy crude oils
- Cement and concrete

.. include:: /_includes/thixotropy_foundations.rst


Model Hierarchy
---------------

::

   Fluidity Family
   │
   ├── FluidityLocal (0D Homogeneous)
   │   └── Scalar stress σ
   │   └── 9 parameters: G, tau_y, K, n_flow, f_eq, f_inf, theta, a, n_rejuv
   │   └── Maxwell-like viscoelasticity
   │
   ├── FluidityNonlocal (1D Spatial)
   │   └── Adds cooperativity length ξ
   │   └── Shear banding resolution
   │   └── Couette/channel flow profiles
   │
   └── FluiditySaramito EVP (Tensorial)
       │
       ├── Minimal Coupling
       │   └── λ = 1/f only
       │   └── Fewer parameters, identifiable
       │
       └── Full Coupling
           └── λ + τ_y(f) aging yield
           └── Wait-time dependent yield stress


When to Use Which Model
-----------------------

.. list-table::
   :widths: 30 23 23 24
   :header-rows: 1

   * - Feature / Use Case
     - FluidityLocal
     - FluidityNonlocal
     - FluiditySaramito EVP
   * - Homogeneous flow
     - ✓ Use this
     - Overkill
     - ✓ If :math:`N_1` needed
   * - Shear banding
     - Cannot capture
     - ✓ Use this
     - ✓ Nonlocal variant
   * - Stress overshoot
     - ✓ Scalar
     - ✓ Scalar
     - ✓ Tensorial
   * - Normal stresses (:math:`N_1`)
     - ✗
     - ✗
     - ✓ Use this
   * - Von Mises yield
     - ✗ (implicit)
     - ✗ (implicit)
     - ✓ Explicit
   * - Aging yield stress
     - ✗
     - ✗
     - ✓ Full coupling
   * - Creep bifurcation
     - ✓
     - ✓
     - ✓ (enhanced)
   * - Parameters
     - 9
     - 10
     - 10-12
   * - Computational cost
     - 1× (baseline)
     - 2-5×
     - 3-5×

**Decision Guide:**

- **Start with FluidityLocal** for exploratory analysis and homogeneous flows
- **Use FluidityNonlocal** when shear banding is observed or expected
- **Use FluiditySaramito** when normal stresses, tensorial stress state, or
  aging-dependent yield stress are important


Quick Comparison
----------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Model
     - Stress Type
     - Key Extension
     - Primary Use
   * - FluidityLocal
     - Scalar :math:`\sigma`
     - Base model
     - Thixotropic flow curves
   * - FluidityNonlocal
     - Scalar :math:`\sigma(y)`
     - Cooperativity :math:`\xi`
     - Shear banding
   * - FluiditySaramitoLocal
     - Tensor [:math:`\tau_{xx}`, :math:`\tau_{yy}`, :math:`\tau_{xy}`]
     - UCM + Von Mises
     - EVP with :math:`N_1`
   * - FluiditySaramitoNonlocal
     - Tensor :math:`\tau(y)`
     - Spatial + tensorial
     - Banding with :math:`N_1`


Key Equations
-------------

**Scalar fluidity evolution** (all models):

.. math::

   \frac{df}{dt} = \frac{f_{\rm eq} - f}{\tau_{\rm age}} + a|\dot{\gamma}|^n (f_\infty - f)

**Maxwell constitutive** (Local/Nonlocal):

.. math::

   \dot{\sigma} = G\dot{\gamma} - f(t)\sigma

**Upper-convected Maxwell with plasticity** (Saramito EVP):

.. math::

   \lambda \overset{\nabla}{\boldsymbol{\tau}} + \alpha(\boldsymbol{\tau})\boldsymbol{\tau} = 2\eta_p \mathbf{D}

where :math:`\lambda = 1/f` and :math:`\alpha = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|)`.


Quick Start
-----------

**Local (homogeneous) model:**

.. code-block:: python

   from rheojax.models.fluidity import FluidityLocal

   model = FluidityLocal()
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Simulate startup with stress overshoot
   t, stress, fluidity = model.simulate_startup(t, gamma_dot=1.0, t_wait=100)

**Nonlocal (shear banding) model:**

.. code-block:: python

   from rheojax.models.fluidity import FluidityNonlocal

   model = FluidityNonlocal(N_y=51, H=1e-3, xi=1e-5)
   result = model.simulate_startup(t, gamma_dot=0.1)

   # Check for shear banding
   is_banded, cv, ratio = model.detect_shear_bands()

**Saramito EVP (tensorial) model:**

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal

   # Minimal coupling (most identifiable)
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Note: tensorial stress (τ_xx, τ_yy, τ_xy) is tracked internally;
   # access N1 via transient simulations (simulate_startup, simulate_laos)


Protocol-Specific Recommendations
---------------------------------

Different experimental protocols and material types are best served by different
models in the Fluidity family. Use this guide to select the appropriate variant.

**By Experimental Protocol:**

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Protocol
     - Recommended Model
     - Rationale
   * - **Standard rheometry** (cone-plate, parallel plate)
     - FluidityLocal or FluiditySaramitoLocal
     - Homogeneous flow assumption valid; no spatial resolution needed
   * - **LAOS with** :math:`N_1` **extraction**
     - FluiditySaramitoLocal
     - Tensorial stress required for first normal stress difference
   * - **Microfluidic confinement** (:math:`H \sim \xi`)
     - FluidityNonlocal
     - Gap-dependent flow curves; spatial fluidity profiles
   * - **Wide-gap Couette** :math:`(R_o - R_i)/R_i > 0.1`
     - FluidityNonlocal (with curvature)
     - Stress gradient matters; velocity profiles accessible
   * - **Startup with velocity profiles** (PIV, USV)
     - FluiditySaramitoNonlocal
     - Validates spatial predictions; extracts :math:`\xi`
   * - **Creep bifurcation** tests
     - FluidityLocal or FluiditySaramitoLocal
     - Homogeneous; bifurcation point identifies :math:`\tau_y`
   * - **Extensional flow** (CaBER, filament stretching)
     - FluiditySaramitoLocal
     - Tensorial formulation handles uniaxial extension

**By Material Type:**

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Material
     - Recommended Model
     - Notes
   * - **Carbopol gel**
     - FluiditySaramitoLocal (minimal)
     - Well-characterized simple yield stress fluid; weak thixotropy;
       minimal coupling sufficient
   * - **Concentrated emulsion** (mayonnaise, cosmetics)
     - FluiditySaramitoLocal (minimal)
     - Moderate :math:`N_1`; clear yield; standard thixotropy
   * - **Emulsion in microchannel**
     - FluidityNonlocal
     - Strong confinement effects; :math:`\xi \sim` 10-50 :math:`\mu\text{m}` typically
   * - **Waxy crude oil**
     - FluiditySaramitoLocal (full)
     - Strong aging-yield coupling; :math:`\tau_y` increases significantly with rest
   * - **Drilling mud**
     - FluiditySaramitoLocal (full) or DMT
     - Complex thixotropy; may need aging yield coupling
   * - **Cement/concrete**
     - FluiditySaramitoLocal (full)
     - Hydration-dependent aging; :math:`\tau_y` evolves with time
   * - **Colloidal glass** near jamming
     - FluidityNonlocal or HL Trap
     - Cooperativity important; may need statistical mechanics model

**Quick Decision Flowchart:**

::

   Start
     │
     ├── Need tensorial stress or N_1? ──Yes──► FluiditySaramito*
     │                                              │
     No                                             ├── Spatial profiles? ──Yes──► Nonlocal
     │                                              │
     │                                              └── Homogeneous ──► Local
     │
     ├── Shear banding or confinement? ──Yes──► FluidityNonlocal
     │
     No
     │
     └── Homogeneous thixotropy ──► FluidityLocal


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   fluidity_local
   fluidity_nonlocal
   saramito_evp


References
----------

1. Coussot, P., Nguyen, Q. D., Huynh, H. T., and Bonn, D. (2002). "Viscosity bifurcation
   in thixotropic, yielding fluids." *J. Rheol.*, 46(3), 573-589.
   https://doi.org/10.1122/1.1459447

2. Bocquet, L., Colin, A., and Ajdari, A. (2009). "Kinetic theory of plastic flow
   in soft glassy materials." *Phys. Rev. Lett.*, 103, 036001.
   https://doi.org/10.1103/PhysRevLett.103.036001

3. Saramito, P. (2007). "A new constitutive equation for elastoviscoplastic
   fluid flows." *J. Non-Newtonian Fluid Mech.*, 145, 1-14.
   https://doi.org/10.1016/j.jnnfm.2007.04.004

4. de Souza Mendes, P. R. & Thompson, R. L. (2012). "A critical overview of
   elasto-viscoplastic thixotropic modeling." *J. Non-Newtonian Fluid Mech.*,
   187-188, 8-15.
   https://doi.org/10.1016/j.jnnfm.2012.08.006


See Also
--------

- :doc:`/models/dmt/index` — DMT structural-kinetics models (scalar structure parameter)
- :doc:`/models/sgr/sgr_conventional` — SGR for thermally-activated soft glasses
- :doc:`/models/ikh/index` — IKH models with kinematic hardening
- :doc:`/models/hl/hebraud_lequeux` — Hébraud-Lequeux mean-field model
