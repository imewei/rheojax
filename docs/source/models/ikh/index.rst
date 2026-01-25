Isotropic-Kinematic Hardening (IKH) Models
==========================================

This section documents the Isotropic-Kinematic Hardening (IKH) family of models
for thixotropic elasto-viscoplastic (TEvp) materials.


Overview
--------

The IKH family provides comprehensive constitutive equations for complex fluids
that exhibit:

- **Yield stress behavior** with structure-dependent yielding
- **Thixotropy** (time-dependent structure buildup and breakdown)
- **Viscoelasticity** (stress relaxation, creep)
- **Kinematic hardening** (Bauschinger effect, directional memory)

These models are particularly well-suited for:

- Waxy crude oils (pipeline restart, cold flow assurance)
- Drilling fluids and muds (borehole stability, pump circulation)
- Greases and lubricants (NLGI grades, bearing applications)
- Colloidal gels (bidisperse systems, hierarchical structure)
- Structured emulsions (dense emulsions, foams)
- Thixotropic cements and pastes (self-leveling, 3D printing)

Both models include comprehensive **Industrial Applications** sections with
typical parameter ranges from field studies, and **Parameter Estimation Methods**
covering sequential fitting, multi-start optimization, Bayesian inference,
and regularization techniques for ill-conditioned problems


Model Hierarchy
---------------

::

   IKH Family
   │
   ├── MIKH (Single Mode)
   │   └── 11 parameters
   │   └── Single structural timescale
   │   └── Exponential recovery
   │
   └── ML-IKH (Multi-Mode)
       ├── Per-Mode Yield: 7N+1 parameters
       │   └── N independent yield surfaces
       │   └── Parallel mechanical connection
       │
       └── Weighted-Sum Yield: 6+3N parameters
           └── Single global yield surface
           └── Distributed kinetics


When to Use Which Model
-----------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Behavior
     - Single Mode (MIKH)
     - Multi-Mode (ML-IKH)
   * - Exponential recovery
     - ✓ Use this
     - Overkill
   * - Stretched-exponential recovery
     - Poor fit
     - ✓ Use this
   * - Single structural population
     - ✓ Use this
     - Overkill
   * - Hierarchical structure
     - Poor fit
     - ✓ Use this
   * - Few parameters needed
     - ✓ Use this
     - More params
   * - Complex aging behavior
     - Limited
     - ✓ Use this


Key Features
------------

**Physical Foundation:**

- Built on classical plasticity theory (Armstrong-Frederick kinematic hardening)
- Incorporates structural kinetics for thixotropy (Goodeve-Moore framework)
- Maxwell viscoelasticity for liquid-like long-time behavior
- Perzyna regularization for smooth yield transitions

**Industrial Applications:**

- Quantitative parameter ranges from field studies and laboratory characterization
- Application-specific guidance for pipeline operations, drilling, lubrication
- Mode selection rules for multi-timescale materials (β → N mapping)

**Parameter Estimation:**

- Sequential fitting strategies exploiting timescale separation
- Multi-start global optimization for complex parameter landscapes
- Bayesian inference with NLSQ warm-start and prior selection guidance
- Regularization methods for correlated parameters

**Numerical Implementation:**

- Two formulations: ODE (for creep/relaxation) and return mapping (for startup/LAOS)
- JAX-accelerated kernels for efficient computation
- Full Bayesian inference support via NumPyro

**Supported Protocols:**

- Flow curve (steady state)
- Startup shear
- Stress relaxation
- Creep
- Small amplitude oscillatory shear (SAOS)
- Large amplitude oscillatory shear (LAOS)


Quick Start
-----------

**Single-mode model:**

.. code-block:: python

   from rheojax.models import MIKH

   model = MIKH()
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)
   model.parameters.set_value("tau_thix", 10.0)

   # Predict flow curve
   sigma = model.predict_flow_curve(gamma_dot)

**Multi-mode model:**

.. code-block:: python

   from rheojax.models import MLIKH

   model = MLIKH(n_modes=3, yield_mode='weighted_sum')
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)

   # Set distributed timescales
   for i, tau in enumerate([0.1, 1.0, 10.0], 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   mikh
   ml_ikh


References
----------

1. Dimitriou, C.J. & McKinley, G.H. (2014). "A comprehensive constitutive law for
   waxy crude oil: a thixotropic yield stress fluid." *Soft Matter*, 10, 6619-6644.

2. Geri, M. et al. (2017). "Thermokinematic memory and the thixotropic
   elasto‐viscoplasticity of waxy crude oils." *J. Rheol.*, 61(3), 427-454.

3. Wei, Y., Solomon, M.J., & Larson, R.G. (2018). "A multimode structural kinetics
   constitutive equation for the transient rheology of thixotropic elasto‐viscoplastic
   fluids." *J. Rheol.*, 62(1), 321-342.
