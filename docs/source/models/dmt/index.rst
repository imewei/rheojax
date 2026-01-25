DMT Thixotropic Models
======================

This section documents the de Souza Mendes-Thompson (DMT) family of models
for thixotropic yield-stress materials.


Overview
--------

The DMT family provides comprehensive constitutive equations for complex fluids
that exhibit:

- **Yield stress behavior** with structure-dependent yielding
- **Thixotropy** (time-dependent structure buildup and breakdown)
- **Optional viscoelasticity** (Maxwell backbone for stress overshoot and relaxation)
- **Shear banding** (via nonlocal diffusion extension)

These models are particularly well-suited for:

- Colloidal gels and suspensions
- Structured emulsions and foams
- Drilling fluids and muds
- Waxy crude oils
- Thixotropic pastes and slurries


Model Hierarchy
---------------

::

   DMT Family
   │
   ├── DMTLocal (Homogeneous)
   │   ├── closure="exponential"
   │   │   └── Smooth viscosity transition
   │   │
   │   └── closure="herschel_bulkley"
   │       └── Explicit yield stress
   │
   └── DMTNonlocal (Spatial)
       └── Structure diffusion for shear banding
       └── Couette/channel flow profiles


When to Use Which Model
-----------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Behavior
     - DMTLocal
     - DMTNonlocal
   * - Homogeneous flow
     - ✓ Use this
     - Overkill
   * - Shear banding
     - Cannot capture
     - ✓ Use this
   * - Stress overshoot
     - ✓ (with elasticity)
     - ✓ (with elasticity)
   * - Delayed yielding
     - ✓ Use this
     - ✓ Use this
   * - Few parameters
     - ✓ Use this
     - More params


Key Features
------------

**Physical Foundation:**

- Structure parameter λ ∈ [0, 1] tracks microstructural organization
- Competing buildup (aging) and breakdown (shear-induced) kinetics
- Multiple viscosity closures: exponential or Herschel-Bulkley
- Optional Maxwell backbone for viscoelastic effects
- Fluidity interpretation with cooperativity length scale

**Theoretical Extensions:**

- **Fluidity-Maxwell formulation**: Jeffreys/Oldroyd-B backbone with structure-dependent
  relaxation and retardation times for true stress relaxation and SAOS moduli
- **Nonlocal fluidity**: Spatial diffusion for shear band regularization with
  cooperativity length ξ ~ √(D_λ · t_eq)
- **Complete protocol equations**: Full mathematical derivations for all rheological
  tests with closed-form solutions where available

**Numerical Implementation:**

- JAX-accelerated kernels with ``jax.lax.scan`` integration
- Papanastasiou regularization for smooth yield behavior
- Full Bayesian inference support via NumPyro

**Supported Protocols:**

- Flow curve (steady state) with viscosity bifurcation
- Startup shear with stress overshoot mechanism
- Stress relaxation after cessation (arrested by structure recovery)
- Creep with delayed yielding and avalanche effect
- Small amplitude oscillatory shear (SAOS) with Maxwell moduli
- Large amplitude oscillatory shear (LAOS) with Fourier/Chebyshev analysis


Quick Start
-----------

**Exponential closure:**

.. code-block:: python

   from rheojax.models import DMTLocal

   model = DMTLocal(closure="exponential", include_elasticity=True)
   model.fit(gamma_dot, stress, test_mode="flow_curve")

**Herschel-Bulkley closure:**

.. code-block:: python

   from rheojax.models import DMTLocal

   model = DMTLocal(closure="herschel_bulkley", include_elasticity=True)
   model.fit(gamma_dot, stress, test_mode="flow_curve")

**Nonlocal for shear banding:**

.. code-block:: python

   from rheojax.models import DMTNonlocal

   model = DMTNonlocal(closure="exponential", n_points=51, gap_width=1e-3)
   result = model.simulate_steady_shear(gamma_dot_avg=10.0, t_end=500.0)
   banding = model.detect_banding(result, threshold=0.1)


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   dmt


References
----------

1. de Souza Mendes, P. R. (2009). "Modeling the thixotropic behavior of
   structured fluids." *J. Non-Newtonian Fluid Mech.*, 164, 66-75.

2. de Souza Mendes, P. R. & Thompson, R. L. (2012). "A critical overview of
   elasto-viscoplastic thixotropic modeling." *J. Non-Newtonian Fluid Mech.*,
   187-188, 8-15.

3. Thompson, R. L. & de Souza Mendes, P. R. (2014). "Thixotropic behavior of
   elasto-viscoplastic materials." *Phys. Fluids*, 26, 023101.
