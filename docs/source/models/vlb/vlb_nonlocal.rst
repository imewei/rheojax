.. _vlb_nonlocal:

======================================
VLBNonlocal — Shear Banding PDE
======================================

Quick Reference
===============

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Class**
     - ``VLBNonlocal``
   * - **Physics**
     - Nonlocal VLB with tensor diffusion for shear banding
   * - **Key Parameters**
     - :math:`G_0, k_d^0, \eta_s, D_\mu` (+ :math:`\nu` for Bell, :math:`L_{\max}` for FENE)
   * - **Protocols**
     - FLOW_CURVE (steady shear PDE), STARTUP, CREEP
   * - **Key Features**
     - Spatially-resolved 1D PDE, shear banding, cooperativity length
   * - **Reference**
     - Dhont (1999). *PRE* 60, 4534; Vernerey *et al.* (2017). *JMPS* 107, 1-20

**Import:**

.. code-block:: python

   from rheojax.models import VLBNonlocal


Overview
========

``VLBNonlocal`` solves the VLB constitutive equation in one spatial dimension,
modeling the gap of a Couette rheometer.  The distribution tensor
:math:`\boldsymbol{\mu}(y, t)` varies across the gap, coupled by a diffusion
term that represents cooperative rearrangements in the network.

The PDE is:

.. math::

   \frac{\partial \boldsymbol{\mu}}{\partial t}
   = k_d(\mathbf{I} - \boldsymbol{\mu})
   + \mathbf{L} \cdot \boldsymbol{\mu}
   + \boldsymbol{\mu} \cdot \mathbf{L}^T
   + D_\mu \nabla^2 \boldsymbol{\mu}

where :math:`D_\mu` (m\ :sup:`2`/s) is the distribution tensor diffusivity.

**Shear banding** arises when the Bell breakage rate creates a non-monotonic
(S-shaped) constitutive curve :math:`\sigma(\dot\gamma)`.  In the unstable
region, the flow spontaneously separates into coexisting high- and
low-shear-rate bands.  The diffusion term regularizes the interface,
setting its width to the cooperativity length:

.. math::

   \xi = \sqrt{D_\mu / k_d^0}


Constructor
===========

.. code-block:: python

   model = VLBNonlocal(
       breakage="constant",     # or "bell" for banding
       stress_type="linear",    # or "fene"
       n_points=51,             # spatial grid resolution
       gap_width=1e-3,          # gap width in meters
   )


Parameters
==========

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Parameter
     - When Present
     - Units
     - Description
   * - :math:`G_0`
     - Always
     - Pa
     - Network modulus
   * - :math:`k_d^0`
     - Always
     - 1/s
     - Unstressed dissociation rate
   * - :math:`\eta_s`
     - Always
     - Pa·s
     - Solvent viscosity (regularization)
   * - :math:`D_\mu`
     - Always
     - m\ :sup:`2`/s
     - Distribution tensor diffusivity
   * - :math:`\nu`
     - ``breakage="bell"``
     - —
     - Force sensitivity (Bell model)
   * - :math:`L_{\max}`
     - ``stress_type="fene"``
     - —
     - Maximum chain extensibility (FENE-P)


Simulation Methods
==================

``simulate_steady_shear``
-------------------------

Approach to steady state under imposed average shear rate.  Uses a stress
feedback controller to enforce the velocity constraint.

.. code-block:: python

   result = model.simulate_steady_shear(
       gamma_dot_avg=1.0,    # imposed average shear rate
       t_end=100.0,          # simulation time
       dt=1.0,               # output time step
       perturbation=0.05,    # initial symmetry-breaking noise
   )

   # Returns dict with keys:
   # 't': time array
   # 'y': spatial grid
   # 'mu_xy': mu_xy profiles (N_t, N_y)
   # 'gamma_dot': shear rate profiles (N_t, N_y)
   # 'stress': wall stress Sigma(t)


``simulate_startup``
--------------------

Startup from rest (equivalent to steady shear with small perturbation).

.. code-block:: python

   result = model.simulate_startup(
       gamma_dot_avg=1.0, t_end=50.0, dt=0.5
   )


``simulate_creep``
------------------

Stress-controlled creep with spatial resolution.

.. code-block:: python

   result = model.simulate_creep(
       sigma_0=100.0, t_end=100.0, dt=0.1
   )


Banding Detection
=================

.. code-block:: python

   banding = model.detect_banding(result, threshold=0.1)

   if banding["is_banding"]:
       print(f"Band contrast: {banding['band_contrast']:.1f}")
       print(f"Band width: {banding['band_width']*1e3:.2f} mm")
       print(f"Band location: {banding['band_location']*1e3:.2f} mm")

The detection checks the spatial variation of the final shear rate profile.
``threshold`` is the relative standard deviation cutoff.


Velocity Profile
================

.. code-block:: python

   v = model.get_velocity_profile(result)
   # v(0) = 0, v(H) = gamma_dot_avg * H

The velocity profile is computed by integrating the shear rate:
:math:`v(y) = \int_0^y \dot\gamma(y')\,dy'`.


Cooperativity Length
====================

The cooperativity length :math:`\xi` sets the shear band interface width:

.. code-block:: python

   xi = model.get_cooperativity_length()
   print(f"Cooperativity length: {xi*1e6:.1f} um")


Grid Resolution
===============

The number of spatial grid points (``n_points``) should resolve the band
interface.  A rule of thumb is :math:`\Delta y < \xi / 3`:

.. code-block:: python

   # Estimate required grid points
   xi = model.get_cooperativity_length()
   n_min = int(3 * model.gap_width / xi) + 1

   # Recreate with sufficient resolution
   model = VLBNonlocal(breakage="bell", n_points=max(n_min, 51))

For convergence studies, compare results with 25, 51, and 101 points.


Boundary Conditions
===================

The model uses **Neumann (zero-flux)** boundary conditions:

.. math::

   \left.\frac{\partial \boldsymbol{\mu}}{\partial y}\right|_{y=0}
   = \left.\frac{\partial \boldsymbol{\mu}}{\partial y}\right|_{y=H} = 0

This means the structure tensor gradient vanishes at the walls.


Limitations
===========

- ``VLBNonlocal`` does **not** support ``fit()`` or ``fit_bayesian()``.
  Use the ``simulate_*`` methods directly for forward predictions.
- Computational cost scales as :math:`O(N_y \times N_t)` where :math:`N_y`
  is the number of spatial points.
- Only 1D (planar Couette) geometry is supported.


API Reference
=============

.. autoclass:: rheojax.models.vlb.VLBNonlocal
   :members:
   :undoc-members:
   :show-inheritance:
