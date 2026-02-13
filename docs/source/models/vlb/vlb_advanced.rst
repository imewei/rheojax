.. _vlb_advanced:

==============================================
VLB Advanced Theory & Numerical Methods
==============================================

This page documents the theoretical foundations for VLB extensions and
the numerical methods used in their implementation.
Four extensions are implemented in RheoJAX; one (Langevin chains) remains
as a theory reference for future development.

For usage, see :doc:`vlb_variant` (Bell, FENE-P, Temperature) and
:doc:`vlb_nonlocal` (spatial PDE with banding).


.. admonition:: Implementation Status

   .. list-table::
      :widths: 30 20 50
      :header-rows: 1

      * - Extension
        - Status
        - Classes
      * - Bell breakage
        - Implemented
        - :class:`~rheojax.models.vlb.VLBVariant`, :class:`~rheojax.models.vlb.VLBNonlocal`
      * - FENE-P stress
        - Implemented
        - :class:`~rheojax.models.vlb.VLBVariant`, :class:`~rheojax.models.vlb.VLBNonlocal`
      * - Temperature dependence
        - Implemented
        - :class:`~rheojax.models.vlb.VLBVariant`
      * - Nonlocal PDE
        - Implemented
        - :class:`~rheojax.models.vlb.VLBNonlocal`
      * - Langevin chains
        - Theory reference
        - (not yet implemented)


Chain Distribution Decomposition
==================================

The chain distribution function :math:`\varphi(\mathbf{r},t)` admits a
factorization into density and shape:

.. math::

   \varphi(\mathbf{r},t) = c(t) \cdot P(\mathbf{r},t)

where :math:`c(t) = \int \varphi \, d^3r` is the total chain density
(number of elastically active chains per unit volume) and :math:`P(\mathbf{r},t)`
is the normalized probability density of the end-to-end vector.

**Angle-bracket operator:**  For any quantity :math:`f(\mathbf{r})`, the
ensemble average over the distribution is:

.. math::

   \langle f \rangle = \frac{1}{c} \int f(\mathbf{r}) \, \varphi(\mathbf{r},t) \, d^3r
   = \int f(\mathbf{r}) \, P(\mathbf{r},t) \, d^3r

The distribution tensor is therefore:

.. math::

   \boldsymbol{\mu} = \frac{3}{\langle r_0^2 \rangle} \langle \mathbf{r} \otimes \mathbf{r} \rangle

**Master evolution equation:**  The Smoluchowski equation for :math:`\varphi`
under affine convection with source/sink kinetics:

.. math::

   \frac{\partial \varphi}{\partial t}
   + \nabla_r \cdot (\mathbf{L} \cdot \mathbf{r} \, \varphi)
   = k_a \varphi_0 - k_d \varphi

Taking the second moment of this equation (multiplying by
:math:`\mathbf{r} \otimes \mathbf{r}` and integrating) directly yields the
distribution tensor evolution equation used in the VLB model.

At equilibrium :math:`k_a = k_d` (detailed balance), and the equilibrium
distribution :math:`\varphi_0` is isotropic Gaussian with
:math:`\boldsymbol{\mu}_{eq} = \mathbf{I}`.


.. _vlb-numerical-methods:

Numerical Methods
=================

Semi-Implicit Exponential Integrator
--------------------------------------

For the ODE-based protocols (LAOS, startup/relaxation with nonlinear
:math:`k_d`), RheoJAX uses the ``diffrax.Tsit5`` explicit Runge-Kutta solver
with adaptive stepping.  For finite-element or custom integration contexts,
a **semi-implicit exponential integrator** is recommended:

**Step 1 — Explicit convective update:**

.. math::

   \boldsymbol{\mu}^* = \boldsymbol{\mu}^n
   + \Delta t \bigl(\mathbf{D} \cdot \boldsymbol{\mu}^n
   + \boldsymbol{\mu}^n \cdot \mathbf{D}\bigr)

**Step 2 — Exponential kinetic decay:**

.. math::

   \boldsymbol{\mu}^{n+1} = \mathbf{I} + (\boldsymbol{\mu}^* - \mathbf{I})
   e^{-k_d \Delta t}

This scheme is **exact** for the kinetic part (exponential decay toward
:math:`\mathbf{I}`) and explicit for the convective part, providing
unconditional stability for the relaxation term regardless of :math:`k_d \Delta t`.

Symmetry Enforcement
---------------------

After each time step, enforce the symmetry of the distribution tensor:

.. math::

   \boldsymbol{\mu} \leftarrow \frac{1}{2}(\boldsymbol{\mu} + \boldsymbol{\mu}^T)

This is necessary because floating-point arithmetic can introduce asymmetric
errors, particularly in the off-diagonal components.  In JAX:

.. code-block:: python

   mu = 0.5 * (mu + mu.T)

Time Step Selection
--------------------

For stability and accuracy, the time step should satisfy:

.. math::

   \Delta t < \min\!\left(\frac{1}{|\mathbf{D}|}, \quad
   \frac{\pi}{10 \omega}\right)

where :math:`|\mathbf{D}|` is the largest eigenvalue of the deformation rate
tensor.  The second condition ensures at least 20 time points per oscillation
cycle in LAOS/SAOS simulations.

For multi-network models with widely separated :math:`k_d` values, the stiffest
mode (largest :math:`k_d`) sets the stability limit.  The semi-implicit scheme
avoids this restriction for the kinetic term.

Steady-State Detection
-----------------------

For flow curve and other steady-state protocols, convergence is declared when:

.. math::

   \frac{|\dot{\boldsymbol{\mu}}|}{|\boldsymbol{\mu}|} < \varepsilon

where :math:`\varepsilon = 10^{-8}` is the default tolerance and
:math:`|\cdot|` is the Frobenius norm.  For multi-network models, all modes
must satisfy this criterion.

Multiple Time-Scale Handling
-----------------------------

Multi-network VLB models with :math:`k_d` values spanning several decades
create stiff ODE systems.  RheoJAX handles this via:

1. **Adaptive stepping** (``diffrax.PIDController``): automatic step size
   reduction near fast transients
2. **Mode-ordered initialization**: log-spaced :math:`k_d` values ensure
   well-separated time scales
3. **Separate analytical/ODE split**: fast modes that have reached steady state
   can be treated analytically while slow modes continue integrating


Force-Dependent Dissociation Rate
==================================

Physical Motivation
-------------------

In real networks, the bond breaking rate depends on the force applied to the
chain.  Under tension, bonds break faster (slip-bond behavior); under
compression, they may break slower (catch-bond behavior).  This
force-dependence introduces shear thinning, stress overshoot, and nonlinear
LAOS response.

Bell Model
----------

The simplest force-dependent model (Bell, 1978):

.. math::

   k_d(\boldsymbol{\mu}) = k_d^0 \exp\!\left(\nu \cdot (\lambda_c - 1)\right)

where:

- :math:`k_d^0` is the unstressed dissociation rate
- :math:`\nu` is the force sensitivity parameter (dimensionless)
- :math:`\lambda_c = \sqrt{\text{tr}(\boldsymbol{\mu})/3}` is the normalized
  average chain stretch

**Effect on predictions:**

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Protocol
     - Constant :math:`k_d`
     - Bell :math:`k_d`
   * - Flow curve
     - Newtonian
     - Shear thinning (:math:`\eta \propto \dot{\gamma}^{n-1}`)
   * - Startup
     - Monotonic
     - Stress overshoot
   * - LAOS :math:`\sigma_{12}`
     - Linear (no harmonics)
     - Nonlinear (:math:`I_3 > 0`)
   * - Extension
     - Singularity at :math:`\dot{\varepsilon} = k_d/2`
     - Softening before singularity

**New parameter:** :math:`\nu` (force sensitivity).

- :math:`\nu = 0`: recovers constant :math:`k_d`
- :math:`\nu \sim 1` - 5: moderate force sensitivity
- :math:`\nu > 10`: strong force weakening (catch-bond to slip-bond transition)

**Evolution equation becomes nonlinear:**

.. math::

   \dot{\boldsymbol{\mu}} = k_d(\boldsymbol{\mu})(\mathbf{I} - \boldsymbol{\mu})
   + \mathbf{L} \cdot \boldsymbol{\mu}
   + \boldsymbol{\mu} \cdot \mathbf{L}^T

This requires ODE integration for all protocols (no more analytical solutions).

**Comparison with TNT Bell:**

The VLB Bell model is mathematically equivalent to the
:class:`~rheojax.models.tnt.TNTSingleMode` with ``breakage="bell"`` at the
Gaussian chain level.  The VLB formulation is preferred when extending to
Langevin chains (see below).


Slip-Bond and Catch-Bond
------------------------

**Slip-bond** (:math:`k_d` increases with force):

.. math::

   k_d(F) = k_d^0 \exp\!\left(\frac{F \cdot \delta}{k_B T}\right)

where :math:`\delta` is the distance to the transition state.

**Catch-bond** (:math:`k_d` first decreases then increases with force):

.. math::

   k_d(F) = k_{slip} \exp\!\left(\frac{F \delta_s}{k_B T}\right)
   + k_{catch} \exp\!\left(-\frac{F \delta_c}{k_B T}\right)

Catch-bond behavior has been observed in biological systems (selectin-ligand,
fibrin) and in some synthetic networks with dual-lock mechanisms.


Langevin Chain Finite Extensibility
====================================

Physical Motivation
-------------------

Gaussian chain statistics assume unlimited extensibility.  Real polymer chains
have a finite contour length :math:`L_{max} = N_K b` (product of Kuhn segments
and Kuhn length).  As chains approach full extension, the restoring force
diverges.

Langevin Force Law
------------------

For a freely jointed chain with :math:`N_K` segments, the force-extension
relation is given by the inverse Langevin function:

.. math::

   \mathbf{F} = \frac{k_B T}{b} \mathcal{L}^{-1}\!\left(\frac{r}{N_K b}\right)
   \hat{\mathbf{r}}

where :math:`\mathcal{L}(x) = \coth(x) - 1/x` is the Langevin function.

**Stress modification:**

.. math::

   \boldsymbol{\sigma} = G_0 \, f\!\left(\text{tr}(\boldsymbol{\mu})\right)
   (\boldsymbol{\mu} - \mathbf{I})

where :math:`f` is a strain-amplification (FENE-like) function:

.. math::

   f(\text{tr}(\boldsymbol{\mu})) = \frac{L_{max}^2}{L_{max}^2 - \text{tr}(\boldsymbol{\mu}) + 3}

**Effects:**

- Bounds the extensional stress (regularizes the :math:`\dot{\varepsilon} = k_d/2`
  singularity)
- Strain hardening at large deformations
- Affects :math:`N_1` predictions and LAOS harmonics

**New parameter:** :math:`L_{max}` (maximum chain extensibility, typically 3-100).

**Comparison with TNT FENE-P:**

The Langevin/FENE-P stress modification is identical to the FENE variant of
:class:`~rheojax.models.tnt.TNTSingleMode`.  The VLB formulation with the
distribution tensor provides a more natural route to combining FENE with
force-dependent :math:`k_d`.


Padé Approximation
------------------

The inverse Langevin function is typically approximated by a Padé form for
computational efficiency:

.. math::

   \mathcal{L}^{-1}(x) \approx x \frac{3 - x^2}{1 - x^2}

This is exact at :math:`x = 0` and :math:`x \to 1`, and accurate to within
4% for all :math:`x \in [0, 1)`.


Nonlocal Formulation
====================

Physical Motivation
-------------------

In spatially heterogeneous flows (e.g., Couette, pipe, channel), the
distribution tensor can vary across the gap.  Shear banding — the coexistence
of high-shear and low-shear bands — is a common phenomenon in transient
networks with force-dependent :math:`k_d`.

Nonlocal Evolution Equation
----------------------------

The distribution tensor acquires a spatial diffusion term:

.. math::

   \frac{\partial \boldsymbol{\mu}}{\partial t}
   = k_d(\boldsymbol{\mu})(\mathbf{I} - \boldsymbol{\mu})
   + \mathbf{L} \cdot \boldsymbol{\mu}
   + \boldsymbol{\mu} \cdot \mathbf{L}^T
   + D_\mu \nabla^2 \boldsymbol{\mu}

where :math:`D_\mu` is a diffusivity with dimensions m\ :sup:`2`/s.

**Cooperativity length:** :math:`\xi = \sqrt{D_\mu / k_d}` sets the
characteristic width of shear band interfaces.

**Coupled 1D system (gap direction** :math:`y` **):**

.. math::

   \frac{\partial \mu_{xy}}{\partial t} &= -k_d \mu_{xy}
   + \dot{\gamma}(y)\mu_{yy} + D_\mu \frac{\partial^2 \mu_{xy}}{\partial y^2} \\
   \frac{\partial \dot{\gamma}}{\partial y} &= \text{(momentum balance)}

This is a PDE system similar to the existing
:class:`~rheojax.models.fluidity.FluidityNonlocal` and
:class:`~rheojax.models.dmt.DMTNonlocal`.

**New parameters:** :math:`D_\mu` (diffusivity), :math:`n_{points}` (spatial grid),
:math:`L_{gap}` (gap width).


Temperature Dependence
======================

Arrhenius Bond Kinetics
-----------------------

The dissociation rate follows Arrhenius kinetics:

.. math::

   k_d(T) = k_d^0 \exp\!\left(-\frac{E_a}{k_B T}\right)

where :math:`E_a` is the activation energy for bond dissociation.

**Modulus temperature dependence:**

.. math::

   G_0(T) = c(T) k_B T

where :math:`c(T)` is the chain density (may depend on temperature if bonds
have different association constants at different temperatures).

**Application:**  Rheological master curves.  By fitting :math:`k_d(T)` at
multiple temperatures, one obtains :math:`E_a`, which can be compared with
calorimetric data (DSC) or molecular simulations.

WLF/VFT Kinetics
-----------------

For glass-forming systems, the Arrhenius model breaks down and one uses the
Williams-Landel-Ferry (WLF) or Vogel-Fulcher-Tammann (VFT) form:

.. math::

   \log \frac{k_d(T)}{k_d(T_r)} = \frac{-C_1(T - T_r)}{C_2 + T - T_r}


Connection to Finite Elements
=============================

The VLB evolution equation is a first-order tensorial ODE at each material
point.  In a finite element (FE) context:

1. **Gauss-point state:** Each integration point stores
   :math:`\boldsymbol{\mu}` as internal variables
2. **Stress computation:** Given :math:`\boldsymbol{\mu}` and
   :math:`\mathbf{L}`, compute :math:`\boldsymbol{\sigma} = G_0(\boldsymbol{\mu} - \mathbf{I})`
3. **State update:** Integrate :math:`\dot{\boldsymbol{\mu}}` over the time
   step using implicit or semi-implicit schemes
4. **Algorithmic tangent:** The material tangent
   :math:`\partial \boldsymbol{\sigma}/\partial \boldsymbol{\varepsilon}`
   is needed for Newton-Raphson iteration in the FE solver

The VLB formulation is particularly well-suited for FE implementation because
:math:`\boldsymbol{\mu}` has a clear physical interpretation and the evolution
equation is relatively simple.

**Integration schemes:**

- **Exponential map** (exact for constant :math:`\mathbf{L}`):
  :math:`\boldsymbol{\mu}^{n+1} = \mathbf{I} + (\boldsymbol{\mu}^n - \mathbf{I}) e^{-k_d \Delta t} + \ldots`
- **Semi-implicit** (stable for large :math:`\Delta t`):
  kinetic term implicit, convective term explicit
- **Fully implicit** (most stable, requires tangent linearization)

The JAX automatic differentiation framework in RheoJAX provides the algorithmic
tangent "for free" via ``jax.jacfwd`` or ``jax.jacrev``.


Summary of Extensions
=====================

.. list-table::
   :widths: 20 15 20 45
   :header-rows: 1

   * - Extension
     - New Params
     - New Physics
     - Status
   * - Bell :math:`k_d`
     - :math:`\nu`
     - Shear thinning, overshoot, LAOS harmonics
     - Implemented (:class:`~rheojax.models.vlb.VLBVariant`)
   * - Langevin/FENE
     - :math:`L_{max}`
     - Extensional hardening, bounded stress
     - Implemented (:class:`~rheojax.models.vlb.VLBVariant`)
   * - Nonlocal
     - :math:`D_\mu`, :math:`n_{pts}`, :math:`L_{gap}`
     - Shear banding
     - Implemented (:class:`~rheojax.models.vlb.VLBNonlocal`)
   * - Temperature
     - :math:`E_a` or :math:`C_1, C_2, T_r`
     - TTS, Arrhenius/WLF
     - Implemented (:class:`~rheojax.models.vlb.VLBVariant`)
   * - Catch-bond
     - :math:`k_s, k_c, \delta_s, \delta_c`
     - Non-monotonic :math:`k_d(F)`
     - Future
   * - FE coupling
     - (mesh)
     - Spatial mechanics
     - External (via JAX tangent)


References
==========

1. Vernerey, F.J. (2018). "Transient response of nonlinear polymer networks:
   A kinetic theory." *J. Mech. Phys. Solids*, 115, 230-247.
   https://doi.org/10.1016/j.jmps.2018.02.018
   :download:`PDF <../../../reference/vernerey_2018_tnt_kinetic_theory.pdf>`

2. Bell, G.I. (1978). "Models for the specific adhesion of cells to cells."
   *Science*, 200(4342), 618-627.
   https://doi.org/10.1126/science.347575

3. Rubinstein, M. & Semenov, A.N. (2001). "Dynamics of entangled solutions of
   associating polymers." *Macromolecules*, 34(4), 1058-1068.
   https://doi.org/10.1021/ma0013049

4. Meng, F. & Pritchard, R.H. & Terentjev, E.M. (2016). "Stress relaxation,
   dynamics, and plasticity of transient polymer networks."
   *Macromolecules*, 49(7), 2843-2852.
   https://doi.org/10.1021/acs.macromol.5b02667

5. Long, R., Qi, H.J. & Dunn, M.L. (2013). "Modeling the mechanics of
   covalently adaptable polymer networks with temperature-dependent bond
   exchange reactions." *Soft Matter*, 9, 4083-4096.
   https://doi.org/10.1039/C3SM27945F

6. Hui, C.-Y., Cui, F., Zehnder, A. & Vernerey, F.J. (2021). "Physically
   motivated models of polymer networks with dynamic cross-links: comparative
   study and future outlook." *Proc. R. Soc. A*, 477, 20210608.
   https://doi.org/10.1098/rspa.2021.0608
   :download:`PDF <../../../reference/hui_2021_dynamic_crosslinks_review.pdf>`
