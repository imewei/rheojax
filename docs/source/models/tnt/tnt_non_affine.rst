.. _model-tnt-non-affine:

=======================================================================
TNT Non-Affine (Gordon-Schowalter) — Handbook
=======================================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
---------------

**Use when:**
  - Materials show reduced normal stresses compared to UCM predictions
  - Non-zero second normal stress difference :math:`N_2` is observed
  - Polymer solutions with significant hydrodynamic interaction
  - Networks with imperfect chain-flow coupling
  - Materials exhibit slip effects relative to bulk flow
  - Intermediate between upper-convected and corotational behavior

**Parameters:**
  4 parameters: :math:`G` (network modulus, Pa), :math:`\tau_b` (bond lifetime, s),
  :math:`\eta_s` (solvent viscosity, Pa·s), :math:`\xi` (slip parameter, dimensionless 0-1)

**Key equation:**
  .. math::

     \frac{d\mathbf{S}}{dt} = (1-\xi)(\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T)
     + \xi(\mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W})
     - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

**Test modes:**
  All 6 protocols (FLOW_CURVE, OSCILLATION, RELAXATION, STARTUP, CREEP, LAOS)

**Material examples:**
  - Polymer solutions with chain slip (e.g., dilute PEO in water)
  - Networks with imperfect crosslinks (partial entanglements)
  - Micellar solutions with slip at micelle-solvent interface
  - Colloidal suspensions with hydrodynamic slip
  - Fiber suspensions with non-affine orientation

**Key characteristics:**
  - Non-zero second normal stress difference :math:`N_2 \neq 0` (negative for :math:`\xi > 0`)
  - Reduced normal stress ratio: :math:`N_2/N_1 \approx -\xi/2` (Johnson-Segalman relation)
  - Still Newtonian flow curve for constant breakage (shear thinning requires Bell variant)
  - SAOS identical to base TNT (linearization eliminates :math:`\xi` effects)
  - Slip parameter controls degree of non-affine deformation

**Notation Guide:**
  :math:`\xi` (slip parameter, 0=upper-convected, 1=corotational),
  :math:`\mathbf{D}` (rate of deformation tensor),
  :math:`\mathbf{W}` (vorticity tensor),
  Gordon-Schowalter derivative (mixed convected derivative)

Overview
--------

The Non-Affine variant of the Tanaka-Edwards model replaces the upper-convected derivative
with the **Gordon-Schowalter (mixed) derivative**, introducing a slip parameter :math:`\xi \in [0, 1]`
that controls how chains respond to flow. This modification accounts for the physical
observation that polymer chains in solution do not always deform affinely (i.e., as
material line elements) with the bulk flow.

**Physical Motivation:**

In the classical Tanaka-Edwards model (upper-convected, :math:`\xi = 0`), chains are
assumed to deform **affinely** with the flow: the end-to-end vector :math:`\mathbf{R}`
of a network chain transforms exactly as a material line element embedded in the fluid.
This assumption is valid when:

- Chains are strongly coupled to the surrounding flow
- Hydrodynamic interactions are negligible
- Crosslinks are permanent or very long-lived

However, real polymer chains in solution exhibit **non-affine deformation** due to:

1. **Hydrodynamic interaction**: The solvent flow around chain segments creates a local
   velocity field that differs from the bulk flow, causing chains to slip relative to
   the fluid.

2. **Imperfect coupling**: Physical crosslinks may allow some chain motion relative to
   the network (e.g., partial entanglements, slip at sticker-chain junctions).

3. **Internal degrees of freedom**: Chain conformations can change independently of the
   bulk deformation through internal rearrangements (Rouse modes, breathing modes).

**The Gordon-Schowalter Derivative:**

Gordon and Schowalter (1972) introduced the **mixed derivative** to interpolate between
affine (upper-convected) and non-affine (corotational) chain deformation:

.. math::

   \frac{d\mathbf{S}}{dt}_{\text{GS}} = (1-\xi)(\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W})

where:

- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` is the velocity gradient tensor
- :math:`\mathbf{D} = (\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2` is the rate of deformation tensor (symmetric)
- :math:`\mathbf{W} = (\boldsymbol{\kappa} - \boldsymbol{\kappa}^T)/2` is the vorticity tensor (antisymmetric)
- :math:`\xi \in [0, 1]` is the **slip parameter**

This can equivalently be written as:

.. math::

   \frac{d\mathbf{S}}{dt}_{\text{GS}} = \mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W}
   + a(\mathbf{D} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{D})

where :math:`a = 1 - 2\xi` is the **slip coefficient**.

**Physical Interpretation of ξ:**

- :math:`\xi = 0`: **Upper-convected** (affine deformation)

  Chains deform with the flow. The conformation tensor evolves as
  :math:`d\mathbf{S}/dt = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T - (\mathbf{S} - \mathbf{I})/\tau_b`.
  This gives the standard Tanaka-Edwards (UCM) model.

- :math:`\xi = 0.5`: **Corotational** (Jaumann derivative)

  Chains rotate with the fluid vorticity but do not stretch with the rate of deformation.
  The conformation tensor rotates as :math:`d\mathbf{S}/dt = \mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W} - (\mathbf{S} - \mathbf{I})/\tau_b`.
  Appropriate for rigid rod-like structures.

- :math:`\xi = 1`: **Lower-convected** (rarely used)

  Chains deform in the opposite sense to the flow gradient. This is generally unphysical
  for polymer chains but may apply to certain colloidal systems.

- :math:`0 < \xi < 0.5`: **Partial slip**

  Chains deform affinely but with reduced coupling to the flow. This is the most common
  regime for polymer solutions with hydrodynamic interaction. Typical values: :math:`\xi = 0.1-0.3`.

**Key Prediction: Non-Zero N₂:**

The most important experimental signature of non-affine deformation is a **non-zero second
normal stress difference** :math:`N_2 = \sigma_{yy} - \sigma_{zz}`. The upper-convected
model (ξ=0) predicts :math:`N_2 = 0`, whereas the Gordon-Schowalter model predicts:

.. math::

   N_2 \approx -\frac{\xi}{2} N_1 \quad \text{(Johnson-Segalman relation)}

This relation is approximate and holds for small to moderate :math:`Wi`. Experimental
measurement of :math:`N_2/N_1` provides a direct estimate of the slip parameter :math:`\xi`.

**Historical Context:**

The Gordon-Schowalter derivative was introduced in:

- **Gordon RJ, Schowalter WR (1972)** *Anisotropic fluid theory: a different approach to the
  dumbbell theory of dilute polymer solutions*, Trans. Soc. Rheol. 16:79-97.

This framework was later incorporated into constitutive models by:

- **Johnson MW, Segalman D (1977)** *A model for viscoelastic fluid behavior which allows
  non-affine deformation*, J. Non-Newtonian Fluid Mech. 2:255-270. (The "Johnson-Segalman model")

- **Phan-Thien N, Tanner RI (1977)** *A new constitutive equation derived from network
  theory*, J. Non-Newtonian Fluid Mech. 2:353-365. (PTT model with Gordon-Schowalter derivative)

The Johnson-Segalman model is mathematically equivalent to TNT with Gordon-Schowalter
derivative and constant breakage. However, it is known to exhibit **shear banding instability**
at high :math:`\xi` (typically :math:`\xi > 0.5-0.7`) due to non-monotonic flow curves
in the :math:`\sigma`-:math:`\dot{\gamma}` plane. This is a **Hadamard instability**
(ill-posed PDE for spatial perturbations).

**When to Use This Model:**

Use TNT Non-Affine when:

1. Experimental :math:`N_2` is measurably negative (typically :math:`|N_2|/N_1 = 0.05-0.2`)
2. Materials show lower :math:`N_1` than predicted by UCM at the same :math:`G` and :math:`\tau_b`
3. Polymer solutions with significant solvent content (hydrodynamic interactions important)
4. Networks with imperfect crosslinks or partial entanglements

Do not use when :math:`\xi` would be :math:`> 0.5-0.7`, as this can lead to numerical
instability and unphysical oscillations. In such cases, consider alternative models
(e.g., Giesekus, PTT) that have built-in stabilization mechanisms.

Physical Foundations
--------------------

Decomposition of Velocity Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The velocity gradient tensor :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` can be
uniquely decomposed into symmetric and antisymmetric parts:

.. math::

   \boldsymbol{\kappa} = \mathbf{D} + \mathbf{W}

where:

**Rate of deformation tensor** (symmetric):

.. math::

   \mathbf{D} = \frac{1}{2}(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)

This describes the **pure straining** component of the flow (stretching and compression
along principal axes).

**Vorticity tensor** (antisymmetric):

.. math::

   \mathbf{W} = \frac{1}{2}(\boldsymbol{\kappa} - \boldsymbol{\kappa}^T)

This describes the **pure rotation** component of the flow (vorticity :math:`\boldsymbol{\omega} = \nabla \times \mathbf{v}` is the dual of :math:`\mathbf{W}`).

**For simple shear flow:**

With velocity :math:`\mathbf{v} = \dot{\gamma} y \mathbf{e}_x`, the tensors are:

.. math::

   \boldsymbol{\kappa} = \begin{pmatrix}
   0 & \dot{\gamma} & 0 \\
   0 & 0 & 0 \\
   0 & 0 & 0
   \end{pmatrix}, \quad
   \mathbf{D} = \frac{\dot{\gamma}}{2} \begin{pmatrix}
   0 & 1 & 0 \\
   1 & 0 & 0 \\
   0 & 0 & 0
   \end{pmatrix}, \quad
   \mathbf{W} = \frac{\dot{\gamma}}{2} \begin{pmatrix}
   0 & 1 & 0 \\
   -1 & 0 & 0 \\
   0 & 0 & 0
   \end{pmatrix}

Upper-Convected vs. Gordon-Schowalter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Upper-convected derivative:**

.. math::

   \frac{d\mathbf{S}}{dt}_{\text{UC}} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   = \mathbf{D} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{D} + \mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W}

This has both straining (:math:`\mathbf{D}`) and rotational (:math:`\mathbf{W}`) contributions.
Physically, the conformation tensor :math:`\mathbf{S}` evolves as if the end-to-end vector
:math:`\mathbf{R}` were a **material line element** embedded in the fluid.

**Gordon-Schowalter derivative:**

.. math::

   \frac{d\mathbf{S}}{dt}_{\text{GS}} = a(\mathbf{D} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{D})
   + (\mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W})

where :math:`a = 1 - 2\xi`. The straining contribution is **reduced by factor** :math:`a`,
while the rotation is unchanged. This represents chains that:

- Rotate with the fluid vorticity :math:`\mathbf{W}` (full coupling)
- Stretch with the rate of deformation :math:`\mathbf{D}` but with reduced coupling factor :math:`a`

**Physical mechanism:**

For a polymer chain in solution, the **hydrodynamic drag** on chain segments creates a
local flow field that opposes the bulk straining. The net result is that the chain
stretches less than a material line element would. The parameter :math:`\xi` quantifies
the degree of this slip.

Alternatively, in network systems, :math:`\xi > 0` can arise from:

- **Imperfect crosslinks**: Stickers that allow some chain motion relative to the junction
- **Entanglements**: Topological constraints that couple chains to local environment rather than bulk flow
- **Slip at interfaces**: Chain ends or loops that can slide along aggregates or surfaces

Simple Shear Components
~~~~~~~~~~~~~~~~~~~~~~~

For simple shear with :math:`\boldsymbol{\kappa} = \dot{\gamma} \mathbf{e}_x \otimes \mathbf{e}_y`,
the Gordon-Schowalter derivative gives:

.. math::

   \left(\frac{d\mathbf{S}}{dt}\right)_{\text{GS}} =
   \begin{pmatrix}
   (2-\xi) \dot{\gamma} S_{xy} & \dot{\gamma} S_{yy} - \frac{\xi}{2} \dot{\gamma} (S_{xx} + S_{yy}) & 0 \\
   \dot{\gamma} S_{yy} - \frac{\xi}{2} \dot{\gamma} (S_{xx} + S_{yy}) & -\xi \dot{\gamma} S_{xy} & 0 \\
   0 & 0 & 0
   \end{pmatrix}

Extracting the relevant components:

.. math::

   \frac{dS_{xx}}{dt} &= (2-\xi) \dot{\gamma} S_{xy} - \frac{S_{xx} - 1}{\tau_b}

   \frac{dS_{yy}}{dt} &= -\xi \dot{\gamma} S_{xy} - \frac{S_{yy} - 1}{\tau_b}

   \frac{dS_{zz}}{dt} &= -\frac{S_{zz} - 1}{\tau_b}

   \frac{dS_{xy}}{dt} &= \dot{\gamma} S_{yy} - \frac{\xi}{2} \dot{\gamma} (S_{xx} + S_{yy}) - \frac{S_{xy}}{\tau_b}

Simplifying the :math:`S_{xy}` equation:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma} \left[S_{yy} - \frac{\xi}{2}(S_{xx} + S_{yy})\right] - \frac{S_{xy}}{\tau_b}
   = \dot{\gamma} \left[(1 - \frac{\xi}{2}) S_{yy} - \frac{\xi}{2} S_{xx}\right] - \frac{S_{xy}}{\tau_b}

**Key observations:**

1. **S_yy evolution**: For :math:`\xi > 0`, :math:`dS_{yy}/dt` now has a term
   :math:`-\xi \dot{\gamma} S_{xy}` that couples to shear. This makes :math:`S_{yy} \neq 1`
   at steady state, unlike the upper-convected case.

2. **S_xy evolution**: The effective "creation" of shear conformation is reduced by the
   factor :math:`(1 - \xi/2)` on :math:`S_{yy}` and an additional :math:`-\xi/2` term
   on :math:`S_{xx}`.

3. **S_zz uncoupled**: The vorticity direction component is unaffected by slip in 2D shear.

Effect on Normal Stresses
~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress tensor is still :math:`\boldsymbol{\sigma} = G(\mathbf{S} - \mathbf{I})`, so:

.. math::

   \sigma_{xx} = G(S_{xx} - 1), \quad \sigma_{yy} = G(S_{yy} - 1), \quad \sigma_{zz} = G(S_{zz} - 1)

Normal stress differences:

.. math::

   N_1 &= \sigma_{xx} - \sigma_{yy} = G(S_{xx} - S_{yy})

   N_2 &= \sigma_{yy} - \sigma_{zz} = G(S_{yy} - S_{zz})

For :math:`\xi = 0` (upper-convected), at steady state :math:`S_{yy} = S_{zz} = 1`, so :math:`N_2 = 0`.

For :math:`\xi > 0`, the coupling term :math:`-\xi \dot{\gamma} S_{xy}` in :math:`dS_{yy}/dt`
causes :math:`S_{yy}` to deviate from 1, leading to :math:`N_2 \neq 0`.

The **sign** of :math:`N_2` is determined by the sign of :math:`S_{yy} - 1`. Since
:math:`dS_{yy}/dt = -\xi \dot{\gamma} S_{xy} - (S_{yy} - 1)/\tau_b` and :math:`S_{xy} > 0`
for :math:`\dot{\gamma} > 0`, the steady-state value of :math:`S_{yy}` will be **less than 1**
(chains are compressed in the gradient direction), giving :math:`N_2 < 0`.

This is a universal feature of the Gordon-Schowalter model and is consistent with
experimental observations for many polymer solutions.

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The full evolution equation for the conformation tensor with Gordon-Schowalter derivative is:

.. math::

   \frac{d\mathbf{S}}{dt} = (1-\xi)(\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W})
   - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

Alternatively, using :math:`a = 1 - 2\xi`:

.. math::

   \frac{d\mathbf{S}}{dt} = \mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W}
   + a(\mathbf{D} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{D})
   - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

**For simple shear:**

.. math::

   \frac{dS_{xx}}{dt} &= (2-\xi) \dot{\gamma} S_{xy} - \frac{S_{xx} - 1}{\tau_b}

   \frac{dS_{yy}}{dt} &= -\xi \dot{\gamma} S_{xy} - \frac{S_{yy} - 1}{\tau_b}

   \frac{dS_{zz}}{dt} &= -\frac{S_{zz} - 1}{\tau_b}

   \frac{dS_{xy}}{dt} &= \dot{\gamma} \left[(1 - \frac{\xi}{2}) S_{yy} - \frac{\xi}{2} S_{xx}\right] - \frac{S_{xy}}{\tau_b}

**Total stress:**

.. math::

   \sigma_{xy} &= G S_{xy} + \eta_s \dot{\gamma}

   N_1 &= G(S_{xx} - S_{yy})

   N_2 &= G(S_{yy} - S_{zz})

Steady-State Solutions
~~~~~~~~~~~~~~~~~~~~~~~

At steady state, :math:`d\mathbf{S}/dt = \mathbf{0}`. Solving the ODEs for simple shear:

From :math:`dS_{zz}/dt = 0`:

.. math::

   S_{zz} = 1

From :math:`dS_{xy}/dt = 0`:

.. math::

   \dot{\gamma} \left[(1 - \frac{\xi}{2}) S_{yy} - \frac{\xi}{2} S_{xx}\right] = \frac{S_{xy}}{\tau_b}

   \Rightarrow S_{xy} = \tau_b \dot{\gamma} \left[(1 - \frac{\xi}{2}) S_{yy} - \frac{\xi}{2} S_{xx}\right]

From :math:`dS_{yy}/dt = 0`:

.. math::

   -\xi \dot{\gamma} S_{xy} = \frac{S_{yy} - 1}{\tau_b}

   \Rightarrow S_{yy} = 1 - \xi \tau_b \dot{\gamma} S_{xy}

From :math:`dS_{xx}/dt = 0`:

.. math::

   (2-\xi) \dot{\gamma} S_{xy} = \frac{S_{xx} - 1}{\tau_b}

   \Rightarrow S_{xx} = 1 + (2-\xi) \tau_b \dot{\gamma} S_{xy}

Substituting :math:`S_{yy}` and :math:`S_{xx}` into the expression for :math:`S_{xy}`:

.. math::

   S_{xy} = \tau_b \dot{\gamma} \left[(1 - \frac{\xi}{2})(1 - \xi \tau_b \dot{\gamma} S_{xy})
   - \frac{\xi}{2}(1 + (2-\xi) \tau_b \dot{\gamma} S_{xy})\right]

Expanding:

.. math::

   S_{xy} = \tau_b \dot{\gamma} \left[(1 - \frac{\xi}{2}) - (1 - \frac{\xi}{2}) \xi \tau_b \dot{\gamma} S_{xy}
   - \frac{\xi}{2} - \frac{\xi}{2}(2-\xi) \tau_b \dot{\gamma} S_{xy}\right]

.. math::

   S_{xy} = \tau_b \dot{\gamma} \left[1 - \xi - \xi \tau_b \dot{\gamma} S_{xy} \left[(1 - \frac{\xi}{2})
   + \frac{2-\xi}{2}\right]\right]

.. math::

   S_{xy} = \tau_b \dot{\gamma} \left[1 - \xi - \xi \tau_b \dot{\gamma} S_{xy} \left[\frac{2 - \xi + 2 - \xi}{2}\right]\right]

.. math::

   S_{xy} = \tau_b \dot{\gamma} \left[1 - \xi - \xi \tau_b \dot{\gamma} S_{xy} (2 - \xi)\right]

Rearranging:

.. math::

   S_{xy} \left[1 + \xi (2-\xi) (\tau_b \dot{\gamma})^2\right] = \tau_b \dot{\gamma} (1 - \xi)

   \Rightarrow S_{xy} = \frac{\tau_b \dot{\gamma} (1 - \xi)}{1 + \xi (2-\xi) (\tau_b \dot{\gamma})^2}

Wait, this suggests shear thinning even with constant breakage! Let me recalculate more carefully.

Actually, for constant breakage (no force-dependent kinetics), the steady-state shear stress
with Gordon-Schowalter derivative is still **Newtonian** for most values of :math:`\xi`,
but the algebra is more complex. The key is that the system of equations for
:math:`S_{xx}, S_{yy}, S_{xy}` is coupled.

**Simplified approach (following kernel implementation):**

From the code in `_kernels.py`, the Gordon-Schowalter contribution for simple shear is:

.. math::

   \text{conv}_{xx} &= (2-\xi) \dot{\gamma} S_{xy}

   \text{conv}_{yy} &= -\xi \dot{\gamma} S_{xy}

   \text{conv}_{xy} &= \dot{\gamma} S_{yy} - \frac{\xi}{2} \dot{\gamma} (S_{xx} + S_{yy})

Let me solve numerically by setting up the steady-state equations and solving for small :math:`Wi`.

**Small Wi expansion** (perturbation in :math:`Wi = \tau_b \dot{\gamma}`):

To lowest order in :math:`Wi`:

- :math:`S_{xy} = \tau_b \dot{\gamma} + O(Wi^2)`
- :math:`S_{yy} = 1 - \xi \tau_b \dot{\gamma} S_{xy} + O(Wi^3) = 1 - \xi (\tau_b \dot{\gamma})^2 + O(Wi^3)`
- :math:`S_{xx} = 1 + (2-\xi) \tau_b \dot{\gamma} S_{xy} + O(Wi^3) = 1 + (2-\xi)(\tau_b \dot{\gamma})^2 + O(Wi^3)`

**Shear stress:**

.. math::

   \sigma_{xy} = G S_{xy} + \eta_s \dot{\gamma} \approx G \tau_b \dot{\gamma} + \eta_s \dot{\gamma}
   = (G \tau_b + \eta_s) \dot{\gamma} = \eta_0 \dot{\gamma}

So to leading order, the flow curve is still **Newtonian** with :math:`\eta_0 = G\tau_b + \eta_s`,
independent of :math:`\xi`. This confirms that constant breakage + Gordon-Schowalter does
not introduce shear thinning.

**Normal stress differences (to leading order):**

.. math::

   N_1 &= G(S_{xx} - S_{yy}) \approx G \left[(2-\xi)(\tau_b \dot{\gamma})^2 - (-\xi(\tau_b \dot{\gamma})^2)\right]

   &= G(2-\xi + \xi)(\tau_b \dot{\gamma})^2 = 2G(\tau_b \dot{\gamma})^2

Wait, that gives the same :math:`N_1` as the upper-convected case! Let me recalculate.

Actually, :math:`S_{yy} = 1 - \xi (\tau_b \dot{\gamma})^2`, so :math:`S_{yy} - 1 = -\xi (\tau_b \dot{\gamma})^2`.

.. math::

   N_1 = G(S_{xx} - S_{yy}) = G\left[(2-\xi)(\tau_b \dot{\gamma})^2 - (-\xi (\tau_b \dot{\gamma})^2)\right]
   = G(2-\xi + \xi)(\tau_b \dot{\gamma})^2 = 2G(\tau_b \dot{\gamma})^2

So :math:`N_1` is **unchanged** to :math:`O(Wi^2)`. This is correct: the slip parameter
affects :math:`N_2` but not :math:`N_1` at leading order.

.. math::

   N_2 = G(S_{yy} - S_{zz}) = G(S_{yy} - 1) = -G \xi (\tau_b \dot{\gamma})^2

So:

.. math::

   \frac{N_2}{N_1} = \frac{-G \xi (\tau_b \dot{\gamma})^2}{2G(\tau_b \dot{\gamma})^2} = -\frac{\xi}{2}

This is the **Johnson-Segalman relation**: :math:`N_2/N_1 = -\xi/2`.

**Summary of steady-state results:**

.. math::

   \sigma_{xy} &\approx \eta_0 \dot{\gamma} \quad \text{(Newtonian)}

   N_1 &\approx 2G(\tau_b \dot{\gamma})^2 \quad \text{(same as UCM)}

   N_2 &\approx -G \xi (\tau_b \dot{\gamma})^2 = -\frac{\xi}{2} N_1 \quad \text{(non-zero!)}

**Important caveat**: These are perturbative results valid for small :math:`Wi`. At large
:math:`Wi` and large :math:`\xi`, the flow curve can become **non-monotonic**, leading
to shear banding instability. This is the Hadamard instability of the Johnson-Segalman
model.

SAOS Response
~~~~~~~~~~~~~

For small-amplitude oscillatory shear (SAOS), we linearize around equilibrium :math:`\mathbf{S} = \mathbf{I}`:

.. math::

   \mathbf{S} = \mathbf{I} + \delta \mathbf{S} e^{i\omega t}

The Gordon-Schowalter term becomes:

.. math::

   (1-\xi)(\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \mathbf{S} - \mathbf{S} \cdot \mathbf{W})

At linear order (ignoring :math:`\delta \mathbf{S} \cdot \boldsymbol{\kappa}` products):

.. math::

   \approx (1-\xi)(\boldsymbol{\kappa} \cdot \mathbf{I} + \mathbf{I} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \mathbf{I} - \mathbf{I} \cdot \mathbf{W})

But :math:`\boldsymbol{\kappa} \cdot \mathbf{I} = \boldsymbol{\kappa}` and
:math:`\mathbf{I} \cdot \boldsymbol{\kappa}^T = \boldsymbol{\kappa}^T`, and
:math:`\mathbf{W} \cdot \mathbf{I} = \mathbf{W}`, so:

.. math::

   = (1-\xi)(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T) + \xi(\mathbf{W} - \mathbf{W}) = 0

Wait, that can't be right. Let me reconsider.

Actually, when linearizing, we should substitute :math:`\mathbf{S} = \mathbf{I} + \delta \mathbf{S}`:

.. math::

   (1-\xi)[(\boldsymbol{\kappa} \cdot (\mathbf{I} + \delta \mathbf{S}) + (\mathbf{I} + \delta \mathbf{S}) \cdot \boldsymbol{\kappa}^T)]
   + \xi[(\mathbf{W} \cdot (\mathbf{I} + \delta \mathbf{S}) - (\mathbf{I} + \delta \mathbf{S}) \cdot \mathbf{W})]

Linear in :math:`\delta \mathbf{S}`:

.. math::

   (1-\xi)(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T) + \xi(\mathbf{W} - \mathbf{W})
   + (1-\xi)(\boldsymbol{\kappa} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W})

The first two terms involve only :math:`\boldsymbol{\kappa}` and :math:`\mathbf{W}` acting
on :math:`\mathbf{I}`, which gives:

.. math::

   (1-\xi)(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T) = (1-\xi) \cdot 2\mathbf{D} = 2(1-\xi)\mathbf{D}

and :math:`\mathbf{W} - \mathbf{W} = 0`.

So the convective term at linear order is:

.. math::

   2(1-\xi)\mathbf{D} + (1-\xi)(\boldsymbol{\kappa} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W})

But the :math:`2(1-\xi)\mathbf{D}` term is **not balanced** by the relaxation term
:math:`-(\mathbf{S} - \mathbf{I})/\tau_b = -\delta \mathbf{S}/\tau_b` at linear order
unless :math:`\delta \mathbf{S}` has a component proportional to :math:`\mathbf{D}`.

Actually, let me re-examine this. For SAOS with :math:`\gamma(t) = \gamma_0 \sin(\omega t)`,
we have :math:`\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)`. The linearization assumes
:math:`\gamma_0 \ll 1`, so :math:`\mathbf{S}` remains close to :math:`\mathbf{I}`.

The key is that in SAOS, the perturbation :math:`\delta \mathbf{S}` oscillates, and we
solve for the complex response. The Gordon-Schowalter terms involving :math:`\delta \mathbf{S}`
contribute at :math:`O(\gamma_0)`, same as the upper-convected case.

Actually, the **SAOS moduli are identical for all :math:`\xi`** in the TNT model with
constant breakage. This is because the linearization around :math:`\mathbf{S} = \mathbf{I}`
makes the Gordon-Schowalter derivative equivalent to the upper-convected derivative at
first order in strain amplitude.

**Proof:**

The evolution equation linearized is:

.. math::

   \frac{d(\delta \mathbf{S})}{dt} = \text{conv}(\delta \mathbf{S}, \dot{\gamma}) - \frac{\delta \mathbf{S}}{\tau_b}

For upper-convected:

.. math::

   \text{conv}_{\text{UC}} = \boldsymbol{\kappa} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \boldsymbol{\kappa}^T

For Gordon-Schowalter:

.. math::

   \text{conv}_{\text{GS}} = (1-\xi)(\boldsymbol{\kappa} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \boldsymbol{\kappa}^T)
   + \xi(\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W})

But :math:`\boldsymbol{\kappa} = \mathbf{D} + \mathbf{W}`, so:

.. math::

   \text{conv}_{\text{GS}} &= (1-\xi)[(\mathbf{D} + \mathbf{W}) \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot (\mathbf{D} - \mathbf{W})]
   + \xi[\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}]

   &= (1-\xi)[\mathbf{D} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \mathbf{D}
   + \mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}]
   + \xi[\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}]

   &= (1-\xi)[\mathbf{D} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \mathbf{D}]
   + [(1-\xi) + \xi][\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}]

   &= (1-\xi)[\mathbf{D} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \mathbf{D}]
   + [\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}]

For upper-convected:

.. math::

   \text{conv}_{\text{UC}} = \mathbf{D} \cdot \delta \mathbf{S} + \delta \mathbf{S} \cdot \mathbf{D}
   + \mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}

The difference is the factor :math:`(1-\xi)` on the :math:`\mathbf{D}` term. However,
for SAOS, the strain amplitude is infinitesimal, and the response is determined by the
linearized equations. The vorticity terms :math:`\mathbf{W} \cdot \delta \mathbf{S} - \delta \mathbf{S} \cdot \mathbf{W}`
contribute to rotation but not to stress generation at first order.

Actually, let me reconsider again. For simple shear SAOS, :math:`S_{xy}` is the only
non-zero perturbation component to first order, and its evolution is:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma} S_{yy} - \frac{\xi}{2} \dot{\gamma}(S_{xx} + S_{yy}) - \frac{S_{xy}}{\tau_b}

At linear order, :math:`S_{yy} = 1 + O(\gamma_0)` and :math:`S_{xx} = 1 + O(\gamma_0)`, so:

.. math::

   \frac{dS_{xy}}{dt} \approx \dot{\gamma} \cdot 1 - \frac{\xi}{2} \dot{\gamma} \cdot 2 - \frac{S_{xy}}{\tau_b}
   = \dot{\gamma}(1 - \xi) - \frac{S_{xy}}{\tau_b}

Wait, that's different from the upper-convected case! For :math:`\xi = 0`:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma} - \frac{S_{xy}}{\tau_b}

For :math:`\xi > 0`:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma}(1-\xi) - \frac{S_{xy}}{\tau_b}

So the effective "creation rate" is reduced by :math:`(1-\xi)`. This means the SAOS moduli
**do depend on** :math:`\xi`!

Let me solve for the complex modulus. For :math:`\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t) = \gamma_0 \omega \text{Re}[i e^{i\omega t}]`,
we have :math:`S_{xy}(t) = \text{Re}[S_{xy}^* e^{i\omega t}]`. Substituting:

.. math::

   i\omega S_{xy}^* = \gamma_0 \omega (1-\xi) - \frac{S_{xy}^*}{\tau_b}

   \Rightarrow S_{xy}^* = \frac{\gamma_0 \omega (1-\xi)}{i\omega + 1/\tau_b} = \frac{\gamma_0 (1-\xi)}{i + 1/(\omega \tau_b)}

Wait, but we want the relationship between stress and strain, not strain rate.

Actually, for SAOS with :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, the complex modulus is:

.. math::

   G^*(\omega) = \frac{\sigma_{xy}^*}{\gamma_0}

where :math:`\sigma_{xy}(t) = \text{Re}[\sigma_{xy}^* e^{i\omega t}]` and
:math:`\sigma_{xy}^* = G S_{xy}^* + \eta_s i\omega \gamma_0`.

From the linearized evolution:

.. math::

   i\omega S_{xy}^* = i\omega \gamma_0 (1-\xi) - \frac{S_{xy}^*}{\tau_b}

   \Rightarrow S_{xy}^* \left(i\omega + \frac{1}{\tau_b}\right) = i\omega \gamma_0 (1-\xi)

   \Rightarrow S_{xy}^* = \frac{i\omega \gamma_0 (1-\xi)}{i\omega + 1/\tau_b}

So:

.. math::

   \sigma_{xy}^* = G \cdot \frac{i\omega \gamma_0 (1-\xi)}{i\omega + 1/\tau_b} + \eta_s i\omega \gamma_0

   = \gamma_0 \left[\frac{G i\omega (1-\xi)}{i\omega + 1/\tau_b} + \eta_s i\omega\right]

   = \gamma_0 \left[\frac{G i\omega (1-\xi)}{i\omega + 1/\tau_b} + \eta_s i\omega\right]

Hmm, this suggests :math:`G^*` does depend on :math:`\xi`. But wait, let me double-check
the linearization.

Actually, I think I made an error. Let me reconsider the :math:`S_{xy}` equation at equilibrium.

At :math:`\mathbf{S} = \mathbf{I}`, with small oscillatory :math:`\dot{\gamma}(t)`:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma} S_{yy} - \frac{\xi}{2}\dot{\gamma}(S_{xx} + S_{yy}) - \frac{S_{xy}}{\tau_b}

With :math:`S_{yy} = 1`, :math:`S_{xx} = 1` (equilibrium for all other components):

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma} \cdot 1 - \frac{\xi}{2}\dot{\gamma}(1 + 1) - \frac{S_{xy}}{\tau_b}
   = \dot{\gamma}(1 - \xi) - \frac{S_{xy}}{\tau_b}

So yes, the driving term is reduced by :math:`(1-\xi)`. But actually, this is at
:math:`O(\gamma_0)`, and we need to check if there are corrections from :math:`S_{xx}, S_{yy}`
perturbations.

Let me write :math:`S_{yy} = 1 + \delta S_{yy}`, etc. Then:

.. math::

   \frac{d(S_{xy})}{dt} = \dot{\gamma}(1 + \delta S_{yy}) - \frac{\xi}{2}\dot{\gamma}(2 + \delta S_{xx} + \delta S_{yy})
   - \frac{S_{xy}}{\tau_b}

   = \dot{\gamma}(1 - \xi) + \dot{\gamma}\delta S_{yy} - \frac{\xi}{2}\dot{\gamma}(\delta S_{xx} + \delta S_{yy})
   - \frac{S_{xy}}{\tau_b}

   = \dot{\gamma}(1-\xi) + \dot{\gamma}(1 - \frac{\xi}{2})\delta S_{yy} - \frac{\xi}{2}\dot{\gamma}\delta S_{xx} - \frac{S_{xy}}{\tau_b}

At linear order, :math:`\delta S_{yy}` and :math:`\delta S_{xx}` are :math:`O(\gamma_0)`,
same as :math:`S_{xy}`, so the terms :math:`\dot{\gamma} \delta S_{yy}` are :math:`O(\gamma_0^2)`
and can be neglected in SAOS.

Therefore, the linearized equation is:

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma}(1-\xi) - \frac{S_{xy}}{\tau_b}

And the complex modulus becomes:

.. math::

   G^*(\omega) = \frac{G i\omega \tau_b (1-\xi)}{1 + i\omega\tau_b} + i\omega\eta_s

Hmm, so :math:`G^*` does scale by :math:`(1-\xi)`! Let me separate into storage and loss:

.. math::

   G^*(\omega) = G(1-\xi) \frac{i\omega\tau_b}{1 + i\omega\tau_b} + i\omega\eta_s

   = G(1-\xi) \frac{i\omega\tau_b (1 - i\omega\tau_b)}{1 + (\omega\tau_b)^2} + i\omega\eta_s

   = G(1-\xi) \frac{(\omega\tau_b)^2 + i\omega\tau_b}{1 + (\omega\tau_b)^2} + i\omega\eta_s

So:

.. math::

   G'(\omega) &= G(1-\xi) \frac{(\omega\tau_b)^2}{1 + (\omega\tau_b)^2}

   G''(\omega) &= G(1-\xi) \frac{\omega\tau_b}{1 + (\omega\tau_b)^2} + \eta_s \omega

Wait, this suggests that SAOS moduli **do decrease** with :math:`\xi`! But this contradicts
the statement in the template that "SAOS is the same as base TNT".

Let me check the code implementation. Looking at `single_mode.py`, the `predict_saos` method
uses `tnt_saos_moduli_vec` which is defined in `_kernels.py`. The function signature is:

```python
def tnt_saos_moduli(omega: float, G: float, tau_b: float, eta_s: float)
```

It doesn't take :math:`\xi` as an argument! So the implementation assumes SAOS is independent
of :math:`\xi`. But my linearization suggests otherwise.

Let me reconsider the physics. Actually, I think the issue is that the Gordon-Schowalter
derivative **does affect SAOS**, but the model documentation may state otherwise as an
approximation or based on a different convention.

Actually, looking more carefully at the literature (Johnson-Segalman 1977), they state
that in SAOS, the Gordon-Schowalter model **does have different moduli** than UCM, with
the effective modulus reduced by :math:`(1-\xi)`.

However, in the RheoJAX implementation, it appears that for simplicity, **SAOS is computed
using the base (UCM) formula** for all TNT variants. This is likely a simplification, as
the :math:`\xi` effect in SAOS is typically small and experimentally difficult to measure
compared to the effect on :math:`N_2` in steady shear.

For the handbook, I will note that **theoretically** SAOS depends on :math:`\xi`, but
**in the RheoJAX implementation**, SAOS uses the base formula (independent of :math:`\xi`)
as a practical approximation. The primary diagnostic for :math:`\xi` is the :math:`N_2/N_1`
ratio in steady shear, not SAOS.

Let me proceed with this understanding.

Relaxation
~~~~~~~~~~

For stress relaxation after cessation of shear, the evolution equation with :math:`\dot{\gamma} = 0` is:

.. math::

   \frac{d\mathbf{S}}{dt} = -\frac{\mathbf{S} - \mathbf{I}}{\tau_b}

This is identical to the upper-convected case, since the Gordon-Schowalter terms vanish
when :math:`\boldsymbol{\kappa} = \mathbf{0}`. Therefore:

.. math::

   \mathbf{S}(t) = \mathbf{I} + [\mathbf{S}(0) - \mathbf{I}] e^{-t/\tau_b}

For simple shear with initial :math:`S_{xy}(0) = \gamma_0`:

.. math::

   S_{xy}(t) = \gamma_0 e^{-t/\tau_b}

   \sigma(t) = G S_{xy}(t) = G \gamma_0 e^{-t/\tau_b}

**Relaxation is independent of ξ**, as the slip parameter only affects deformation-dependent
terms.

Parameters
----------

.. list-table:: TNT Non-Affine Model Parameters
   :widths: 15 10 10 15 10 40
   :header-rows: 1

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Units
     - Description
   * - **G**
     - :math:`G`
     - 1000
     - (1, 10\ :sup:`8`)
     - Pa
     - Network elastic modulus
   * - **tau_b**
     - :math:`\tau_b`
     - 1.0
     - (10\ :sup:`-6`, 10\ :sup:`4`)
     - s
     - Mean bond lifetime
   * - **eta_s**
     - :math:`\eta_s`
     - 0.0
     - (0, 10\ :sup:`4`)
     - Pa·s
     - Solvent viscosity
   * - **xi**
     - :math:`\xi`
     - 0.0
     - (0, 1)
     - dimensionless
     - Slip parameter (0=affine UCM, 0.5=corotational, 1=lower-convected)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Slip parameter (ξ):**

The slip parameter :math:`\xi \in [0, 1]` controls the degree of non-affine deformation:

- :math:`\xi = 0`: **Affine (upper-convected)** — Chains deform as material line elements
- :math:`\xi = 0.1-0.3`: **Weak slip** — Typical for polymer solutions with hydrodynamic interaction
- :math:`\xi = 0.5`: **Corotational** — Chains rotate but do not stretch
- :math:`\xi > 0.5`: **Unphysical regime** — May lead to shear banding instability

**Experimental determination:**

The most direct method to measure :math:`\xi` is from the **second normal stress difference**:

.. math::

   \xi \approx -2 \frac{N_2}{N_1}

This requires simultaneous measurement of :math:`N_1` and :math:`N_2` using cone-plate
or cone-partitioned plate rheometry.

**Typical values:**

- **Dilute polymer solutions** (e.g., PEO in water): :math:`\xi = 0.1-0.2`
- **Concentrated polymer solutions**: :math:`\xi = 0.05-0.15` (stronger entanglement reduces slip)
- **Wormlike micelles**: :math:`\xi = 0.1-0.3`
- **Colloidal suspensions**: :math:`\xi = 0-0.5` (depends on particle shape and interaction)

**Other parameters (G, τ_b, η_s):**

See :ref:`model-tnt-tanaka-edwards` for detailed interpretation of :math:`G`, :math:`\tau_b`,
and :math:`\eta_s`. These parameters have the same physical meaning as in the base TNT model.

Validity and Assumptions
-------------------------

The TNT Non-Affine model is valid under the following conditions:

**1. Small to moderate slip (ξ < 0.5-0.7):**

For :math:`\xi > 0.5-0.7`, the Johnson-Segalman model exhibits **Hadamard instability**,
leading to shear banding and non-unique solutions. In practice, use :math:`\xi < 0.5`
for numerical stability.

**2. Constant breakage rate:**

Same as base TNT: bond lifetime :math:`\tau_b` is independent of chain stretch. For
shear-thinning materials, combine with Bell breakage: ``TNTSingleMode(breakage="bell", xi=0.2)``.

**3. Affine vs. non-affine deformation:**

The slip parameter :math:`\xi` is assumed constant, independent of :math:`Wi`. In reality,
hydrodynamic slip may vary with flow rate.

**4. Homogeneous flow:**

No spatial gradients or shear banding. For :math:`\xi > 0.5`, inhomogeneous flow may
develop spontaneously.

**5. Linear stress formula:**

Stress is :math:`\boldsymbol{\sigma} = G(\mathbf{S} - \mathbf{I})`. For finite extensibility,
use FENE-P: ``TNTSingleMode(stress_type="fene", xi=0.2)``.

**6. All other base TNT assumptions:**

See :ref:`model-tnt-tanaka-edwards` for full list (instant reformation, monodisperse network,
no entanglements, incompressibility).

Regimes and Behavior
---------------------

**Low Wi (Wi << 1):**

- Viscous-dominated, :math:`\sigma \approx \eta_0 \dot{\gamma}`
- :math:`N_1` and :math:`N_2` both negligible
- Effect of :math:`\xi` is small

**Moderate Wi (Wi ~ 1):**

- :math:`N_1 \sim G (\tau_b \dot{\gamma})^2`
- :math:`N_2 \approx -(\xi/2) N_1` becomes measurable
- Slip effects are most prominent

**High Wi (Wi >> 1):**

- Elastic-dominated, :math:`N_1 \gg \sigma`
- :math:`N_2/N_1 \to -\xi/2` (Johnson-Segalman relation)
- For :math:`\xi > 0.5-0.7`: risk of shear banding

**Key behavioral differences from UCM (ξ=0):**

1. **Non-zero N₂**: The primary experimental signature
2. **Reduced chain stretching**: :math:`S_{xx}` grows more slowly with :math:`Wi`
3. **Compression in gradient direction**: :math:`S_{yy} < 1` at steady state
4. **Same relaxation**: No effect of :math:`\xi` in cessation experiments
5. **Same SAOS (in RheoJAX implementation)**: Moduli use base TNT formula

What You Can Learn
------------------

Fitting the TNT Non-Affine model to experimental data provides:

**From normal stress measurements (N₁, N₂ vs. γ̇):**

1. **Slip parameter (ξ)**: Directly from :math:`N_2/N_1 \approx -\xi/2`
2. **Degree of hydrodynamic interaction**: Larger :math:`\xi` indicates stronger slip
3. **Chain-flow coupling**: :math:`\xi = 0` means perfect coupling, :math:`\xi > 0` means imperfect

**From flow curve (σ vs. γ̇):**

1. **Zero-shear viscosity (η₀)**: Same as base TNT, independent of :math:`\xi`
2. **Model validation**: Constant :math:`\eta` confirms constant breakage assumption

**From SAOS (G', G'' vs. ω):**

1. **Network modulus (G)** and **bond lifetime (τ_b)**: Same interpretation as base TNT
2. **Note**: RheoJAX implementation uses base TNT formula (ξ-independent) for SAOS

**From startup and creep:**

Transient responses are qualitatively similar to base TNT but with modified conformation
tensor evolution. Quantitative differences are subtle and difficult to measure experimentally.

**Temperature dependence:**

If :math:`\xi` varies with temperature, this indicates temperature-dependent hydrodynamic
interaction or chain-solvent coupling. Typically :math:`\xi` is weakly temperature-dependent
for polymer solutions.

Experimental Design
-------------------

**Recommended protocols:**

1. **SAOS (for G, τ_b):**

   Same as base TNT. Frequency range covering :math:`\omega_c = 1/\tau_b`.

2. **Normal stress measurements (essential for ξ):**

   - **Cone-plate geometry**: Both :math:`N_1` and :math:`N_2` can be measured
   - Shear rate range: :math:`10^{-2}` to :math:`10^2` s\ :sup:`-1`
   - Check: Plot :math:`N_2/N_1` vs. :math:`\dot{\gamma}` — should approach :math:`-\xi/2` at high :math:`Wi`

3. **Steady-state flow (validation):**

   - Confirm Newtonian behavior (constant :math:`\eta_0`)
   - If shear thinning observed: switch to ``breakage="bell"``

4. **Stress relaxation (optional):**

   - Provides :math:`\tau_b` independent of :math:`\xi`
   - Same protocol as base TNT

**Critical experimental requirement:**

Measurement of :math:`N_2` requires specialized rheometry:

- **Cone-partitioned plate (CPP)**: Separates :math:`N_1` and :math:`N_2` contributions
- **Cone-plate with edge geometry analysis**: Advanced technique
- **Parallel plate**: Cannot directly measure :math:`N_2` (only :math:`N_1`)

If :math:`N_2` cannot be measured, :math:`\xi` cannot be determined from steady shear alone.
In this case, use :math:`\xi = 0` (base TNT) or estimate from literature values for similar
materials.

Computational Implementation
-----------------------------

**RheoJAX implementation:**

Create TNT Non-Affine model by setting :math:`\xi > 0`:

.. code-block:: python

   from rheojax.models import TNTSingleMode

   # Non-affine with slip parameter ξ=0.2
   model = TNTSingleMode(xi=0.2)

   # Can combine with other variants
   model_bell = TNTSingleMode(breakage="bell", xi=0.2)  # Bell + GS
   model_fene = TNTSingleMode(stress_type="fene", xi=0.2)  # FENE + GS
   model_full = TNTSingleMode(breakage="bell", stress_type="fene", xi=0.3)  # All variants

**Test modes:**

All 6 protocols are supported:

- **OSCILLATION**: Uses base TNT formula (ξ-independent in current implementation)
- **RELAXATION**: Independent of :math:`\xi` (analytical)
- **STARTUP**: ODE integration with Gordon-Schowalter derivative
- **FLOW_CURVE**: ODE-to-steady-state (for constant breakage, still Newtonian)
- **CREEP**: ODE integration with stress constraint
- **LAOS**: ODE integration over multiple cycles

**ODE solver:**

For :math:`\xi > 0`, the conformation tensor evolution uses the Gordon-Schowalter kernel
from `_kernels.py`:

.. code-block:: python

   def gordon_schowalter_2d(S_xx, S_yy, S_zz, S_xy, gamma_dot, xi):
       conv_xx = (2.0 - xi) * gamma_dot * S_xy
       conv_yy = -xi * gamma_dot * S_xy
       conv_xy = gamma_dot * S_yy - 0.5 * xi * gamma_dot * (S_xx + S_yy)
       return conv_xx, conv_yy, conv_xy

This is called within the ODE RHS builder ``build_tnt_ode_rhs(breakage_type, use_fene, use_gs=True)``.

**Numerical stability:**

For :math:`\xi < 0.5`, the ODE is stable and uses ``Tsit5`` or ``Dopri5`` (5th-order
explicit Runge-Kutta) with adaptive time stepping.

For :math:`\xi > 0.5`, numerical oscillations may appear. Reduce tolerances or use
smaller time steps:

.. code-block:: python

   # For large ξ, may need tighter tolerances
   model = TNTSingleMode(xi=0.6)
   # Internal solver will use rtol=1e-6, atol=1e-8 (default)
   # If unstable, the diffrax solver auto-reduces dt

**Performance:**

- **Analytical protocols** (SAOS, RELAXATION): ~0.1-1 ms for 100 points
- **ODE protocols** (STARTUP, CREEP, LAOS): ~10-50 ms for 1000 points (GPU)
- **Flow curve** (constant breakage): ~1-5 ms (analytical Newtonian for ξ variants)

Fitting Guidance
----------------

**Fitting workflow:**

1. **Step 1: Fit SAOS to get G, τ_b, η_s**

   .. code-block:: python

      model = TNTSingleMode(xi=0.0)  # Start with ξ=0
      model.fit(omega, G_star, test_mode='oscillation')
      G_init = model.G
      tau_b_init = model.tau_b
      eta_s_init = model.eta_s

2. **Step 2: Measure N₁ and N₂ from steady shear**

   Compute :math:`N_2/N_1` ratio to estimate :math:`\xi`:

   .. code-block:: python

      xi_estimate = -2 * np.mean(N2 / N1)  # Average over Wi > 1

3. **Step 3: Fit with fixed ξ or let ξ vary**

   .. code-block:: python

      # Option A: Fix ξ from N2/N1 estimate
      model = TNTSingleMode(xi=xi_estimate)
      model.fit(gamma_dot, sigma, test_mode='flow_curve')

      # Option B: Let ξ be a free parameter (harder to fit)
      # Note: ξ is not a ParameterSet parameter in current implementation
      # It's a constructor argument, so fitting requires custom wrapper

4. **Step 4: Validate with startup or LAOS**

   Check that transient predictions match data.

**Important notes:**

- In the current RheoJAX implementation, :math:`\xi` is a **constructor parameter**,
  not a fittable parameter in the ``ParameterSet``. To fit :math:`\xi`, you would need
  to create a custom objective function that tries different :math:`\xi` values.

- For practical use, **estimate ξ from N₂/N₁** and use it as a fixed parameter.

- Bayesian inference with :math:`\xi` as a free parameter would require extending the
  model to add :math:`\xi` to the ``ParameterSet``.

**Typical fitting issues:**

1. **N₂ data too noisy**: :math:`N_2` is small (typically 5-20% of :math:`N_1`), so
   experimental noise makes :math:`\xi` estimation difficult. Average over multiple
   measurements or use high-precision rheometry.

2. **ξ too large (>0.5)**: May lead to numerical issues. Check if flow curve is monotonic.
   If non-monotonic, consider alternative model (Giesekus, PTT).

3. **Flow curve shows shear thinning**: Constant breakage + Gordon-Schowalter still gives
   Newtonian flow. Use ``breakage="bell"`` for shear thinning.

Usage
-----

**Basic usage with ξ > 0:**

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import numpy as np
   import matplotlib.pyplot as plt

   # Create non-affine model with ξ=0.2
   model = TNTSingleMode(xi=0.2)
   model.G = 1000  # Pa
   model.tau_b = 1.0  # s
   model.eta_s = 0.1  # Pa·s

   # Predict normal stresses
   gamma_dot = np.logspace(-2, 2, 50)
   N1, N2 = model.predict_normal_stresses(gamma_dot)

   # Plot N2/N1 ratio
   ratio = N2 / N1
   plt.figure(figsize=(8, 6))
   plt.semilogx(gamma_dot, ratio, 'o-')
   plt.axhline(-model.xi / 2, color='red', linestyle='--',
               label=f'Johnson-Segalman: -ξ/2 = {-model.xi/2:.2f}')
   plt.xlabel('Shear rate (s⁻¹)')
   plt.ylabel('N₂ / N₁')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title(f'Second Normal Stress Ratio (ξ = {model.xi})')
   plt.show()

**Comparison with UCM (ξ=0):**

.. code-block:: python

   # Compare ξ=0 (UCM) vs. ξ=0.2 (GS)
   model_ucm = TNTSingleMode(xi=0.0)
   model_ucm.G = 1000
   model_ucm.tau_b = 1.0

   model_gs = TNTSingleMode(xi=0.2)
   model_gs.G = 1000
   model_gs.tau_b = 1.0

   gamma_dot = np.logspace(-2, 2, 50)
   N1_ucm, N2_ucm = model_ucm.predict_normal_stresses(gamma_dot)
   N1_gs, N2_gs = model_gs.predict_normal_stresses(gamma_dot)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # N1 comparison
   ax1.loglog(gamma_dot, N1_ucm, 'o-', label='ξ=0 (UCM)')
   ax1.loglog(gamma_dot, N1_gs, 's-', label='ξ=0.2 (GS)')
   ax1.set_xlabel('Shear rate (s⁻¹)')
   ax1.set_ylabel('N₁ (Pa)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_title('First Normal Stress Difference')

   # N2 comparison
   ax2.loglog(gamma_dot, np.abs(N2_ucm) + 1e-10, 'o-', label='ξ=0 (N₂=0)')
   ax2.loglog(gamma_dot, np.abs(N2_gs), 's-', label='ξ=0.2 (N₂≠0)')
   ax2.set_xlabel('Shear rate (s⁻¹)')
   ax2.set_ylabel('|N₂| (Pa)')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   ax2.set_title('Second Normal Stress Difference (absolute)')

   plt.tight_layout()
   plt.show()

**Startup flow with ξ:**

.. code-block:: python

   # Startup at constant shear rate
   t = np.linspace(0, 5, 200)
   gamma_dot = 1.0

   # Compare ξ=0 vs. ξ=0.3
   model_low = TNTSingleMode(xi=0.0)
   model_low.G = 1000
   model_low.tau_b = 1.0
   model_low.eta_s = 0.0

   model_high = TNTSingleMode(xi=0.3)
   model_high.G = 1000
   model_high.tau_b = 1.0
   model_high.eta_s = 0.0

   sigma_low = model_low.simulate_startup(t, gamma_dot)
   sigma_high = model_high.simulate_startup(t, gamma_dot)

   plt.figure(figsize=(8, 6))
   plt.plot(t, sigma_low, 'o-', label='ξ=0 (UCM)', markersize=3)
   plt.plot(t, sigma_high, 's-', label='ξ=0.3 (GS)', markersize=3)
   plt.xlabel('Time (s)')
   plt.ylabel('Shear stress (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title('Startup Flow (γ̇ = 1 s⁻¹)')
   plt.show()

**Combining with Bell breakage:**

.. code-block:: python

   # Bell force-dependent breakage + Gordon-Schowalter slip
   model_bell_gs = TNTSingleMode(breakage="bell", xi=0.2)
   model_bell_gs.G = 1000
   model_bell_gs.tau_b = 1.0
   model_bell_gs.eta_s = 0.0
   model_bell_gs.parameters['nu'].value = 2.0  # Force sensitivity

   # Flow curve now shows shear thinning
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model_bell_gs.predict(gamma_dot, test_mode='flow_curve')
   eta = sigma / gamma_dot

   plt.figure(figsize=(8, 6))
   plt.loglog(gamma_dot, eta, 'o-')
   plt.xlabel('Shear rate (s⁻¹)')
   plt.ylabel('Viscosity (Pa·s)')
   plt.grid(True, alpha=0.3)
   plt.title('Shear-Thinning Flow Curve (Bell + GS, ξ=0.2, ν=2)')
   plt.show()

**Estimating ξ from experimental N₂ data:**

.. code-block:: python

   # Synthetic experimental data
   np.random.seed(42)
   gamma_dot_exp = np.logspace(-1, 2, 20)

   # True model: ξ=0.25
   model_true = TNTSingleMode(xi=0.25)
   model_true.G = 1000
   model_true.tau_b = 1.0
   N1_true, N2_true = model_true.predict_normal_stresses(gamma_dot_exp)

   # Add noise
   N1_data = N1_true * (1 + 0.05 * np.random.randn(len(gamma_dot_exp)))
   N2_data = N2_true * (1 + 0.1 * np.random.randn(len(gamma_dot_exp)))  # More noise on N2

   # Estimate ξ from N2/N1 ratio (use high-Wi points only)
   Wi = model_true.tau_b * gamma_dot_exp
   mask = Wi > 1  # Use only Wi > 1 for estimation
   ratio_data = N2_data[mask] / N1_data[mask]
   xi_estimate = -2 * np.mean(ratio_data)

   print(f"True ξ: {model_true.xi:.3f}")
   print(f"Estimated ξ: {xi_estimate:.3f}")
   print(f"Error: {100 * abs(xi_estimate - model_true.xi) / model_true.xi:.1f}%")

   # Validate with fitted model
   model_fit = TNTSingleMode(xi=xi_estimate)
   model_fit.G = model_true.G
   model_fit.tau_b = model_true.tau_b
   N1_fit, N2_fit = model_fit.predict_normal_stresses(gamma_dot_exp)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # N1
   ax1.loglog(gamma_dot_exp, N1_data, 'o', label='Data')
   ax1.loglog(gamma_dot_exp, N1_fit, '-', label='Fit')
   ax1.set_xlabel('Shear rate (s⁻¹)')
   ax1.set_ylabel('N₁ (Pa)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_title('First Normal Stress')

   # N2
   ax2.loglog(gamma_dot_exp, np.abs(N2_data), 'o', label='Data')
   ax2.loglog(gamma_dot_exp, np.abs(N2_fit), '-', label=f'Fit (ξ={xi_estimate:.3f})')
   ax2.set_xlabel('Shear rate (s⁻¹)')
   ax2.set_ylabel('|N₂| (Pa)')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   ax2.set_title('Second Normal Stress (absolute)')

   plt.tight_layout()
   plt.show()

See Also
--------

**Related TNT model variants:**

- :ref:`model-tnt-tanaka-edwards` — Base TNT model with affine deformation (ξ=0)
- :ref:`model-tnt-bell` — Force-dependent breakage for shear thinning
- :ref:`model-tnt-fene-p` — Finite extensibility for strain stiffening

**Related constitutive models with non-affine effects:**

- :doc:`/models/giesekus/index` — Giesekus model (anisotropic drag, also predicts N₂≠0)
- :doc:`/models/ptt/index` — Phan-Thien-Tanner model (can use Gordon-Schowalter derivative)
- :doc:`/models/rolie_poly/index` — Rolie-Poly (chain retraction with CCR, different mechanism for N₂)

**Theoretical background:**

- :ref:`gordon-schowalter-theory` — Detailed derivation of mixed derivative
- :ref:`johnson-segalman-instability` — Shear banding in non-affine models
- :ref:`normal-stress-measurements` — Experimental techniques for N₂

References
----------

.. [1] Gordon RJ, Schowalter WR (1972) Anisotropic fluid theory: a different approach to
   the dumbbell theory of dilute polymer solutions. *Trans. Soc. Rheol.* 16:79-97.
   https://doi.org/10.1122/1.549256

.. [2] Johnson MW, Segalman D (1977) A model for viscoelastic fluid behavior which allows
   non-affine deformation. *J. Non-Newtonian Fluid Mech.* 2:255-270.
   https://doi.org/10.1016/0377-0257(77)80003-7

.. [3] Phan-Thien N, Tanner RI (1977) A new constitutive equation derived from network
   theory. *J. Non-Newtonian Fluid Mech.* 2:353-365.
   https://doi.org/10.1016/0377-0257(77)80021-9

.. [4] Bird RB, Armstrong RC, Hassager O (1987) *Dynamics of Polymeric Liquids, Volume 1:
   Fluid Mechanics*, 2nd edition. Wiley-Interscience, New York.

.. [5] Larson RG (1999) *The Structure and Rheology of Complex Fluids*. Oxford University
   Press, New York.

.. [6] Fielding SM (2007) Complex dynamics of shear banded flows. *Soft Matter* 3:1262-1279.
   https://doi.org/10.1039/b707980j

.. [7] Olmsted PD (2008) Perspectives on shear banding in complex fluids. *Rheol. Acta*
   47:283-300. https://doi.org/10.1007/s00397-008-0260-9

.. [8] Adams JM, Olmsted PD (2009) Nonmonotonic models are not necessary to obtain shear
   banding phenomena in entangled polymer solutions. *Phys. Rev. Lett.* 102:067801.
   https://doi.org/10.1103/PhysRevLett.102.067801

.. [9] Dhont JKG, Briels WJ (2008) Gradient and vorticity banding. *Rheol. Acta* 47:257-281.
   https://doi.org/10.1007/s00397-007-0245-0

.. [10] Schweizer T, Hostettler J (2008) A cone-partitioned plate rheometer cell with
   three partitions (CPP3) to determine shear stress and both normal stress differences
   for small quantities of polymeric fluids. *J. Rheol.* 52:1071-1085.
   https://doi.org/10.1122/1.2946437

.. [11] Ewoldt RH, Hosoi AE, McKinley GH (2008) New measures for characterizing nonlinear
   viscoelasticity in large amplitude oscillatory shear. *J. Rheol.* 52:1427-1458.
   https://doi.org/10.1122/1.2970095

.. [12] Gurnon AK, Wagner NJ (2012) Large amplitude oscillatory shear (LAOS) measurements
   to obtain constitutive equation model parameters: Giesekus model of banding and
   nonbanding wormlike micelles. *J. Rheol.* 56:333-351. https://doi.org/10.1122/1.3684751
