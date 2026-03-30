.. _fluidity_saramito_equations:

==============================================================================
Mathematical Foundations: Fluidity and Saramito Models — Equation Verification
==============================================================================

This document provides the **complete governing equations** for the Fluidity
and Saramito model families implemented in RheoJAX, cross-referenced against
published literature. It serves as a verification reference for the
implementations in:

- ``rheojax/models/fluidity/_kernels.py`` (Scalar and Nonlocal Fluidity)
- ``rheojax/models/fluidity/saramito/_kernels.py`` (Fluidity-Saramito EVP)

.. contents:: Contents
   :local:
   :depth: 2


1. Scalar Fluidity Model
========================

**References:**

- Coussot, P., Nguyen, Q.D., Huynh, H.T., Bonn, D. (2002).
  *Viscosity bifurcation in thixotropic, yielding fluids.*
  J. Rheol. **46**\(3), 573--589.

- de Souza Mendes, P.R. (2009).
  *Modeling the thixotropic behavior of structured fluids.*
  J. Non-Newtonian Fluid Mech. **164**, 66--75.

- de Souza Mendes, P.R., Thompson, R.L. (2012).
  *A critical overview of elasto-viscoplastic thixotropic modeling.*
  J. Non-Newtonian Fluid Mech. **187--188**, 8--15.


1.1 State Variables
-------------------

The local fluidity model tracks two state variables:

- :math:`\sigma(t)` — deviatoric stress (Pa)
- :math:`f(t)` — fluidity (1/(Pa·s)), the scalar inverse of a structural
  viscosity

The fluidity represents the degree of structural breakdown: high fluidity
corresponds to a fluid-like (rejuvenated) state, low fluidity to a solid-like
(aged) state.


1.2 Stress Evolution (Rate-Controlled)
---------------------------------------

Under imposed constant shear rate :math:`\dot\gamma`:

.. math::

   \frac{d\sigma}{dt} = G\!\left(\dot\gamma - \sigma\, f\right)

where:

- :math:`G` — elastic modulus (Pa)
- :math:`\dot\gamma` — applied shear rate (1/s)
- :math:`\sigma\, f` — plastic strain rate contribution

**Physical interpretation:** The stress evolves via a Maxwell-like
viscoelastic balance. The elastic loading rate is :math:`G\dot\gamma` and the
relaxation rate is :math:`G\sigma f = \sigma / \lambda` where
:math:`\lambda = 1/(Gf)` is the effective relaxation time.

**Implementation:** ``fluidity_local_ode_rhs`` in ``_kernels.py``, line 198:
``d_sigma = G * (gamma_dot - sigma * f_safe)`` — **matches published form.**


1.3 Fluidity Evolution (Aging + Rejuvenation)
----------------------------------------------

.. math::

   \frac{df}{dt} = \underbrace{\frac{f_\text{eq} - f}{\theta}}_{\text{aging}}
   \;+\; \underbrace{a\,|\dot\gamma|^{n_r}\,(f_\infty - f)}_{\text{rejuvenation}}

where:

- :math:`f_\text{eq}` — equilibrium (low-shear) fluidity; the aging limit
- :math:`f_\infty` — high-shear fluidity; the rejuvenation limit
- :math:`\theta` — structural relaxation time (aging timescale, s)
- :math:`a` — rejuvenation amplitude (dimensionless)
- :math:`n_r` — rejuvenation exponent (dimensionless)

**Physical interpretation:**

- **Aging term** :math:`(f_\text{eq} - f)/\theta`: drives fluidity toward
  the rest-state value :math:`f_\text{eq}` on timescale :math:`\theta`.
  This represents spontaneous structural build-up (gelation, aging).

- **Rejuvenation term** :math:`a|\dot\gamma|^{n_r}(f_\infty - f)`: drives
  fluidity toward the flow-state value :math:`f_\infty` at a rate
  proportional to :math:`|\dot\gamma|^{n_r}`. This represents
  flow-induced structural breakdown.

**Origin:** This form generalizes the Coussot et al. (2002) model, which
used a simpler :math:`df/dt = 1/\theta_0 - \alpha f \dot\gamma` form
(equivalent with suitable parameter mapping). The aging/rejuvenation
decomposition with separate limits :math:`f_\text{eq}` and :math:`f_\infty`
follows de Souza Mendes (2009).

**Implementation:** ``fluidity_local_ode_rhs`` in ``_kernels.py``, lines
201--209 — **matches published form.** Note the ``1e-6`` floor on
:math:`|\dot\gamma|` to prevent gradient divergence when
:math:`n_r < 1`.


1.4 Steady-State Flow Curve
----------------------------

Setting :math:`df/dt = 0`:

.. math::

   f_\text{ss} = \frac{f_\text{eq}/\theta + a|\dot\gamma|^{n_r}\, f_\infty}
   {1/\theta + a|\dot\gamma|^{n_r}}

The steady-state stress is then:

.. math::

   \sigma_\text{ss} = \tau_y + \frac{|\dot\gamma|}{f_\text{ss}}

where :math:`\tau_y` is the yield stress providing the finite stress plateau
as :math:`\dot\gamma \to 0`.

**Implementation:** ``fluidity_local_steady_state`` in ``_kernels.py``,
lines 488--498 — **matches published form.**

**Viscosity bifurcation (Coussot et al. 2002):** For certain parameter
ranges, the steady-state flow curve :math:`\sigma(\dot\gamma)` is
non-monotonic, producing a region of negative slope. Under stress-controlled
conditions, this leads to *viscosity bifurcation*: below a critical stress
:math:`\sigma_c`, the material jams (fluidity collapses to
:math:`f_\text{eq}`); above :math:`\sigma_c`, the material flows (fluidity
jumps to :math:`\sim f_\infty`). There are no stable steady states at
intermediate shear rates.


1.5 Creep Equations (Stress-Controlled)
----------------------------------------

Under constant applied stress :math:`\sigma_\text{app}`:

.. math::

   \frac{d\gamma}{dt} &= \sigma_\text{app}\, f

   \frac{df}{dt} &= \frac{f_\text{eq} - f}{\theta}
   + a\,|\sigma_\text{app}\, f|^{n_r}\,(f_\infty - f)

**Physical interpretation:** The strain rate is directly
:math:`\dot\gamma = \sigma f` (fluidity definition). The fluidity
evolution is the same aging/rejuvenation equation, with the driving
rate :math:`|\dot\gamma| = |\sigma_\text{app}\, f|` replacing the
imposed shear rate.

**Creep bifurcation:** If :math:`\sigma_\text{app} < \sigma_c` (critical
stress), fluidity decays to :math:`f_\text{eq}` and the material arrests.
If :math:`\sigma_\text{app} > \sigma_c`, rejuvenation wins and the
material flows indefinitely with accelerating strain rate.

**Implementation:** ``fluidity_local_creep_ode_rhs`` in ``_kernels.py``,
lines 260--271 — **matches published form.**


2. Nonlocal Fluidity Model
============================

**References:**

- Goyon, J., Colin, A., Ovarlez, G., Ajdari, A., Bocquet, L. (2008).
  *Spatial cooperativity in soft glassy flows.*
  Nature **454**, 84--87.

- Bocquet, L., Colin, A., Ajdari, A. (2009).
  *Kinetic theory of plastic flow in soft glassy materials.*
  Phys. Rev. Lett. **103**, 036001.

- Ovarlez, G., Cohen-Addad, S., Krishan, K., Goyon, J., Coussot, P. (2012).
  *On the existence of a simple yield stress fluid behavior.*
  J. Non-Newtonian Fluid Mech. **177--178**, 19--28.


2.1 Nonlocal Fluidity PDE
---------------------------

The nonlocal model extends the scalar fluidity to a spatially-resolved
field :math:`f(y, t)` across the flow gap:

.. math::

   \frac{\partial f}{\partial t}
   = \frac{f_\text{loc}(\sigma) - f}{\theta}
   + \xi^2 \frac{\partial^2 f}{\partial y^2}

where:

- :math:`f_\text{loc}(\sigma)` — local equilibrium fluidity from the
  Herschel-Bulkley flow curve (see below)
- :math:`\theta` — structural relaxation time (s)
- :math:`\xi` — cooperativity length (m), the key nonlocal parameter
- :math:`y` — position across the gap

The term :math:`\xi^2 \nabla^2 f` is a **diffusion** of fluidity: plastic
rearrangements at one location trigger rearrangements nearby, with the
spatial extent characterized by :math:`\xi`. Bocquet et al. (2009) derived
this from a kinetic theory of plastic events (STZ-like).


2.2 Local Equilibrium Fluidity
-------------------------------

From the Herschel-Bulkley flow curve :math:`\sigma = \tau_y + K\dot\gamma^n`:

.. math::

   f_\text{loc}(\sigma)
   = \left(\frac{\max(0,\, |\sigma| - \tau_y)}{K}\right)^{1/n}

This gives the shear rate that corresponds to stress :math:`\sigma` on the
HB flow curve. When :math:`|\sigma| < \tau_y`, the local fluidity is zero
(no flow).

**Implementation:** ``f_loc_herschel_bulkley`` in ``_kernels.py``, lines
60--92. Uses a softplus smoothing of :math:`\max(0, x)` for
differentiability — **matches published form** with smooth regularization.


2.3 Boundary Conditions
-------------------------

**Neumann (zero-flux)** boundary conditions at both walls:

.. math::

   \frac{\partial f}{\partial y}\bigg|_{y=0} = 0, \qquad
   \frac{\partial f}{\partial y}\bigg|_{y=H} = 0

This means no fluidity flux through the walls. It is the standard choice
from Goyon et al. (2008).

**Implementation:** ``laplacian_1d_neumann`` in ``_kernels.py``, lines
116--132. Uses ghost points for second-order finite differences:
:math:`\nabla^2 f|_{i=0} = 2(f_1 - f_0)/\Delta y^2` — **matches standard
FD implementation of Neumann BCs.**


2.4 Bulk Stress Evolution
--------------------------

For rate-controlled Couette flow with applied :math:`\dot\gamma`:

.. math::

   \frac{d\Sigma}{dt} = G\!\left(\dot\gamma - \Sigma\,\langle f \rangle\right)

where :math:`\Sigma` is the macroscopic (gap-averaged) stress and
:math:`\langle f \rangle = (1/N_y)\sum_i f_i` is the mean fluidity.

**Implementation:** ``fluidity_nonlocal_pde_rhs`` in ``_kernels.py``,
line 346 — **matches published form.**


2.5 Shear Banding Predictions
-------------------------------

The nonlocal model predicts **shear banding** when the cooperativity
length :math:`\xi` is comparable to or larger than the gap width :math:`H`:

- When :math:`\xi/H \ll 1`: local behavior recovered, sharp bands
- When :math:`\xi/H \sim 1`: smooth gradients, finite-size effects
- When :math:`\xi/H \gg 1`: homogeneous flow (nonlocality smears out bands)

The coefficient of variation :math:`\text{CV} = \text{std}(f)/\text{mean}(f)`
serves as a banding metric: CV > 0.3 indicates significant banding.

**Implementation:** ``shear_banding_cv`` and ``banding_ratio`` in
``_kernels.py`` — **standard metrics.**


2.6 Steady-State Flow Curve (Homogeneous)
-------------------------------------------

For homogeneous steady state (:math:`\partial f/\partial t = 0`,
:math:`\nabla^2 f = 0`), the nonlocal model reduces to the
Herschel-Bulkley flow curve:

.. math::

   \sigma_\text{ss} = \tau_y + K\,|\dot\gamma|^n

**Implementation:** ``fluidity_nonlocal_steady_state`` in ``_kernels.py``,
line 540 — **matches published form.**


3. Saramito Elastoviscoplastic Model
======================================

**References:**

- Saramito, P. (2007).
  *A new constitutive equation for elastoviscoplastic fluid flows.*
  J. Non-Newtonian Fluid Mech. **145**, 1--14.

- Saramito, P. (2009).
  *A new elastoviscoplastic model based on the Herschel-Bulkley
  viscoplastic model.*
  J. Non-Newtonian Fluid Mech. **158**, 154--161.

- Fraggedakis, D., Dimakopoulos, Y., Tsamopoulos, J. (2016).
  *Yielding the yield stress analysis: A thorough comparison of recently
  proposed elasto-visco-plastic (EVP) fluid models.*
  J. Non-Newtonian Fluid Mech. **236**, 104--122.


3.1 Tensorial Constitutive Equation
-------------------------------------

The Saramito (2007, 2009) constitutive equation is:

.. math::

   \lambda\,\overset{\nabla}{\boldsymbol{\tau}}
   + \alpha(\boldsymbol{\tau})\,\boldsymbol{\tau}
   = 2\eta_p\,\mathbf{D}

where:

- :math:`\boldsymbol{\tau}` — extra (polymeric) stress tensor (Pa)
- :math:`\overset{\nabla}{\boldsymbol{\tau}}` — upper-convected derivative
  of :math:`\boldsymbol{\tau}`
- :math:`\lambda` — relaxation time (s)
- :math:`\eta_p = G\lambda` — polymeric viscosity (Pa·s)
- :math:`\mathbf{D} = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)`
  — rate of deformation tensor
- :math:`\alpha(\boldsymbol{\tau})` — Von Mises plasticity function

**Below yield** (:math:`|\boldsymbol{\tau}_d| < \tau_y`):
:math:`\alpha = 0`, so the equation becomes
:math:`\lambda\overset{\nabla}{\boldsymbol{\tau}} = 2\eta_p\mathbf{D}`,
which is the **Upper-Convected Maxwell (UCM) model** — the material
behaves as a viscoelastic solid.

**Above yield** (:math:`|\boldsymbol{\tau}_d| > \tau_y`):
:math:`\alpha > 0`, and the material flows as a **viscoelastic
Oldroyd-type fluid** with yield stress.


3.2 Von Mises Plasticity Parameter
------------------------------------

.. math::

   \alpha(\boldsymbol{\tau})
   = \max\!\left(0,\; 1 - \frac{\tau_y}{|\boldsymbol{\tau}_d|}\right)

where the Von Mises equivalent stress is:

.. math::

   |\boldsymbol{\tau}_d|
   = \sqrt{\tfrac{1}{2}\,\boldsymbol{\tau}_d : \boldsymbol{\tau}_d}

and :math:`\boldsymbol{\tau}_d` is the deviatoric part of the extra stress.

**Properties of** :math:`\alpha`:

- :math:`\alpha = 0` when :math:`|\boldsymbol{\tau}_d| \le \tau_y`
  (below yield, purely elastic)
- :math:`0 < \alpha < 1` when :math:`|\boldsymbol{\tau}_d| > \tau_y`
  (above yield, partial plastic flow)
- :math:`\alpha \to 1` when
  :math:`|\boldsymbol{\tau}_d| \gg \tau_y` (far above yield, mostly
  viscous)

**Implementation:** ``saramito_plasticity_alpha`` in
``saramito/_kernels.py``, lines 97--118. Uses softplus smoothing
for differentiability — **matches published form** with smooth
regularization.


3.3 Upper-Convected Derivative
-------------------------------

The upper-convected (Oldroyd) derivative is defined as:

.. math::

   \overset{\nabla}{\boldsymbol{\tau}}
   = \frac{D\boldsymbol{\tau}}{Dt}
   - \mathbf{L} \cdot \boldsymbol{\tau}
   - \boldsymbol{\tau} \cdot \mathbf{L}^T

where :math:`\mathbf{L} = \nabla\mathbf{u}` is the velocity gradient tensor.


3.4 Simple Shear Component Equations
--------------------------------------

For simple shear flow with velocity :math:`\mathbf{u} = (\dot\gamma y, 0, 0)`:

.. math::

   \mathbf{L} = \begin{pmatrix} 0 & \dot\gamma & 0 \\
   0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}, \qquad
   \mathbf{D} = \frac{1}{2}\begin{pmatrix} 0 & \dot\gamma & 0 \\
   \dot\gamma & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

The velocity gradient coupling terms are:

.. math::

   (\mathbf{L}\cdot\boldsymbol{\tau} + \boldsymbol{\tau}\cdot\mathbf{L}^T)_{xx}
   &= 2\dot\gamma\,\tau_{xy}

   (\mathbf{L}\cdot\boldsymbol{\tau} + \boldsymbol{\tau}\cdot\mathbf{L}^T)_{yy}
   &= 0

   (\mathbf{L}\cdot\boldsymbol{\tau} + \boldsymbol{\tau}\cdot\mathbf{L}^T)_{xy}
   &= \dot\gamma\,\tau_{yy}

**Derivation of** :math:`(\mathbf{L}\cdot\boldsymbol{\tau})_{xy}`:

.. math::

   (\mathbf{L}\cdot\boldsymbol{\tau})_{xy}
   = L_{xx}\tau_{xy} + L_{xy}\tau_{yy}
   = 0 \cdot \tau_{xy} + \dot\gamma \cdot \tau_{yy}
   = \dot\gamma\,\tau_{yy}

and :math:`(\boldsymbol{\tau}\cdot\mathbf{L}^T)_{xy} = \tau_{xx}L_{xy}^T + \tau_{xy}L_{yy}^T = 0`
since :math:`\mathbf{L}^T = \begin{pmatrix} 0 & 0 \\ \dot\gamma & 0 \end{pmatrix}` gives
:math:`L^T_{1,y} = \dot\gamma` only in the (yx) entry.

**Implementation:** ``upper_convected_2d`` in ``saramito/_kernels.py``,
lines 186--190 — **matches derived form.**


3.5 Full ODE System in Simple Shear
-------------------------------------

Rearranging the constitutive equation
:math:`\lambda\overset{\nabla}{\boldsymbol{\tau}} + \alpha\boldsymbol{\tau} = 2\eta_p\mathbf{D}`
for :math:`d\boldsymbol{\tau}/dt`:

.. math::

   \frac{d\boldsymbol{\tau}}{dt}
   = \mathbf{L}\cdot\boldsymbol{\tau}
   + \boldsymbol{\tau}\cdot\mathbf{L}^T
   + \frac{1}{\lambda}\left(2\eta_p\,\mathbf{D}
   - \alpha\,\boldsymbol{\tau}\right)

Since :math:`\eta_p = G\lambda`, we have :math:`2\eta_p\mathbf{D}/\lambda = 2G\mathbf{D}`.
In simple shear, the only nonzero contribution from :math:`2G\mathbf{D}` is
:math:`2G D_{xy} = G\dot\gamma` in the (xy) component.

Also, :math:`\alpha/\lambda = \alpha\, f` when we define :math:`\lambda = 1/f`.

The component ODEs become:

.. math::

   \frac{d\tau_{xx}}{dt} &= 2\dot\gamma\,\tau_{xy}
   - \alpha\, f\, \tau_{xx}

   \frac{d\tau_{yy}}{dt} &= -\alpha\, f\, \tau_{yy}

   \frac{d\tau_{xy}}{dt} &= \dot\gamma\,\tau_{yy}
   + G\dot\gamma - \alpha\, f\, \tau_{xy}

**Term-by-term verification against implementation:**

+--------------------+-----------------------------------------+---------------------------------------+
| Component          | Published equation                      | Code (``saramito_local_ode_rhs``)     |
+====================+=========================================+=======================================+
| :math:`d\tau_{xx}` | :math:`2\dot\gamma\tau_{xy}             | ``conv_xx - alpha * f * tau_xx``       |
|                    | - \alpha f \tau_{xx}`                   | where ``conv_xx = 2*gdot*tau_xy``     |
+--------------------+-----------------------------------------+---------------------------------------+
| :math:`d\tau_{yy}` | :math:`-\alpha f \tau_{yy}`             | ``conv_yy - alpha * f * tau_yy``      |
|                    |                                         | where ``conv_yy = 0.0``               |
+--------------------+-----------------------------------------+---------------------------------------+
| :math:`d\tau_{xy}` | :math:`\dot\gamma\tau_{yy}              | ``conv_xy + G*gdot - alpha*f*tau_xy`` |
|                    | + G\dot\gamma - \alpha f \tau_{xy}`     | where ``conv_xy = gdot*tau_yy``       |
+--------------------+-----------------------------------------+---------------------------------------+

**All three components match the published Saramito (2007) equations.**


3.6 Von Mises Stress (2D Traceless Tensor)
--------------------------------------------

For a traceless deviatoric tensor with
:math:`\tau_{zz} = -(\tau_{xx} + \tau_{yy})`:

.. math::

   |\boldsymbol{\tau}|
   = \sqrt{\frac{1}{2}\left(\tau_{xx}^2 + \tau_{yy}^2 + \tau_{zz}^2
   + 2\tau_{xy}^2\right)}

**Implementation:** ``von_mises_stress_2d`` in ``saramito/_kernels.py``,
lines 56--67 — **matches published form** with :math:`\epsilon = 10^{-30}`
guard against sqrt(0).


3.7 Steady-State in Simple Shear
----------------------------------

At steady state with constant :math:`\dot\gamma` and :math:`\alpha \approx 1`
(far above yield):

.. math::

   \tau_{xy,\text{ss}} &= \tau_y + K\,|\dot\gamma|^n
   \quad\text{(Herschel-Bulkley)}

   N_1 = \tau_{xx} - \tau_{yy} &= 2\lambda\,\dot\gamma\,\tau_{xy}
   \quad\text{(UCM normal stress)}

The first normal stress difference :math:`N_1` arises from the
upper-convected derivative coupling and is a signature of elasticity.

**Implementation:** ``saramito_steady_state_full`` in
``saramito/_kernels.py``, lines 808--811 — **matches published form.**


3.8 Creep Equations (Stress-Controlled)
----------------------------------------

Under constant applied stress :math:`\sigma_\text{app}`:

.. math::

   \frac{d\gamma}{dt} &= \alpha\!\left(\sigma_\text{app}\right)\, f\,
   \sigma_\text{app}

   \frac{df}{dt} &= \frac{f_\text{age} - f}{t_a}
   + b\,|\dot\gamma|^{n_r}\,(f_\text{flow} - f)

where the plasticity parameter for scalar stress simplifies to:

.. math::

   \alpha = \max\!\left(0,\; 1 - \frac{\tau_y}{|\sigma_\text{app}|}\right)

**Creep bifurcation:**

- :math:`\sigma_\text{app} < \tau_y`: :math:`\alpha = 0`, no plastic flow,
  bounded elastic deformation only.
- :math:`\sigma_\text{app} > \tau_y`: :math:`\alpha > 0`, creep with
  potential acceleration if rejuvenation dominates aging.

**Implementation:** ``saramito_local_creep_ode_rhs`` in
``saramito/_kernels.py``, lines 514--533 — **matches published form.**


4. Coupled Fluidity-Saramito Model
====================================

**References:**

- Dimitriou, C.J., McKinley, G.H. (2019).
  *A canonical framework for modeling elasto-viscoplasticity in complex
  fluids.*
  J. Non-Newtonian Fluid Mech. **265**, 116--132.

- de Souza Mendes, P.R., Thompson, R.L. (2019).
  *Time-dependent yield stress materials.*
  Curr. Opin. Colloid Interface Sci. **43**, 15--25.


4.1 Fluidity-Relaxation Time Coupling
---------------------------------------

The key coupling between fluidity and the Saramito tensorial model is:

.. math::

   \lambda(f) = \frac{1}{f}

As fluidity evolves (aging increases :math:`\lambda`, rejuvenation
decreases :math:`\lambda`), the viscoelastic relaxation time changes
dynamically. This makes the Saramito model **thixotropic**.


4.2 Fluidity Evolution for Saramito
-------------------------------------

.. math::

   \frac{df}{dt}
   = \frac{f_\text{age} - f}{t_a}
   + b\,|\text{driving}|^{n_r}\,(f_\text{flow} - f)

where the driving rate depends on protocol:

- **Rate-controlled:** :math:`\text{driving} = |\dot\gamma|`
- **Stress-controlled:** :math:`\text{driving} = |\dot\gamma| = |\alpha\, f\, \sigma|`
  (plastic strain rate)
- **Relaxation** (:math:`\dot\gamma = 0`):
  :math:`\text{driving} = 0` (pure aging)

**Parameters:**

- :math:`f_\text{age}` — equilibrium fluidity under aging (low value, solid-like)
- :math:`f_\text{flow}` — high-shear fluidity limit (high value, fluid-like)
- :math:`t_a` — aging timescale (s)
- :math:`b` — rejuvenation amplitude
- :math:`n_r` — rejuvenation exponent

**Implementation:** ``fluidity_evolution_saramito`` in
``saramito/_kernels.py``, lines 236--249 — **matches published form.**


4.3 Dynamic Yield Stress Coupling
-----------------------------------

For **full coupling mode**, the yield stress depends on fluidity:

.. math::

   \tau_y(f) = \tau_{y,0} + \frac{a_y}{f^m}

where:

- :math:`\tau_{y,0}` — base (minimum) yield stress (Pa)
- :math:`a_y` — coupling coefficient
- :math:`m` — coupling exponent

**Physical interpretation:** An aged state (low :math:`f`) has high yield
stress (strong microstructure), while a rejuvenated state (high :math:`f`)
has low yield stress (broken structure). This captures structure-dependent
yield stress observed in many thixotropic materials.

For **minimal coupling mode**: :math:`\tau_y = \tau_{y,0}` (constant).

**Implementation:** ``yield_stress_from_fluidity`` in
``saramito/_kernels.py``, lines 282--288 — **matches published form.**


4.4 Complete ODE System (Rate-Controlled)
------------------------------------------

State vector: :math:`\mathbf{y} = [\tau_{xx}, \tau_{yy}, \tau_{xy}, f, \gamma]`

.. math::

   \frac{d\tau_{xx}}{dt} &= 2\dot\gamma\,\tau_{xy}
   - \alpha(\boldsymbol{\tau}, \tau_y(f))\, f\, \tau_{xx}

   \frac{d\tau_{yy}}{dt} &= -\alpha\, f\, \tau_{yy}

   \frac{d\tau_{xy}}{dt} &= \dot\gamma\,\tau_{yy} + G\dot\gamma
   - \alpha\, f\, \tau_{xy}

   \frac{df}{dt} &= \frac{f_\text{age} - f}{t_a}
   + b\,|\dot\gamma|^{n_r}\,(f_\text{flow} - f)

   \frac{d\gamma}{dt} &= \dot\gamma

where :math:`\alpha = \max(0, 1 - \tau_y(f)/|\boldsymbol{\tau}_d|)`.

**Implementation:** ``saramito_local_ode_rhs`` in
``saramito/_kernels.py``, lines 392--442 — **all five components match.**


4.5 Nonlocal Saramito (with Spatial Diffusion)
------------------------------------------------

The nonlocal extension adds fluidity diffusion across the gap:

.. math::

   \frac{\partial f_i}{\partial t}
   = \left[\frac{f_\text{age} - f_i}{t_a}
   + b\,|\dot\gamma_i|^{n_r}(f_\text{flow} - f_i)\right]
   + D_f\,\nabla^2 f_i

where:

- :math:`D_f = \xi^2 / t_a` — fluidity diffusivity (m :math:`^2` /s)
- :math:`\xi` — cooperativity length (m)
- :math:`\dot\gamma_i = \alpha_i\, f_i\, \tau_{xy}` — local plastic
  shear rate
- Neumann BCs: :math:`\partial f/\partial y = 0` at walls

**Implementation:** ``saramito_nonlocal_pde_rhs`` in
``saramito/_kernels.py``, lines 854--965 — **matches published form.**


5. Verification Summary
========================

+--------------------------------------+-----------+----------------------------------------------+
| Equation                             | Status    | Notes                                        |
+======================================+===========+==============================================+
| Stress evolution (local fluidity)    | VERIFIED  | Maxwell form with plastic term               |
+--------------------------------------+-----------+----------------------------------------------+
| Fluidity evolution (aging+rejuv)     | VERIFIED  | Generalized Coussot/de Souza Mendes          |
+--------------------------------------+-----------+----------------------------------------------+
| Steady-state fluidity                | VERIFIED  | Algebraic from df/dt=0                       |
+--------------------------------------+-----------+----------------------------------------------+
| Steady-state flow curve              | VERIFIED  | HB with fluidity-weighted viscous term       |
+--------------------------------------+-----------+----------------------------------------------+
| Creep equations (local)              | VERIFIED  | Strain rate = sigma * f                      |
+--------------------------------------+-----------+----------------------------------------------+
| Nonlocal fluidity PDE                | VERIFIED  | Goyon/Bocquet diffusion term                 |
+--------------------------------------+-----------+----------------------------------------------+
| Neumann BCs (Laplacian)              | VERIFIED  | Ghost-point FD, 2nd order                    |
+--------------------------------------+-----------+----------------------------------------------+
| Saramito constitutive (tensor)       | VERIFIED  | UCM + Von Mises plasticity                   |
+--------------------------------------+-----------+----------------------------------------------+
| Von Mises alpha parameter            | VERIFIED  | max(0, 1 - tau_y/|tau|) with softplus        |
+--------------------------------------+-----------+----------------------------------------------+
| Upper-convected derivative (shear)   | VERIFIED  | 3 components correct                         |
+--------------------------------------+-----------+----------------------------------------------+
| Saramito ODE system (5 components)   | VERIFIED  | All terms match Saramito (2007)              |
+--------------------------------------+-----------+----------------------------------------------+
| Saramito creep                       | VERIFIED  | alpha * f * sigma plastic rate               |
+--------------------------------------+-----------+----------------------------------------------+
| Fluidity-lambda coupling             | VERIFIED  | lambda = 1/f                                 |
+--------------------------------------+-----------+----------------------------------------------+
| Dynamic yield stress tau_y(f)        | VERIFIED  | tau_y0 + a_y / f^m                           |
+--------------------------------------+-----------+----------------------------------------------+
| Nonlocal Saramito PDE                | VERIFIED  | Diffusivity D_f = xi^2/t_a                   |
+--------------------------------------+-----------+----------------------------------------------+
| Steady-state N1                      | VERIFIED  | 2*lambda*gdot*tau_xy (UCM result)            |
+--------------------------------------+-----------+----------------------------------------------+

**All 16 equations verified against published literature.**


6. References
==============

.. [Bocquet2009] Bocquet, L., Colin, A., Ajdari, A. (2009). Kinetic theory
   of plastic flow in soft glassy materials. *Phys. Rev. Lett.* **103**,
   036001.

.. [Coussot2002] Coussot, P., Nguyen, Q.D., Huynh, H.T., Bonn, D. (2002).
   Viscosity bifurcation in thixotropic, yielding fluids. *J. Rheol.*
   **46**\(3), 573--589.

.. [deSouzaMendes2009] de Souza Mendes, P.R. (2009). Modeling the
   thixotropic behavior of structured fluids. *J. Non-Newtonian Fluid Mech.*
   **164**, 66--75.

.. [Dimitriou2019] Dimitriou, C.J., McKinley, G.H. (2019). A canonical
   framework for modeling elasto-viscoplasticity in complex fluids.
   *J. Non-Newtonian Fluid Mech.* **265**, 116--132.

.. [Fraggedakis2016] Fraggedakis, D., Dimakopoulos, Y., Tsamopoulos, J.
   (2016). Yielding the yield stress analysis. *J. Non-Newtonian Fluid Mech.*
   **236**, 104--122.

.. [Goyon2008] Goyon, J., Colin, A., Ovarlez, G., Ajdari, A., Bocquet, L.
   (2008). Spatial cooperativity in soft glassy flows. *Nature* **454**,
   84--87.

.. [Saramito2007] Saramito, P. (2007). A new constitutive equation for
   elastoviscoplastic fluid flows. *J. Non-Newtonian Fluid Mech.* **145**,
   1--14.

.. [Saramito2009] Saramito, P. (2009). A new elastoviscoplastic model
   based on the Herschel-Bulkley viscoplastic model. *J. Non-Newtonian
   Fluid Mech.* **158**, 154--161.
