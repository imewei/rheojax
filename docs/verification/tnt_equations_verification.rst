.. _tnt-equations-verification:

==========================================================================
Transient Network Theory (TNT) — Mathematical Foundations & Verification
==========================================================================

This document records the published governing equations for each TNT model
variant, cross-referenced against the RheoJAX implementation in
``rheojax/models/tnt/``.  Sources are cited inline; web-accessible links
are listed at the end.

.. contents:: Models
   :local:
   :depth: 2

----

1. Green-Tobolsky / Tanaka-Edwards (Basic Transient Network)
=============================================================

**Primary references:**

- Green, M.S. & Tobolsky, A.V. (1946). *J. Chem. Phys.* **14**, 80-92.
- Tanaka, F. & Edwards, S.F. (1992). *Macromolecules* **25**, 1516-1523.
- Lodge, A.S. (1956). *Trans. Faraday Soc.* **52**, 120-130.

1.1 Chain Distribution / Conformation Tensor
---------------------------------------------

Green & Tobolsky (1946) conceived a network of chains with temporary
junctions that are steadily created and destroyed.  Tanaka & Edwards (1992)
formalised this using the **conformation tensor**

.. math::

   \mathbf{S} = \frac{\langle \mathbf{R} \otimes \mathbf{R} \rangle}{R_0^2}

where :math:`\mathbf{R}` is the end-to-end vector of a network strand and
:math:`R_0 = \sqrt{\langle R^2 \rangle_0}` is the equilibrium
root-mean-square distance.  At equilibrium :math:`\mathbf{S} = \mathbf{I}`.

1.2 Creation and Destruction Rates
------------------------------------

**Constant-rate (basic model):**

- Destruction (breakage) rate: :math:`\beta = 1/\tau_b`
- Creation (reformation) rate: :math:`g_0 = 1/\tau_b`

At equilibrium the two rates balance, maintaining
:math:`\mathbf{S} = \mathbf{I}`.

**Force-dependent detachment (Tanaka-Edwards, Bell):**

If a force :math:`f` acts on a chain bound at time :math:`t_0`, the
breakage rate at time :math:`t` is (Tanaka & Edwards 1992, Eq. 2; see also
Bell 1978):

.. math::

   \beta(f) = \omega_0 \exp\!\left(-\frac{W_b - f \cdot a}{k_B T}\right)
            = \beta_0 \exp\!\left(\frac{f \cdot a}{k_B T}\right)

where :math:`\omega_0` is the thermal vibration frequency, :math:`W_b` is
the bond dissociation energy, and :math:`a` is the activation length.  In
the conformation-tensor formulation this becomes:

.. math::

   \beta(\mathbf{S})
     = \frac{1}{\tau_b}\exp\!\bigl[\nu\,(\text{stretch} - 1)\bigr],
   \qquad
   \text{stretch} = \sqrt{\operatorname{tr}(\mathbf{S})/3}

with :math:`\nu` a dimensionless force-sensitivity parameter
(:math:`\nu = 0` recovers constant breakage).

**Implementation:** ``_kernels.py``, functions ``breakage_constant``,
``breakage_bell``, ``breakage_power_law``.

1.3 Constitutive Equation (Conformation Tensor Evolution)
----------------------------------------------------------

.. math::

   \frac{d\mathbf{S}}{dt}
     = \boldsymbol{\kappa}\cdot\mathbf{S}
       + \mathbf{S}\cdot\boldsymbol{\kappa}^T
       + g_0\,\mathbf{I}
       - \beta(\mathbf{S})\,\mathbf{S}

where :math:`\boldsymbol{\kappa} = (\nabla\mathbf{v})^T` is the velocity
gradient tensor.  The first two terms are the **upper-convected derivative**
(affine deformation); the third is chain creation; the fourth is chain
destruction.

Equivalently, writing as relaxation toward equilibrium:

.. math::

   \frac{d\mathbf{S}}{dt}
     = \boldsymbol{\kappa}\cdot\mathbf{S}
       + \mathbf{S}\cdot\boldsymbol{\kappa}^T
       - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

This is mathematically identical to the **upper-convected Maxwell (UCM)**
model.

**Implementation:** ``_kernels.py``, ``tnt_single_mode_ode_rhs`` (lines
588-638) and ``build_tnt_ode_rhs`` factory (lines 823-881).

1.4 Stress Tensor
------------------

.. math::

   \boldsymbol{\sigma} = G\,(\mathbf{S} - \mathbf{I}) + 2\eta_s\,\mathbf{D}

where :math:`G \approx n_{\text{chains}} k_B T` is the network modulus and
:math:`\mathbf{D} = (\boldsymbol{\kappa}+\boldsymbol{\kappa}^T)/2`.

In simple shear (:math:`\boldsymbol{\kappa} = \dot\gamma\,\mathbf{e}_x\otimes\mathbf{e}_y`):

.. math::

   \sigma_{xy} &= G\,S_{xy} + \eta_s\dot\gamma \\
   N_1 &= G(S_{xx} - S_{yy}) \\
   N_2 &= G(S_{yy} - S_{zz}) = 0 \quad\text{(upper-convected)}

**Implementation:** ``_kernels.py``, ``stress_linear_xy``,
``stress_fene_xy``.

1.5 Simple Shear Component Equations
--------------------------------------

.. math::

   \frac{dS_{xx}}{dt} &= 2\dot\gamma\,S_{xy}
     - \frac{S_{xx}-1}{\tau_b} \\[4pt]
   \frac{dS_{yy}}{dt} &= -\frac{S_{yy}-1}{\tau_b} \\[4pt]
   \frac{dS_{zz}}{dt} &= -\frac{S_{zz}-1}{\tau_b} \\[4pt]
   \frac{dS_{xy}}{dt} &= \dot\gamma\,S_{yy} - \frac{S_{xy}}{\tau_b}

1.6 Steady-State Solutions (Simple Shear)
-------------------------------------------

.. math::

   S_{xy} &= \tau_b\dot\gamma \\
   S_{xx} &= 1 + 2(\tau_b\dot\gamma)^2 \\
   S_{yy} &= S_{zz} = 1

**Flow curve (Newtonian):**

.. math::

   \sigma_{xy} = (G\tau_b + \eta_s)\,\dot\gamma = \eta_0\,\dot\gamma

**First normal stress difference:**

.. math::

   N_1 = 2G(\tau_b\dot\gamma)^2, \qquad \Psi_1 = 2G\tau_b^2

**Implementation:** ``_kernels.py``, ``tnt_base_steady_conformation``,
``tnt_base_steady_stress``, ``tnt_base_steady_n1``.

1.7 Relaxation Modulus
-----------------------

.. math::

   G(t) = G\,\exp(-t/\tau_b)

Single Maxwell exponential.

**Implementation:** ``_kernels.py``, ``tnt_base_relaxation``.

1.8 SAOS Moduli
----------------

.. math::

   G'(\omega)  &= G\,\frac{(\omega\tau_b)^2}{1+(\omega\tau_b)^2} \\[4pt]
   G''(\omega) &= G\,\frac{\omega\tau_b}{1+(\omega\tau_b)^2} + \eta_s\omega

**Implementation:** ``_kernels.py``, ``tnt_saos_moduli``.

1.9 Parameters
---------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G`
     - Pa
     - Network modulus (:math:`\approx n_\text{chains} k_B T`)
   * - :math:`\tau_b`
     - s
     - Bond lifetime (reciprocal breakage rate)
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity

Derived: :math:`\eta_0 = G\tau_b + \eta_s` (zero-shear viscosity).

----

2. Tanaka-Edwards Model with Force-Dependent Breakage
=======================================================

**Primary references:**

- Tanaka, F. & Edwards, S.F. (1992). *Macromolecules* **25**, 1516-1523.
- Bell, G.I. (1978). *Science* **200**, 618-627.

2.1 Breakage Rate (Kramers/Bell)
----------------------------------

From Tanaka & Edwards, the thermally-activated breakage rate in the
presence of a force :math:`f` is:

.. math::

   \beta(f) = \omega_0\,\exp\!\Bigl[-\frac{W_b}{k_BT}\Bigr]
              \cdot\exp\!\Bigl[\frac{f\cdot a}{k_BT}\Bigr]
            = \frac{1}{\tau_b}\,\exp\!\Bigl[\frac{f\cdot a}{k_BT}\Bigr]

In the conformation tensor description, the end-to-end distance proxy is
the **stretch ratio** :math:`\lambda = \sqrt{\operatorname{tr}(\mathbf{S})/3}`,
giving the coarse-grained Bell form:

.. math::

   \beta(\mathbf{S}) = \frac{1}{\tau_b}\,\exp\!\bigl[\nu(\lambda - 1)\bigr]

At equilibrium (:math:`\lambda=1`): :math:`\beta = 1/\tau_b`.

**Power-law variant:**

.. math::

   \beta(\mathbf{S}) = \frac{1}{\tau_b}\,\lambda^m

**Implementation:** ``_kernels.py``, ``breakage_bell`` (line 71),
``breakage_power_law`` (line 103).

2.2 ODE System (Bell breakage, simple shear)
----------------------------------------------

.. math::

   \frac{dS_{xx}}{dt} &= 2\dot\gamma S_{xy} + g_0 - \beta(\mathbf{S})\,S_{xx}\\
   \frac{dS_{yy}}{dt} &= g_0 - \beta(\mathbf{S})\,S_{yy}\\
   \frac{dS_{zz}}{dt} &= g_0 - \beta(\mathbf{S})\,S_{zz}\\
   \frac{dS_{xy}}{dt} &= \dot\gamma\,S_{yy} - \beta(\mathbf{S})\,S_{xy}

with :math:`g_0 = 1/\tau_b` (creation rate unaffected by force).

**Key consequence:** Force-dependent breakage produces
**shear thinning** (faster breakage at high strain reduces effective
relaxation time) and **stress overshoot** with non-exponential relaxation.

**Implementation:** ``_kernels.py``, ``build_tnt_ode_rhs(breakage_type="bell")``.

----

3. Cates Reptation-Reaction Model (Living Polymers)
=====================================================

**Primary references:**

- Cates, M.E. (1987). *Macromolecules* **20**, 2289-2296.
- Cates, M.E. (1990). *J. Phys. Chem.* **94**, 371-375.
- Turner, M.S. & Cates, M.E. (1991). *Langmuir* **7**, 1590.
- Chen, V., Drucker, C.T., Love, C., Peterson, J. & Peterson, J.D. (2024).
  arXiv:2407.07213 (analytic series solution).

3.1 Physical Mechanism
------------------------

Wormlike micelles are entangled living polymers that relax stress via two
mechanisms operating simultaneously:

1. **Reptation** (curvilinear diffusion along the tube): time scale
   :math:`\tau_\text{rep} \sim L^3/(\pi^2 D)`.
2. **Reversible scission** (random breaking/recombination): time scale
   :math:`\tau_\text{break}`.

Breaking reorganises tube segments: interior segments become end segments
(which relax quickly) and vice versa.

3.2 Reptation Spectrum (Unbreakable Chains)
---------------------------------------------

For permanent entangled polymers (Doi-Edwards):

.. math::

   G(t) = G_0 \sum_{p\;\text{odd}} \frac{8}{\pi^2 p^2}
          \exp\!\left(-\frac{p^2 t}{\tau_\text{rep}}\right)

3.3 Breaking Parameter
------------------------

.. math::

   \zeta = \frac{\tau_\text{break}}{\tau_\text{rep}}

3.4 Effective Relaxation Time (Fast-Breaking Limit)
-----------------------------------------------------

When :math:`\zeta \ll 1` (fast-breaking), Cates (1987) showed that the
stress relaxation becomes near-single-exponential with the **geometric
mean** relaxation time:

.. math::

   \boxed{\tau_d = \sqrt{\tau_\text{rep}\cdot\tau_\text{break}}}

**Physical derivation:** Reptation requires diffusion over contour length
:math:`L`. Breaking cuts the micelle every :math:`\tau_\text{break}` into
pieces of size :math:`\sim\sqrt{D\tau_\text{break}}`.  Setting this equal
to the tube escape distance gives
:math:`\tau_d \sim \sqrt{\tau_\text{rep}\,\tau_\text{break}}`.

**Implementation:** ``_kernels.py``, ``tnt_cates_effective_tau`` (line 560).

3.5 Stress Relaxation
-----------------------

**Fast-breaking regime** (:math:`\zeta \ll 1`):

.. math::

   G(t) \approx G_0\,\exp\!\left(-\sqrt{2t/\tau_\text{break}}\right)

This stretched exponential is well approximated by a single Maxwell mode
with time constant :math:`\tau_d`.

**Slow-breaking regime** (:math:`\zeta \gg 1`):

Standard reptation spectrum (Eq. in Sec. 3.2).

3.6 Constitutive Equation (Fast-Breaking UCM)
------------------------------------------------

In the fast-breaking limit the system reduces to a **single-mode UCM** with
:math:`\tau_b \to \tau_d`:

.. math::

   \frac{d\mathbf{S}}{dt}
     = \boldsymbol{\kappa}\cdot\mathbf{S}
       + \mathbf{S}\cdot\boldsymbol{\kappa}^T
       - \frac{\mathbf{S}-\mathbf{I}}{\tau_d}

.. math::

   \boldsymbol{\sigma} = G_0(\mathbf{S}-\mathbf{I}) + 2\eta_s\mathbf{D}

**Implementation:** ``cates.py`` delegates to the base single-mode ODE with
:math:`\tau_b = \tau_d = \sqrt{\tau_\text{rep}\tau_\text{break}}`.

3.7 SAOS Moduli
-----------------

Single Maxwell mode with :math:`\tau_d`:

.. math::

   G'(\omega) &= G_0\,\frac{(\omega\tau_d)^2}{1+(\omega\tau_d)^2}\\[4pt]
   G''(\omega) &= G_0\,\frac{\omega\tau_d}{1+(\omega\tau_d)^2}
                  + \eta_s\omega

3.8 Cole-Cole Semicircle
--------------------------

Plotting :math:`G''` vs :math:`G'` parametrically in :math:`\omega` (with
:math:`\eta_s=0`) gives a **perfect semicircle**:

.. math::

   \left(G' - \frac{G_0}{2}\right)^{\!2} + (G'')^2
     = \left(\frac{G_0}{2}\right)^{\!2}

- Centre at :math:`(G_0/2,\;0)`
- Radius :math:`G_0/2`
- Passes through the origin (:math:`\omega\to 0`) and :math:`(G_0,0)`
  (:math:`\omega\to\infty`)

Deviations indicate intermediate breaking, branching (Y-junctions), or
polydispersity.

More precisely, at arbitrary :math:`\zeta`:

.. math::

   G^*(\omega) \approx \frac{G_0}{1 + \sqrt{\tau_\text{break}/(2i\omega)}}

(Turner & Cates 1991).

**Implementation:** Verified via ``tnt_saos_moduli_vec`` called with
effective :math:`\tau_d`.

3.9 Steady Shear (Non-Monotonic Flow Curve)
---------------------------------------------

.. math::

   \sigma_{xy} = G_0\,\frac{\tau_d\dot\gamma}{1+(\tau_d\dot\gamma)^2}
                 + \eta_s\dot\gamma

The network contribution is **non-monotonic** with a maximum at
:math:`\dot\gamma = 1/\tau_d` (constitutive instability leading to
**shear banding** for :math:`\text{Wi}_d > 1`).

3.10 Parameters
-----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G_0`
     - Pa
     - Plateau modulus (:math:`\sim k_BT/\xi^3`, :math:`\xi` = mesh size)
   * - :math:`\tau_\text{rep}`
     - s
     - Reptation time
   * - :math:`\tau_\text{break}`
     - s
     - Mean scission/breakage time
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity

Derived: :math:`\tau_d = \sqrt{\tau_\text{rep}\tau_\text{break}}`,
:math:`\eta_0 = G_0\tau_d + \eta_s`.

----

4. Sticky Rouse Model
=======================

**Primary references:**

- Baxandall, L.G. (1989). *Macromolecules* **22**, 1982.
- Leibler, L., Rubinstein, M. & Colby, R.H. (1991). *Macromolecules*
  **24**, 4701-4707.
- Rubinstein, M. & Semenov, A.N. (1998). *Macromolecules* **31**,
  1373-1385.
- Rubinstein, M. & Semenov, A.N. (2001). *Macromolecules* **34**,
  1058-1068.

4.1 Physical Picture
---------------------

A flexible chain of :math:`N` Kuhn segments carries :math:`N_s` evenly
spaced reversible association sites ("stickers").  Between stickers, the
chain behaves as a Rouse sub-chain.  The sticker lifetime :math:`\tau_s`
imposes a **minimum effective relaxation time** for all modes.

4.2 Rouse Mode Spectrum
-------------------------

For a standard Rouse chain, mode :math:`k` has:

.. math::

   \tau_{R,k} = \frac{\tau_{R,1}}{k^2},
   \qquad k = 1,2,\ldots,N

where :math:`\tau_{R,1}` is the longest Rouse time.

4.3 Effective Relaxation Times (Sticker Renormalisation)
---------------------------------------------------------

Sticker exchange slows all modes faster than :math:`\tau_s`.  Two
formulations appear in the literature:

**Additive (Leibler-Rubinstein-Colby, 1991; used in RheoJAX docs):**

.. math::

   \tau_{\text{eff},k} = \tau_{R,k} + \tau_s

**Max (phenomenological, used in RheoJAX implementation):**

.. math::

   \tau_{\text{eff},k} = \max(\tau_{R,k},\;\tau_s)

Both produce the same qualitative physics: modes with
:math:`\tau_{R,k} < \tau_s` are *pinned* at :math:`\tau_s`, while slow
modes (:math:`\tau_{R,k} > \tau_s`) are essentially unaffected.

.. note::

   **Discrepancy found:** The RheoJAX documentation (``tnt_sticky_rouse.rst``
   line 27) states :math:`\tau_{\text{eff},k} = \tau_{R,k} + \tau_s`
   (additive), whereas the Python source (``sticky_rouse.py`` lines 14-15)
   uses :math:`\max(\tau_{R,k},\tau_s)`.  Both are valid physical
   approximations.  The additive form more faithfully follows
   Leibler-Rubinstein-Colby; the max form is more commonly used in recent
   computational works.  **Recommend unifying docs and code.**

4.4 Multi-Mode Constitutive Equation
--------------------------------------

Each mode :math:`k` evolves independently:

.. math::

   \frac{d\mathbf{S}_k}{dt}
     = \boldsymbol{\kappa}\cdot\mathbf{S}_k
       + \mathbf{S}_k\cdot\boldsymbol{\kappa}^T
       - \frac{\mathbf{S}_k - \mathbf{I}}{\tau_{\text{eff},k}}

Total stress:

.. math::

   \boldsymbol{\sigma} = \sum_{k=1}^{N} G_k\,(\mathbf{S}_k - \mathbf{I})
                         + 2\eta_s\mathbf{D}

**Implementation:** ``_kernels.py``, ``tnt_multimode_ode_rhs``.

4.5 Relaxation Modulus
-----------------------

.. math::

   G(t) = \sum_{k=1}^{N} G_k\,\exp\!\left(-\frac{t}{\tau_{\text{eff},k}}\right)

At intermediate times (when multiple modes have
:math:`\tau_{\text{eff},k} \approx \tau_s`), this yields Rouse-like
power-law decay :math:`G(t) \sim t^{-1/2}`.

**Implementation:** ``_kernels.py``, ``tnt_multimode_relaxation``.

4.6 SAOS Moduli
-----------------

.. math::

   G'(\omega)  &= \sum_k G_k\,\frac{(\omega\tau_{\text{eff},k})^2}
                   {1+(\omega\tau_{\text{eff},k})^2}\\[4pt]
   G''(\omega) &= \sum_k G_k\,\frac{\omega\tau_{\text{eff},k}}
                   {1+(\omega\tau_{\text{eff},k})^2} + \eta_s\omega

**Characteristic signatures:**

- :math:`G' \sim \omega^{1/2}` at intermediate frequencies (Rouse regime)
- Sticky plateau at :math:`1/\tau_s < \omega < 1/\tau_{R,N}`
- Terminal flow (:math:`G' \sim \omega^2`) at :math:`\omega \ll 1/\tau_{\text{eff},1}`

**Implementation:** ``_kernels.py``, ``tnt_multimode_saos_moduli``.

4.7 Parameters
----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G_k`
     - Pa
     - Modulus of mode :math:`k`
   * - :math:`\tau_{R,k}`
     - s
     - Rouse relaxation time of mode :math:`k`
   * - :math:`\tau_s`
     - s
     - Sticker lifetime
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity

Derived: :math:`G_N^{(0)} = \sum_k G_k` (plateau modulus),
:math:`\eta_0 = \sum_k G_k\tau_{\text{eff},k} + \eta_s`.

----

5. Loop-Bridge Model (Telechelic Networks)
============================================

**Primary references:**

- Annable, T., Buscall, R., Ettelaie, R. & Whittlestone, D. (1993).
  *J. Rheol.* **37**, 695-726.
- Tanaka, F. & Edwards, S.F. (1992). *Macromolecules* **25**, 1516-1523.
- Leibler, L., Rubinstein, M. & Colby, R.H. (1991). *Macromolecules*
  **24**, 4701-4707.
- Bell, G.I. (1978). *Science* **200**, 618-627.

5.1 Physical Picture
---------------------

Telechelic chains have associating end-groups that can exist in two
populations:

- **Bridges** (fraction :math:`f_B`): Both ends attached to *different*
  junctions — **load-bearing**.
- **Loops** (fraction :math:`f_L = 1 - f_B`): Both ends on the *same*
  junction — **non-load-bearing**.

The dynamic equilibrium between these populations determines the modulus
and produces the distinctive shear-thickening-then-thinning behaviour
observed experimentally (Annable et al. 1993).

5.2 Bridge Fraction Kinetics
------------------------------

.. math::

   \frac{df_B}{dt}
     = \underbrace{\frac{1-f_B}{\tau_a}}_{\text{loop}\to\text{bridge}}
       - \underbrace{f_B\,\beta(\mathbf{S})}_{\text{bridge}\to\text{loop}}

where:

- :math:`\tau_a` = loop-to-bridge association time
- :math:`\beta(\mathbf{S})` = force-dependent bridge detachment rate

**Equilibrium bridge fraction** (at :math:`\mathbf{S}=\mathbf{I}`,
:math:`\beta = 1/\tau_b`):

.. math::

   f_{B,\text{eq}} = \frac{\tau_b}{\tau_a + \tau_b}

**Implementation:** ``loop_bridge.py``, ``_loop_bridge_ode_rhs`` (line 146).

5.3 Force-Dependent Detachment (Bell Model)
---------------------------------------------

.. math::

   \beta(\mathbf{S}) = \frac{1}{\tau_b}\,\exp\!\Bigl[
     \nu\bigl(\sqrt{\operatorname{tr}(\mathbf{S})/3} - 1\bigr)\Bigr]

At equilibrium: :math:`\beta = 1/\tau_b`.
Under stretch: detachment accelerates exponentially.

5.4 Conformation Tensor Evolution (Bridges Only)
--------------------------------------------------

Bridges evolve under flow with creation and force-activated destruction:

.. math::

   \frac{d\mathbf{S}}{dt}
     = \boldsymbol{\kappa}\cdot\mathbf{S}
       + \mathbf{S}\cdot\boldsymbol{\kappa}^T
       + g_0\,\mathbf{I}
       - \beta(\mathbf{S})\,\mathbf{S}

with :math:`g_0 = 1/\tau_b`.

**Implementation:** ``loop_bridge.py``, ``_loop_bridge_ode_rhs``
(lines 150-161).

5.5 Stress Tensor
-------------------

Only bridges contribute to elastic stress:

.. math::

   \boldsymbol{\sigma}
     = f_B\,G\,(\mathbf{S} - \mathbf{I}) + 2\eta_s\,\mathbf{D}

**Implementation:** Loop-bridge predict methods multiply the base stress
by :math:`f_B`.

5.6 Shear Thickening Mechanism
---------------------------------

The non-monotonic viscosity arises from competing effects:

1. **Low shear rates** (:math:`\dot\gamma \ll 1/\tau_b`): Bridges are at
   equilibrium; viscosity :math:`\eta \approx f_{B,\text{eq}} G\tau_b + \eta_s`.

2. **Moderate shear rates**: Flow-induced stretching increases the
   detachment rate :math:`\beta`, but the *bridge fraction can increase*
   if the reattachment rate (:math:`1/\tau_a`) exceeds the enhanced
   detachment.  More bridges → higher modulus → **shear thickening**.

3. **High shear rates** (:math:`\dot\gamma \gg 1/\tau_b`): Force-activated
   detachment dominates; bridges are destroyed faster than they form.
   Bridge fraction drops → **shear thinning**.

This produces the characteristic non-monotonic viscosity observed by Annable
et al. (1993) in HEUR thickeners: weak thickening followed by strong thinning.

5.7 SAOS Moduli (Linearised)
------------------------------

In the linear regime (:math:`\mathbf{S} \approx \mathbf{I}`), the loop-bridge
model reduces to a single Maxwell mode with effective modulus :math:`f_{B,\text{eq}} G`
and relaxation time :math:`\tau_b`:

.. math::

   G'(\omega)  &= f_{B,\text{eq}}\,G\,
     \frac{(\omega\tau_b)^2}{1+(\omega\tau_b)^2}\\[4pt]
   G''(\omega) &= f_{B,\text{eq}}\,G\,
     \frac{\omega\tau_b}{1+(\omega\tau_b)^2} + \eta_s\omega

5.8 Parameters
----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G`
     - Pa
     - Network modulus (all bridges active)
   * - :math:`\tau_b`
     - s
     - Bridge lifetime (detachment timescale)
   * - :math:`\tau_a`
     - s
     - Loop-to-bridge association time
   * - :math:`\nu`
     - dimensionless
     - Force sensitivity (Bell exponent)
   * - :math:`f_{B,\text{eq}}`
     - dimensionless
     - Equilibrium bridge fraction
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity

Derived: :math:`\eta_0 = f_{B,\text{eq}}\,G\,\tau_b + \eta_s`,
:math:`f_{B,\text{eq}} = \tau_b/(\tau_a+\tau_b)`.

----

Verification Notes
==================

Discrepancy: Sticky Rouse Effective Time
-----------------------------------------

The documentation uses the **additive** form
:math:`\tau_{\text{eff},k} = \tau_{R,k} + \tau_s`, while the code uses
**max**: :math:`\tau_{\text{eff},k} = \max(\tau_{R,k},\tau_s)`.  Both are
physically motivated but numerically distinct.  Recommend aligning the
documentation with the code (or vice versa) and adding a note about the
alternative.

All Other Equations: Verified
------------------------------

- Green-Tobolsky/Tanaka-Edwards constitutive equation, steady states,
  SAOS moduli, and relaxation modulus match published forms exactly.
- Cates geometric-mean :math:`\tau_d`, Cole-Cole semicircle, and UCM
  constitutive equation match Cates (1987, 1990) and Turner & Cates (1991).
- Loop-Bridge kinetics, Bell detachment, and stress coupling to bridge
  fraction match Tanaka & Edwards (1992) and Annable et al. (1993).
- Bell force-dependent breakage matches Bell (1978) Kramers form.

----

Sources
=======

Web-accessible references:

- `Probing the Molecular Mechanism of Viscoelastic Relaxation in Transient
  Networks (PMC 2023) <https://pmc.ncbi.nlm.nih.gov/articles/PMC10743357/>`_
- `Transient Network at Large Deformations: Elastic-Plastic Transition
  (PMC 2016) <https://pmc.ncbi.nlm.nih.gov/articles/PMC6432060/>`_
- `Analytic Solution for the Linear Rheology of Living Polymers
  (arXiv 2024) <https://arxiv.org/html/2407.07213>`_
- `Rheological Characterization and Theoretical Modeling for Dynamically
  Associating Polymers (PMC 2022) <https://pmc.ncbi.nlm.nih.gov/articles/PMC9523779/>`_
- `Annable et al. (1993) J. Rheol.
  <https://pubs.aip.org/sor/jor/article/37/4/695/238774/>`_
- `Tanaka & Edwards (1992) Macromolecules
  <https://pubs.acs.org/doi/10.1021/ma00031a024>`_
- `Rubinstein & Semenov (2001) Macromolecules
  <https://pubs.acs.org/doi/abs/10.1021/ma0013049>`_
- `Sticky Rouse Model for Dual Networks (2022)
  <https://pubs.acs.org/doi/10.1021/acs.macromol.1c02059>`_
- `Structure and Rheology of Wormlike Micelles (Rheol. Acta)
  <https://link.springer.com/article/10.1007/BF00396041>`_
