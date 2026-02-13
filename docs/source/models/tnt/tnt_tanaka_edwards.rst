.. _model-tnt-tanaka-edwards:

=============================================================
TNT Tanaka-Edwards (Basic Transient Network) — Handbook
=============================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
---------------

**Use when:**
  - Simple associating polymers with reversible physical crosslinks
  - Telechelic networks (e.g., PEG-PEO telechelics)
  - Dilute wormlike micelles (initial approximation)
  - Initial model for any transient network system
  - Maxwell-like behavior expected (single relaxation time)

**Parameters:**
  3 parameters: :math:`G` (network modulus, Pa), :math:`\tau_b` (bond lifetime, s),
  :math:`\eta_s` (solvent viscosity, Pa·s)

**Key equation:**
  .. math::

     \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
     - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

**Test modes:**
  All 6 protocols (FLOW_CURVE, OSCILLATION, RELAXATION, STARTUP, CREEP, LAOS)

**Material examples:**
  - Hydrophobically modified ethoxylated urethane (HEUR) solutions
  - PVA-borax gels
  - Telechelic PEG-PEO associative polymers
  - Reversible gelatin networks
  - Casein micelle dispersions (dilute)

**Key characteristics:**
  - Newtonian steady-state flow (no shear thinning without variants)
  - Quadratic normal stress difference: :math:`N_1 \propto \dot{\gamma}^2`
  - Single exponential stress relaxation
  - Stress overshoot in startup flow
  - Maxwell-type SAOS response

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`\mathbf{S}`
     - dimensionless
     - Conformation tensor (3×3 symmetric, normalized second moment of end-to-end vector)
   * - :math:`G`
     - Pa
     - Network elastic modulus (related to chain density via :math:`G \approx n_{chains} k_B T`)
   * - :math:`\tau_b`
     - s
     - Bond lifetime — mean time between bond breaking and reformation events
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity (non-network contribution to total viscosity)
   * - :math:`\eta_0`
     - Pa·s
     - Zero-shear viscosity: :math:`\eta_0 = G \tau_b + \eta_s`
   * - :math:`\boldsymbol{\kappa}`
     - 1/s
     - Velocity gradient tensor: :math:`\kappa_{ij} = \partial v_i / \partial x_j`
   * - :math:`\mathbf{D}`
     - 1/s
     - Rate of deformation tensor: :math:`\mathbf{D} = (\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2`
   * - :math:`\dot{\gamma}`
     - 1/s
     - Shear rate (scalar, for simple shear flow)
   * - :math:`Wi`
     - dimensionless
     - Weissenberg number: :math:`Wi = \tau_b \dot{\gamma}` (ratio of relaxation to flow timescales)
   * - :math:`\mathbf{I}`
     - dimensionless
     - Identity tensor (equilibrium conformation)
   * - :math:`\mathbf{R}`
     - m
     - End-to-end vector of network chain between crosslinks
   * - :math:`R_0`
     - m
     - Equilibrium root-mean-square end-to-end distance
   * - :math:`N_1`
     - Pa
     - First normal stress difference: :math:`N_1 = \sigma_{xx} - \sigma_{yy}`
   * - :math:`N_2`
     - Pa
     - Second normal stress difference: :math:`N_2 = \sigma_{yy} - \sigma_{zz}`

Overview
--------

The Tanaka-Edwards model is the foundational transient network theory for viscoelastic
materials where polymer chains are connected by reversible physical crosslinks. Originally
proposed by Green and Tobolsky (1946) in their seminal work on elastomers with
interchanging bonds, and later formalized by Tanaka and Edwards (1992) using a
conformation tensor approach, this model describes materials where bonds break and reform
with a constant rate :math:`1/\tau_b`, independent of chain stretch or applied force.

At equilibrium, bonds break and reform at equal rates, maintaining a constant network
structure. Under deformation, chains between crosslinks stretch and orient, generating
stress proportional to the deviation of the conformation tensor :math:`\mathbf{S}` from
its equilibrium value (the identity tensor :math:`\mathbf{I}`). When a bond breaks, the
chain relaxes instantly to its equilibrium conformation; when a new bond forms, the chain
joins the network with the current equilibrium state. This kinetic balance gives rise to
a single relaxation mode with Maxwell-like linear viscoelastic behavior.

The model predicts:

- **Linear viscoelasticity**: Single-mode Maxwell response in SAOS with
  :math:`G' \sim \omega^2` and :math:`G'' \sim \omega` at low frequencies
- **Steady-state flow**: Newtonian behavior (constant viscosity :math:`\eta_0`)
- **Normal stresses**: Quadratic in shear rate (:math:`N_1 = 2G(\tau_b \dot{\gamma})^2`)
- **Transient response**: Stress overshoot in startup flow, single exponential relaxation
- **LAOS**: Nonlinear harmonics from conformation tensor evolution

Despite its simplicity (only 3 parameters), the Tanaka-Edwards model provides the
molecular foundation for understanding more complex transient network materials and serves
as a starting point for extensions incorporating force-dependent breakage, finite
extensibility, and multiple relaxation modes.

Historical Context
~~~~~~~~~~~~~~~~~~

The development of transient network theory spans several decades:

1. **Green & Tobolsky (1946)**: First transient network model for rubber-like materials
   with reversible crosslinks. They introduced the concept of a network with bonds that
   can break and reform, leading to stress relaxation without permanent deformation.

2. **Yamamoto (1956)**: Provided a more rigorous kinetic theory based on statistical
   mechanics, deriving the evolution equations for network structure under deformation.

3. **Lodge (1956)**: Developed the network model framework emphasizing the role of
   entanglements and temporary junctions in polymer melts and concentrated solutions.

4. **Tanaka & Edwards (1992)**: Formulated the conformation tensor approach used in
   modern TNT models, providing a unified framework for transient networks with various
   kinetic mechanisms. Their work established the connection between molecular-scale
   bond dynamics and macroscopic rheological response.

5. **Modern developments**: The TNT framework has been extended to incorporate
   force-dependent breakage (Bell model), finite chain extensibility (FENE-P), multiple
   species, and Rouse dynamics for chain relaxation.

The Tanaka-Edwards model is mathematically equivalent to the upper-convected Maxwell
(UCM) model, but derives from molecular network kinetics rather than continuum mechanical
arguments. This molecular foundation provides physical interpretation of the parameters
and enables systematic extensions to more complex network behaviors.

Physical Foundations
--------------------

Mechanical Analogue
~~~~~~~~~~~~~~~~~~~

The Tanaka-Edwards model can be represented as a simple mechanical analogue:

.. code-block:: text

    ┌─────────┐
    │ Spring  │──── Network contribution (modulus G)
    │  (G)    │
    └────┬────┘
         │
    ┌────┴────┐
    │ Dashpot │──── Network relaxation (viscosity η_p = G·τ_b)
    │ (η_p)   │
    └─────────┘
         ║
         ║  (parallel)
         ║
    ┌─────────┐
    │ Dashpot │──── Solvent contribution (viscosity η_s)
    │ (η_s)   │
    └─────────┘

This is equivalent to the **Jeffreys model** or **Oldroyd-B model** (when solvent
viscosity is included). The spring represents the elastic response of stretched network
chains. The series dashpot represents stress relaxation due to bond breaking (at rate
:math:`1/\tau_b`). The parallel dashpot represents the contribution of the solvent and
any free (non-network) chains.

**Physical interpretation:**

- **Spring (G)**: When chains are stretched, they store elastic energy. The modulus
  :math:`G \approx n_{chains} k_B T` where :math:`n_{chains}` is the number density of
  elastically active chains.

- **Series dashpot (η_p = G·τ_b)**: Bonds break randomly at rate :math:`1/\tau_b`,
  releasing stored elastic energy and allowing chains to relax. The longer the bond
  lifetime, the higher the viscosity contribution from the network.

- **Parallel dashpot (η_s)**: Solvent and free chains provide immediate viscous
  resistance to flow, independent of network dynamics.

Network Kinetics
~~~~~~~~~~~~~~~~

The transient network is characterized by the following kinetic processes:

**Bond breakage and reformation:**

- **Breakage rate**: :math:`1/\tau_b` (constant, independent of chain conformation)
- **Reformation rate**: :math:`1/\tau_b` (maintains equilibrium at zero stress)
- **Active chain fraction**: :math:`\phi = 1` (all chains are connected, constant)

**Kinetic balance at equilibrium:**

At equilibrium (zero applied stress), bonds break and reform at equal rates:

.. math::

   \text{breakage rate} = \text{reformation rate} = \frac{1}{\tau_b}

This maintains a constant network structure with isotropic conformation
:math:`\mathbf{S} = \mathbf{I}`.

**Under deformation:**

When the material is deformed, the conformation tensor :math:`\mathbf{S}` deviates from
:math:`\mathbf{I}`. Chains become stretched and oriented:

1. **Bond breaks**: The chain instantly relaxes to equilibrium conformation
   :math:`\mathbf{S} = \mathbf{I}` and contributes :math:`-(\mathbf{S} - \mathbf{I})/\tau_b`
   to the rate of change of the average conformation.

2. **Bond forms**: The chain joins the network with the current equilibrium conformation
   :math:`\mathbf{I}` and contributes :math:`+(\mathbf{S} - \mathbf{I})/\tau_b` to
   restore equilibrium.

3. **Deformation**: The velocity gradient :math:`\boldsymbol{\kappa}` continuously
   stretches and rotates the conformation tensor, contributing
   :math:`\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T`.

The competition between deformation (which stretches chains) and bond breaking (which
relaxes chains) determines the steady-state network structure and stress.

**Constant breakage assumption:**

The key simplification of the Tanaka-Edwards model is that the breakage rate
:math:`1/\tau_b` is **independent of chain stretch**. This is appropriate when:

- Bonds are weak enough that thermal fluctuations dominate over mechanical stress
- The activation energy for bond breaking is independent of chain conformation
- The applied stress is much smaller than the bond dissociation energy

For materials where stress significantly affects bond lifetime (e.g., strong shear
thinning), force-dependent breakage models (Bell, catch-slip) are required.

Conformation Tensor
~~~~~~~~~~~~~~~~~~~

The conformation tensor :math:`\mathbf{S}` is the normalized second moment of the
end-to-end vector :math:`\mathbf{R}` of network chains:

.. math::

   \mathbf{S} = \frac{\langle \mathbf{R} \otimes \mathbf{R} \rangle}{R_0^2}

where :math:`R_0 = \sqrt{\langle R^2 \rangle_0}` is the equilibrium root-mean-square
end-to-end distance (related to the number of Kuhn segments per chain).

**Physical meaning:**

- :math:`\mathbf{S} = \mathbf{I}`: Chains are unstretched and isotropically oriented
  (equilibrium state)
- :math:`S_{ii} > 1`: Chains are stretched along direction :math:`i`
- :math:`S_{ii} < 1`: Chains are compressed along direction :math:`i`
- :math:`S_{ij} \neq 0` (off-diagonal): Chains are oriented at an angle to the principal
  axes

**In simple shear flow:**

For shear in the :math:`xy`-plane with velocity :math:`v_x = \dot{\gamma} y`:

- :math:`S_{xx}`: Chain extension in flow direction (increases with
  :math:`Wi = \tau_b \dot{\gamma}`)
- :math:`S_{yy}`: Chain extension in gradient direction (decreases slightly)
- :math:`S_{zz}`: Chain extension in vorticity direction (unchanged for 2D flow,
  :math:`S_{zz} = 1`)
- :math:`S_{xy}`: Chain orientation (proportional to shear stress)

**Stress-conformation relation:**

The stress tensor is proportional to the deviation of :math:`\mathbf{S}` from equilibrium:

.. math::

   \boldsymbol{\sigma}_{network} = G (\mathbf{S} - \mathbf{I})

This is the **Giesekus-type constitutive relation**, derived from entropic elasticity of
network chains. The total stress includes the solvent contribution:

.. math::

   \boldsymbol{\sigma}_{total} = G (\mathbf{S} - \mathbf{I}) + 2 \eta_s \mathbf{D}

where :math:`\mathbf{D} = (\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2` is the rate
of deformation tensor.

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The evolution of the conformation tensor is governed by the **upper-convected derivative**:

.. math::

   \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   - \frac{\mathbf{S} - \mathbf{I}}{\tau_b}

where:

- :math:`\frac{d\mathbf{S}}{dt}` is the material derivative (rate of change following the flow)
- :math:`\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T`
  is the upper-convected term (describes affine deformation and rotation of the
  conformation tensor with the flow)
- :math:`-(\mathbf{S} - \mathbf{I})/\tau_b` is the relaxation term (bond breaking returns
  chains to equilibrium at rate :math:`1/\tau_b`)

**Affine deformation assumption:**

The upper-convected derivative assumes that network chains deform **affinely** with the
bulk flow, i.e., the end-to-end vector :math:`\mathbf{R}` transforms as a material line
element. This is valid when:

- Chains are much smaller than any length scale of flow variation
- Chains are well-connected to the surrounding network
- Chain relaxation is much slower than local flow rearrangements

**2D simple shear flow:**

For simple shear with :math:`\boldsymbol{\kappa} = \dot{\gamma} \mathbf{e}_x \otimes \mathbf{e}_y`,
the conformation tensor evolution reduces to:

.. math::

   \frac{dS_{xx}}{dt} &= 2 \dot{\gamma} S_{xy} - \frac{S_{xx} - 1}{\tau_b}

   \frac{dS_{yy}}{dt} &= -\frac{S_{yy} - 1}{\tau_b}

   \frac{dS_{zz}}{dt} &= -\frac{S_{zz} - 1}{\tau_b}

   \frac{dS_{xy}}{dt} &= \dot{\gamma} S_{yy} - \frac{S_{xy}}{\tau_b}

**Total stress:**

The total shear stress and normal stress differences are:

.. math::

   \sigma_{xy} &= G S_{xy} + \eta_s \dot{\gamma}

   N_1 &= \sigma_{xx} - \sigma_{yy} = G (S_{xx} - S_{yy})

   N_2 &= \sigma_{yy} - \sigma_{zz} = G (S_{yy} - S_{zz})

For the Tanaka-Edwards model with constant breakage, :math:`N_2 = 0` in simple shear
(since :math:`S_{yy} = S_{zz}`).

Steady-State Solutions
~~~~~~~~~~~~~~~~~~~~~~

At steady state, :math:`d\mathbf{S}/dt = \mathbf{0}`. Solving the ODEs for simple shear:

**Conformation tensor components:**

.. math::

   S_{xy} &= \tau_b \dot{\gamma}

   S_{yy} &= 1

   S_{zz} &= 1

   S_{xx} &= 1 + 2 (\tau_b \dot{\gamma})^2

**Shear stress (flow curve):**

.. math::

   \sigma_{xy} = G \tau_b \dot{\gamma} + \eta_s \dot{\gamma} = \eta_0 \dot{\gamma}

where :math:`\eta_0 = G \tau_b + \eta_s` is the **zero-shear viscosity**. The flow curve
is **Newtonian** (constant viscosity).

**First normal stress difference:**

.. math::

   N_1 = G (S_{xx} - S_{yy}) = 2 G (\tau_b \dot{\gamma})^2

This is **quadratic in shear rate**, consistent with Maxwell-type viscoelasticity. The
normal stress coefficient :math:`\Psi_1 = N_1 / \dot{\gamma}^2 = 2 G \tau_b^2` is constant.

**Second normal stress difference:**

.. math::

   N_2 = G (S_{yy} - S_{zz}) = 0

The Tanaka-Edwards model predicts zero second normal stress difference in simple shear.

Physical Insight: :math:`N_2 = 0`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prediction :math:`N_2 = 0` in simple shear is a direct consequence of the
**upper-convected derivative** (affine kinematics). The gradient direction (:math:`y`)
and vorticity direction (:math:`z`) are equivalent because the affine deformation acts
only through :math:`\boldsymbol{\kappa}`, which has no :math:`yy` or :math:`zz`
components in simple shear. Therefore :math:`S_{yy}(t) = S_{zz}(t)` for all :math:`t`,
giving :math:`N_2 = G(S_{yy} - S_{zz}) = 0`.

Experimental observation of :math:`N_2 \neq 0` is a clear indicator that **non-affine
deformation** is present. See :doc:`tnt_non_affine` for the Gordon-Schowalter extension
that breaks this symmetry.

Stress Relaxation Non-Exponentiality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the base Tanaka-Edwards model, stress relaxation is a perfect single exponential
:math:`G(t) = G e^{-t/\tau_b}`. However, even within the TNT framework, non-exponential
relaxation arises naturally from several extensions:

- **Bell breakage**: Initially stretched chains have shorter :math:`\tau_b^{\text{eff}}`,
  producing fast initial decay that slows as chains relax toward equilibrium. This mimics
  a **broad spectrum** without requiring multiple modes.
- **Multi-mode** (StickyRouse, MultiSpecies): Explicit multi-exponential decay from
  discrete :math:`\{G_k, \tau_k\}` spectrum.
- **Cates model**: :math:`G(t) \sim \exp(-\sqrt{2t/\tau_b})` in the fast-breaking regime
  — a stretched exponential from scission/recombination dynamics.

**Weissenberg number dependence:**

The dimensionless Weissenberg number :math:`Wi = \tau_b \dot{\gamma}` controls the
relative magnitude of elastic and viscous stresses:

- :math:`Wi \ll 1`: Viscous-dominated, :math:`\sigma \approx \eta_0 \dot{\gamma}`
- :math:`Wi \sim 1`: Comparable elastic and viscous contributions
- :math:`Wi \gg 1`: Elastic-dominated, :math:`N_1 \gg \sigma_{xy}`

SAOS Response
~~~~~~~~~~~~~

For small-amplitude oscillatory shear (SAOS), the conformation tensor is linearized around
equilibrium :math:`\mathbf{S} = \mathbf{I}`:

.. math::

   \mathbf{S} = \mathbf{I} + \delta \mathbf{S} e^{i \omega t}

Substituting into the evolution equation and solving for the complex modulus:

.. math::

   G^*(\omega) = G' + i G''

where the storage and loss moduli are:

.. math::

   G'(\omega) &= G \frac{(\omega \tau_b)^2}{1 + (\omega \tau_b)^2}

   G''(\omega) &= G \frac{\omega \tau_b}{1 + (\omega \tau_b)^2} + \eta_s \omega

**Low-frequency behavior** (:math:`\omega \tau_b \ll 1`):

.. math::

   G'(\omega) &\approx G (\omega \tau_b)^2 \quad \text{(quadratic)}

   G''(\omega) &\approx G \omega \tau_b + \eta_s \omega = \eta_0 \omega \quad \text{(linear)}

This is characteristic of **Maxwell-type viscoelasticity** with a single relaxation mode.

**High-frequency behavior** (:math:`\omega \tau_b \gg 1`):

.. math::

   G'(\omega) &\approx G \quad \text{(plateau)}

   G''(\omega) &\approx \frac{G}{\omega \tau_b} + \eta_s \omega \quad \text{(decreasing + solvent)}

The storage modulus approaches the network modulus :math:`G`, representing the elastic
response of the network before bonds have time to break.

**Crossover frequency:**

The loss tangent :math:`\tan \delta = G'' / G'` equals 1 at the crossover frequency:

.. math::

   \omega_c = \frac{1}{\tau_b}

This provides a direct experimental measure of the bond lifetime.

**Complex viscosity:**

.. math::

   \eta^*(\omega) = \frac{G^*(\omega)}{i \omega} = \eta' - i \eta''

where:

.. math::

   \eta'(\omega) &= \frac{G'(\omega)}{\omega} = G \frac{\omega \tau_b}{1 + (\omega \tau_b)^2} + \eta_s

   \eta''(\omega) &= \frac{G''(\omega)}{\omega} = G \frac{(\omega \tau_b)^2}{\omega [1 + (\omega \tau_b)^2]}

At low frequencies, :math:`\eta'(\omega \to 0) = \eta_0`.

Relaxation
~~~~~~~~~~

For stress relaxation after a step shear strain :math:`\gamma_0` applied at :math:`t = 0`:

**Initial condition:**

.. math::

   \mathbf{S}(t=0) = \mathbf{I} + \gamma_0 (\mathbf{e}_x \otimes \mathbf{e}_y + \mathbf{e}_y \otimes \mathbf{e}_x)

so :math:`S_{xy}(0) = \gamma_0` and all other components are at equilibrium.

**Evolution:**

For :math:`t > 0`, with :math:`\boldsymbol{\kappa} = \mathbf{0}` (no further deformation):

.. math::

   \frac{dS_{xy}}{dt} = -\frac{S_{xy}}{\tau_b}

**Solution:**

.. math::

   S_{xy}(t) = \gamma_0 e^{-t/\tau_b}

**Stress relaxation:**

.. math::

   \sigma(t) = G S_{xy}(t) = G \gamma_0 e^{-t/\tau_b}

This is a **single exponential decay** with relaxation time :math:`\tau_b`. The relaxation
modulus is:

.. math::

   G(t) = \frac{\sigma(t)}{\gamma_0} = G e^{-t/\tau_b}

**Characteristic times:**

- Time to relax to :math:`1/e` of initial stress: :math:`t = \tau_b`
- Time to relax to 5% of initial stress: :math:`t \approx 3 \tau_b`
- Time to relax to 1% of initial stress: :math:`t \approx 4.6 \tau_b`

Startup Flow
~~~~~~~~~~~~

For startup of steady shear at constant :math:`\dot{\gamma}` starting from equilibrium
at :math:`t = 0`:

**Initial condition:**

.. math::

   \mathbf{S}(t=0) = \mathbf{I}

**Evolution:**

The conformation tensor evolves according to the ODEs derived above. For shear stress,
the analytical solution is:

.. math::

   \sigma(t) = G \tau_b \dot{\gamma} [1 - e^{-t/\tau_b}] + \eta_s \dot{\gamma}

This can be rewritten as:

.. math::

   \sigma(t) = \eta_0 \dot{\gamma} [1 - \frac{G \tau_b}{\eta_0} e^{-t/\tau_b}]

**Limiting behavior:**

- **Initial slope** (:math:`t \to 0`):
  :math:`\frac{d\sigma}{dt}\bigg|_{t=0} = G \dot{\gamma}` (elastic response)

- **Steady state** (:math:`t \to \infty`):
  :math:`\sigma_{ss} = \eta_0 \dot{\gamma}` (Newtonian flow)

- **Time to steady state**: :math:`t \approx 5 \tau_b` (>99% of steady state)

**Stress overshoot:**

For the Tanaka-Edwards model with constant breakage, there is **no stress overshoot** in
startup shear. The stress increases monotonically from zero to the steady-state value.
Stress overshoot requires either:

- Force-dependent breakage (shear thinning → overshoot)
- Finite extensibility (strain hardening → overshoot)
- Nonlinear damping (Giesekus-type nonlinearity)

Startup Phase Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The startup transient can be understood in three phases:

1. **Phase 1 — Elastic loading** (:math:`t \ll \tau_b`): Chains stretch affinely with the
   flow. :math:`S_{xy} \approx \dot{\gamma} t` grows linearly. Stress rises as
   :math:`\sigma \approx G \dot{\gamma} t` (purely elastic). No bonds have yet broken.

2. **Phase 2 — Bond turnover onset** (:math:`t \sim \tau_b`): Bonds begin breaking at
   rate :math:`1/\tau_b`. For force-dependent variants (Bell), :math:`\text{tr}(\mathbf{S})`
   rises and :math:`\tau_b^{\text{eff}}` decreases, accelerating destruction beyond creation.
   This produces a stress **overshoot** in Bell/FENE variants (but not in the base
   Tanaka-Edwards model with constant breakage).

3. **Phase 3 — Steady state** (:math:`t \gg \tau_b`): Formation-destruction balance is
   reached. The conformation tensor settles to its steady-state value. For the base model,
   this is :math:`S_{xy} = \tau_b \dot{\gamma}`.

.. note::

   The base Tanaka-Edwards model reaches steady state **monotonically** (no overshoot).
   A stress overshoot requires force-dependent breakage (:doc:`tnt_bell`), finite
   extensibility (:doc:`tnt_fene_p`), or non-affine slip (:doc:`tnt_non_affine`).

**First normal stress difference in startup:**

The analytical solution for :math:`N_1(t)` involves both :math:`S_{xx}` and :math:`S_{yy}`:

.. math::

   N_1(t) = 2 G (\tau_b \dot{\gamma})^2 [1 - e^{-t/\tau_b} - \frac{t}{\tau_b} e^{-t/\tau_b}]

The normal stress exhibits a **transient overshoot** before approaching the steady-state
value :math:`N_1^{ss} = 2 G (\tau_b \dot{\gamma})^2`. The overshoot occurs at
:math:`t_{max} \approx 2 \tau_b`.

Creep
~~~~~

For creep under constant applied stress :math:`\sigma_0`, the shear rate :math:`\dot{\gamma}(t)`
and strain :math:`\gamma(t)` must be solved from the coupled ODEs:

**Governing equations:**

.. math::

   \frac{dS_{xy}}{dt} &= \dot{\gamma} S_{yy} - \frac{S_{xy}}{\tau_b}

   \frac{dS_{yy}}{dt} &= -\frac{S_{yy} - 1}{\tau_b}

   \sigma_0 &= G S_{xy} + \eta_s \dot{\gamma}

**Shear rate from stress balance:**

.. math::

   \dot{\gamma}(t) = \frac{\sigma_0 - G S_{xy}(t)}{\eta_s}

**Strain:**

.. math::

   \gamma(t) = \int_0^t \dot{\gamma}(t') dt'

**Analytical solution:**

The creep compliance :math:`J(t) = \gamma(t) / \sigma_0` is:

.. math::

   J(t) = \frac{1}{G} [1 - e^{-t/\tau_b}] + \frac{t}{\eta_0}

This consists of:

1. **Elastic contribution**: :math:`J_e = 1/G` (instantaneous elastic deformation, approached
   at :math:`t \approx 5\tau_b`)

2. **Viscous contribution**: :math:`J_v(t) = t/\eta_0` (linear in time, steady-state flow)

**Limiting behavior:**

- **Short time** (:math:`t \ll \tau_b`):
  :math:`J(t) \approx t/G \tau_b = t/\eta_p` (network relaxation)

- **Long time** (:math:`t \gg \tau_b`):
  :math:`J(t) \approx 1/G + t/\eta_0` (steady flow with elastic offset)

**Creep recovery:**

If stress is removed at :math:`t = t_1`, the recoverable strain is:

.. math::

   \gamma_{rec} = \frac{\sigma_0}{G} [1 - e^{-t_1/\tau_b}]

The non-recoverable (viscous) strain is:

.. math::

   \gamma_{visc} = \frac{\sigma_0 t_1}{\eta_0}

Creep Rupture (Force-Dependent Extension)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the base Tanaka-Edwards model with constant breakage always reaches a finite
steady-state creep rate :math:`\dot{\gamma}_{ss} = \sigma_0/\eta_0`, the introduction
of force-dependent breakage (Bell variant) creates the possibility of **creep rupture**:

1. Applied stress :math:`\sigma_0` stretches chains → :math:`\text{tr}(\mathbf{S})` increases
2. Stretch increases :math:`k_{\text{off}}` (Bell) → effective :math:`\tau_b` decreases
3. Reduced :math:`\tau_b` means lower effective viscosity → :math:`\dot{\gamma}` increases
4. Higher :math:`\dot{\gamma}` stretches chains further → positive feedback loop
5. Above a critical stress :math:`\sigma_c`, the system undergoes **delayed yielding**
   (creep rate accelerates without bound)

This viscosity bifurcation is analogous to the yielding transition in thixotropic fluids.
See :doc:`tnt_bell` for the full treatment.

LAOS
~~~~

For large-amplitude oscillatory shear (LAOS) with strain :math:`\gamma(t) = \gamma_0 \sin(\omega t)`:

**Strain rate:**

.. math::

   \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**Governing ODEs:**

The conformation tensor components evolve according to:

.. math::

   \frac{dS_{xx}}{dt} &= 2 \gamma_0 \omega \cos(\omega t) S_{xy} - \frac{S_{xx} - 1}{\tau_b}

   \frac{dS_{yy}}{dt} &= -\frac{S_{yy} - 1}{\tau_b}

   \frac{dS_{xy}}{dt} &= \gamma_0 \omega \cos(\omega t) S_{yy} - \frac{S_{xy}}{\tau_b}

**Stress response:**

.. math::

   \sigma(t) = G S_{xy}(t) + \eta_s \gamma_0 \omega \cos(\omega t)

**Fourier decomposition:**

The stress can be decomposed into odd harmonics:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots} [G'_n(\omega, \gamma_0) \sin(n \omega t)
   + G''_n(\omega, \gamma_0) \cos(n \omega t)]

where :math:`G'_1` and :math:`G''_1` are the linear viscoelastic moduli (at small
:math:`\gamma_0`), and higher harmonics :math:`G'_3, G''_3, G'_5, \ldots` characterize
nonlinearity.

**Lissajous curves:**

Plots of :math:`\sigma` vs. :math:`\gamma` (elastic Lissajous) and :math:`\sigma` vs.
:math:`\dot{\gamma}` (viscous Lissajous) reveal:

- **Linear regime** (:math:`\gamma_0 \ll 1`): Elliptical curves
- **Nonlinear regime** (:math:`\gamma_0 \sim 1`): Distorted ellipses with higher harmonics

**Numerical solution:**

LAOS requires numerical integration of the ODEs over multiple cycles until a periodic
steady state is reached. The RheoJAX implementation uses Diffrax with adaptive time
stepping.

LAOS: Linear Stress Response at All Amplitudes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A remarkable property of the base Tanaka-Edwards model (constant breakage, Hookean
chains) is that the stress response in LAOS is **perfectly sinusoidal** regardless of
strain amplitude :math:`\gamma_0`. This means:

- :math:`I_3/I_1 = 0` (no third harmonic) at **all** :math:`\gamma_0`
- The elastic Lissajous curve is always a perfect ellipse
- The Fourier spectrum contains only the fundamental frequency

This occurs because the constitutive equation is **linear in** :math:`\mathbf{S}` (the
upper-convected Maxwell model). The time-varying shear rate
:math:`\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)` enters linearly through the
velocity gradient :math:`\boldsymbol{\kappa}`, and the relaxation term is also linear.

**Consequence for model identification:** If LAOS data show **any** higher harmonics
(:math:`I_3/I_1 > 0`), the base Tanaka-Edwards model is immediately ruled out. One must
invoke nonlinear extensions:

- **Bell breakage** → odd harmonics from :math:`\exp[\nu(\text{tr}\mathbf{S} - 3)]`
- **FENE stress** → harmonics from :math:`L^2/(L^2 - \text{tr}\mathbf{S})` nonlinearity
- **Non-affine slip** → harmonics from Gordon-Schowalter coupling

Parameters
----------

.. list-table:: Tanaka-Edwards Model Parameters
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
     - Network elastic modulus (related to chain density via :math:`G \approx n_{chains} k_B T`)
   * - **tau_b**
     - :math:`\tau_b`
     - 1.0
     - (10\ :sup:`-6`, 10\ :sup:`4`)
     - s
     - Mean bond lifetime (characteristic relaxation time of the network)
   * - :math:`\eta_s`
     - :math:`\eta_s`
     - 0.0
     - (0, 10\ :sup:`4`)
     - Pa·s
     - Solvent viscosity (non-network contribution, can be zero for entangled systems)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Network modulus (G):**

The network modulus is related to the number density of elastically active chains:

.. math::

   G \approx n_{chains} k_B T

where:

- :math:`n_{chains}` is the number of chains per unit volume (m\ :sup:`-3`)
- :math:`k_B = 1.38 \times 10^{-23}` J/K is Boltzmann's constant
- :math:`T` is absolute temperature (K)

For a polymer solution with concentration :math:`c` (g/mL) and molecular weight
:math:`M_w` (g/mol):

.. math::

   n_{chains} \approx \frac{c N_A}{M_w}

where :math:`N_A = 6.02 \times 10^{23}` mol\ :sup:`-1` is Avogadro's number.

**Typical values:**

- Dilute polymer solutions: :math:`G \sim 1-100` Pa
- Semi-dilute solutions: :math:`G \sim 100-10^4` Pa
- Entangled melts and gels: :math:`G \sim 10^4-10^6` Pa

**Bond lifetime (τ_b):**

The bond lifetime :math:`\tau_b` is the mean time between bond breaking and reformation
events. It is related to the activation energy :math:`E_a` for bond dissociation via
Arrhenius kinetics:

.. math::

   \tau_b = \tau_0 \exp\left(\frac{E_a}{k_B T}\right)

where :math:`\tau_0` is a pre-exponential attempt time (typically 10\ :sup:`-9` to
10\ :sup:`-12` s).

**Experimental determination:**

- **SAOS**: Crossover frequency :math:`\omega_c = 1/\tau_b`
- **Stress relaxation**: Time to decay to :math:`1/e` of initial stress
- **DLS**: Characteristic time for structural relaxation
- **Fluorescence recovery**: Bond exchange timescale

**Typical values:**

- Weak physical gels: :math:`\tau_b \sim 10^{-3}-1` s
- Associating polymers: :math:`\tau_b \sim 0.1-100` s
- Strong supramolecular networks: :math:`\tau_b \sim 10-10^4` s

**Solvent viscosity (η_s):**

The solvent viscosity :math:`\eta_s` accounts for viscous dissipation not associated with
network relaxation. It includes:

- Viscosity of the solvent (e.g., water ~ 0.001 Pa·s at 20°C)
- Contribution from free (non-network) chains
- Hydrodynamic interactions

For entangled polymer melts without added solvent, :math:`\eta_s \approx 0` is a
reasonable approximation.

**Zero-shear viscosity:**

The zero-shear viscosity is the sum of network and solvent contributions:

.. math::

   \eta_0 = G \tau_b + \eta_s

This can be measured directly from steady-state flow curves (Newtonian plateau at low
:math:`\dot{\gamma}`).

**Parameter correlations:**

- :math:`G` and :math:`\tau_b` are strongly correlated when fitting SAOS or relaxation
  data (both affect the magnitude and timescale of the response)
- :math:`\eta_s` is most easily determined from high-frequency (or high shear rate)
  behavior where network relaxation is fast

**Bayesian priors:**

For Bayesian inference, recommended priors are:

.. math::

   G &\sim \text{LogNormal}(\log(1000), 2) \quad \text{(broad, uninformative)}

   \tau_b &\sim \text{LogNormal}(\log(1), 2) \quad \text{(centered at 1 s)}

   \eta_s &\sim \text{HalfNormal}(10) \quad \text{(weakly informative, allows zero)}

These can be refined based on prior knowledge of the material system.

Validity and Assumptions
------------------------

The Tanaka-Edwards model is valid under the following assumptions:

**1. Constant breakage rate:**

- Bond lifetime :math:`\tau_b` is independent of chain stretch or applied force
- Appropriate when thermal fluctuations dominate over mechanical stress
- Fails for materials with strong force-dependent bond kinetics (use Bell or catch-slip
  variants instead)

**2. Affine deformation:**

- Network chains deform affinely with the bulk flow (no slip)
- Valid when chains are well-connected and much smaller than flow length scales
- Fails for poorly connected networks or spatially heterogeneous deformation

**3. Instant reformation at equilibrium:**

- When a bond breaks, the chain instantly relaxes to :math:`\mathbf{S} = \mathbf{I}`
- When a bond forms, the chain joins with equilibrium conformation
- Neglects finite Rouse relaxation time (use TNTStickyRouse for chain dynamics)

**4. Linear springs (Hookean elasticity):**

- No finite extensibility limit for network chains
- Stress is proportional to :math:`\mathbf{S} - \mathbf{I}` at all strain levels
- Fails at high strains where chains approach full extension (use FENE-P variant)

**5. Monodisperse network:**

- Single bond lifetime :math:`\tau_b` (single relaxation mode)
- All chains have the same :math:`G` and :math:`R_0`
- For polydisperse systems with multiple relaxation times, use TNTMultiSpecies or sum
  of Maxwell modes

**6. No entanglements beyond crosslinks:**

- Chain relaxation occurs only via bond breaking, not reptation or constraint release
- Appropriate for dilute to semi-dilute solutions
- For entangled melts, additional relaxation mechanisms may be important

**7. Homogeneous deformation:**

- Assumes no shear banding or spatial gradients in structure
- Fails when flow induces heterogeneous microstructure (use 1D nonlocal model if needed)

**8. Incompressibility:**

- Volume is conserved (:math:`\text{tr}(\boldsymbol{\kappa}) = 0`)
- Appropriate for polymer solutions and melts
- May fail for compressible systems (e.g., foams)

Regimes and Behavior
---------------------

The Tanaka-Edwards model exhibits distinct rheological regimes depending on the
dimensionless Weissenberg number :math:`Wi = \tau_b \dot{\gamma}` (for flow) or
dimensionless frequency :math:`\omega \tau_b` (for oscillatory shear).

**Linear viscoelastic regime** (:math:`Wi \ll 1` or :math:`\omega \tau_b \ll 1`):

- **SAOS**: :math:`G' \sim \omega^2`, :math:`G'' \sim \omega` (terminal regime)
- **Flow**: Newtonian behavior, :math:`\sigma = \eta_0 \dot{\gamma}`
- **Normal stress**: Negligible, :math:`N_1 \ll \sigma`
- **Physical picture**: Bonds break and reform faster than chains can be significantly
  stretched

**Crossover regime** (:math:`Wi \sim 1` or :math:`\omega \tau_b \sim 1`):

- **SAOS**: :math:`G' \approx G'' \approx G/2`, :math:`\tan \delta = 1`
- **Flow**: Comparable elastic and viscous stresses
- **Normal stress**: :math:`N_1 \sim \sigma^2 / G`
- **Physical picture**: Deformation and relaxation timescales are matched

**Elastic-dominated regime** (:math:`Wi \gg 1` or :math:`\omega \tau_b \gg 1`):

- **SAOS**: :math:`G' \to G` (rubber-like plateau), :math:`G'' \ll G'`
- **Flow**: Normal stress dominates, :math:`N_1 \gg \sigma`
- **Physical picture**: Chains are highly stretched before bonds break, elastic energy
  storage dominates

**Key behavioral features:**

1. **Newtonian flow curve**: The Tanaka-Edwards model always predicts constant viscosity
   :math:`\eta_0` at steady state. This is a key limitation for materials that exhibit
   shear thinning. For shear-thinning behavior, use force-dependent breakage variants
   (Bell, catch-slip).

2. **Quadratic normal stress**: :math:`N_1 \propto \dot{\gamma}^2` is a universal
   prediction of single-mode Maxwell models. Experimental data showing different scaling
   (e.g., :math:`N_1 \propto \dot{\gamma}^{1.5}` for some wormlike micelles) indicates
   more complex network dynamics.

3. **Single relaxation time**: All transient responses (relaxation, startup, creep) are
   characterized by the single timescale :math:`\tau_b`. Materials with multiple
   relaxation times require multi-mode extensions.

4. **No stress overshoot in startup shear**: Unlike shear-thinning models, the
   Tanaka-Edwards model predicts monotonic stress increase. Experimental overshoot
   indicates force-dependent kinetics or finite extensibility effects.

5. **Zero second normal stress difference**: :math:`N_2 = 0` in simple shear.
   Experimental observation of :math:`N_2 < 0` indicates non-affine deformation or
   additional microstructural effects.

What You Can Learn
------------------

Fitting the Tanaka-Edwards model to experimental data provides the following physical
insights:

**From SAOS (:math:`G'`, :math:`G''` vs. ω):**

1. **Network modulus (G)**: High-frequency plateau of :math:`G'` gives :math:`G` directly.
   This is related to chain density via :math:`G \approx n_{chains} k_B T`, allowing
   estimation of the number of elastically active chains per unit volume.

2. **Bond lifetime (τ_b)**: Crossover frequency :math:`\omega_c` where :math:`G' = G''`
   gives :math:`\tau_b = 1/\omega_c`. This is the characteristic timescale for bond
   breaking and network relaxation.

3. **Zero-shear viscosity (η_0)**: Low-frequency limit :math:`\eta_0 = G'' / \omega` as
   :math:`\omega \to 0` gives :math:`\eta_0 = G \tau_b + \eta_s`.

4. **Solvent viscosity (η_s)**: High-frequency viscous contribution :math:`\eta_s \omega`
   determines the non-network viscosity.

**From stress relaxation (σ vs. t):**

1. **Bond lifetime (τ_b)**: Time to decay to :math:`1/e` of initial stress is
   :math:`\tau_b`. This is the most direct measure of bond dynamics.

2. **Network modulus (G)**: Initial stress :math:`\sigma(0) = G \gamma_0` after step
   strain :math:`\gamma_0`.

**From steady-state flow (σ vs. γ̇):**

1. **Zero-shear viscosity (η_0)**: Slope of the Newtonian flow curve.

2. **Model validity check**: If the flow curve shows shear thinning (decreasing viscosity
   with :math:`\dot{\gamma}`), the Tanaka-Edwards model is **not appropriate**. Switch
   to force-dependent breakage models (Bell, catch-slip).

**From normal stress (N₁ vs. γ̇):**

1. **Bond lifetime (τ_b)**: From :math:`N_1 = 2 G (\tau_b \dot{\gamma})^2`, the ratio
   :math:`N_1 / \sigma^2 = 2 \tau_b / \eta_0` gives a second estimate of :math:`\tau_b`
   that is independent of SAOS.

2. **Consistency check**: If :math:`N_1 / \dot{\gamma}^2` is not constant, the model
   assumptions are violated (force-dependent kinetics or finite extensibility).

**From creep (γ vs. t):**

1. **Compliance (J)**: Long-time slope :math:`1/\eta_0` and elastic offset :math:`1/G`.

2. **Recoverable strain**: Fraction of strain that recovers after stress removal is
   :math:`\gamma_{rec} / \gamma_{total} = (1 - e^{-t_1/\tau_b}) / [1 + G t_1 / (\eta_0 \tau_b)]`.

**From LAOS (harmonics vs. γ_0):**

1. **Onset of nonlinearity**: Strain amplitude :math:`\gamma_0` at which higher harmonics
   become significant. For the Tanaka-Edwards model, nonlinearity appears at
   :math:`\gamma_0 \sim 1` (order-unity strain).

2. **Nonlinear parameter space**: Odd harmonics :math:`G'_3, G''_3, \ldots` characterize
   the shape of the stress-strain response.

**Temperature dependence:**

If data are available at multiple temperatures, Arrhenius analysis of :math:`\tau_b(T)`
gives the activation energy :math:`E_a` for bond dissociation:

.. math::

   \ln(\tau_b) = \ln(\tau_0) + \frac{E_a}{k_B T}

This provides insight into the molecular mechanism of bond breaking (hydrogen bonding,
hydrophobic interactions, electrostatic interactions, etc.).

Experimental Design
-------------------

**Recommended protocols for parameter extraction:**

1. **SAOS (essential):**

   - Frequency range: At least 2 decades around :math:`\omega_c = 1/\tau_b`
   - Recommended: :math:`10^{-2}` to :math:`10^2` rad/s for typical :math:`\tau_b \sim 1` s
   - Strain amplitude: :math:`\gamma_0 < 0.1` (linear regime)
   - Output: :math:`G`, :math:`\tau_b`, :math:`\eta_s`

2. **Stress relaxation (highly informative):**

   - Step strain: :math:`\gamma_0 \sim 0.1-0.5` (linear regime)
   - Time range: At least :math:`5 \tau_b` to reach baseline
   - Output: :math:`\tau_b` (most direct), :math:`G`

3. **Steady-state flow (model validation):**

   - Shear rate range: :math:`10^{-3}` to :math:`10^2` s\ :sup:`-1` (to cover
     :math:`Wi < 1` and :math:`Wi > 1`)
   - Check: Flow curve should be Newtonian (constant :math:`\eta_0`)
   - If shear thinning is observed: model is inappropriate, use Bell variant

4. **Normal stress (optional, for consistency check):**

   - Same shear rate range as flow curve
   - Check: :math:`N_1 / \dot{\gamma}^2 = 2 G \tau_b^2` should be constant
   - Provides independent estimate of :math:`\tau_b`

5. **Startup flow (optional):**

   - Useful for validating transient behavior
   - Apply constant :math:`\dot{\gamma}` from equilibrium
   - For Tanaka-Edwards: no stress overshoot expected (monotonic rise)

6. **Creep (alternative to relaxation):**

   - Apply constant :math:`\sigma_0 < G` (linear regime)
   - Time range: At least :math:`5 \tau_b` to observe steady flow
   - Output: :math:`G`, :math:`\eta_0`, :math:`\tau_b`

7. **LAOS (for advanced characterization):**

   - Strain amplitude sweep: :math:`\gamma_0 = 0.01` to :math:`10`
   - Fixed frequency: :math:`\omega \sim 1/\tau_b` (crossover)
   - Output: Nonlinear parameters, Lissajous curves

**Experimental considerations:**

- **Sample equilibration**: Allow sufficient rest time (:math:`\sim 10 \tau_b`) between
  measurements to restore equilibrium structure

- **Temperature control**: Maintain constant temperature (±0.1°C) as :math:`\tau_b` is
  strongly temperature-dependent

- **Wall slip**: Check for slip (especially at low :math:`\dot{\gamma}`) using parallel
  plate gap variation or roughened surfaces

- **Edge fracture**: At high :math:`Wi`, normal stresses can cause edge instabilities
  in parallel plate geometry

- **Instrument compliance**: Subtract instrument inertia and compliance (especially
  important for low-modulus materials :math:`G < 100` Pa)

**Minimal experimental set for fitting:**

For quick parameter estimation with limited instrument time:

1. SAOS frequency sweep (:math:`\gamma_0 = 0.05`, :math:`10^{-2}` to :math:`10^2` rad/s)
2. Stress relaxation (step :math:`\gamma_0 = 0.2`, measure for :math:`5\tau_b`)

These two experiments are sufficient to extract all three parameters and validate the
model assumptions.

Computational Implementation
-----------------------------

**RheoJAX TNTSingleMode implementation:**

The Tanaka-Edwards model is implemented as ``TNTSingleMode`` with ``breakage="constant"``:

.. code-block:: python

   from rheojax.models import TNTSingleMode

   # Create model with constant breakage (Tanaka-Edwards)
   model = TNTSingleMode(breakage="constant")

   # Default parameters: G=1000 Pa, tau_b=1.0 s, eta_s=0.0 Pa·s
   # Bounds: G in [1, 1e8], tau_b in [1e-6, 1e4], eta_s in [0, 1e4]

**Test modes and predictions:**

1. **OSCILLATION**: Analytical solution for :math:`G'(\omega)` and :math:`G''(\omega)`
2. **RELAXATION**: Analytical solution :math:`\sigma(t) = G \gamma_0 e^{-t/\tau_b}`
3. **STARTUP**: Analytical solution :math:`\sigma(t) = \eta_0 \dot{\gamma} [1 - (G \tau_b / \eta_0) e^{-t/\tau_b}]`
4. **FLOW_CURVE**: Analytical solution :math:`\sigma = \eta_0 \dot{\gamma}`
5. **CREEP**: ODE integration for :math:`\gamma(t)`
6. **LAOS**: ODE integration for :math:`\mathbf{S}(t)` over multiple cycles

**ODE solver details:**

For protocols requiring numerical integration (CREEP, LAOS):

- **Solver**: Diffrax ``Tsit5`` (adaptive 5th-order Runge-Kutta)
- **Tolerances**: ``rtol=1e-6``, ``atol=1e-8`` (configurable)
- **Time stepping**: Adaptive with safety factor 0.9
- **JIT compilation**: All functions are JIT-compiled for GPU acceleration

**Conformation tensor representation:**

The 3×3 symmetric conformation tensor :math:`\mathbf{S}` is stored as a 6-component
vector: ``[S_xx, S_yy, S_zz, S_xy, S_xz, S_yz]``. For 2D simple shear, only the first
4 components are active.

**Numerical stability:**

- At very high :math:`Wi` (:math:`> 10^3`), the conformation tensor can become stiff.
  The adaptive solver automatically reduces time step size to maintain accuracy.

- For LAOS at high :math:`\gamma_0` (:math:`> 10`), the model may predict unphysically
  large chain extension (since there is no finite extensibility limit). Use the FENE-P
  variant if strain stiffening is observed experimentally.

**Performance:**

- **SAOS, RELAXATION, STARTUP, FLOW_CURVE**: Analytical solutions, no ODE integration
  required. Prediction time: ~0.1-1 ms for 100 points (GPU).

- **CREEP, LAOS**: ODE integration required. Prediction time: ~10-100 ms for 1000 time
  points (GPU, depends on stiffness and number of cycles).

**Fitting workflow:**

The recommended workflow uses NLSQ for fast point estimation followed by Bayesian
inference for uncertainty quantification:

.. code-block:: python

   # Step 1: NLSQ fit (fast, deterministic)
   model.fit(omega, G_star, test_mode='oscillation')
   print(f"NLSQ: G={model.G}, tau_b={model.tau_b}, eta_s={model.eta_s}")

   # Step 2: Bayesian fit with NLSQ warm-start
   result = model.fit_bayesian(omega, G_star, test_mode='oscillation',
                                num_warmup=1000, num_samples=2000, num_chains=4)

   # Step 3: Extract credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"95% CI for G: {intervals['G']}")

**Custom parameter bounds:**

For materials with known parameter ranges, custom bounds improve fitting:

.. code-block:: python

   from rheojax.core import Parameter

   model = TNTSingleMode(breakage="constant")
   model.parameters['G'].bounds = (100, 10000)  # Pa, for dilute solutions
   model.parameters['tau_b'].bounds = (0.01, 100)  # s, for fast networks
   model.fit(omega, G_star, test_mode='oscillation')

Fitting Guidance
----------------

**SAOS fitting:**

For small-amplitude oscillatory shear data:

.. code-block:: python

   import numpy as np
   from rheojax.models import TNTSingleMode

   # Prepare data: frequency, complex modulus
   omega = np.logspace(-2, 2, 50)  # 50 points, 10^-2 to 10^2 rad/s
   G_star = G_prime + 1j * G_double_prime  # Complex modulus

   # Option 1: Fit to complex modulus (recommended)
   model = TNTSingleMode(breakage="constant")
   model.fit(omega, G_star, test_mode='oscillation')

   # Option 2: Fit to G' and G'' separately (if complex data not available)
   # Stack G' and G'' as [G'_1, ..., G'_N, G''_1, ..., G''_N]
   y_data = np.concatenate([G_prime, G_double_prime])
   model.fit(omega, y_data, test_mode='oscillation')

**Tips for SAOS fitting:**

- Ensure frequency range covers at least 1 decade below and above :math:`\omega_c = 1/\tau_b`
- If :math:`\tau_b` is unknown, use :math:`\omega = 10^{-3}` to :math:`10^3` rad/s
- Weight low-frequency points more heavily if zero-shear viscosity is important
- Check for instrument compliance (spurious upturn in :math:`G''` at high :math:`\omega`)

**Stress relaxation fitting:**

For step strain relaxation:

.. code-block:: python

   # Prepare data: time, shear stress
   t = np.linspace(0, 10, 200)  # 10 s, 200 points
   sigma = ...  # Measured stress after step strain gamma_0

   # Fit (note: gamma_0 must be provided as a keyword argument)
   model = TNTSingleMode(breakage="constant")
   model.fit(t, sigma, test_mode='relaxation', gamma_0=0.2)

   # Extract parameters
   print(f"tau_b = {model.tau_b:.3f} s")
   print(f"G = {model.G:.0f} Pa")

**Tips for relaxation fitting:**

- Measure for at least :math:`5\tau_b` to reach baseline (stress decays to <1% of initial)
- If :math:`\tau_b` is unknown, measure for at least 100 s
- Subtract residual stress (instrument drift, sample thixotropy) if non-zero at long times
- The fit is relatively insensitive to :math:`\eta_s` (relaxation is dominated by network)

**Flow curve fitting:**

For steady-state shear stress vs. shear rate:

.. code-block:: python

   # Prepare data: shear rate, shear stress
   gamma_dot = np.logspace(-3, 2, 50)  # 10^-3 to 10^2 s^-1
   sigma = ...  # Measured steady-state stress

   # Fit
   model = TNTSingleMode(breakage="constant")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Expected: constant viscosity (Newtonian)
   eta_0 = model.G * model.tau_b + model.eta_s
   print(f"Zero-shear viscosity: {eta_0:.2f} Pa·s")

**Important**: If the flow curve shows shear thinning (decreasing :math:`\sigma / \dot{\gamma}`
with increasing :math:`\dot{\gamma}`), the Tanaka-Edwards model is **not appropriate**.
Switch to force-dependent breakage:

.. code-block:: python

   # For shear-thinning materials
   model = TNTSingleMode(breakage="bell")  # Bell force-dependent breakage
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

**Startup flow fitting:**

For transient shear stress during startup:

.. code-block:: python

   # Prepare data: time, shear stress during startup at constant gamma_dot
   t = np.linspace(0, 10, 200)
   sigma = ...  # Measured stress during startup

   # Fit (gamma_dot must be provided)
   model = TNTSingleMode(breakage="constant")
   model.fit(t, sigma, test_mode='startup', gamma_dot=1.0)

**Creep fitting:**

For strain vs. time under constant stress:

.. code-block:: python

   # Prepare data: time, shear strain
   t = np.linspace(0, 100, 500)
   gamma = ...  # Measured strain under constant sigma_0

   # Fit (sigma_applied must be provided)
   model = TNTSingleMode(breakage="constant")
   model.fit(t, gamma, test_mode='creep', sigma_applied=100.0)

**LAOS fitting:**

For large-amplitude oscillatory shear:

.. code-block:: python

   # Prepare data: time, shear stress (multiple cycles)
   t = np.linspace(0, 10*T, 1000)  # 10 cycles, T = 2*pi/omega
   sigma = ...  # Measured stress

   # Fit (gamma_0 and omega must be provided)
   model = TNTSingleMode(breakage="constant")
   model.fit(t, sigma, test_mode='laos', gamma_0=0.5, omega=1.0)

**Bayesian inference with NLSQ warm-start:**

For parameter uncertainty quantification:

.. code-block:: python

   # Step 1: NLSQ for initial guess
   model = TNTSingleMode(breakage="constant")
   model.fit(omega, G_star, test_mode='oscillation')

   # Step 2: Bayesian with warm-start (4 chains for robust diagnostics)
   result = model.fit_bayesian(omega, G_star, test_mode='oscillation',
                                num_warmup=1000,    # Burn-in
                                num_samples=2000,   # Samples per chain
                                num_chains=4,       # Parallel chains (default)
                                seed=42)            # Reproducibility

   # Step 3: Check diagnostics
   print(f"R-hat: {result.diagnostics['r_hat']}")  # Should be < 1.01
   print(f"ESS: {result.diagnostics['ess']}")      # Should be > 400

   # Step 4: Extract credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"G: {intervals['G'][0]:.0f} - {intervals['G'][1]:.0f} Pa (95% CI)")
   print(f"tau_b: {intervals['tau_b'][0]:.3f} - {intervals['tau_b'][1]:.3f} s (95% CI)")

**Parameter initialization:**

For models that are difficult to fit (e.g., multiple relaxation modes, high noise),
smart initialization improves convergence:

.. code-block:: python

   from rheojax.utils.initialization import initialize_from_saos

   # Extract initial guess from SAOS data
   initial_params = initialize_from_saos(omega, G_star, model_type='maxwell')

   # Set initial values
   model = TNTSingleMode(breakage="constant")
   model.G = initial_params['G']
   model.tau_b = initial_params['tau_b']

   # Fit with improved initial guess
   model.fit(omega, G_star, test_mode='oscillation')

**Common fitting issues:**

1. **Poor fit quality (low R²)**:

   - Check data quality (noise, instrument artifacts)
   - Verify model assumptions (Newtonian flow curve, single relaxation time)
   - Try multi-mode model if data show multiple relaxation times

2. **Parameter bounds hit**:

   - If :math:`G` hits upper bound (10⁸ Pa): check for instrument compliance or
     dimensionless data
   - If :math:`\tau_b` hits lower bound (10⁻⁶ s): relaxation is too fast for instrument
     or model is inappropriate
   - If :math:`\eta_s` hits upper bound: network relaxation may be too slow, or solvent
     dominates

3. **Bayesian convergence issues (R-hat > 1.1, low ESS)**:

   - Increase ``num_warmup`` (try 2000-5000 for difficult posteriors)
   - Check for multimodal posteriors (plot pair plots)
   - Tighten parameter bounds if physically justified
   - Ensure NLSQ warm-start is used (critical for convergence)

4. **Numerical instability in ODE integration (CREEP, LAOS)**:

   - Reduce ``rtol`` and ``atol`` (try 1e-8 and 1e-10)
   - Check for unphysical parameter values (negative :math:`\eta_s`, too large :math:`Wi`)
   - Use implicit solver for stiff systems (not yet implemented in RheoJAX)

Usage
-----

**Basic SAOS prediction and fitting:**

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import numpy as np
   import matplotlib.pyplot as plt

   # Create model with default parameters
   model = TNTSingleMode(breakage="constant")
   print(f"Default: G={model.G} Pa, tau_b={model.tau_b} s, eta_s={model.eta_s} Pa·s")

   # Generate SAOS prediction
   omega = np.logspace(-2, 2, 100)  # 0.01 to 100 rad/s
   G_star = model.predict(omega, test_mode='oscillation')

   # Extract G' and G''
   G_prime = G_star.real
   G_double_prime = G_star.imag

   # Plot
   plt.figure(figsize=(8, 6))
   plt.loglog(omega, G_prime, 'o-', label="G' (storage)")
   plt.loglog(omega, G_double_prime, 's-', label="G'' (loss)")
   plt.xlabel('Angular frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title('Tanaka-Edwards SAOS Response')
   plt.show()

   # Fit to synthetic data (with noise)
   np.random.seed(42)
   noise = 1 + 0.05 * np.random.randn(len(omega))  # 5% noise
   G_star_data = G_star * noise

   model_fit = TNTSingleMode(breakage="constant")
   model_fit.fit(omega, G_star_data, test_mode='oscillation')
   print(f"Fitted: G={model_fit.G:.0f} Pa, tau_b={model_fit.tau_b:.3f} s, eta_s={model_fit.eta_s:.3f} Pa·s")

**Stress relaxation:**

.. code-block:: python

   # Predict stress relaxation after step strain
   t = np.linspace(0, 10, 200)  # 0 to 10 s
   gamma_0 = 0.2  # Step strain (20%)

   sigma = model.predict(t, test_mode='relaxation', gamma_0=gamma_0)

   # Plot
   plt.figure(figsize=(8, 6))
   plt.semilogy(t, sigma, 'o-')
   plt.axhline(sigma[0] / np.e, color='red', linestyle='--',
               label=f'τ_b = {model.tau_b:.2f} s')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title('Stress Relaxation (Tanaka-Edwards)')
   plt.show()

**Startup flow:**

.. code-block:: python

   # Predict stress during startup at constant shear rate
   t = np.linspace(0, 5, 200)  # 0 to 5 s
   gamma_dot = 1.0  # 1 s^-1

   sigma = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Analytical steady-state
   eta_0 = model.G * model.tau_b + model.eta_s
   sigma_ss = eta_0 * gamma_dot

   # Plot
   plt.figure(figsize=(8, 6))
   plt.plot(t, sigma, 'o-', label='Transient')
   plt.axhline(sigma_ss, color='red', linestyle='--', label=f'Steady state ({sigma_ss:.1f} Pa)')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title(f'Startup Flow (γ̇ = {gamma_dot} s⁻¹)')
   plt.show()

**Flow curve:**

.. code-block:: python

   # Predict steady-state flow curve
   gamma_dot = np.logspace(-3, 2, 50)  # 0.001 to 100 s^-1
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   # Calculate viscosity
   eta = sigma / gamma_dot

   # Plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   ax1.loglog(gamma_dot, sigma, 'o-')
   ax1.set_xlabel('Shear rate (s⁻¹)')
   ax1.set_ylabel('Shear stress (Pa)')
   ax1.grid(True, alpha=0.3)
   ax1.set_title('Flow Curve (Tanaka-Edwards)')

   ax2.semilogx(gamma_dot, eta, 'o-')
   ax2.axhline(eta_0, color='red', linestyle='--', label=f'η₀ = {eta_0:.1f} Pa·s')
   ax2.set_xlabel('Shear rate (s⁻¹)')
   ax2.set_ylabel('Viscosity (Pa·s)')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   ax2.set_title('Viscosity (Newtonian)')

   plt.tight_layout()
   plt.show()

**Normal stress difference:**

.. code-block:: python

   # Predict first normal stress difference (requires custom calculation)
   # For Tanaka-Edwards: N_1 = 2 * G * (tau_b * gamma_dot)^2

   gamma_dot = np.logspace(-2, 2, 50)
   N_1 = 2 * model.G * (model.tau_b * gamma_dot)**2

   # Normal stress coefficient
   Psi_1 = N_1 / gamma_dot**2

   # Plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   ax1.loglog(gamma_dot, N_1, 'o-')
   ax1.set_xlabel('Shear rate (s⁻¹)')
   ax1.set_ylabel('N₁ (Pa)')
   ax1.grid(True, alpha=0.3)
   ax1.set_title('First Normal Stress Difference')

   ax2.semilogx(gamma_dot, Psi_1, 'o-')
   ax2.set_xlabel('Shear rate (s⁻¹)')
   ax2.set_ylabel('Ψ₁ (Pa·s²)')
   ax2.grid(True, alpha=0.3)
   ax2.set_title('Normal Stress Coefficient (Constant)')

   plt.tight_layout()
   plt.show()

**Creep simulation:**

.. code-block:: python

   # Predict creep under constant stress
   t = np.linspace(0, 50, 500)  # 0 to 50 s
   sigma_0 = 100.0  # Applied stress (Pa)

   gamma = model.predict(t, test_mode='creep', sigma_applied=sigma_0)

   # Analytical creep compliance
   J_t = (1/model.G) * (1 - np.exp(-t/model.tau_b)) + t / eta_0
   gamma_analytical = sigma_0 * J_t

   # Plot
   plt.figure(figsize=(8, 6))
   plt.plot(t, gamma, 'o-', label='Numerical', markersize=3)
   plt.plot(t, gamma_analytical, 'r--', label='Analytical')
   plt.xlabel('Time (s)')
   plt.ylabel('Strain')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title(f'Creep (σ₀ = {sigma_0} Pa)')
   plt.show()

**LAOS simulation:**

.. code-block:: python

   # Predict LAOS response
   omega = 1.0  # rad/s
   gamma_0 = 0.5  # Strain amplitude
   n_cycles = 10
   T = 2 * np.pi / omega
   t = np.linspace(0, n_cycles * T, 1000)

   # Note: For LAOS, predict() returns stress vs. time
   # Strain must be computed separately
   gamma_t = gamma_0 * np.sin(omega * t)
   sigma = model.predict(t, test_mode='laos', gamma_0=gamma_0, omega=omega)

   # Plot last 2 cycles (steady periodic state)
   t_plot = t[t > (n_cycles - 2) * T]
   gamma_plot = gamma_t[t > (n_cycles - 2) * T]
   sigma_plot = sigma[t > (n_cycles - 2) * T]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # Time series
   ax1.plot(t_plot, sigma_plot, 'o-', markersize=3, label='Stress')
   ax1.set_xlabel('Time (s)')
   ax1.set_ylabel('Stress (Pa)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_title('LAOS Time Series')

   # Lissajous (elastic)
   ax2.plot(gamma_plot, sigma_plot, 'o-', markersize=3)
   ax2.set_xlabel('Strain')
   ax2.set_ylabel('Stress (Pa)')
   ax2.grid(True, alpha=0.3)
   ax2.set_title('Elastic Lissajous Curve')

   plt.tight_layout()
   plt.show()

**Bayesian inference workflow:**

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   # Create synthetic SAOS data
   omega_true = np.logspace(-2, 2, 30)
   G_true, tau_b_true, eta_s_true = 1000, 1.0, 0.1
   model_true = TNTSingleMode(breakage="constant")
   model_true.G = G_true
   model_true.tau_b = tau_b_true
   model_true.eta_s = eta_s_true
   G_star_true = model_true.predict(omega_true, test_mode='oscillation')

   # Add noise
   np.random.seed(42)
   noise_factor = 1 + 0.1 * np.random.randn(len(omega_true))
   G_star_data = G_star_true * noise_factor

   # Step 1: NLSQ fit for initial guess
   model_nlsq = TNTSingleMode(breakage="constant")
   model_nlsq.fit(omega_true, G_star_data, test_mode='oscillation')
   print(f"NLSQ: G={model_nlsq.G:.0f}, tau_b={model_nlsq.tau_b:.3f}, eta_s={model_nlsq.eta_s:.3f}")

   # Step 2: Bayesian inference with NLSQ warm-start
   result = model_nlsq.fit_bayesian(omega_true, G_star_data, test_mode='oscillation',
                                     num_warmup=1000, num_samples=2000, num_chains=4, seed=42)

   # Step 3: Check diagnostics
   print(f"R-hat (G): {result.diagnostics['r_hat']['G']:.4f}")  # Should be < 1.01
   print(f"ESS (G): {result.diagnostics['ess']['G']:.0f}")      # Should be > 400

   # Step 4: Credible intervals
   intervals = model_nlsq.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"G: [{intervals['G'][0]:.0f}, {intervals['G'][1]:.0f}] Pa (true: {G_true})")
   print(f"tau_b: [{intervals['tau_b'][0]:.3f}, {intervals['tau_b'][1]:.3f}] s (true: {tau_b_true})")
   print(f"eta_s: [{intervals['eta_s'][0]:.3f}, {intervals['eta_s'][1]:.3f}] Pa·s (true: {eta_s_true})")

   # Step 5: Posterior predictive check
   # (Sample from posterior and compare predictions to data)
   posterior_samples = result.posterior_samples
   n_samples = min(100, len(posterior_samples['G']))

   plt.figure(figsize=(10, 6))
   for i in range(n_samples):
       model_sample = TNTSingleMode(breakage="constant")
       model_sample.G = posterior_samples['G'][i]
       model_sample.tau_b = posterior_samples['tau_b'][i]
       model_sample.eta_s = posterior_samples['eta_s'][i]
       G_star_sample = model_sample.predict(omega_true, test_mode='oscillation')
       plt.loglog(omega_true, np.abs(G_star_sample), 'gray', alpha=0.1)

   plt.loglog(omega_true, np.abs(G_star_data), 'ro', label='Data')
   plt.loglog(omega_true, np.abs(G_star_true), 'b-', linewidth=2, label='True')
   plt.xlabel('Angular frequency (rad/s)')
   plt.ylabel('|G*| (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title('Posterior Predictive Check (100 samples)')
   plt.show()

**Using BayesianPipeline for streamlined workflow:**

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   # Streamlined workflow with diagnostics and plotting
   pipeline = BayesianPipeline()
   (pipeline.load_data(omega_true, G_star_data, test_mode='oscillation')
            .fit_nlsq('tnt_tanaka_edwards')  # Automatic model selection
            .fit_bayesian(num_warmup=1000, num_samples=2000, num_chains=4)
            .plot_trace()      # MCMC trace plots
            .plot_pair()       # Parameter correlations
            .plot_forest()     # Credible intervals
            .save('tanaka_edwards_results.hdf5'))

See Also
--------

**Related TNT model variants:**

- :doc:`tnt_protocols` — Full protocol equations, cohort formulation, and numerical methods
- :doc:`tnt_knowledge_extraction` — Model identification and fitting guidance
- :ref:`model-tnt-bell` — Force-dependent breakage (Bell model) for shear-thinning networks
- :ref:`model-tnt-fene-p` — Finite extensibility (FENE-P) for strain stiffening
- :ref:`model-tnt-non-affine` — Gordon-Schowalter non-affine motion
- :ref:`model-tnt-sticky-rouse` — Chain Rouse dynamics between crosslinks
- :ref:`model-tnt-multi-species` — Multiple chain types with different bond lifetimes

**Alternative constitutive models:**

- :doc:`/models/giesekus/index` — Giesekus model family (nonlinear damping)
- :doc:`/models/ptt/index` — Phan-Thien-Tanner model (shear thinning via trace of stress)
- :doc:`/models/rolie_poly/index` — Rolie-Poly model (entangled polymer melts)

**Theoretical background:**

- :doc:`/models/tnt/tnt_protocols` — Detailed protocol equations for all test modes
- :doc:`/models/tnt/tnt_knowledge_extraction` — Knowledge extraction guide for TNT models

API Reference
-------------

.. currentmodule:: rheojax.models

.. autoclass:: TNTSingleMode
   :members: fit, predict, fit_bayesian, get_credible_intervals
   :inherited-members:
   :exclude-members: parameters

**Key methods:**

- ``fit(x, y, test_mode, **kwargs)``: NLSQ fit to data
- ``predict(x, test_mode, **kwargs)``: Model prediction
- ``fit_bayesian(x, y, test_mode, **kwargs)``: Bayesian inference with NUTS
- ``get_credible_intervals(samples, credibility)``: Extract credible intervals from posterior

**Parameters (attributes):**

- ``G``: Network elastic modulus (Pa), default 1000, bounds (1, 1e8)
- ``tau_b``: Bond lifetime (s), default 1.0, bounds (1e-6, 1e4)
- ``eta_s``: Solvent viscosity (Pa·s), default 0.0, bounds (0, 1e4)

**Test modes:**

- ``'oscillation'``: Small-amplitude oscillatory shear (SAOS)
- ``'relaxation'``: Stress relaxation after step strain
- ``'startup'``: Startup of steady shear
- ``'flow_curve'``: Steady-state flow curve
- ``'creep'``: Creep under constant stress
- ``'laos'``: Large-amplitude oscillatory shear

References
----------

.. [1] Green MS, Tobolsky AV (1946) A new approach to the theory of relaxing polymeric media.
   *Journal of Chemical Physics* 14:80-92. https://doi.org/10.1063/1.1724109

.. [2] Yamamoto M (1956) Theory of solutions of high polymers. *Journal of the Physical Society
   of Japan* 11:413-421. https://doi.org/10.1143/JPSJ.11.413

.. [3] Lodge AS (1956) A network theory of flow birefringence and stress in concentrated polymer
   solutions. *Transactions of the Faraday Society* 52:120-130. https://doi.org/10.1039/TF9565200120

.. [4] Tanaka F, Edwards SF (1992) Viscoelastic properties of physically cross-linked networks.
   1. Transient network theory. *Macromolecules* 25:1516-1523. https://doi.org/10.1021/ma00031a024

.. [5] Bird RB, Armstrong RC, Hassager O (1987) *Dynamics of Polymeric Liquids, Volume 1: Fluid
   Mechanics*, 2nd edition. Wiley-Interscience, New York.

.. [6] Larson RG (1999) *The Structure and Rheology of Complex Fluids*. Oxford University Press,
   New York.

.. [7] Tschoegl NW (1989) *The Phenomenological Theory of Linear Viscoelastic Behavior*.
   Springer-Verlag, Berlin.

.. [8] Mewis J, Wagner NJ (2012) *Colloidal Suspension Rheology*. Cambridge University Press,
   Cambridge.

.. [9] Rubinstein M, Colby RH (2003) *Polymer Physics*. Oxford University Press, New York.

.. [10] Doi M, Edwards SF (1986) *The Theory of Polymer Dynamics*. Oxford University Press,
   Oxford.

.. [11] Vaccaro A, Marrucci G (2000) A model for the nonlinear rheology of associating polymers.
   *Journal of Non-Newtonian Fluid Mechanics* 92:261-273. https://doi.org/10.1016/S0377-0257(00)00095-1

.. [12] Tripathi A, Tam KC, McKinley GH (2006) Rheology and dynamics of associative polymers in
   shear and extension. *Macromolecules* 39:1981-1999. https://doi.org/10.1021/ma051614x
