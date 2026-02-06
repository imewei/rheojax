.. _vlb_models:

======================================
VLB Transient Network Models
======================================

Quick Reference
===============

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Classes**
     - ``VLBLocal``, ``VLBMultiNetwork``
   * - **Physics**
     - Statistically-based transient network with distribution tensor :math:`\boldsymbol{\mu}`
   * - **Key Parameters**
     - :math:`G_0` (network modulus), :math:`k_d` (dissociation rate)
   * - **Protocols**
     - FLOW_CURVE, STARTUP, RELAXATION, CREEP, OSCILLATION, LAOS
   * - **Key Features**
     - Molecular foundation, all-analytical (single network), uniaxial extension
   * - **Reference**
     - Vernerey, Long & Brighenti (2017). *JMPS* 107, 1-20

**Import:**

.. code-block:: python

   from rheojax.models import VLBLocal, VLBMultiNetwork

**Basic Usage:**

.. code-block:: python

   # Single transient network
   model = VLBLocal()
   model.fit(omega, G_star, test_mode="oscillation")

   # Multi-network (generalized Maxwell via VLB)
   model = VLBMultiNetwork(n_modes=3, include_permanent=True)
   model.fit(omega, G_star, test_mode="oscillation")


Notation Guide
==============

.. list-table::
   :widths: 15 45 15
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`\boldsymbol{\mu}`
     - Distribution tensor (second moment of chain end-to-end vector)
     - dimensionless
   * - :math:`\varphi(\mathbf{r},t)`
     - Chain end-to-end vector distribution function
     - 1/m\ :sup:`3`
   * - :math:`G_0`
     - Network modulus (:math:`= c k_B T` for Gaussian chains)
     - Pa
   * - :math:`k_d`
     - Bond dissociation (detachment) rate
     - 1/s
   * - :math:`k_a`
     - Bond association (attachment) rate (= :math:`k_d` at equilibrium)
     - 1/s
   * - :math:`t_R`
     - Relaxation time (:math:`= 1/k_d`)
     - s
   * - :math:`\eta_0`
     - Zero-shear viscosity (:math:`= G_0/k_d`)
     - Pa·s
   * - :math:`G_e`
     - Permanent (equilibrium) network modulus
     - Pa
   * - :math:`\eta_s`
     - Solvent viscosity
     - Pa·s
   * - :math:`\mathbf{L}`
     - Velocity gradient tensor
     - 1/s
   * - :math:`\mathbf{D}`
     - Rate-of-deformation tensor (:math:`= (\mathbf{L} + \mathbf{L}^T)/2`)
     - 1/s
   * - :math:`\mathbf{W}`
     - Vorticity tensor (:math:`= (\mathbf{L} - \mathbf{L}^T)/2`)
     - 1/s
   * - :math:`c`
     - Number density of elastically active chains
     - 1/m\ :sup:`3`
   * - :math:`\text{Wi}`
     - Weissenberg number (:math:`= \dot{\gamma}/k_d`)
     - dimensionless
   * - :math:`\text{De}`
     - Deborah number (:math:`= \omega/k_d` or :math:`= 1/(k_d \cdot t_{obs})`)
     - dimensionless
   * - :math:`N_1`
     - First normal stress difference (:math:`= \sigma_{xx} - \sigma_{yy}`)
     - Pa
   * - :math:`J(t)`
     - Creep compliance
     - 1/Pa
   * - :math:`\dot{\varepsilon}`
     - Extensional strain rate
     - 1/s
   * - :math:`\eta_E`
     - Extensional (Trouton) viscosity
     - Pa·s


Overview & Historical Context
=============================

**Physical picture.**  Many soft materials — hydrogels, vitrimers, self-healing
polymers, telechelic networks, supramolecular assemblies — derive their
mechanical response from *reversible* (dynamic) cross-links that break and
reform under thermal fluctuations and mechanical load.  At equilibrium the
creation and destruction of bonds balance; under deformation the chain
configuration evolves and generates stress.

**Historical development:**

1. **Green & Tobolsky (1946)** introduced the concept of a transient network
   where chains continuously break and reform.  Under the assumption of
   instantaneous reformation in the unstressed state and constant destruction
   rate, the macroscopic response is Maxwell-like with a single exponential
   relaxation.

2. **Tanaka & Edwards (1992)** formalized the network theory using the
   conformation tensor :math:`\mathbf{S} = \langle \mathbf{Q Q} \rangle` and
   derived ODE evolution equations.  This is the basis for the TNT family in
   RheoJAX.

3. **Vernerey, Long & Brighenti (2017)** returned to the full chain
   distribution function :math:`\varphi(\mathbf{r},t)` and derived the
   distribution tensor :math:`\boldsymbol{\mu}` as its second moment, providing
   a molecular-statistical foundation that naturally connects to entropy, free
   energy, and dissipation.  This is the VLB framework.

**Key insight.**  At the Gaussian-chain level with constant :math:`k_d`, the
VLB and TNT formulations are **mathematically equivalent** — both reduce to
Maxwell viscoelasticity.  The VLB route is preferred when one wishes to
incorporate molecular extensions (Langevin finite extensibility,
force-dependent :math:`k_d`, entropic arguments) because the distribution
tensor :math:`\boldsymbol{\mu}` has a clear statistical-mechanical
interpretation.


Physical Foundations
====================

Chain Distribution Function
---------------------------

Consider a network of elastically active chains, each described by its
end-to-end vector :math:`\mathbf{r}`.  The **chain distribution function**
:math:`\varphi(\mathbf{r},t)` gives the number density of chains with
end-to-end vector :math:`\mathbf{r}` at time :math:`t`.  Its evolution is:

.. math::

   \frac{\partial \varphi}{\partial t}
   + \nabla_r \cdot (\dot{\mathbf{r}} \, \varphi)
   = k_a \varphi_0(\mathbf{r}) - k_d \varphi(\mathbf{r},t)

where:

- :math:`\dot{\mathbf{r}} = \mathbf{L} \cdot \mathbf{r}` is the affine
  convection of the end-to-end vector
- :math:`k_a \varphi_0(\mathbf{r})` represents creation of new chains in the
  equilibrium (isotropic Gaussian) distribution
- :math:`k_d \varphi` represents destruction of existing chains

At equilibrium (:math:`\mathbf{L} = 0`):  :math:`k_a \varphi_0 = k_d \varphi_{eq}`,
hence :math:`k_a = k_d`.


Distribution Tensor
-------------------

The **distribution tensor** is the normalized second moment:

.. math::

   \boldsymbol{\mu} \equiv \frac{\langle \mathbf{r} \otimes \mathbf{r} \rangle}
   {\langle r_0^2 \rangle / 3}
   = \frac{1}{c} \frac{3}{\langle r_0^2 \rangle}
   \int \mathbf{r} \otimes \mathbf{r} \, \varphi(\mathbf{r},t) \, d^3\!r

where :math:`c = \int \varphi \, d^3\!r` is the total chain number density and
:math:`\langle r_0^2 \rangle` is the mean-square end-to-end distance at
equilibrium.

**Properties:**

- :math:`\boldsymbol{\mu}` is symmetric and positive-definite
- At equilibrium: :math:`\boldsymbol{\mu}_{eq} = \mathbf{I}`
- :math:`\text{tr}(\boldsymbol{\mu})/3` measures average chain stretch relative
  to equilibrium


Governing Equations
===================

Evolution of the Distribution Tensor
--------------------------------------

By taking the second moment of the Smoluchowski equation for
:math:`\varphi(\mathbf{r},t)`:

.. math::
   :label: mu_evolution

   \dot{\boldsymbol{\mu}} = k_d(\mathbf{I} - \boldsymbol{\mu})
   + \mathbf{L} \cdot \boldsymbol{\mu}
   + \boldsymbol{\mu} \cdot \mathbf{L}^T

This is the **workhorse equation** of the VLB model.  The three terms represent:

1. **Bond kinetics** :math:`k_d(\mathbf{I} - \boldsymbol{\mu})`:  drives
   :math:`\boldsymbol{\mu}` toward equilibrium :math:`\mathbf{I}` at rate
   :math:`k_d`.

2. **Affine deformation** :math:`\mathbf{L} \cdot \boldsymbol{\mu} +
   \boldsymbol{\mu} \cdot \mathbf{L}^T`:  stretches and rotates chains
   according to the macroscopic flow.

.. note::

   Equation :eq:`mu_evolution` uses the **full velocity gradient**
   :math:`\mathbf{L}`, not the symmetric part :math:`\mathbf{D}`.  In simple
   shear with :math:`L_{12} = \dot{\gamma}`, the components are:

   .. math::

      \dot{\mu}_{xx} &= k_d(1 - \mu_{xx}) + 2\dot{\gamma}\mu_{xy} \\
      \dot{\mu}_{yy} &= k_d(1 - \mu_{yy}) \\
      \dot{\mu}_{zz} &= k_d(1 - \mu_{zz}) \\
      \dot{\mu}_{xy} &= -k_d \mu_{xy} + \dot{\gamma}\mu_{yy}

   The asymmetry (:math:`\dot{\gamma}` appears only via :math:`\mu_{xy}` and
   :math:`\mu_{yy}`) arises because the velocity gradient
   :math:`\mathbf{L}` is not symmetric in simple shear.


Cauchy Stress
-------------

For Gaussian chains the free energy per chain is :math:`\frac{3}{2}k_BT
\frac{r^2}{\langle r_0^2 \rangle}`, giving the network stress:

.. math::
   :label: cauchy_stress

   \boldsymbol{\sigma} = G_0 (\boldsymbol{\mu} - \mathbf{I}) + p\mathbf{I}

where :math:`G_0 = c k_B T` is the network modulus.

**Shear stress:**

.. math::

   \sigma_{12} = G_0 \mu_{xy}

**First normal stress difference:**

.. math::

   N_1 = \sigma_{xx} - \sigma_{yy} = G_0(\mu_{xx} - \mu_{yy})


Stored Energy and Dissipation
-----------------------------

The Helmholtz free energy density of the network is:

.. math::

   \Psi = \frac{1}{2} G_0 \bigl[\text{tr}(\boldsymbol{\mu}) - 3
   - \ln \det(\boldsymbol{\mu})\bigr]

The mechanical dissipation rate is:

.. math::

   \mathcal{D} = G_0 k_d \bigl[\text{tr}(\boldsymbol{\mu}) - 3
   - \ln \det(\boldsymbol{\mu})\bigr] \geq 0

which is non-negative by the convexity of :math:`f(x) = x - \ln x - 1` for
:math:`x > 0`, guaranteeing thermodynamic consistency.


Parameters
==========

VLBLocal Parameters (2)
-----------------------

.. list-table::
   :widths: 12 12 18 10 48
   :header-rows: 1

   * - Name
     - Default
     - Bounds
     - Units
     - Description
   * - :math:`G_0`
     - 1000.0
     - (1, 10\ :sup:`8`)
     - Pa
     - Network modulus. Product of chain density and thermal energy: :math:`G_0 = c k_B T`.
   * - :math:`k_d`
     - 1.0
     - (10\ :sup:`-6`, 10\ :sup:`6`)
     - 1/s
     - Dissociation rate.  Inverse of the characteristic network relaxation time: :math:`t_R = 1/k_d`.

**Derived quantities:**

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Property
     - Expression
     - Physical Meaning
   * - Relaxation time
     - :math:`t_R = 1/k_d`
     - Time for stress to relax to :math:`1/e` of initial value
   * - Zero-shear viscosity
     - :math:`\eta_0 = G_0/k_d`
     - Newtonian plateau viscosity
   * - Crossover frequency
     - :math:`\omega_c = k_d`
     - Frequency where :math:`G' = G''`


VLBMultiNetwork Parameters (2M + 1 or 2M + 2)
----------------------------------------------

For M transient modes:

.. list-table::
   :widths: 15 12 18 10 45
   :header-rows: 1

   * - Name
     - Default
     - Bounds
     - Units
     - Description
   * - :math:`G_I`
     - log-spaced
     - (1, 10\ :sup:`8`)
     - Pa
     - Network modulus for mode I (I = 0..M-1)
   * - :math:`k_{d,I}`
     - log-spaced
     - (10\ :sup:`-6`, 10\ :sup:`6`)
     - 1/s
     - Dissociation rate for mode I
   * - :math:`\eta_s`
     - 0.0
     - (0, 10\ :sup:`4`)
     - Pa·s
     - Solvent viscosity (always present)
   * - :math:`G_e`
     - 0.0
     - (0, 10\ :sup:`8`)
     - Pa
     - Permanent network modulus (only if ``include_permanent=True``)

**Total parameters:** 2M + 1 (without permanent) or 2M + 2 (with permanent).


Special Cases
=============

The VLB model reduces to several well-known models under special conditions:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Condition
     - Resulting Model
     - Details
   * - Single mode, constant :math:`k_d`
     - **Maxwell**
     - :math:`t_R = 1/k_d`, :math:`\eta = G_0/k_d`
   * - :math:`k_d \to 0`
     - **Neo-Hookean solid**
     - Permanent network, no relaxation
   * - :math:`k_d \to \infty`
     - **Newtonian fluid**
     - Instantaneous relaxation, :math:`\eta = G_0/k_d \to 0`
   * - M modes + :math:`G_e`
     - **Standard linear solid** (M=1)
     - Retardation + relaxation times
   * - M modes + :math:`\eta_s`
     - **Generalized Maxwell** (Prony series)
     - :math:`G(t) = \eta_s \delta(t) + \sum G_I e^{-k_{d,I} t}`
   * - M modes + :math:`G_e` + :math:`\eta_s`
     - **Oldroyd-B** (M=1)
     - Solvent + single viscoelastic mode + equilibrium


Protocol Equations
==================

Flow Curve (Steady Shear)
-------------------------

At steady state (:math:`\dot{\boldsymbol{\mu}} = 0`), the shear components give:

.. math::

   \mu_{xy}^{ss} = \frac{\dot{\gamma}}{k_d}, \qquad
   \mu_{xx}^{ss} - \mu_{yy}^{ss} = \frac{2\dot{\gamma}^2}{k_d^2}

**Steady-state viscometric functions:**

.. math::

   \sigma_{12} &= G_0 \frac{\dot{\gamma}}{k_d} = \eta_0 \dot{\gamma} \\
   N_1 &= 2 G_0 \left(\frac{\dot{\gamma}}{k_d}\right)^2

The flow curve is **Newtonian** (constant :math:`k_d`) — a key signature.
Non-Newtonian behavior requires force-dependent :math:`k_d(\boldsymbol{\mu})`
(see :doc:`vlb_extensions`).

**Multi-network:**

.. math::

   \sigma = \left(\sum_{I=0}^{M-1} \frac{G_I}{k_{d,I}} + \eta_s\right) \dot{\gamma}


Startup Shear
-------------

For startup from equilibrium (:math:`\boldsymbol{\mu}(0) = \mathbf{I}`) at
constant :math:`\dot{\gamma}`:

.. math::

   \sigma_{12}(t) = \frac{G_0 \dot{\gamma}}{k_d} \left(1 - e^{-k_d t}\right)

.. math::

   N_1(t) = 2G_0 \left(\frac{\dot{\gamma}}{k_d}\right)^2
   \left(1 - e^{-k_d t}\right)
   - 2G_0 \frac{\dot{\gamma}}{k_d} t \, e^{-k_d t}

**Characteristics:**

- :math:`\sigma_{12}(0) = 0`, monotonic rise to :math:`\sigma^{ss} = G_0 \dot{\gamma}/k_d`
- No stress overshoot (constant :math:`k_d`)
- Time constant is :math:`t_R = 1/k_d` regardless of :math:`\dot{\gamma}`
- :math:`N_1` is non-monotonic when :math:`\dot{\gamma}/k_d > 1`

**Multi-network (superposition):**

.. math::

   \sigma_{12}(t) = \sum_{I=0}^{M-1} \frac{G_I \dot{\gamma}}{k_{d,I}}
   \left(1 - e^{-k_{d,I} t}\right) + \eta_s \dot{\gamma}


Stress Relaxation
-----------------

After step strain :math:`\gamma_0`:

.. math::

   G(t) = G_0 \, e^{-k_d t}

**Characteristics:**

- :math:`G(0) = G_0` (instantaneous modulus)
- :math:`G(\infty) = 0` (liquid-like, no equilibrium modulus)
- Single exponential with time constant :math:`t_R = 1/k_d`
- :math:`\ln G(t)` is linear in :math:`t` with slope :math:`-k_d`

**Multi-network (Prony series):**

.. math::

   G(t) = G_e + \sum_{I=0}^{M-1} G_I \, e^{-k_{d,I} t}


Creep Compliance
----------------

Under constant stress :math:`\sigma_0`:

.. math::

   J(t) = \frac{1 + k_d t}{G_0}

**Characteristics:**

- :math:`J(0) = 1/G_0` (instantaneous elastic compliance)
- :math:`dJ/dt = k_d/G_0 = 1/\eta_0` (viscous flow rate)
- Linear in time — characteristic of Maxwell liquid
- From creep: :math:`G_0 = 1/J(0)`, :math:`k_d = G_0 \cdot dJ/dt`

**Dual-network (1 transient + permanent):**

.. math::

   J(t) = \frac{1}{G_0 + G_e} + \frac{G_0}{G_e(G_0 + G_e)}
   \left(1 - e^{-t/\tau_{ret}}\right)

where :math:`\tau_{ret} = (G_0 + G_e)/(G_e \cdot k_d)` is the retardation time.


Small-Amplitude Oscillatory Shear (SAOS)
-----------------------------------------

For oscillatory strain :math:`\gamma(t) = \gamma_0 e^{i\omega t}`:

.. math::

   G'(\omega) &= G_0 \frac{\omega^2 t_R^2}{1 + \omega^2 t_R^2} \\
   G''(\omega) &= G_0 \frac{\omega t_R}{1 + \omega^2 t_R^2}

where :math:`t_R = 1/k_d`.  This is exactly the **Maxwell model**.

**Crossover:**  :math:`G'(\omega_c) = G''(\omega_c)` at :math:`\omega_c = k_d`,
with :math:`G'(\omega_c) = G''(\omega_c) = G_0/2`.

**Multi-network:**

.. math::

   G'(\omega) &= G_e + \sum_{I=0}^{M-1} G_I \frac{\omega^2/k_{d,I}^2}{1 + \omega^2/k_{d,I}^2} \\
   G''(\omega) &= \eta_s \omega + \sum_{I=0}^{M-1} G_I \frac{\omega/k_{d,I}}{1 + \omega^2/k_{d,I}^2}


Large-Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

Under :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, the full ODE system
:eq:`mu_evolution` is integrated numerically.

**Key result for constant** :math:`k_d`:

- :math:`\sigma_{12}(t)` is **purely linear** (no higher harmonics)
  because the ODE is linear in :math:`\boldsymbol{\mu}` for constant
  :math:`k_d`.
- :math:`N_1(t)` contains a :math:`2\omega` (second harmonic) component even
  at constant :math:`k_d`, because :math:`N_1 = G_0(\mu_{xx} - \mu_{yy})` is
  a quadratic function of the strain.

This means:

- :math:`I_3/I_1 \approx 0` for :math:`\sigma_{12}` (linear response)
- :math:`N_1` oscillates at :math:`2\omega` with a nonzero mean

**LAOS becomes truly nonlinear** when :math:`k_d` depends on
:math:`\boldsymbol{\mu}` (Bell model, see :doc:`vlb_extensions`).


Uniaxial Extension
------------------

For uniaxial extension with rate :math:`\dot{\varepsilon}`, the velocity
gradient is :math:`L_{11} = \dot{\varepsilon}`, :math:`L_{22} = L_{33}
= -\dot{\varepsilon}/2`.

**Steady-state extensional stress:**

.. math::

   \sigma_E = G_0 \dot{\varepsilon}
   \left(\frac{1}{k_d - \dot{\varepsilon}} + \frac{1}{k_d + \dot{\varepsilon}/2}\right)

**Trouton ratio:**

.. math::

   \text{Tr} = \frac{\eta_E}{\eta_0} = \frac{k_d}{k_d - \dot{\varepsilon}}
   + \frac{k_d}{k_d + \dot{\varepsilon}/2}
   - \frac{2k_d}{k_d}

At low extension rates (:math:`\dot{\varepsilon} \ll k_d`):
:math:`\text{Tr} \to 3` (Newtonian Trouton ratio).

**Singularity:**  At :math:`\dot{\varepsilon} = k_d`, the axial component
diverges — this is the extensional catastrophe, analogous to coil-stretch
transition.  Finite extensibility (Langevin chains) regularizes this
singularity (see :doc:`vlb_extensions`).


Multi-Network Model
===================

Physical Picture
----------------

Real polymers often have multiple populations of chains with different
lifetimes, or a combination of reversible and permanent cross-links.
The VLBMultiNetwork model captures this via:

.. math::

   \boldsymbol{\sigma} = \sum_{I=0}^{M-1} G_I (\boldsymbol{\mu}_I - \mathbf{I})
   + G_e (\boldsymbol{\mu}_\infty - \mathbf{I}) + 2\eta_s \mathbf{D}

where each transient mode :math:`I` has its own distribution tensor
:math:`\boldsymbol{\mu}_I` evolving with rate :math:`k_{d,I}`, the permanent
network (:math:`k_d = 0`) maintains equilibrium strain, and the solvent
contributes Newtonian stress.


Relaxation Spectrum
-------------------

The relaxation modulus is a **Prony series**:

.. math::

   G(t) = G_e + \sum_{I=0}^{M-1} G_I \, e^{-k_{d,I} t}

**Fitting strategy:**

1. Start with :math:`M = 1` and increase until residuals plateau
2. Initialize modes at log-spaced :math:`k_d` values spanning the
   experimental frequency range
3. Use SAOS data (broadest frequency window) as the primary fitting target
4. Validate with relaxation and/or startup data


Validity & Assumptions
======================

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Assumption
     - Details & Limitations
   * - **Gaussian chains**
     - Chains follow Gaussian statistics (:math:`P(r) \propto \exp(-3r^2/2\langle r_0^2 \rangle)`). Breaks down for highly stretched chains. See Langevin extension in :doc:`vlb_extensions`.
   * - **Constant** :math:`k_d`
     - Bond lifetime is independent of chain stretch or force.  Results in Newtonian flow curve and linear LAOS. Force-dependent :math:`k_d` introduces shear thinning (see :doc:`vlb_extensions`).
   * - **Affine deformation**
     - Chains deform affinely with the macroscopic flow (:math:`\dot{\mathbf{r}} = \mathbf{L} \cdot \mathbf{r}`).  Non-affine effects (fluctuations, excluded volume) are neglected.
   * - **Incompressibility**
     - Pressure :math:`p` is a Lagrange multiplier; material is assumed incompressible.
   * - **Monodisperse chains**
     - All chains in a given mode have the same :math:`G_0` and :math:`k_d`.  Polydispersity requires multiple modes.
   * - **Isothermal**
     - No temperature dependence.  Temperature enters through :math:`G_0 = c k_B T` and :math:`k_d = k_d^0 \exp(-E_a/k_BT)`.
   * - **No chain entanglement**
     - Chains interact only through cross-links.  Entanglement effects (reptation) are not included.


When to Use VLB
===============

**Ideal materials:**

- **Hydrogels** with dynamic cross-links (e.g., boronate ester, Diels-Alder)
- **Vitrimers** and covalent adaptable networks (CAN)
- **Telechelic polymers** with associating end groups
- **Self-healing polymers** with reversible bonds
- **Supramolecular networks** with hydrogen bonding or metal-ligand coordination
- **Physical gels** (PVA-borax, gelatin, agarose at low concentrations)

**Use VLBLocal when:**

- Material shows single-exponential relaxation
- SAOS data has a clear crossover frequency
- Two parameters (:math:`G_0, k_d`) suffice
- You want the simplest physically-grounded model

**Use VLBMultiNetwork when:**

- Relaxation spectrum is broad (multiple time scales)
- SAOS shows features at multiple frequencies
- Material has both reversible and permanent cross-links
- You need an equilibrium modulus (:math:`G_e > 0`)

**Do NOT use VLB when:**

- Material shows significant shear thinning (need force-dependent :math:`k_d`)
- Stress overshoot is present (need Bell variant or similar)
- Material is entangled (use reptation-based models)
- Yield stress behavior is dominant (use DMT or Fluidity models)


What You Can Learn
==================

From VLBLocal Parameters
-------------------------

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Physical Insight
   * - :math:`G_0`
     - 10 - 10\ :sup:`6` Pa
     - Network stiffness.  :math:`G_0 = c k_B T`, so higher :math:`G_0` means more active chains.  Compare with rubber elasticity theory.
   * - :math:`k_d`
     - 10\ :sup:`-3` - 10\ :sup:`3` 1/s
     - Bond kinetics.  Small :math:`k_d` = long-lived bonds (permanent-like).  Large :math:`k_d` = fast turnover (liquid-like).

**Material classification by** :math:`k_d`:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - :math:`k_d` Range
     - Material Type
     - Examples
   * - :math:`< 10^{-2}` 1/s
     - Permanent-like gel
     - Chemical gels, vulcanized rubber
   * - :math:`10^{-2}` — :math:`1` 1/s
     - Physical gel
     - PVA-borax, gelatin, associative polymers
   * - :math:`1` — :math:`10^2` 1/s
     - Viscoelastic liquid
     - Telechelic polymers, supramolecular networks
   * - :math:`> 10^2` 1/s
     - Liquid-like
     - Dilute associative solutions, fast-exchanging systems


From Multi-Network Spectrum
---------------------------

The relaxation spectrum :math:`\{(G_I, t_{R,I})\}` encodes the distribution
of bond lifetimes in the network:

- **Well-separated modes**: distinct bond populations with different chemistry
- **Closely-spaced modes**: quasi-continuous distribution (polydispersity)
- **Dominant mode**: controls the terminal relaxation
- **G_e > 0**: permanent cross-links present (solid-like long-time behavior)
- **η_s > 0**: un-networked polymer or solvent background


Cross-Protocol Validation
-------------------------

Consistency checks between protocols:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Check
     - Criterion
     - Failing Suggests
   * - :math:`\eta_0` from flow = :math:`\eta_0` from SAOS
     - :math:`\eta_{flow} = \lim_{\omega \to 0} G''/\omega`
     - Shear-rate dependence of :math:`k_d`
   * - :math:`G_0` from relaxation = :math:`G_0` from SAOS
     - :math:`G(0) = \lim_{\omega \to \infty} G'`
     - Multiple time scales present
   * - :math:`t_R` from relaxation = :math:`1/\omega_c`
     - :math:`-1/\text{slope}[\ln G(t)] = 1/\omega_c`
     - Non-Maxwell relaxation
   * - :math:`\sigma^{ss}` from startup = :math:`\sigma` from flow
     - :math:`\sigma(t \to \infty) = \eta_0 \dot{\gamma}`
     - Time-dependent structure


API Reference
=============

.. autoclass:: rheojax.models.vlb.VLBLocal
   :members:
   :inherited-members:

.. autoclass:: rheojax.models.vlb.VLBMultiNetwork
   :members:
   :inherited-members:


References
==========

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1-20.

2. Green, M.S. & Tobolsky, A.V. (1946). "A New Approach to the Theory of
   Relaxing Polymeric Media." *J. Chem. Phys.*, 14(2), 80-92.

3. Tanaka, F. & Edwards, S.F. (1992). "Viscoelastic properties of physically
   crosslinked networks." *J. Non-Newtonian Fluid Mech.*, 43(2-3), 247-271.

4. Vernerey, F.J. (2018). "Transient response of nonlinear polymer networks:
   A kinetic theory." *J. Mech. Phys. Solids*, 115, 230-247.

5. Long, R., Qi, H.J. & Dunn, M.L. (2013). "Modeling the mechanics of
   covalently adaptable polymer networks with temperature-dependent bond
   exchange reactions." *Soft Matter*, 9, 4083-4096.
