.. _model-tnt-bell:

===========================================================
TNT Bell (Force-Dependent Breakage) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

---------------
Quick Reference
---------------

**Use when:**

- Associating polymer networks where bond breakage accelerates with chain stretch or applied force
- Systems exhibiting shear thinning due to force-sensitive crosslinks
- Materials with slip-bond behavior (force accelerates dissociation)
- Networks where chain tension controls dissociation kinetics
- Biological gels showing force-dependent mechanics (fibrin, collagen)
- Vitrimers or dynamic covalent networks with mechanically activated exchange

**Parameters:**

- **4 base parameters**: :math:`G` (modulus, Pa), :math:`\tau_b` (bond lifetime, s), :math:`\nu` (force sensitivity, dimensionless), :math:`\eta_s` (solvent viscosity, Pa·s)
- Typical ranges: :math:`G \in [1, 10^8]` Pa, :math:`\tau_b \in [10^{-6}, 10^4]` s, :math:`\nu \in [0.01, 20]`, :math:`\eta_s \in [0, 10^4]` Pa·s

**Key equation:**

Bell breakage rate with exponential force dependence:

.. math::

   k_{\text{off}}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]

where :math:`\text{tr}(\mathbf{S}) - 3` measures mean chain stretch above equilibrium.

**Test modes:**

All 6 protocols supported: FLOW_CURVE, STARTUP, OSCILLATION, RELAXATION, CREEP, LAOS

**Material examples:**

- Fibrin networks (blood clots with force-sensitive fibrinogen cross-linking)
- Collagen gels (mechanosensitive extracellular matrix)
- Vitrimers (exchangeable covalent networks with force-activated exchange)
- Catch-bond protein assemblies (selectin-ligand, integrin-ECM)
- Force-sensitive supramolecular hydrogels (hydrogen bonds, metal-ligand)
- Telechelic associating polymers under strong flow

**Key characteristics:**

- Shear thinning viscosity :math:`\eta \sim \dot{\gamma}^{-n}` with :math:`n` controlled by :math:`\nu`
- Strain-rate-dependent stress overshoot in startup (Bell signature)
- Stretch-dependent relaxation (non-exponential decay)
- Linear SAOS identical to Maxwell model (Bell effect is second-order in strain)
- Flow curve requires numerical root-finding (no closed-form solution)

--------------
Notation Guide
--------------

.. list-table:: Mathematical Symbols
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`\mathbf{S}`
     - dimensionless
     - Conformation tensor (mean square end-to-end distance normalized by equilibrium)
   * - :math:`G`
     - Pa
     - Network elastic modulus (plateau modulus)
   * - :math:`\tau_b`
     - s
     - Equilibrium bond lifetime (zero-force reference)
   * - :math:`\nu`
     - dimensionless
     - Force sensitivity parameter (Bell exponent)
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity contribution
   * - :math:`k_{\text{off}}(\mathbf{S})`
     - :math:`s^{-1}`
     - Force-dependent bond breakage rate
   * - :math:`\text{tr}(\mathbf{S})`
     - dimensionless
     - Trace of conformation tensor (3 times mean square stretch)
   * - :math:`\boldsymbol{\kappa}`
     - :math:`s^{-1}`
     - Velocity gradient tensor
   * - :math:`\mathbf{D}`
     - :math:`s^{-1}`
     - Rate of deformation tensor (symmetric part of :math:`\boldsymbol{\kappa}`)
   * - :math:`\boldsymbol{\sigma}`
     - Pa
     - Total stress tensor
   * - :math:`\boldsymbol{\sigma}_p`
     - Pa
     - Polymeric (network) stress: :math:`G(\mathbf{S} - \mathbf{I})`
   * - :math:`\mathrm{Wi}`
     - dimensionless
     - Weissenberg number: :math:`\tau_b \dot{\gamma}` (startup) or :math:`\tau_b \omega` (SAOS)
   * - :math:`F_{\text{chain}}`
     - :math:`k_B T / a`
     - Effective chain force (entropic spring tension)
   * - :math:`N_1`
     - Pa
     - First normal stress difference: :math:`\sigma_{xx} - \sigma_{yy}`
   * - :math:`d_b`
     - nm
     - Barrier distance in bond potential (transition state location)
   * - :math:`k_0`
     - :math:`s^{-1}`
     - Zero-force breakage rate: :math:`1/\tau_b`
   * - :math:`E_a`
     - J
     - Activation energy for bond dissociation (without force)

--------
Overview
--------

Historical Context
------------------

The Bell model, introduced by G.I. Bell in 1978 for cell adhesion kinetics, describes how applied force accelerates bond dissociation by tilting the energy landscape. Originally developed for single-molecule force spectroscopy, it has been adapted to continuum polymer network models where chain tension (measured by the conformation tensor :math:`\mathbf{S}`) controls the breakage rate.

The Tanaka-Edwards transient network theory (1992) provides the base framework with constant breakage rate :math:`1/\tau_b`. The **Bell variant** extends this by making the breakage rate exponentially dependent on chain stretch:

.. math::

   k_{\text{off}}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]

This modification introduces **shear thinning** and **stretch-dependent dynamics** absent in the base model.

Physical Picture
----------------

In a transient network:

1. **Chains** connect junction points (physical or chemical crosslinks)
2. **Stretch** accumulates as flow deforms the network affinely
3. **Tension** increases the probability of bond rupture exponentially
4. **Breakage** accelerates with chain force (slip-bond mechanism)
5. **Reformation** occurs into the equilibrium (unstretched) state :math:`\mathbf{I}`

The Bell mechanism captures the intuition that **pulling on a bond makes it break faster**. The force sensitivity :math:`\nu` quantifies how much the activation barrier is lowered by chain tension.

Exponential Sensitivity
-----------------------

The Bell model assumes Kramers' escape rate:

.. math::

   k_{\text{off}}(F) = k_0 \exp\left(\frac{F \cdot d_b}{k_B T}\right)

where:

- :math:`F` is the applied force on the bond
- :math:`d_b` is the distance to the transition state
- :math:`k_B T` is thermal energy

In the continuum coarse-graining:

- **Force proxy**: :math:`F \propto (\text{tr}(\mathbf{S}) - 3)` measures mean chain stretch
- **Bell exponent**: :math:`\nu = d_b / a` where :math:`a` is the Kuhn length
- **Typical values**: :math:`\nu \sim 0.5-5` for physical bonds, :math:`\nu \sim 5-20` for weak non-covalent interactions

Distinguished from Constant-Breakage
-------------------------------------

.. list-table:: Comparison with Tanaka-Edwards Base Model
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - Tanaka-Edwards (Constant)
     - Bell (Force-Dependent)
   * - Breakage rate
     - :math:`k_{\text{off}} = 1/\tau_b`
     - :math:`k_{\text{off}} = (1/\tau_b) \exp[\nu(\text{tr}\mathbf{S} - 3)]`
   * - Steady shear viscosity
     - :math:`\eta \sim \text{const}` (Newtonian)
     - :math:`\eta \sim \dot{\gamma}^{-n}` (shear thinning)
   * - Startup overshoot
     - Strain independent: :math:`\gamma_{\text{peak}} \sim 1`
     - Decreases with :math:`\nu`: :math:`\gamma_{\text{peak}} \sim 1/\sqrt{\nu}`
   * - Relaxation
     - Single exponential
     - Stretched exponential (faster initial decay)
   * - SAOS (linear)
     - Maxwell
     - Maxwell (identical in linear regime)
   * - Parameter coupling
     - Independent :math:`G, \tau_b`
     - Strong :math:`\nu`-:math:`\tau_b` correlation

--------------------
Physical Foundations
--------------------

Kramers' Escape Problem
------------------------

A bond is modeled as a particle in a potential well :math:`U(x)` with barrier at :math:`x = d_b`:

.. math::

   U(x) = U_0 - F \cdot x

Applied force :math:`F` tilts the potential, lowering the effective barrier:

.. math::

   \Delta U_{\text{eff}} = \Delta U_0 - F \cdot d_b

The Arrhenius escape rate becomes:

.. math::

   k_{\text{off}}(F) = k_0 \exp\left(-\frac{\Delta U_{\text{eff}}}{k_B T}\right) = k_0 \exp\left(\frac{F \cdot d_b}{k_B T}\right)

This is the **Bell model** for single-molecule kinetics (dynamic force spectroscopy).

Coarse-Graining to Continuum
-----------------------------

For a network of :math:`N` chains with mean conformation tensor :math:`\mathbf{S}`:

1. **Entropic force**: :math:`F_{\text{chain}} \sim k_B T (\langle R^2 \rangle / R_0^2 - 1) \sim \text{tr}(\mathbf{S}) - 3`
2. **Mean-field assumption**: All chains experience average stretch
3. **Effective force**: Replace :math:`F` with :math:`(\text{tr}(\mathbf{S}) - 3)`
4. **Bell rate**:

   .. math::

      k_{\text{off}}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]

   where :math:`\nu = d_b / a` is dimensionless (barrier distance over Kuhn length).

Mechanical Analog
-----------------

The Bell TNT model can be visualized as:

.. code-block:: text

    ┌─────────┬────────────────────┐
    │  Spring │  Force-Dependent   │
    │   (G)   │  Dashpot (η_eff(S))│
    └─────────┴────────────────────┘
         ∥
    Solvent viscosity (η_s)

where the **dashpot viscosity** :math:`\eta_{\text{eff}} \sim G \tau_b \exp[-\nu(\text{tr}\mathbf{S} - 3)]` **decreases** with chain stretch (shear thinning).

Material Examples
-----------------

**Biological networks:**

- **Fibrin** (blood clots): :math:`\nu \sim 1-3`, force-sensitive cross-linking by Factor XIII
- **Collagen**: Mechanosensitive cross-links in extracellular matrix
- **Actin networks**: Some binding proteins exhibit force-dependent unbinding

**Synthetic polymers:**

- **Vitrimers**: Exchangeable bonds with force-dependent exchange kinetics
- **Supramolecular hydrogels**: Hydrogen bonds, :math:`\nu \sim 0.5-2`
- **Metal-ligand coordination**: :math:`\nu \sim 2-10` depending on coordination number
- **Telechelic polymers**: Associating end-groups with force-sensitive dissociation

-------------------
Governing Equations
-------------------

Conformation Tensor Evolution
------------------------------

The conformation tensor :math:`\mathbf{S}` evolves under:

.. math::

   \frac{D\mathbf{S}}{Dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T - k_{\text{off}}(\mathbf{S}) \left(\mathbf{S} - \mathbf{I}\right)

where:

- :math:`\frac{D}{Dt}` is the material derivative
- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` is the velocity gradient
- :math:`k_{\text{off}}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]`

**Physical interpretation:**

- First two terms: Affine chain stretch by flow
- Last term: Relaxation by bond breakage (rate depends exponentially on stretch)

Total Stress
------------

.. math::

   \boldsymbol{\sigma} = G \left(\mathbf{S} - \mathbf{I}\right) + 2 \eta_s \mathbf{D}

where:

- :math:`G(\mathbf{S} - \mathbf{I})` is the polymeric (network) stress
- :math:`2\eta_s \mathbf{D}` is the Newtonian solvent contribution

Steady Shear Flow
-----------------

For simple shear :math:`\boldsymbol{\kappa} = \dot{\gamma} \mathbf{e}_x \otimes \mathbf{e}_y`, at steady state :math:`\frac{D\mathbf{S}}{Dt} = 0`:

.. math::

   \begin{aligned}
   0 &= 2 \dot{\gamma} S_{xy} - k_{\text{off}}(\mathbf{S}) (S_{xx} - 1) \\
   0 &= \dot{\gamma} S_{yy} - k_{\text{off}}(\mathbf{S}) (S_{xy}) \\
   0 &= -2 \dot{\gamma} S_{xy} - k_{\text{off}}(\mathbf{S}) (S_{yy} - 1)
   \end{aligned}

where :math:`k_{\text{off}}(\mathbf{S}) = \frac{1}{\tau_b} \exp[\nu(S_{xx} + S_{yy} + S_{zz} - 3)]`.

**Critical difference from constant breakage:** This is a **nonlinear implicit system** with no closed-form solution. Numerical root-finding is required.

Flow Curve
----------

The shear stress at steady state is:

.. math::

   \sigma_{xy} = G S_{xy} + \eta_s \dot{\gamma}

where :math:`S_{xy}` is obtained by solving the coupled nonlinear system above.

**Typical behavior:**

- **Low rates** (:math:`\mathrm{Wi} \ll 1`): :math:`\sigma \sim G \tau_b \dot{\gamma}` (Newtonian)
- **High rates** (:math:`\mathrm{Wi} \gg 1`): :math:`\sigma \sim G \tau_b^{1/(1+\nu)} \dot{\gamma}^{\nu/(1+\nu)}` (shear thinning)

The viscosity follows approximate power-law:

.. math::

   \eta(\dot{\gamma}) \sim \eta_0 \left(1 + (\tau_b \dot{\gamma})^{1+\nu}\right)^{-\nu/(1+\nu)}

Small Amplitude Oscillatory Shear (SAOS)
-----------------------------------------

Linearizing around :math:`\mathbf{S} = \mathbf{I} + \delta \mathbf{S}` with :math:`\gamma(t) = \gamma_0 e^{i\omega t}`:

.. math::

   k_{\text{off}}(\mathbf{I} + \delta \mathbf{S}) \approx \frac{1}{\tau_b} \left[1 + \nu \text{tr}(\delta \mathbf{S}) + O(\delta \mathbf{S}^2)\right]

The linear response gives **standard Maxwell behavior**:

.. math::

   G^*(\omega) = G \frac{i \omega \tau_b}{1 + i \omega \tau_b}

The Bell correction enters only at second order (:math:`\gamma_0^2`), so SAOS **cannot distinguish** Bell from constant breakage in the linear regime.

Startup of Steady Shear
------------------------

Starting from :math:`\mathbf{S}(0) = \mathbf{I}`, impose :math:`\dot{\gamma}` at :math:`t > 0`. The ODE system must be integrated numerically:

.. math::

   \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T - \frac{1}{\tau_b} \exp[\nu(\text{tr}\mathbf{S} - 3)] (\mathbf{S} - \mathbf{I})

**Key predictions:**

- **Stress overshoot** at strain :math:`\gamma_{\text{peak}} \sim 1 / \sqrt{\nu}` (decreases with force sensitivity)
- **Peak stress** scales as :math:`\sigma_{\text{peak}} \sim G \mathrm{Wi}^{1/(1+\nu)}`
- **Approach to steady state** faster than constant breakage (stretch accelerates relaxation)

Stress Relaxation
------------------

After step strain :math:`\gamma_0`, with :math:`\boldsymbol{\kappa} = 0` for :math:`t > 0`:

.. math::

   \frac{d\mathbf{S}}{dt} = -\frac{1}{\tau_b} \exp[\nu(\text{tr}\mathbf{S} - 3)] (\mathbf{S} - \mathbf{I})

**For small strains** (:math:`\gamma_0 \ll 1`): Single exponential :math:`\sigma(t) \sim G \gamma_0 e^{-t/\tau_b}` (Bell term negligible)

**For large strains** (:math:`\gamma_0 > 1`): **Stretched exponential** or faster-than-exponential decay due to initial high :math:`k_{\text{off}}`.

Creep
-----

Under constant stress :math:`\sigma_0`, solve the 5-state ODE system:

.. math::

   \begin{aligned}
   \frac{d\mathbf{S}}{dt} &= \dot{\gamma}(t) \left(\mathbf{e}_x \otimes \mathbf{e}_y + \mathbf{e}_y \otimes \mathbf{e}_x\right) \cdot \mathbf{S} + \ldots - k_{\text{off}}(\mathbf{S})(\mathbf{S} - \mathbf{I}) \\
   \frac{d\gamma}{dt} &= \dot{\gamma}(t)
   \end{aligned}

where :math:`\dot{\gamma}(t)` is determined implicitly by :math:`\sigma_{xy} = \sigma_0`.

Large Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

Impose :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, integrate the full nonlinear ODE:

.. math::

   \frac{d\mathbf{S}}{dt} = \gamma_0 \omega \cos(\omega t) (\mathbf{e}_x \otimes \mathbf{e}_y + \mathbf{e}_y \otimes \mathbf{e}_x) \cdot \mathbf{S} - k_{\text{off}}(\mathbf{S})(\mathbf{S} - \mathbf{I})

Extract stress :math:`\sigma(t) = G S_{xy}(t)` and perform Fourier decomposition to get higher harmonics:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots} \left[G_n'(\omega, \gamma_0) \sin(n\omega t) + G_n''(\omega, \gamma_0) \cos(n\omega t)\right]

**Bell effect:** Nonlinearity from :math:`\exp[\nu(\text{tr}\mathbf{S} - 3)]` generates odd harmonics (:math:`n = 3, 5, 7, \ldots`).

---------------
Parameter Table
---------------

.. list-table:: TNT Bell Model Parameters
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Units
     - Description
   * - Network modulus
     - :math:`G`
     - 1000.0
     - :math:`[1, 10^8]`
     - Pa
     - Elastic modulus of the network (plateau modulus)
   * - Bond lifetime
     - :math:`\tau_b`
     - 1.0
     - :math:`[10^{-6}, 10^4]`
     - s
     - Equilibrium bond lifetime at zero force
   * - Force sensitivity
     - :math:`\nu`
     - 1.0
     - :math:`[0.01, 20]`
     - dimensionless
     - Bell exponent (barrier distance / Kuhn length)
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0
     - :math:`[0, 10^4]`
     - Pa·s
     - Newtonian solvent contribution (high-frequency viscosity)

**Parameter dependencies:**

- :math:`\tau_b` and :math:`\nu` are strongly correlated (both affect thinning)
- :math:`G` and :math:`\tau_b` set the zero-shear viscosity: :math:`\eta_0 = G \tau_b`
- :math:`\eta_s` becomes important at :math:`\dot{\gamma} \gg 1/\tau_b`

------------------------
Parameter Interpretation
------------------------

Force Sensitivity :math:`\nu`
------------------------------

**Physical meaning:** Dimensionless ratio :math:`\nu = d_b / a` where:

- :math:`d_b`: Distance to transition state in bond potential (nm)
- :math:`a`: Kuhn length of the polymer chain (nm)

**Typical values:**

- :math:`\nu \to 0`: No force sensitivity (recovers Tanaka-Edwards)
- :math:`\nu \sim 0.1-1`: Weak sensitivity (mild shear thinning)
- :math:`\nu \sim 1-5`: Moderate sensitivity (physical bonds, hydrogen bonding)
- :math:`\nu \sim 5-20`: Strong sensitivity (weak non-covalent, metal-ligand)

**Effect on rheology:**

.. math::

   \eta(\dot{\gamma}) \sim \eta_0 (\tau_b \dot{\gamma})^{-\nu/(1+\nu)}

Shear thinning exponent: :math:`n = \nu/(1+\nu)`.

Network Modulus :math:`G`
--------------------------

Determines the **plateau modulus** in linear viscoelasticity:

.. math::

   G = \frac{\rho k_B T}{M_c}

where:

- :math:`\rho`: Polymer density
- :math:`M_c`: Molecular weight between crosslinks

**From SAOS:** :math:`G = \lim_{\omega \to \infty} G'(\omega)`.

Bond Lifetime :math:`\tau_b`
-----------------------------

The zero-force relaxation time:

.. math::

   \tau_b = \frac{1}{k_0} = \frac{1}{k_{\text{off}}(\mathbf{S} = \mathbf{I})}

**From SAOS:** Crossover frequency :math:`\omega_c = 1/\tau_b` where :math:`G' = G''`.

**Relation to activation energy:**

.. math::

   \tau_b = \tau_0 \exp\left(\frac{\Delta U_0}{k_B T}\right)

where :math:`\tau_0 \sim 10^{-12}` s (attempt time) and :math:`\Delta U_0` is the barrier height.

Solvent Viscosity :math:`\eta_s`
---------------------------------

**Physical origin:**

- Newtonian contribution from unentangled solvent or free chains
- Provides high-frequency damping

**Identifiability:**

- Clear at :math:`\omega \gg 1/\tau_b` in SAOS: :math:`G'' \to \eta_s \omega`
- At low rates, :math:`\eta_s` is masked by network viscosity :math:`G \tau_b`

Parameter Correlations
----------------------

.. list-table:: Correlation Structure
   :widths: 30 30 40
   :header-rows: 1

   * - Parameter Pair
     - Correlation
     - Mitigation Strategy
   * - :math:`\tau_b` vs :math:`\nu`
     - Strong (both control thinning rate)
     - Fit SAOS first for :math:`\tau_b`, then flow curve for :math:`\nu`
   * - :math:`G` vs :math:`\tau_b`
     - Moderate (both set :math:`\eta_0`)
     - Use plateau modulus constraint from SAOS
   * - :math:`G` vs :math:`\nu`
     - Weak
     - Independent fitting possible
   * - :math:`\eta_s` vs others
     - Weak (only at high rates)
     - Fix :math:`\eta_s = 0` unless high-frequency data available

-------------------------
Validity and Assumptions
-------------------------

Core Assumptions
----------------

1. **Affine deformation:** Chains deform with the macroscopic flow (no slip, no reptation)
2. **Gaussian statistics:** Chains are entropic springs with :math:`F \sim (R - R_0)`
3. **Instantaneous reformation:** Broken chains immediately rejoin into equilibrium state :math:`\mathbf{S} = \mathbf{I}`
4. **Mean-field:** All chains experience the same average conformation
5. **Force-stretch proportionality:** :math:`F \propto \text{tr}(\mathbf{S}) - 3`

Validity Ranges
---------------

**Material classes:**

- ✓ Associating polymers with reversible bonds
- ✓ Physical gels (low crosslink density)
- ✗ Entangled melts (need reptation)
- ✗ Permanent networks (bonds don't break)

**Deformation regimes:**

- ✓ Linear viscoelasticity (:math:`\gamma_0 \ll 1`)
- ✓ Weakly nonlinear (:math:`\gamma_0 \sim 1`)
- ✗ Large strains near chain extensibility limit (use FENE variant)

**Time scales:**

- ✓ Dynamics slower than bond vibration (:math:`\omega \ll 10^{12}` rad/s)
- ✗ Sub-nanosecond dynamics (molecular detail needed)

Known Limitations
-----------------

1. **No finite extensibility:** Chains can stretch indefinitely. For :math:`\text{tr}(\mathbf{S}) \gg 10`, use FENE-Bell hybrid.
2. **No non-affine motion:** Shear banding or localized flow requires non-affine variant.
3. **No stretch-dependent creation:** Assumes reformation into :math:`\mathbf{S} = \mathbf{I}`. For oriented reformation, use stretch-creation variant.
4. **Mean-field:** Fluctuations neglected (important near gel point).

When to Use Alternatives
-------------------------

.. list-table:: Model Selection Guide
   :widths: 40 60
   :header-rows: 1

   * - Observation
     - Recommended Variant
   * - Stress saturates at high rates
     - FENE-Bell (finite extensibility)
   * - Shear banding observed
     - Non-affine Bell
   * - Second normal stress difference :math:`N_2 \neq 0`
     - Non-affine or add Giesekus-like term
   * - Thixotropy (time-dependent)
     - Add fluidity evolution (SGR-Bell hybrid)
   * - Strong strain hardening
     - FENE-Bell or Pom-Pom

-----------------------
Regimes and Behavior
-----------------------

Linear Regime (:math:`\mathrm{Wi} \ll 1`)
------------------------------------------

For :math:`\omega \tau_b \ll 1` or :math:`\dot{\gamma} \tau_b \ll 1`:

- Conformation near equilibrium: :math:`\mathbf{S} \approx \mathbf{I}`
- Bell correction negligible: :math:`\exp[\nu(\text{tr}\mathbf{S} - 3)] \approx 1`
- **Identical to Maxwell model:**

  .. math::

     G^*(\omega) = G \frac{i \omega \tau_b}{1 + i \omega \tau_b}

**Plateau modulus:** :math:`G_N^0 = G`

**Zero-shear viscosity:** :math:`\eta_0 = G \tau_b`

Weakly Nonlinear Regime (:math:`\mathrm{Wi} \sim 1`)
-----------------------------------------------------

Onset of shear thinning:

.. math::

   \eta(\dot{\gamma}) \approx \eta_0 \left[1 - \frac{\nu}{2} (\tau_b \dot{\gamma})^2 + \ldots\right]

**Startup overshoot:** Peak stress at :math:`\gamma \sim 1` (slightly reduced from constant breakage).

**First normal stress difference:**

.. math::

   N_1 \approx 2 G \tau_b^2 \dot{\gamma}^2

Strongly Nonlinear Regime (:math:`\mathrm{Wi} \gg 1`)
------------------------------------------------------

**Shear thinning power-law:**

.. math::

   \eta \sim G \tau_b (\tau_b \dot{\gamma})^{-\nu/(1+\nu)}

For :math:`\nu = 1`: :math:`\eta \sim \dot{\gamma}^{-0.5}` (square-root thinning).

**Conformation stretch:**

.. math::

   \text{tr}(\mathbf{S}) \sim (\tau_b \dot{\gamma})^{1/(1+\nu)}

**Effective relaxation time:**

.. math::

   \tau_{\text{eff}} = \frac{\tau_b}{\exp[\nu(\text{tr}\mathbf{S} - 3)]} \ll \tau_b

Chains break much faster under high stretch.

------------------
What You Can Learn
------------------

From SAOS (Linear Viscoelasticity)
-----------------------------------

1. **Plateau modulus:** :math:`G = G'_{\text{plateau}}`
2. **Relaxation time:** :math:`\tau_b = 1/\omega_c` where :math:`G' = G''`
3. **Zero-shear viscosity:** :math:`\eta_0 = G \tau_b`

**Note:** SAOS **cannot** distinguish Bell from constant breakage (need nonlinear tests).

From Flow Curve (Steady Shear)
-------------------------------

1. **Force sensitivity:** :math:`\nu` from shear thinning slope :math:`n = \nu/(1+\nu)`

   .. math::

      \log \eta \sim -n \log \dot{\gamma}

2. **Verification:** Check if :math:`\eta_0 = G \tau_b` matches SAOS
3. **Solvent viscosity:** :math:`\eta_s` from high-rate plateau (if observed)

From Startup Tests
------------------

1. **Overshoot strain:** :math:`\gamma_{\text{peak}} \sim 1/\sqrt{\nu}` gives independent :math:`\nu` estimate
2. **Rate dependence:** Peak stress scaling confirms :math:`\sigma_{\text{peak}} \sim \dot{\gamma}^{1/(1+\nu)}`
3. **Cross-validation:** Compare :math:`\nu` from thinning vs overshoot

---------------------
Experimental Design
---------------------

Protocol Selection
------------------

**Recommended test sequence:**

1. **SAOS** (strain amplitude sweep + frequency sweep)
2. **Flow curve** (steady shear rate sweep)
3. **Startup tests** at 3-5 rates
4. **Optional:** Step strain relaxation

Best Practices for Measuring :math:`\nu`
-----------------------------------------

**Method 1: Flow curve slope**

.. math::

   \nu = \frac{n}{1 - n} \quad \text{where} \quad n = -\frac{d \log \eta}{d \log \dot{\gamma}}

**Method 2: Startup overshoot strain**

.. math::

   \nu \sim \frac{1}{\gamma_{\text{peak}}^2}

**Method 3: Joint fitting**

Simultaneous NLSQ fit to flow curve + startup curves.

-------------------------------
Computational Implementation
-------------------------------

JAX Architecture
----------------

**Key function:**

.. code-block:: python

   def breakage_bell(S: Array, tau_b: float, nu: float) -> float:
       trace_S = jnp.trace(S)
       return (1.0 / tau_b) * jnp.exp(nu * (trace_S - 3.0))

---------------
Fitting Guidance
---------------

NLSQ Strategy
-------------

**Sequential fitting (recommended):**

1. Fit SAOS first (fix :math:`G`, :math:`\tau_b`, :math:`\eta_s`)
2. Fit flow curve with :math:`G`, :math:`\tau_b` fixed (optimize :math:`\nu` only)
3. Refine with startup (joint optimization)

--------------
Usage Examples
--------------

Example 1: Basic NLSQ Fitting
------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   import numpy as np

   model = TNTSingleMode(breakage="bell")
   result = model.fit(gamma_dot, sigma_true, test_mode='flow_curve')

Example 2: Bayesian Inference
------------------------------

.. code-block:: python

   result = model.fit_bayesian(
       gamma_dot, sigma_obs,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

Example 3: Startup Simulation
------------------------------

.. code-block:: python

   t, S = model.simulate_startup(gamma_dot=10.0, t_end=10.0)
   stress = model.G * S[:, 0, 1]

----------------------------------
Composition with Other Variants
----------------------------------

Bell + FENE
-----------

.. code-block:: python

   model = TNTSingleMode(breakage="bell", stress_type="fene")

--------
See Also
--------

- :ref:`model-tnt-tanaka-edwards`
- :ref:`model-tnt-fene-p`
- :ref:`model-tnt-non-affine`

-------------
API Reference
-------------

.. autoclass:: rheojax.models.tnt.TNTSingleMode
   :members:
   :noindex:

----------
References
----------

1. **Bell, G.I.** (1978). Science 200, 618-627.
2. **Tanaka, F., & Edwards, S.F.** (1992). JNFM 43, 247-271.
3. **Evans, E., & Ritchie, K.** (1997). Biophys J 72, 1541-1555.
4. **Hänggi, P., et al.** (1990). Rev Mod Phys 62, 251-341.
5. **Vaccaro, A., & Marrucci, G.** (2000). JNFM 92, 261-273.
6. **Tripathi, A., et al.** (2006). Macromolecules 39, 1981-1999.
