.. _hvnm_advanced:

=====================================================
HVNM Advanced Theory & Numerical Methods
=====================================================

This page documents the thermodynamic foundations, interphase physics,
damage mechanics, and numerical methods underlying the HVNM.  For the
constitutive equations, see :doc:`hvnm`.  For protocol derivations,
see :doc:`hvnm_protocols`.  For the unfilled HVM theory, see
:doc:`/models/hvm/hvm_advanced`.


.. _hvnm-thermodynamics:

Thermodynamic Framework
========================

Helmholtz Free Energy
---------------------

The total Helmholtz free energy density extends the HVM
(:ref:`hvm-thermodynamics`) with the interphase contribution and
interfacial damage:

.. math::

   \Psi_{tot} = (1-D)\,\Psi_P(\mathbf{B}_{amp})
   + \Psi_E[\boldsymbol{\mu}^E, \boldsymbol{\mu}^E_{nat}]
   + \Psi_D[\boldsymbol{\mu}^D]
   + (1-D_{int})\,\Psi_I[\boldsymbol{\mu}^I, \boldsymbol{\mu}^I_{nat}]
   + p(\det\mathbf{F} - 1)

Each subnetwork energy is a Gaussian chain model (neo-Hookean):

**Permanent network** (hydrodynamically amplified):

.. math::

   \Psi_P = \frac{G_P X(\phi)}{2}\left(\text{tr}(\mathbf{B}) - 3\right)

where :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2` is the Guth-Gold
strain amplification factor.

**Exchangeable (vitrimer) network:**

.. math::

   \Psi_E = \frac{G_E}{2}\,\text{tr}\!\left(\boldsymbol{\mu}^E
   - \boldsymbol{\mu}^E_{nat}\right)

**Dissociative (physical) network:**

.. math::

   \Psi_D = \frac{G_D}{2}\,\text{tr}(\boldsymbol{\mu}^D - \mathbf{I})

**Interphase network** (NEW):

.. math::

   \Psi_I = \frac{G_{I,eff}}{2}\,\text{tr}\!\left(\boldsymbol{\mu}^I
   - \boldsymbol{\mu}^I_{nat}\right)

The new scalar damage variable :math:`D_{int} \in [0,1]` couples to
:math:`\Psi_I`, representing interfacial debonding.  :math:`D` and
:math:`D_{int}` are **independent**: permanent network scission and
interfacial failure are distinct mechanisms.


Clausius-Duhem Derivation
--------------------------

Applying the Clausius-Duhem procedure to :math:`\Psi_{tot}`, the
total Cauchy stress is:

.. math::

   \boldsymbol{\sigma}_{tot} =
   \underbrace{(1-D)\,\tilde{G}_P\,(\mathbf{B}_{amp} - \mathbf{I})}_{\text{Permanent}}
   + \underbrace{G_E (\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})}_{\text{Exchangeable}}
   + \underbrace{G_D (\boldsymbol{\mu}^D - \mathbf{I})}_{\text{Dissociative}}
   + \underbrace{(1-D_{int})\,G_{I,eff} (\boldsymbol{\mu}^I
   - \boldsymbol{\mu}^I_{nat})}_{\text{Interphase}}
   - p\mathbf{I}

where :math:`\tilde{G}_P = G_P X(\phi)` is the amplified permanent modulus.

The remaining terms yield **four dissipation contributions**, each
individually non-negative:

**Exchangeable network dissipation:**

.. math::

   \mathcal{D}_{exch} = \frac{G_E}{2} k_{BER}^{mat}
   \text{tr}\!\left[(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})^2
   \cdot (\boldsymbol{\mu}^E_{nat})^{-1}\right] \geq 0

**Dissociative network dissipation:**

.. math::

   \mathcal{D}_{diss} = \frac{G_D}{2} k_d^D
   \text{tr}(\boldsymbol{\mu}^D - \mathbf{I})^2 \geq 0

**Interphase network dissipation** (NEW):

.. math::

   \mathcal{D}_{int} = \frac{G_{I,eff}}{2} k_{BER}^{int}
   \text{tr}\!\left[(\boldsymbol{\mu}^I - \boldsymbol{\mu}^I_{nat})^2
   \cdot (\boldsymbol{\mu}^I_{nat})^{-1}\right] \geq 0

**Damage dissipation** (dual):

.. math::

   \mathcal{D}_{dam} = \Psi_P \dot{D}
   + \Psi_I \dot{D}_{int} \geq 0

Satisfied because :math:`\Psi_P, \Psi_I \geq 0` and
:math:`\dot{D}, \dot{D}_{int} \geq 0` (damage is irreversible or
controlled by self-healing).


.. _hvnm-interphase-model:

Nanoparticle Interphase Model
==============================

Three-Layer Interphase Structure
---------------------------------

Following Papon et al. (2012) and confirmed by NMR and scattering
studies (Berriot et al. 2002, Kim et al. 2024), the interphase
around each nanoparticle consists of three concentric layers:

**Glassy layer** (thickness :math:`\delta_g \sim 1`--2 nm):
Chains strongly adsorbed/bonded to the NP surface with dynamics
:math:`>100\times` slower than bulk.  In the HVNM, this layer is
treated as part of the rigid NP inclusion, increasing the effective
filler fraction:

.. math::

   \phi_{eff} = \phi\left(1 + \frac{\delta_g}{R_{NP}}\right)^3

**Mobile interphase layer** (thickness :math:`\delta_m \sim 5`--20 nm):
Chains with intermediate dynamics -- slower than bulk but not glassy.
This is the interphase subnetwork (I) in the HVNM, with volume fraction:

.. math::

   \phi_I = \phi_{eff}\left[\left(1 + \frac{\delta_m}{R_{NP} + \delta_g}
   \right)^3 - 1\right]

For small :math:`\delta_m / R_{NP}`, this reduces to
:math:`\phi_I \approx 3\phi_{eff}\,\delta_m / (R_{NP} + \delta_g)`.

**Bulk matrix** (remainder):
Unperturbed dynamics, described by the P, E, D subnetworks.

.. note::

   The RheoJAX implementation uses the simplified formula
   :math:`\phi_I = \phi[(R_{NP} + \delta_m)^3/R_{NP}^3 - 1]`,
   which absorbs :math:`\delta_g` into :math:`R_{NP}`.


Temperature-Dependent Interphase Thickness
-------------------------------------------

The mobile interphase thickness decreases with temperature following a
WLF-type dependence:

.. math::

   \delta_m(T) = \delta_m^0 \cdot \left(
   \frac{T_g^{int} - T_\infty}{T - T_\infty}\right)^{1/\alpha_{int}}

where :math:`T_g^{int}` is the elevated glass transition in the
interphase, :math:`T_\infty = T_g^{int} - C_2^{int}` is the Vogel
temperature, and :math:`\alpha_{int} \sim 0.5`--1 is a scaling
exponent.

**Physical consequence:** For :math:`T \gg T_g^{int}`,
:math:`\delta_m \to 0` and the interphase contribution vanishes,
recovering the unfilled HVM.  Near :math:`T_g^{int}`, the interphase
is thick and NP reinforcement is maximized.


Interphase Modulus
-------------------

The interphase modulus is expressed as:

.. math::

   G_I = \beta_I \cdot G_E

where :math:`\beta_I \in [1.5, 10]` is the reinforcement ratio,
dependent on:

- **Surface chemistry:** Covalently grafted NPs
  (:math:`\beta_I \sim 3`--10) vs. physically adsorbed
  (:math:`\beta_I \sim 1.5`--3)
- **Chain-NP interaction strength:** Quantified by the Flory-Huggins
  interaction parameter :math:`\chi_{NP}`
- **Temperature:** :math:`\beta_I` decreases toward 1 as :math:`T`
  increases well above :math:`T_g^{int}`

The effective interphase modulus is:

.. math::

   G_{I,eff} = \beta_I G_E \phi_I


Interphase Percolation
-----------------------

When :math:`\phi_I` exceeds the percolation threshold
:math:`\phi_I^{perc} \approx 0.15`--0.30, a connected interphase
network forms spanning the entire sample.  This produces dramatic
stiffening and slowing of dynamics.  The percolation-enhanced modulus
is:

.. math::

   G_I^{eff}(\phi_I) = G_I \cdot \phi_I \cdot \left[1 + \kappa
   \left(\frac{\phi_I - \phi_I^{perc}}{\phi_I^{perc}}\right)^+\right]

where :math:`\kappa` is the percolation enhancement factor and
:math:`(\cdot)^+` denotes the Macaulay bracket.  Below percolation
(:math:`\phi_I < \phi_I^{perc}`), the interphase acts as isolated shells
around NPs.  Above percolation, the connected network adds geometric
stiffening.


Strain Amplification in the Interphase
---------------------------------------

The interphase chains experience amplified strain because the rigid NP
cores do not deform.  The effective velocity gradient experienced by the
interphase is:

.. math::

   \mathbf{L}^I = X_I(\phi)\,\mathbf{L}

where :math:`X_I = X(\phi_{eff})` uses the effective NP volume fraction
(including glassy layer).  This amplification appears in the affine terms
of the interphase evolution equation:

.. math::

   \dot{\boldsymbol{\mu}}^I = X_I\,(\mathbf{L}\boldsymbol{\mu}^I
   + \boldsymbol{\mu}^I\mathbf{L}^T)
   + k_{BER}^{int}\,(\boldsymbol{\mu}^I_{nat} - \boldsymbol{\mu}^I)

**Physical consequence:** Strain amplification means interphase chains
reach large stretch (and damage threshold) earlier than bulk matrix chains
at the same macroscopic strain.  This explains the early onset of
nonlinearity in filled systems.


Non-Affine Interphase Dynamics
-------------------------------

An alternative to strain amplification introduces a **monomer-particle
friction parameter** :math:`\xi_{NP}` that produces non-affine, partially
suppressed deformation:

.. math::

   \dot{\boldsymbol{\mu}}^I = (1 - \xi_{NP})\,
   (\mathbf{L}\boldsymbol{\mu}^I + \boldsymbol{\mu}^I\mathbf{L}^T)
   + k_{BER}^{int}\,(\boldsymbol{\mu}^I_{nat} - \boldsymbol{\mu}^I)

where :math:`\xi_{NP} \in [0, 1]`:

- :math:`\xi_{NP} = 0`: Full affine deformation (all strain transmitted
  to interphase)
- :math:`\xi_{NP} = 1`: Completely pinned layer (no deformation)

The two approaches (strain amplification and friction) are complementary:
:math:`X_I > 1` models the *geometric* concentration of strain;
:math:`\xi_{NP}` models the *dynamical* suppression.  They can be
combined as :math:`(1 - \xi_{NP}) X_I`, but this introduces a
degeneracy.  The recommended default is to use :math:`X_I` alone.


.. _hvnm-damage-mechanics:

Enhanced Damage Mechanics
==========================

Two Damage Variables
--------------------

The HVNM carries two independent scalar damage variables:

**Permanent network damage** :math:`D` (chain scission, identical to HVM):

.. math::

   \dot{D} = \Gamma_0\,\bigl(\lambda_{eff}^{perm} - \lambda_{crit}\bigr)^+
   \cdot (1 - D)

**Interfacial damage** :math:`D_{int}` (debonding/desorption/interfacial
bond rupture):

.. math::

   \dot{D}_{int} = \Gamma_0^{int}\,
   \bigl(\lambda_{chain}^{int} - \lambda_{crit}^{int}\bigr)^+
   \cdot (1 - D_{int})
   - h_{int}(T)\,(D_{int})^{n_h}

where the interfacial chain stretch is:

.. math::

   \lambda_{chain}^{int} = \sqrt{\frac{\text{tr}(\boldsymbol{\mu}^I)}{3}}

Key differences from permanent damage:

- **Lower critical stretch** :math:`\lambda_{crit}^{int} < \lambda_{crit}`:
  Confined interphase chains have less extensibility, and the interface
  concentrates stress due to modulus mismatch.

- **Self-healing term** :math:`-h_{int}(T)\,(D_{int})^{n_h}`:
  Above :math:`T_v^{int}`, interfacial BER can reform broken bonds,
  reducing :math:`D_{int}` over time.  The healing rate follows TST:

  .. math::

     h_{int}(T) = h_0 \exp\!\left(-\frac{E_a^{heal}}{k_B T}\right)

  The exponent :math:`n_h \in [0.5, 1]` controls healing kinetics
  shape (:math:`n_h = 1` for first-order, :math:`n_h = 0.5` for
  diffusion-limited).

This makes interfacial damage **reversible** above :math:`T_v^{int}`
but **irreversible** below it -- the hallmark of vitrimer
nanocomposites.


Weissenberg Number Fracture Criterion
--------------------------------------

The competition between loading rate and BER rate determines whether
the material flows or fractures, captured by dual Weissenberg numbers:

.. math::

   \text{Wi}^{mat} = \frac{\dot{\epsilon}}{k_{BER}^{mat}}, \qquad
   \text{Wi}^{int} = \frac{\dot{\epsilon}}{k_{BER}^{int}}

**Three fracture regimes:**

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Regime
     - Condition
     - Behavior
   * - I: Flow
     - Wi\ :sup:`mat` :math:`\ll 1`, Wi\ :sup:`int` :math:`\ll 1`
     - All networks relax faster than load accumulates. Viscous flow, no damage.
   * - II: Partial
     - Wi\ :sup:`mat` :math:`\ll 1`, Wi\ :sup:`int` :math:`\gtrsim 1`
     - Matrix flows but interphase cannot relax. :math:`D_{int}` grows. Ductile.
   * - III: Brittle
     - Wi\ :sup:`mat` :math:`\gg 1`, Wi\ :sup:`int` :math:`\gg 1`
     - Neither network relaxes. Both :math:`D` and :math:`D_{int}` grow. Thermoset-like.

**Crack tip analysis:** Near a propagating crack tip, the local strain
rate diverges as :math:`\dot{\epsilon} \sim v_{crack}/r`.  The fracture
process zone size is:

.. math::

   r_{fz}^{mat} = \frac{v_{crack}}{k_{BER}^{mat}}, \qquad
   r_{fz}^{int} = \frac{v_{crack}}{k_{BER}^{int}}

Since :math:`k_{BER}^{int} < k_{BER}^{mat}`, the interphase fracture
zone is *larger* (:math:`r_{fz}^{int} > r_{fz}^{mat}`), meaning
interfacial damage extends further ahead of the crack tip.  This
dissipates energy and is the micromechanical origin of **NP toughening**
in vitrimer nanocomposites.


Cooperative Shielding (4-Network)
----------------------------------

The HVM's cooperative shielding concept is extended to include the
interphase.  The effective permanent chain stretch is:

.. math::

   \lambda_{eff}^{perm} = \lambda_{chain}^{perm} \cdot
   \frac{(1-D)\,\tilde{G}_P}
   {(1-D)\,\tilde{G}_P + G_E(t) + G_D(t)
   + (1-D_{int})\,G_{I,eff}(t)}

where :math:`G_\alpha(t)` represents each network's current load share.

**NP toughening mechanism:** The interphase carries load that would
otherwise stress the permanent network, delaying :math:`D > 0`.  When
:math:`D_{int}` grows (interface fails), shielding decreases and load
transfers to the permanent network, causing delayed permanent
damage -- a **cascading failure** that produces gradual, ductile failure.


Payne Effect
-------------

The Payne effect (strain-amplitude-dependent modulus drop in filled
rubbers) emerges naturally from the HVNM:

1. **Small** :math:`\gamma_0`: Interphase intact (:math:`D_{int} \approx 0`),
   full modulus :math:`G' = G_P X + G_E + G_D + G_{I,eff} X_I`
2. **Increasing** :math:`\gamma_0`: Interphase natural state begins
   tracking deformation via BER, reducing :math:`\sigma_I`
3. **Large** :math:`\gamma_0`: :math:`\sigma_I \to 0` at steady state,
   :math:`G'` drops to unfilled level

The critical strain for Payne onset is:

.. math::

   \gamma_c \approx \frac{\lambda_{crit}^{int} - 1}{X_I(\phi)}

Higher :math:`\phi` lowers the onset strain through the :math:`X_I`
amplification factor.  See :ref:`hvnm-laos` for detailed LAOS analysis.


Mullins Effect
--------------

Under cyclic loading, the HVNM predicts stress softening through three
mechanisms:

- **Irreversible** (from :math:`D`): Permanent chain scission softens
  the elastic response permanently.
- **Partially reversible** (from :math:`D_{int}`): Interfacial damage
  reduces interphase stress, but self-healing above :math:`T_v^{int}`
  partially restores it between cycles.
- **Fully reversible** (from BER): Bond exchange in transient networks
  relaxes stress during each cycle.

The Mullins effect in vitrimers is **temperature-dependent**: above
:math:`T_v^{int}`, softening is partially recovered between cycles
(due to self-healing); below :math:`T_v^{int}`, it is permanent.
See :ref:`hvnm-cyclic` for the full cyclic loading analysis.


.. _hvnm-diffusion-mode:

Diffusion-Limited Slow Mode
=============================

Karim, Vernerey & Sain (Macromolecules, 2025) identified that the
long-term mechanical response of vitrimers requires a **constant-rate
kinetic term** representing diffusion-driven chain dynamics slower than
the BER timescale.  In a nanocomposite, this is physically motivated by:

- Reptation-like chain diffusion through the network
  (:math:`\tau_{diff} \gg \tau_{BER}`)
- Slow chain extraction from the interphase as chains desorb from NP
  surfaces
- Long-range topological reorganization requiring multiple sequential
  BER events

The HVNM adds a constant background rate :math:`k_{diff}` to both
kinetics:

.. math::

   k_{eff}^{mat} = k_{BER}^{mat}(T,\boldsymbol{\sigma}^E)
   + k_{diff}^{mat}(T)

.. math::

   k_{eff}^{int} = k_{BER}^{int}(T,\boldsymbol{\sigma}^I)
   + k_{diff}^{int}(T)

The diffusion rates follow simple Arrhenius:

.. math::

   k_{diff}^{mat} = k_{diff,0}^{mat}\exp\!\left(
   -\frac{E_a^{diff}}{k_B T}\right), \qquad
   k_{diff}^{int} = k_{diff,0}^{int}\exp\!\left(
   -\frac{E_a^{diff,int}}{k_B T}\right)

**Key properties:**

- The diffusion activation energy :math:`E_a^{diff}` is typically
  1.5--3:math:`\times` larger than :math:`E_a^{mat}`, reflecting the
  higher barrier for coordinated multi-bond rearrangement.
- The diffusion rate is **stress-independent** (not TST-activated)
  because it represents thermally driven Brownian motion.
- :math:`k_{diff}^{int} \ll k_{diff}^{mat}` because chain extraction
  from NP surfaces is entropically penalized.

**When to enable:** The ``include_diffusion=True`` flag should be used
when stress relaxation data shows a long-time tail beyond the BER
relaxation, or when creep data shows continuing slow deformation at
:math:`t \gg 1/k_{BER}`.


.. _hvnm-numerical:

Numerical Implementation
=========================

**ODE state vector:** 17 components in simple shear (18 with
interfacial damage):

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Index
     - Component
     - Description
   * - 0--2
     - :math:`\mu^E_{xx}, \mu^E_{xy}, \mu^E_{yy}`
     - Exchangeable distribution tensor
   * - 3--5
     - :math:`\mu^{E,nat}_{xx}, \mu^{E,nat}_{xy}, \mu^{E,nat}_{yy}`
     - Exchangeable natural state
   * - 6--8
     - :math:`\mu^D_{xx}, \mu^D_{xy}, \mu^D_{yy}`
     - Dissociative distribution tensor
   * - 9--11
     - :math:`\mu^I_{xx}, \mu^I_{xy}, \mu^I_{yy}`
     - Interphase distribution tensor
   * - 12--14
     - :math:`\mu^{I,nat}_{xx}, \mu^{I,nat}_{xy}, \mu^{I,nat}_{yy}`
     - Interphase natural state
   * - 15
     - :math:`\gamma`
     - Accumulated strain
   * - 16
     - :math:`D`
     - Permanent network damage
   * - 17
     - :math:`D_{int}`
     - Interfacial damage (if ``include_interfacial_damage=True``)

**ODE solver:** diffrax ``Tsit5`` (explicit 5th-order Runge-Kutta) with
``PIDController`` adaptive stepping (``rtol=1e-8``, ``atol=1e-10``).

.. note::

   Implicit solvers (e.g., Kvaerno5) were tested but produce
   ``TracerBoolConversionError`` due to lineax LU transpose checks
   during JAX tracing.  Tsit5 is the recommended solver.

**Stiffness at high** :math:`\phi`:
High :math:`\phi` amplifies the affine deformation (:math:`X_I \dot{\gamma}`),
creating stiff ODEs.  If the solver diverges, increase ``max_steps`` or
reduce the shear rate.

**Square-root guard:** The BER rate computation uses:

.. code-block:: python

   safe_stretch = jnp.sqrt(jnp.maximum(stretch_invariant, 0.0) + 1e-30)

This prevents infinite gradients at :math:`\sigma_{VM} = 0`.

**Initial conditions:** All tensors at identity
(:math:`\mu_{xx} = \mu_{yy} = 1`, :math:`\mu_{xy} = 0`),
:math:`\gamma = 0`, :math:`D = D_{int} = 0`.


.. _hvnm-phi-zero:

:math:`\phi = 0` Recovery Verification
========================================

When :math:`\phi = 0`, the HVNM must recover the HVM exactly.
Mathematically:

1. :math:`X(\phi=0) = 1` and :math:`X_I = X(\phi_{eff}=0) = 1`:
   No strain amplification.
2. :math:`\phi_I = 0 \cdot [(\ldots)^3 - 1] = 0`:
   Zero interphase volume fraction.
3. :math:`G_{I,eff} = \beta_I G_E \cdot 0 = 0`:
   No interphase modulus contribution.
4. The :math:`D_{int}` equation decouples (no interphase stress to drive
   damage).
5. The :math:`\boldsymbol{\mu}^I` equations decouple (zero prefactor in
   stress).

The remaining equations are identical to HVM with :math:`D = 0`,
:math:`X = 1`.  This is verified numerically to **machine precision**
(relative error :math:`< 10^{-14}`) across all six protocols in the
test suite.


References
===========

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1--20.

2. Vernerey, F.J. (2018). "Transient response of nonlinear polymer networks:
   A kinetic theory." *J. Mech. Phys. Solids*, 115, 230--247.

3. Vernerey, F.J., Brighenti, R., Long, R. & Shen, T. (2018). "Statistical
   Damage Mechanics of Polymer Networks." *Macromolecules*, 51(17), 6609--6622.

4. Meng, F., Saed, M.O. & Terentjev, E.M. (2019). "Elasticity and Relaxation
   in Full and Partial Vitrimer Networks." *Macromolecules*, 52(19), 7423--7429.

5. Shen, T., Song, Z., Cai, S. & Vernerey, F.J. (2021). "Nonsteady fracture
   of transient networks: The case of vitrimer." *PNAS*, 118(29), e2105974118.

6. Song, Z., Wang, Z. & Cai, S. (2021). "Mechanics of vitrimer with hybrid
   networks." *Mech. Mater.*, 153, 103687.

7. Papon, A., Montes, H., Lequeux, F. et al. (2012). "Glass-transition
   temperature gradient in nanocomposites: Evidence from nuclear magnetic
   resonance and differential scanning calorimetry." *Soft Matter*, 8(15),
   4090--4096.

8. Berriot, J., Montes, H., Lequeux, F. et al. (2002). "Filler-elastomer
   interaction in model filled rubbers, a :sup:`1`\ H NMR study."
   *J. Non-Crystalline Solids*, 307--310, 719--724.

9. Duan, P., Zhao, H., Chen, Q. et al. (2023). "Insights into Uniaxial
   Tension and Relaxation of Nanorod-Filled Polymer Vitrimer Nanocomposites:
   A Molecular Dynamics Simulation." *Macromolecules*, 56(11), 4468--4481.

10. Li, Z., Zhao, H., Duan, P., Zhang, L. & Liu, J. (2024). "Manipulating
    the Properties of Polymer Vitrimer Nanocomposites by Designing Dual Dynamic
    Covalent Bonds." *Langmuir*, 40(14), 7550--7560.

11. Duan, P., Zhao, H., Liu, M. et al. (2024). "Molecular Insights into the
    Topological Transition, Fracture, and Self-Healing Behavior of Vitrimer
    Composites with Exchangeable Interfaces." *Macromolecules*, 57, 7561--7573.

12. Kim, J., Thompson, B.R., Tominaga, T. et al. (2024). "Suppression of
    Segmental Chain Dynamics on a Particle's Surface in Well-Dispersed Polymer
    Nanocomposites." *ACS Macro Lett.*, 13(6), 720--725.

13. Karim, M.R., Vernerey, F. & Sain, T. (2025). "Constitutive Modeling of
    Vitrimers and Their Nanocomposites Based on Transient Network Theory."
    *Macromolecules*, 58(10), 4899--4912.

14. Li, Z., Zhao, H., Zhang, L., Liu, J. et al. (2025). "Designing
    All-Vitrimer Nanocomposites to Combine Low Energy Consumption, Mechanical
    Robust and Recyclability." *Nano Energy*, 142, 111199.

15. Alkhoury, K., Chester, S.A. & Vernerey, F.J. (2025). "Dynamic networks
    containing multiple bond types." *J. Mech. Phys. Solids*.

16. Wagner, R.J. & Silberstein, M.N. (2025). "A foundational framework for
    the mesoscale modeling of dynamic elastomers and gels." *J. Mech. Phys.
    Solids*, 194, 105914.

17. Hayashi, M. & Ricarte, R.G. (2025). "Towards the next development of
    vitrimers: Recent key topics for the practical application and understanding
    of the fundamental physics." *Prog. Polym. Sci.*, 170, 102026.

18. Zhao, H., Wei, X., Fang, Y. et al. (2022). "Molecular dynamics simulation
    of the structural, mechanical, and reprocessing properties of vitrimers
    based on a dynamic covalent polymer network." *Macromolecules*, 55,
    1091--1103.
