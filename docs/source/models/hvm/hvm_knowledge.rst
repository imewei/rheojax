HVM Knowledge Extraction Guide
===============================

This guide explains how to extract physical insights about vitrimer materials
from HVM model parameters and predictions.


What Knowledge Can Be Extracted
-------------------------------

**Structural parameters:**

- Crosslink densities from moduli: :math:`c_i = G_i / (k_B T)`
- Exchange fraction: :math:`f_E = G_E / (G_P + G_E)`
- Network architecture: permanent vs exchangeable vs physical

**Kinetic parameters:**

- Activation energy :math:`E_a` from multi-temperature fits (Arrhenius plot)
- TST attempt frequency :math:`\nu_0` from rate prefactor
- Mechanochemical coupling :math:`V_{act}` from nonlinear startup

**Material classification:**

- Thermoset (:math:`G_P \gg G_E`): dominated by permanent crosslinks
- Partial vitrimer (:math:`G_E \sim G_P`): mixed permanent + exchangeable
- Vitrimer liquid (:math:`G_P \approx 0`): fully exchangeable network
- Full HVM (:math:`G_D > 0`): additional physical crosslinks

**Predictive capabilities:**

- Temperature-rate superposition for processing windows
- Topology freezing transition temperature :math:`T_v`
- Stress relaxation vs permanent elastic memory


Parameter-to-Physics Map
------------------------

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - Parameter
     - Physical Meaning
     - How to Determine
   * - :math:`G_P`
     - Permanent crosslink density
     - Low-frequency SAOS plateau: :math:`G'(\omega \to 0) = G_P`
   * - :math:`G_E`
     - Exchangeable crosslink density
     - Difference: :math:`G'(\omega \to \infty) - G_P - G_D = G_E`
   * - :math:`G_D`
     - Physical bond density
     - Second loss peak position and height in :math:`G''(\omega)`
   * - :math:`E_a`
     - BER activation barrier
     - Arrhenius fit of :math:`k_{BER,0}` vs :math:`1/T` from multi-T relaxation
   * - :math:`V_{act}`
     - Mechanochemical coupling
     - Stress overshoot magnitude in startup shear
   * - :math:`\nu_0`
     - Bond exchange attempt rate
     - Arrhenius intercept (hard to determine independently)
   * - :math:`k_d^D`
     - Physical bond lifetime
     - Second relaxation time in bi-exponential fit
   * - :math:`\Gamma_0`
     - Damage sensitivity
     - Strain softening rate under large deformation
   * - :math:`\lambda_{crit}`
     - Damage onset threshold
     - Strain at which softening begins


Diagnostic Decision Tree
-------------------------

::

   Does SAOS show a low-frequency plateau?
   |
   +-- Yes: G_P > 0 (permanent crosslinks present)
   |   |
   |   +-- Single relaxation peak in G''?
   |   |   +-- Yes: Partial vitrimer (G_D = 0)
   |   |   +-- No (two peaks): Full HVM (G_D > 0)
   |   |
   |   +-- Is plateau modulus T-dependent?
   |       +-- No: Covalent permanent network
   |       +-- Yes: May have T-dependent damage
   |
   +-- No: G_P ~ 0 (vitrimer liquid or pure physical)
       |
       +-- Relaxation fully exponential?
       |   +-- Yes: Maxwell-like, use VLBLocal
       |   +-- No (stretched): TST kinetics active
       |
       +-- Does stress relax to zero?
           +-- Yes: No permanent network
           +-- No: Hidden G_P, re-fit with G_P > 0


Multi-Protocol Fitting Strategy
--------------------------------

The recommended fitting workflow exploits information content of each protocol:

1. **SAOS first** (linear regime, analytical):

   - Identify :math:`G_P` from low-:math:`\omega` plateau
   - Identify :math:`G_P + G_E + G_D` from high-:math:`\omega` plateau
   - Locate loss peaks for :math:`\tau_{E,eff}` and :math:`\tau_D`
   - Fix :math:`T` at experimental value

2. **Relaxation** (confirm time constants):

   - Verify bi-exponential + plateau structure
   - Confirm :math:`G(0^+) \approx G_P + G_E + G_D`
   - Confirm :math:`G(\infty) \approx G_P`

3. **Multi-temperature SAOS** (activation energy):

   - Fit :math:`k_{BER,0}(T)` at 3+ temperatures
   - Extract :math:`E_a` from Arrhenius plot slope: :math:`E_a = -R \cdot d(\ln k_{BER,0}) / d(1/T)`
   - Extract :math:`\nu_0` from Arrhenius intercept

4. **Startup** (TST parameters):

   - Fit :math:`V_{act}` from stress overshoot magnitude and position
   - High :math:`V_{act}` = strong mechanochemical coupling = prominent overshoot
   - Validate against SAOS parameters

5. **Creep** (long-time behavior):

   - Verify elastic jump: :math:`J(0^+) = 1/G_{tot}`
   - Check long-time compliance: :math:`J(\infty) \to 1/G_P` (with permanent network)
   - Identify vitrimer plastic creep at intermediate times


Common Pitfalls
---------------

**Factor-of-2 confusion:**

A standard Maxwell fit to E-network relaxation data yields
:math:`\tau_{fit} = 1/(2 k_{BER,0})`, not :math:`1/k_{BER,0}`. Always account
for this factor when converting fitted time constants to BER rates. Use
``model.get_vitrimer_relaxation_time()`` to get the correct :math:`\tau_{E,eff}`.

**Unbounded permanent stress:**

The P-network stress :math:`\sigma_P = G_P \gamma` grows without bound in
steady shear. This is physically correct (permanent crosslinks store elastic
energy) but means flow curve predictions diverge unless you examine the
viscous contribution :math:`\sigma_D` separately. Use ``return_components=True``
in flow curve predictions.

**Parameter identifiability:**

With single-protocol data at one temperature, several parameters may be
correlated:

- :math:`\nu_0` and :math:`E_a` are coupled: both affect :math:`k_{BER,0}`.
  Resolve with multi-T data.
- :math:`G_E` and :math:`\tau_{E,eff}` can trade off in SAOS. Fix one using
  relaxation data.
- :math:`V_{act}` is only identifiable from nonlinear data (startup, LAOS).

**Temperature vs vitrimer regime:**

At low T (``classify_vitrimer_regime() == "glassy"``), exchange is frozen and
the model behaves as a neo-Hookean + Maxwell solid. All vitrimer-specific
behavior vanishes. Use ``model.compute_ber_rate_at_equilibrium()`` to check
whether BER is active at your experimental temperature.


Vitrimer vs Conventional Transient Network
------------------------------------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - VLB / TNT (Conventional)
     - HVM (Vitrimer)
   * - Natural state
     - Fixed (:math:`\mathbf{I}`)
     - Evolving (:math:`\boldsymbol{\mu}^E_{nat}`)
   * - Steady-state stress
     - :math:`\sigma = \eta \dot{\gamma}`
     - :math:`\sigma_E = 0` (BER erases all E-stress)
   * - Permanent memory
     - None (fully relaxes)
     - :math:`G_P` plateau preserved
   * - Relaxation
     - Single exponential
     - Bi-exponential + plateau
   * - Bond exchange
     - Dissociative (network breaks)
     - Associative (topology changes, network intact)
   * - Temperature
     - :math:`k_d \sim T` (simple)
     - Arrhenius :math:`k_{BER} \sim e^{-E_a/RT}` (TST)
