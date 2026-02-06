HVNM Knowledge Extraction Guide
================================

This guide explains how to extract physical knowledge from HVNM model
parameters and fitting results.


What Knowledge Can Be Extracted
-------------------------------

**Interphase Characterization:**

- :math:`\phi_I`: Interphase volume fraction (from multi-:math:`\phi` SAOS)
- :math:`\delta_m`: Mobile interphase thickness (from :math:`\phi_I` vs NP geometry)
- :math:`\beta_I`: Reinforcement ratio (surface chemistry / confinement strength)

**Dual Activation Energies:**

- :math:`E_a^{mat}`: Matrix activation energy (from multi-T relaxation)
- :math:`E_a^{int}`: Interfacial activation energy (from multi-T relaxation)
- :math:`\Delta E_a^{surf} = E_a^{int} - E_a^{mat}`: Surface confinement penalty

**Strain Amplification:**

- :math:`X(\phi)` from modulus vs :math:`\phi` calibration
- Deviation from Guth-Gold suggests non-spherical NPs or aggregation

**Two Topological Freezing Temperatures:**

- :math:`T_v^{mat}`: Matrix vitrimer freezing (BER arrest)
- :math:`T_v^{int} > T_v^{mat}`: Interfacial freezing (higher barrier)

**Payne Onset Strain:**

- :math:`\gamma_c^{NC} = \gamma_c / X_I`: Reduced critical strain from amplification


Parameter-to-Physics Map
-------------------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Parameters
     - Derived Quantity
     - Physical Meaning
   * - :math:`\phi, R_{NP}, \delta_m`
     - :math:`\phi_I`
     - Interphase fraction → NP dispersion quality
   * - :math:`\beta_I`
     - :math:`G_{I,eff}`
     - Surface chemistry / confinement strength
   * - :math:`E_a^{int} - E_a^{mat}`
     - :math:`\Delta E_a^{surf}`
     - Surface confinement penalty
   * - :math:`V_{act}^{int} / V_{act}^{mat}`
     - Ratio
     - Interfacial mechanochemical coupling
   * - :math:`G_P \cdot X(\phi)`
     - Effective plateau
     - Actual permanent modulus with amplification


Diagnostic Decision Tree
-------------------------

1. **Single relaxation mode in G''** → use HVM (no interphase needed)
2. **Two relaxation modes + phi dependence** → use HVNM
3. **Third slow mode in G''** → ``include_diffusion=True``
4. **Stress softening in cyclic tests** → ``include_interfacial_damage=True``
5. **Monotonic G'(phi) matching Guth-Gold** → standard HVNM
6. **G'(phi) deviates from Guth-Gold** → investigate NP aggregation


Multi-Protocol Fitting Strategy
--------------------------------

1. **SAOS first**: Identify :math:`G_P`, :math:`G_E`, :math:`G_{I,eff}`, mode timescales
2. **Multi-phi SAOS**: Extract :math:`\beta_I`, :math:`\delta_m` (interphase geometry)
3. **Relaxation**: Confirm 4-mode spectrum (:math:`\tau_D, \tau_E, \tau_I, \infty`)
4. **Multi-T relaxation**: Extract :math:`E_a^{mat}`, :math:`E_a^{int}` (dual Arrhenius)
5. **Startup**: Identify :math:`V_{act}^{mat}`, :math:`V_{act}^{int}` (TST coupling)
6. **LAOS amplitude sweep**: Confirm Payne onset at :math:`\gamma_c / X_I`


When to Use HVNM vs HVM
-------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Use HVM if
     - Use HVNM if
   * - Unfilled vitrimer
     - NP-filled vitrimer
   * - Single relaxation mode
     - Multi-timescale relaxation
   * - No :math:`\phi` dependence
     - Modulus depends on :math:`\phi`
   * - No Payne effect
     - Payne effect observed
   * - :math:`T_v^{mat}` sufficient
     - Two freezing temperatures needed
