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


Common Pitfalls
----------------

**Dual factor-of-2 confusion:**

Both the E-network and I-network exhibit the :ref:`factor-of-2 <hvnm-dual-factor-of-2>`:
:math:`\tau_{E,eff} = 1/(2k_{BER,0}^{mat})` and
:math:`\tau_{I,eff} = 1/(2k_{BER,0}^{int})`.  A naive Maxwell fit to SAOS data
will yield :math:`\tau_{fit} = \tau_{eff}`, not the true bond exchange time.
When converting to BER rates, multiply the fitted time constant by 2:
:math:`k_{BER,0} = 1/(2\tau_{fit})`.  See also the HVM derivation
(:ref:`hvm-factor-of-2`).

**Guth-Gold deviations:**

The Guth-Gold formula :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2` is accurate
for well-dispersed spherical NPs at low to moderate loading (:math:`\phi < 0.3`).
Deviations can indicate:

- **Higher-than-predicted modulus**: NP aggregation (effective larger particles)
- **Lower-than-predicted modulus**: poor NP-matrix adhesion
- **Phi-dependent exponent**: non-spherical NPs (rods, platelets)

Plot :math:`G'/G'_{unfilled}` vs :math:`\phi` and compare with Guth-Gold to
diagnose.

**Parameter identifiability with** :math:`\phi`:

From single-:math:`\phi` SAOS data alone, :math:`\beta_I` and :math:`\phi_I`
are correlated — only the product :math:`G_{I,eff} = \beta_I G_E \phi_I`
is identifiable.  Multi-phi SAOS data separates them because :math:`\phi_I`
varies with :math:`\phi` (via NP geometry) while :math:`\beta_I` does not.

**Frozen interphase at low** :math:`T`:

The interphase typically has higher activation energy than the matrix
(:math:`E_a^{int} > E_a^{mat}`), so it freezes at a higher temperature.  Below
:math:`T_v^{int}`, the I-network behaves as an elastic spring and its
contribution becomes indistinguishable from an enhanced :math:`G_P`.  Check
:math:`k_{BER,0}^{int}` at the experimental temperature before attributing
a high plateau to permanent crosslinks alone.


NP-Surface Characterization
-----------------------------

The HVNM interphase parameters encode NP-surface chemistry:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Physical Meaning
     - How to Determine
   * - :math:`\beta_I`
     - Surface binding strength
     - Multi-:math:`\phi` SAOS: fit :math:`G_{I,eff}` vs :math:`\phi_I`
   * - :math:`E_a^{int}`
     - Surface exchange barrier
     - Multi-T relaxation of slow mode
   * - :math:`\delta_m`
     - Interphase thickness
     - :math:`\phi_I = \phi[(R_{NP}+\delta_m)^3/R_{NP}^3 - 1]`; fit from multi-:math:`\phi`
   * - :math:`V_{act}^{int}`
     - Interfacial mechanochemistry
     - Stress overshoot in startup at high :math:`\phi`

**Interpreting** :math:`\Delta E_a^{surf} = E_a^{int} - E_a^{mat}`:

- :math:`\Delta E_a \approx 0`: interphase exchanges as fast as matrix
  (weak NP-polymer interaction)
- :math:`\Delta E_a \sim 20` - 50 kJ/mol: moderate surface confinement
  (typical for silica in epoxy vitrimers)
- :math:`\Delta E_a > 80` kJ/mol: strong confinement (chemically grafted NPs)


Payne Effect Interpretation
-----------------------------

The Payne effect — the decrease of :math:`G'` with increasing strain amplitude
in filled rubbers — is naturally captured by HVNM:

1. At small :math:`\gamma_0`: all networks respond linearly,
   :math:`G' = G_P X + G_E + G_D + G_{I,eff} X_I` (full modulus)
2. As :math:`\gamma_0` increases: the I-network natural state begins tracking
   the deformation via BER, reducing :math:`\sigma_I`
3. At large :math:`\gamma_0`: :math:`\sigma_I \to 0` at steady state,
   and :math:`G'` drops to the unfilled level

The **onset strain** is reduced by strain amplification:
:math:`\gamma_c^{NC} = \gamma_c / X_I`, where :math:`\gamma_c` is the onset
for the unfilled material.  Higher :math:`\phi` lowers the onset strain.


Worked Example: Identifying :math:`\phi_I` from Multi-:math:`\phi` SAOS
-------------------------------------------------------------------------

**Procedure:**

1. Prepare samples at :math:`\phi =` 0, 0.05, 0.10, 0.15, 0.20
2. Fit SAOS with HVNM at each :math:`\phi`, extracting :math:`G_{I,eff}(\phi)`
3. Compute theoretical :math:`\phi_I(\phi)` from NP geometry:
   :math:`\phi_I = \phi[(R_{NP}+\delta_m)^3/R_{NP}^3 - 1]`
4. Plot :math:`G_{I,eff}` vs :math:`\phi_I` — slope gives :math:`\beta_I G_E`
5. With :math:`G_E` known from the :math:`\phi=0` fit, extract :math:`\beta_I`
6. From the :math:`\phi_I(\phi)` relationship, extract :math:`\delta_m`

**Validation:** The unfilled (:math:`\phi=0`) fit should match HVM exactly,
and :math:`\beta_I` should be independent of :math:`\phi`.


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


.. _hvnm-cross-protocol-validation:

Troubleshooting
----------------

**Modulus doesn't match Guth-Gold scaling:**
Guth-Gold :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2` assumes well-dispersed
spherical NPs.  Deviations indicate NP aggregation (higher-than-predicted
modulus) or non-spherical particles (different amplification exponents).
Plot :math:`G'/G'_{\phi=0}` vs :math:`\phi` and compare with the quadratic.

**Interphase appears frozen at experimental temperature:**
The interphase activation energy :math:`E_a^{int}` is typically higher than
:math:`E_a^{mat}`.  If :math:`k_{BER,0}^{int} < 10^{-6}` s\ :sup:`-1`, the
interphase is effectively elastic on experimental timescales.  To model this,
use a high :math:`E_a^{int}` (up to 250 kJ/mol) rather than reducing
:math:`\nu_0^{int}` (which may violate parameter bounds).

**ODE solver diverges at high phi:**
High :math:`\phi` amplifies the affine deformation (:math:`X_I \dot{\gamma}`),
creating stiff ODEs.  Increase ``max_steps`` or reduce the shear rate.
See :ref:`hvnm-numerical` for solver details.

**phi=0 gives slightly different results from HVM:**
This should not happen — HVNM with :math:`\phi = 0` is verified to recover
HVM to machine precision.  If discrepancy occurs, check that :math:`\phi` is
exactly 0.0 (not a small nonzero value).  See :ref:`hvnm-phi-zero` for the
mathematical proof.

**Parameter identifiability with limited data:**
With single-:math:`\phi` SAOS data, :math:`\beta_I` and :math:`\phi_I` are
correlated (only their product :math:`G_{I,eff}` matters for SAOS).  Multi-phi
SAOS data is needed to separate these.  Similarly, :math:`\nu_0^{int}` and
:math:`E_a^{int}` require multi-temperature data for independent estimation.

**Interfacial damage makes results irreversible:**
If :math:`D_{int}` accumulates but you expect recovery, ensure self-healing
is properly configured.  Check that :math:`T > T_v^{int}` (healing is
Arrhenius-activated).  See :ref:`hvnm-damage-mechanics` for the healing model.

**Relaxation has unexplained slow tail:**
A long-time tail slower than any Maxwell mode may indicate diffusion-limited
exchange.  Try ``include_diffusion=True`` and fit :math:`k_{diff}`.
See :ref:`hvnm-diffusion-mode`.


Cross-Protocol Validation
--------------------------

Use multiple protocols to validate the HVNM fit:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Check
     - Criterion
     - Failing Suggests
   * - :math:`G_P X` from SAOS = :math:`G(\infty)` from relaxation
     - :math:`\lim_{\omega \to 0} G' \approx G(t \to \infty)`
     - Incorrect :math:`G_P` or :math:`\phi`
   * - :math:`\tau_{I,eff}` from SAOS = :math:`\tau_I` from relaxation
     - Loss peak frequency :math:`\approx 1/\tau_{I,eff}`
     - Interphase TST distorting linear regime
   * - :math:`G_P X` increases with :math:`\phi` as Guth-Gold
     - :math:`G'(\omega \to 0)` vs :math:`\phi` follows quadratic
     - NP aggregation or non-spherical particles
   * - :math:`\sigma_I \to 0` at steady state
     - I-network stress vanishes in long startup
     - :math:`k_{BER}^{int}` too slow at this :math:`T`

This is analogous to the HVM cross-protocol workflow
(:ref:`hvm-cross-protocol-validation`).


Application Examples
---------------------

**Multi-phi SAOS workflow:**
Prepare samples at :math:`\phi =` 0, 0.05, 0.10, 0.15, 0.20.  Fit SAOS at
each :math:`\phi` to extract :math:`G_{I,eff}(\phi)`.  Plot vs computed
:math:`\phi_I` to determine :math:`\beta_I` (slope) and validate the
Guth-Gold amplification.  The :math:`\phi = 0` fit should exactly match HVM.

**Temperature sweep for dual** :math:`E_a`:
Fit SAOS at 3+ temperatures.  The E-network loss peak shifts as
:math:`\tau_E(T) = 1/(2k_{BER,0}^{mat}(T))` — Arrhenius slope gives
:math:`E_a^{mat}`.  The I-network loss peak shifts independently —
its Arrhenius slope gives :math:`E_a^{int}`.  Expect
:math:`E_a^{int} > E_a^{mat}` for confined interphase.

**Cyclic loading analysis:**
Perform strain-amplitude sweeps at fixed :math:`\omega`.  The Payne onset
occurs at :math:`\gamma_c^{NC} = \gamma_c / X_I` — lower than the unfilled
material by the strain amplification factor.  If
``include_interfacial_damage=True``, the modulus does not fully recover on
unloading (Mullins effect).  Recovery timescale depends on :math:`T` through
the self-healing rate :math:`h_{int}(T)`.

**Interphase thickness from geometry:**
With known :math:`R_{NP}` (TEM) and :math:`\phi_I` (from multi-phi SAOS fit),
solve :math:`\phi_I = \phi[(R_{NP}+\delta_m)^3/R_{NP}^3 - 1]` for
:math:`\delta_m`.  Typical values: :math:`\delta_m \sim 2\text{-}20` nm for
polymer-NP interphases.  Compare with the Kuhn length for consistency.
