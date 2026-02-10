.. _model-zener:

Zener (Standard Linear Solid)
=============================

Quick Reference
---------------

- **Use when:** Viscoelastic solid with finite equilibrium modulus, creep-recovery tests
- **Parameters:** 3 (Ge, Gm, :math:`\eta`)
- **Key equation:** :math:`G(t) = G_s + G_p \exp(-t/\tau)` where :math:`\tau = \eta_p/G_p`
- **Test modes:** Oscillation, relaxation, creep, flow curve
- **Material examples:** Cross-linked PDMS, vulcanized rubber, hydrogels, biological tissues

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`G_s`
     - Equilibrium modulus (Pa). Permanent stiffness at infinite time.
   * - :math:`G_p`
     - Maxwell arm modulus (Pa). Relaxation strength.
   * - :math:`\eta_p`
     - Maxwell arm viscosity (Pa·s). Controls relaxation timescale.
   * - :math:`\tau`
     - Relaxation time (s), :math:`\tau = \eta_p/G_p`.

Overview
--------

The :class:`rheojax.models.Zener` model—also known as the Standard Linear Solid—adds a
Maxwell arm in parallel with an equilibrium spring. It is the simplest element that can
capture both an instantaneous modulus and a finite equilibrium plateau, making it a
workhorse for creep-recovery tests, stress relaxation in solids, and instrument compliance
corrections.

Physical Foundations
--------------------

Mechanical Analogue
~~~~~~~~~~~~~~~~~~~

The Zener model consists of a **Maxwell element** (spring :math:`G_p` in series with dashpot :math:`\eta_p`) connected in **parallel** with an equilibrium spring :math:`G_s`:

.. code-block:: text

   ┌──────────────────────┐
   │    Spring Gs         │  ← Equilibrium (parallel)
   └──────────┬───────────┘
              │
   ┌──────────┴───────────┐
   │  ┌────┐    ┌──────┐  │
   │  │ Gp │────│ ηp   │  │  ← Maxwell arm (series)
   │  └────┘    └──────┘  │
   └──────────────────────┘

   Total stress: σ_total = σ_spring + σ_Maxwell
   Same strain: γ_spring = γ_Maxwell = γ

The parallel configuration means:

- **Stress is additive**: :math:`\sigma(t) = \sigma_s(t) + \sigma_{\text{Maxwell}}(t)`
- **Strain is identical**: Both branches experience the same deformation :math:`\gamma(t)`

Alternative nomenclature:
   - **Standard Linear Solid (SLS)**: Emphasizes 3-parameter linear model
   - **Zener model**: Named after Clarence Zener (anelasticity in metals)
   - **Voigt-Kelvin-Maxwell (VKM)**: Descriptive of element arrangement

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Zener model represents materials with **two distinct energy storage/dissipation mechanisms**:

**Equilibrium spring** :math:`G_s` **(parallel branch)**:
   - Permanent cross-links in chemical gels (covalent bonds)
   - Long-lived entanglements in high-MW polymers
   - Crystalline regions in semi-crystalline polymers
   - Physical bonds (hydrogen bonds, ionic cross-links)
   - **Stores energy indefinitely** → finite equilibrium modulus

**Maxwell arm** :math:`G_p + \eta_p` **(viscoelastic branch)**:
   - Temporary network junctions that relax
   - Chain reptation through transient entanglements
   - Viscous dissipation from molecular rearrangements
   - **Relaxes stress** over timescale :math:`\tau = \eta_p / G_p`

**Physical meaning**:
   At short times/high frequencies, both branches contribute (:math:`G_{\text{instant}} = G_s + G_p`). At long times/low frequencies, only the equilibrium spring remains (:math:`G_{\text{equilib}} = G_s`).

Material Examples with Typical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative Zener parameters
   :header-rows: 1
   :widths: 28 15 15 15 12 15

   * - Material
     - :math:`G_s` (Pa)
     - :math:`G_p` (Pa)
     - :math:`\eta_p` (Pa·s)
     - :math:`\tau` (s)
     - Ref
   * - Crosslinked PDMS
     - :math:`5 \times 10^4`
     - :math:`2 \times 10^5`
     - :math:`2 \times 10^4`
     - 0.1
     - [1]
   * - Vulcanized rubber (NR)
     - :math:`1 \times 10^6`
     - :math:`3 \times 10^6`
     - :math:`3 \times 10^7`
     - 10
     - [2]
   * - PVA hydrogel (5 wt%)
     - :math:`2 \times 10^3`
     - :math:`5 \times 10^3`
     - :math:`5 \times 10^2`
     - 0.1
     - [3]
   * - Biological tissue (skin)
     - :math:`1 \times 10^4`
     - :math:`5 \times 10^4`
     - :math:`1 \times 10^4`
     - 0.2
     - [4]
   * - Filled elastomer (CB 40 phr)
     - :math:`3 \times 10^5`
     - :math:`7 \times 10^5`
     - :math:`7 \times 10^6`
     - 10
     - [2]
   * - Epoxy (cured)
     - :math:`2 \times 10^9`
     - :math:`5 \times 10^8`
     - :math:`5 \times 10^1^0`
     - 100
     - [5]

**Note**: NR = natural rubber, CB = carbon black, phr = parts per hundred rubber

Connection to Polymer Network Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **chemically crosslinked elastomers**, the equilibrium modulus follows **affine network theory**:

.. math::

   G_s = G_e = \nu k_B T

where:
   - :math:`\nu` = number density of elastically effective network strands (mol/:math:`m^3`)
   - :math:`k_B` = Boltzmann constant
   - :math:`T` = absolute temperature
   - Equivalent: :math:`G_e = \rho RT / M_c` where :math:`M_c` = molecular weight between cross-links

For **physical gels** (transient networks):
   - :math:`G_s` depends on cross-link lifetime and thermal energy
   - Lower :math:`G_s` than chemical gels (weaker associations)
   - Example: Alginate gel :math:`G_s \sim 10^3` Pa vs epoxy :math:`G_s \sim 10^9` Pa

**Molecular weight between cross-links**:

.. math::

   M_c = \frac{\rho RT}{G_s}

Typical values:
   - Rubber: :math:`M_c \approx 5000-10000` g/mol
   - PDMS elastomer: :math:`M_c \approx 10000-20000` g/mol
   - Epoxy thermoset: :math:`M_c \approx 500-1000` g/mol (tightly cross-linked)

Governing Equations
-------------------

Mathematical Derivation
~~~~~~~~~~~~~~~~~~~~~~~

Starting from the mechanical analogue with **parallel connection**:

**Step 1**: Express parallel branch (equilibrium spring)
   :math:`\sigma_s = G_s \gamma`

**Step 2**: Express Maxwell branch (series spring + dashpot)
   For series elements: :math:`\gamma_{\text{Maxwell}} = \gamma_{G_p} + \gamma_{\eta_p}`

   Spring: :math:`\sigma_p = G_p \gamma_{G_p}` → :math:`\dot{\sigma}_p = G_p \dot{\gamma}_{G_p}`

   Dashpot: :math:`\sigma_p = \eta_p \dot{\gamma}_{\eta_p}`

   Total Maxwell strain rate: :math:`\dot{\gamma} = \frac{\dot{\sigma}_p}{G_p} + \frac{\sigma_p}{\eta_p}`

**Step 3**: Define relaxation times
   :math:`\tau_\epsilon = \eta_p / G_p` (relaxation time at constant strain)

   :math:`\tau_\sigma = \eta_p / (G_s + G_p)` (relaxation time at constant stress)

**Step 4**: Total stress (parallel)
   :math:`\sigma = \sigma_s + \sigma_p = G_s \gamma + \sigma_p`

**Step 5**: Differentiate and substitute
   :math:`\dot{\sigma} = G_s \dot{\gamma} + \dot{\sigma}_p`

   From Maxwell branch: :math:`\dot{\sigma}_p = G_p \left(\dot{\gamma} - \frac{\sigma_p}{\eta_p}\right)`

   Substitute :math:`\sigma_p = \sigma - G_s \gamma`:

   .. math::

      \dot{\sigma} = G_s \dot{\gamma} + G_p \dot{\gamma} - \frac{G_p}{\eta_p}(\sigma - G_s \gamma)

**Step 6**: Rearrange to constitutive form

Differential form:

.. math::

   \sigma(t) + \frac{\eta_p}{G_p} \dot{\sigma}(t)
   = G_s \gamma(t) + \left(G_s + G_p\right)\frac{\eta_p}{G_p} \dot{\gamma}(t)

Or equivalently:

.. math::

   \sigma + \tau_\epsilon \dot{\sigma} = G_s (\gamma + \tau_\sigma \dot{\gamma})

where :math:`\tau_\epsilon = \eta_p/G_p` and :math:`\tau_\sigma = \eta_p/(G_s + G_p)`.

Stress Relaxation Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For step strain :math:`\gamma_0` at :math:`t=0`, with :math:`\dot{\gamma}=0` for :math:`t>0`:

The ODE :math:`\sigma + \tau_\epsilon \dot{\sigma} = G_s \gamma_0` has solution:

.. math::

   G(t) = \frac{\sigma(t)}{\gamma_0} = G_s + G_p e^{-t/\tau_\epsilon}

**Interpretation**:
   - Initial modulus: :math:`G(0) = G_s + G_p` (instantaneous response)
   - Equilibrium modulus: :math:`G(\infty) = G_s` (long-time plateau)
   - Relaxation strength: :math:`\Delta G = G_p` (magnitude of decay)
   - Relaxation time: :math:`\tau_\epsilon = \eta_p / G_p`

Fourier Transform to Frequency Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For harmonic excitation :math:`\gamma(t) = \gamma_0 e^{i\omega t}`:

**Step 1**: Apply Fourier transform
   :math:`\dot{\sigma} = i\omega \sigma`, :math:`\dot{\gamma} = i\omega \gamma`

**Step 2**: Substitute into constitutive equation
   :math:`\sigma(1 + i\omega\tau_\epsilon) = G_s \gamma(1 + i\omega\tau_\sigma)`

**Step 3**: Define complex modulus
   .. math::

      G^*(\omega) = \frac{\sigma}{\gamma} = G_s \frac{1 + i\omega\tau_\sigma}{1 + i\omega\tau_\epsilon}

**Step 4**: Multiply by conjugate and separate

Complex modulus:

.. math::

   G^*(\omega) = G_s + \frac{i\omega \eta_p G_p}{G_p + i \omega \eta_p}.

Storage and loss moduli:

.. math::

   G'(\omega) = G_s + G_p \frac{(\omega \tau_\epsilon)^2}{1 + (\omega \tau_\epsilon)^2}

   G''(\omega) = G_p \frac{\omega \tau_\epsilon}{1 + (\omega \tau_\epsilon)^2}

**Alternative form** (more compact):

.. math::

   G'(\omega) = G_s + \frac{G_p (\omega \tau_\epsilon)^2}{1 + (\omega \tau_\epsilon)^2}, \qquad
   G''(\omega) = \frac{G_p \omega \tau_\epsilon}{1 + (\omega \tau_\epsilon)^2}

Mathematical Significance
~~~~~~~~~~~~~~~~~~~~~~~~~

**Rational function form**: The complex modulus is a **first-order rational function** (ratio of polynomials), characteristic of single-relaxation-time models.

**Loss tangent minimum**: Unlike Maxwell model (monotonic tan :math:`\delta`), Zener exhibits a **minimum** in loss tangent:

.. math::

   \tan\delta(\omega) = \frac{G''}{G'} = \frac{G_p \omega \tau_\epsilon}{G_s (1 + (\omega\tau_\epsilon)^2) + G_p (\omega\tau_\epsilon)^2}

At low :math:`\omega`: :math:`\tan\delta \sim \omega` (solid-like, tan :math:`\delta` → 0)

At high :math:`\omega`: :math:`\tan\delta \sim 1/\omega` (glassy, tan :math:`\delta` → 0)

At intermediate :math:`\omega`: **maximum dissipation** (tan :math:`\delta` peaks near :math:`\omega \approx 1/\tau_\epsilon`)

**Creep compliance**: Unlike Maxwell, Zener predicts **bounded creep**:

.. math::

   J(t) = \frac{1}{G_s + G_p} + \frac{G_p}{G_s(G_s + G_p)} \left(1 - e^{-t/\tau_\sigma}\right)

At :math:`t \to \infty`: :math:`J_e = 1/G_s` (finite equilibrium compliance, no flow).

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 24 20 56

   * - Name
     - Units
     - Description / Constraints
   * - ``Ge``
     - Pa
     - Equilibrium modulus; > 0 to retain solid plateau.
   * - ``Gm``
     - Pa
     - Maxwell spring modulus; > 0 controls relaxation magnitude.
   * - ``eta``
     - Pa·s
     - Maxwell dashpot viscosity; > 0 sets relaxation time :math:`\tau = \eta_p/G_p`.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Gs (Equilibrium Modulus)**:
   - **Physical meaning**: Long-time elastic response from permanent network structure
   - **Molecular origin**: Cross-link density in rubbers, crystallinity in semicrystalline polymers
   - **Typical ranges**:
      - Soft hydrogels: :math:`10^2 - 10^4` Pa
      - Rubbers/elastomers: :math:`10^5 - 10^7` Pa
      - Thermosets (epoxy): :math:`10^9 - 10^{10}` Pa
   - **Scaling**: :math:`G_s = \rho RT / M_c` (affine network theory)
   - **Diagnostic**: :math:`G_s = \lim_{\omega \to 0} G'(\omega)` (low-frequency plateau)

**Gp (Maxwell Arm Modulus)**:
   - **Physical meaning**: Relaxation strength (magnitude of stress decay)
   - **Molecular origin**: Transient entanglements or weak physical bonds that relax
   - **Typical ranges**: Similar to :math:`G_s` (often :math:`G_p \approx G_s` to :math:`5G_s`)
   - **Relation to initial modulus**: :math:`G_0 = G_s + G_p`
   - **Diagnostic**: :math:`G_p = G(0) - G(\infty)` from relaxation data

:math:`\etap` **(Maxwell Dashpot Viscosity)**:
   - **Physical meaning**: Controls timescale of stress relaxation
   - **Molecular origin**: Chain friction, reptation, or bond breakage/reformation
   - **Typical ranges**: :math:`10^2 - 10^{10}` Pa·s (highly material-dependent)
   - **Derived relaxation times**:
      - :math:`\tau_\epsilon = \eta_p / G_p` (relaxation at constant strain)
      - :math:`\tau_\sigma = \eta_p / (G_s + G_p)` (relaxation at constant stress)
   - **Diagnostic**: :math:`\tau_\epsilon^{-1} \approx \omega_{\max}` where :math:`G''(\omega)` peaks

**Important parameter relations**:

.. math::

   \tau_\sigma = \tau_\epsilon \cdot \frac{G_p}{G_s + G_p} < \tau_\epsilon

   \text{Relaxation strength: } \Delta G = G_p = G(0) - G(\infty)

Relation to Molecular Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **crosslinked networks**:

**Cross-link density**:

.. math::

   \nu_c = \frac{G_s}{RT} \quad (\text{mol/m}^3)

   M_c = \frac{\rho}{2\nu_c} = \frac{\rho RT}{2 G_s}

Factor of 2 from tetrafunctional cross-links (each cross-link connects 4 chains).

**Degree of cross-linking**:

.. math::

   \text{Cross-link density (mol\%)} = \frac{M_0}{M_c} \times 100\%

where :math:`M_0` = monomer molecular weight.

For **filled elastomers**:

Filler reinforcement increases both :math:`G_s` and :math:`G_p`:

.. math::

   G_s^{\text{filled}} = G_s^{\text{unfilled}} \cdot (1 + 2.5\phi + 14.1\phi^2)

where :math:`\phi` = filler volume fraction (Guth-Gold equation for rigid spheres).

Validity and Assumptions
------------------------

- Linear viscoelasticity: yes
- Small amplitude: yes (strain < critical strain :math:`\gamma_c`)
- Isothermal: yes
- Data/test modes: **relaxation, creep, oscillation** (all three work well)
- Additional assumptions: single relaxation time

Limitations
~~~~~~~~~~~

**Single relaxation time**:
   Real materials exhibit **continuous distributions** :math:`H(\tau)`. Zener adequate when:
   - One dominant relaxation process
   - Data span < 2-3 decades in time/frequency
   - Material is simple (monodisperse polymer, single cross-link type)

**No terminal flow**:
   :math:`G_e = G_s > 0` means material **never flows**. This fails for:
   - Uncrosslinked polymers (use Maxwell or Burgers instead)
   - Physical gels that eventually relax (use fractional models)

**Linear regime only**:
   Zener assumes :math:`\sigma \propto \gamma`. Fails for:
   - Large strains (yielding, strain-stiffening)
   - Nonlinear materials (LAOS required)

When to Upgrade
~~~~~~~~~~~~~~~

**From Zener to more complex models**:

- **Broad relaxation spectra** → Generalized Maxwell (Prony series) or Fractional Zener (FZSS)
- **Multiple relaxation processes** → Multi-mode Zener (parallel Zener elements)
- **Power-law relaxation** → Fractional Zener Solid-Solid (FZSS), see :doc:`../fractional/fractional_zener_ss`
- **Terminal flow** → Burgers (4-element), Fractional Maxwell Liquid
- **Nonlinear behavior** → Gent model, LAOS analysis

Regimes and Behavior
--------------------

Limiting Cases
~~~~~~~~~~~~~~

**Low frequency (** :math:`\omega` **→ 0, terminal region)**:

.. math::

   G'(\omega) \to G_s \quad (\text{equilibrium plateau})

   G''(\omega) \approx \frac{G_p \eta_p}{G_s^2} \omega \sim \omega

**Interpretation**: Elastic solid-like (G' > G''), but with residual dissipation from Maxwell arm.

**High frequency (** :math:`\omega` **→ ∞, glassy region)**:

.. math::

   G'(\omega) \to G_s + G_p = G_0 \quad (\text{instantaneous modulus})

   G''(\omega) \to 0

**Interpretation**: Fully elastic (Maxwell arm frozen, acts as spring).

**Crossover frequency** :math:`\omega_c \approx 1/\tau_\epsilon`:

.. math::

   G''(\omega_c) \approx \frac{G_p}{2} \quad (\text{peak in loss modulus})

This defines the characteristic relaxation timescale.

Asymptotic Behavior Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Frequency-dependent regimes
   :header-rows: 1
   :widths: 22 26 26 26

   * - Regime
     - :math:`G'(\omega)`
     - :math:`G''(\omega)`
     - Physical interpretation
   * - Low :math:`\omega \ll 1/\tau_\epsilon`
     - :math:`\to G_s`
     - :math:`\sim \omega`
     - Elastic solid (G' > G'')
   * - :math:`\omega \approx 1/\tau_\epsilon`
     - :math:`\approx G_s + G_p/2`
     - :math:`\approx G_p/2` (peak)
     - Maximum dissipation
   * - High :math:`\omega \gg 1/\tau_\epsilon`
     - :math:`\to G_s + G_p`
     - :math:`\to 0`
     - Glassy solid (fully elastic)

Special Cases (Parameter Limits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case 1**: :math:`G_s \to 0` (remove equilibrium spring)
   Zener → **Maxwell model** (viscoelastic liquid)

   :math:`G(t) = G_p e^{-t\cdot G_p/\eta_p}` (exponential decay to zero)

**Case 2**: :math:`\eta_p \to \infty` (dashpot becomes rigid)
   Zener → **Two springs in parallel**

   :math:`G(t) = G_s + G_p` (purely elastic, no relaxation)

**Case 3**: :math:`G_p \to 0` (remove Maxwell spring)
   Zener → **Kelvin-Voigt model** (spring + dashpot in parallel)

   Good for creep but poor for relaxation (doesn't capture instantaneous response)

**Case 4**: :math:`G_s = G_p` (symmetric model)
   Common simplification: :math:`G(t) = G_s (1 + e^{-t/\tau})`, :math:`\tau = \eta_p/G_s`

Diagnostic Signatures
~~~~~~~~~~~~~~~~~~~~~

**1. Loss tangent minimum**:
   Unlike Maxwell (monotonic decrease), Zener exhibits tan :math:`\delta` **peak** then **decrease** at high :math:`\omega`.

**2. Two plateaus in** :math:`G'(\omega)`:
   - Low-frequency: :math:`G' \approx G_s`
   - High-frequency: :math:`G' \approx G_s + G_p`

**3. Creep recovery**:
   After removing stress, material **partially recovers**:

   .. math::

      \gamma_{\text{recovered}} = \gamma_{\text{total}} \cdot \frac{G_p}{G_s + G_p}

   Fraction recovered = :math:`G_p / (G_s + G_p)`

What You Can Learn
------------------

This section explains how to extract material insights and process guidance from fitted Zener parameters.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Gs (Equilibrium Modulus)**:
   Fitted :math:`G_s` reveals the permanent network structure:

   - **Low values (<** :math:`10^3` **Pa)**: Weak hydrogels, low cross-link density, soft tissues
   - **Moderate values (** :math:`10^4-10^6` **Pa)**: Elastomers, biological tissues, filled rubbers
   - **High values (>** :math:`10^7` **Pa)**: Thermosets, highly crosslinked networks, rigid materials

   *For researchers*: Calculate cross-link density from :math:`G_s`:

   .. math::

      \nu_c = \frac{G_s}{RT} \quad (\text{mol/m}^3), \qquad M_c = \frac{\rho RT}{2G_s} \quad (\text{g/mol})

   *For practitioners*: :math:`G_s` indicates dimensional stability under sustained load. Higher :math:`G_s` means less creep deformation.

**Gp (Maxwell Arm Modulus)**:
   Fitted :math:`G_p` quantifies transient network contribution:

   - **Low** :math:`G_p/G_s` **ratio (<0.5)**: Strongly crosslinked, minimal relaxation
   - **Moderate** :math:`G_p/G_s` **(0.5-2)**: Balanced network (permanent + transient)
   - **High** :math:`G_p/G_s` **(>2)**: Weak permanent structure, strong transient bonds

   *For researchers*: :math:`G_p/(G_s + G_p)` = fraction of strain recovered in creep-recovery test

   *For practitioners*: High :math:`G_p` means strong viscous dissipation (damping), useful for vibration isolation

:math:`\etap` **(Maxwell Dashpot Viscosity)**:
   Fitted :math:`\eta_p` controls stress relaxation timescale:

   - **Short** :math:`\tau_\epsilon = \eta_p/G_p` **(<0.1 s)**: Fast relaxation, minimal processing history
   - **Moderate** :math:`\tau_\epsilon` **(0.1-100 s)**: Typical elastomers, biological timescales
   - **Long** :math:`\tau_\epsilon` **(>100 s)**: Slow relaxation, shape memory effects

   *For researchers*: :math:`\eta_p` relates to chain friction and entanglement dynamics

   *For practitioners*: Processing Deborah number :math:`De = \tau_\epsilon \cdot \dot{\gamma}_{process}` determines if elastic effects are significant

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Zener Parameters
   :header-rows: 1
   :widths: 22 22 28 28

   * - Parameter Pattern
     - Material Type
     - Examples
     - Key Property
   * - High :math:`G_s`, low :math:`G_p`
     - Strongly crosslinked
     - Thermosets, vulcanized rubber
     - Minimal creep, high recovery
   * - Moderate :math:`G_s`, high :math:`G_p`
     - Physical gels, tissues
     - Gelatin, collagen, skin
     - Significant relaxation
   * - Low :math:`G_s`, high :math:`G_p`
     - Weak gels
     - Alginate, low cross-link
     - Poor dimensional stability
   * - :math:`G_s \approx G_p`
     - Balanced viscoelastic
     - Filled elastomers
     - Equal elastic/viscous contribution

Network Structure Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cross-link density from** :math:`G_s`:

For elastomers at temperature :math:`T`:

.. math::

   M_c = \frac{\rho RT}{2 G_s}

Example: PDMS with :math:`G_s = 5 \times 10^4` Pa at 298 K, :math:`\rho = 965` kg/:math:`m^3`:

.. math::

   M_c = \frac{965 \times 8.314 \times 298}{2 \times 5 \times 10^4} \approx 24000 \text{ g/mol}

**Creep recovery fraction:**

.. math::

   \text{Recovery} = \frac{G_p}{G_s + G_p}

Example: :math:`G_s = 5 \times 10^4` Pa, :math:`G_p = 2 \times 10^5` Pa:

.. math::

   \text{Recovery} = \frac{2 \times 10^5}{5 \times 10^4 + 2 \times 10^5} = 0.80 = 80\%

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

**Warning signs in fitted parameters:**

- **If** :math:`G_s \to 0`: Material is liquid, not solid → use Maxwell or Burgers model
- **If** :math:`G_p \gg G_s` **(>10×)**: Weak network, consider fractional models (FZSS)
- **If** :math:`\tau_\epsilon` **outside data range**: Expand frequency sweep or use time-temperature superposition
- **If** :math:`R^2 < 0.95`: Single relaxation time inadequate → use Generalized Maxwell or fractional models
- **If** :math:`\eta_p > 10^{12}` **Pa·s**: Unrealistic viscosity, check bounds and initialization

**Material quality checks:**

- **Cross-link uniformity**: Low :math:`G_p/G_s` ratio indicates uniform crosslinking
- **Cure completion**: :math:`G_s` should not change with waiting time if fully cured
- **Filler dispersion**: High :math:`G_s` and :math:`G_p` indicate good filler-polymer interaction

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Quality Control (Elastomers):**
   - Track :math:`G_s` batch-to-batch to verify crosslink density consistency
   - Monitor :math:`\tau_\epsilon` for cure optimization (shorter → faster production)
   - Use :math:`G_p/(G_s + G_p)` to classify grades (soft/medium/hard)

**Material Design (Hydrogels):**
   - Target :math:`G_s = 10^3-10^4` Pa for soft tissue mimics
   - Adjust crosslinker ratio to achieve desired :math:`M_c`
   - Use :math:`\tau_\epsilon` to match physiological timescales (0.1-10 s)

**Process Optimization:**
   - Calculate :math:`De = \tau_\epsilon \cdot \dot{\gamma}` for molding/extrusion
   - If :math:`De > 1`: Reduce processing rate or increase temperature
   - If :math:`De < 0.1`: Can increase throughput without elastic defects

**Failure Prediction:**
   - Low :math:`G_s` indicates weak network → creep failure risk
   - High :math:`\tau_\epsilon` indicates stress accumulation → relaxation cracking risk
   - Recovery fraction < 50% suggests permanent deformation under load

Experimental Design
-------------------

Recommended Test Modes
~~~~~~~~~~~~~~~~~~~~~~

**1. Small Amplitude Oscillatory Shear (SAOS) - Frequency Sweep**

**Why optimal for Zener**:
   - Captures both low-frequency plateau (:math:`G_s`) and high-frequency plateau (:math:`G_s + G_p`)
   - :math:`G''(\omega)` peak location gives :math:`\tau_\epsilon` directly
   - Simultaneous fit of 3 parameters from single experiment

**Protocol**:
   - Amplitude sweep first: Determine LVR (typically :math:`\gamma_0 = 0.5-5\%` for elastomers)
   - Frequency range: **Must span** :math:`1/\tau_\epsilon` → recommend 3-4 decades
   - Example: :math:`10^{-2}` to :math:`10^2` rad/s for :math:`\tau \approx 1` s
   - Points per decade: 8-12 (logarithmic spacing)
   - Temperature control: ±0.1°C

**Expected features**:
   - :math:`G'` levels off at low :math:`\omega` (plateau = :math:`G_s`)
   - :math:`G''` shows **clear peak** at :math:`\omega \approx 1/\tau_\epsilon`
   - :math:`G' > G''` across all frequencies (solid-like, tan :math:`\delta` < 1)

**2. Stress Relaxation Test**

**Why optimal for Zener**:
   - Direct measurement of :math:`G(t) = G_s + G_p e^{-t/\tau}`
   - Easy parameter extraction: plateau = :math:`G_s`, initial = :math:`G_s + G_p`
   - Validates exponential relaxation assumption

**Protocol**:
   - Step strain: :math:`\gamma_0 = 1-5\%` (within LVR)
   - Rise time: < :math:`0.05\tau_\epsilon` (instrument limitation)
   - Duration: :math:`5-10\tau_\epsilon` to observe plateau
   - **Critical**: Measure long enough to reach :math:`G(\infty) = G_s` (many users stop too early!)

**Data analysis**:
   - Plot :math:`G(t)` vs :math:`t` → exponential decay to plateau
   - Non-linear regression: fit :math:`G(t) = G_s + G_p e^{-t/\tau}`
   - Extract all 3 parameters directly

**3. Creep-Recovery Test**

**Why useful for Zener**:
   - **Gold standard** for determining equilibrium compliance :math:`J_e = 1/G_s`
   - Recovery phase quantifies :math:`G_p / (G_s + G_p)` ratio
   - Excellent for viscoelastic solids

**Protocol**:
   - **Creep phase**: Apply constant stress :math:`\sigma_0` within LVR
      - Duration: :math:`5-10\tau_\sigma` until :math:`J(t)` plateaus
      - :math:`J_e = 1/G_s` (equilibrium compliance)

   - **Recovery phase**: Remove stress, measure strain recovery
      - Duration: :math:`5-10\tau_\sigma` (same timescale as creep)
      - Recovered strain / total strain = :math:`G_p / (G_s + G_p)`

**Advantages**:
   - Very sensitive to network structure (cross-links)
   - Distinguishes elastic vs viscous contributions
   - Common in polymer engineering (ASTM D2990)

Sample Preparation Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Crosslinked elastomers**:
   - Cure completely (post-cure if needed) to avoid time-dependent cross-linking during test
   - Compression molding: :math:`T > T_g + 50°C`, 5-10 MPa pressure
   - Punch disks: 8 mm (strain-controlled rheometer) or 25 mm (stress-controlled)
   - Check for air bubbles (reduce void content < 1%)

**Hydrogels**:
   - Maintain hydration: Use humidity chamber or immersion cell
   - Equilibrate in buffer/water for 24 h before testing
   - Avoid evaporation (can increase modulus by 10× in 1 hour!)
   - Typical geometry: Parallel plates with roughened surface (prevent slip)

**Biological tissues**:
   - Keep hydrated in physiological saline
   - Test within 2-4 hours post-mortem (degradation)
   - Temperature: 37°C ± 0.5°C (physiological)
   - Low strain amplitudes: :math:`\gamma_0 = 0.1-1\%` (tissues are fragile)

**Thermosets (epoxy, polyurethane)**:
   - Cure cycle must be complete (DSC to check residual cure)
   - Machine samples to required geometry (precision ±0.01 mm)
   - High modulus → torsional rheometer or DMA (not rotational)

Common Experimental Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Troubleshooting experimental issues
   :header-rows: 1
   :widths: 28 36 36

   * - Artifact
     - Symptom
     - Solution
   * - Wall slip
     - Anomalously low moduli, non-reproducible
     - Sandblasted/serrated plates, adhesive bonding, reduce gap
   * - Sample swelling/drying
     - :math:`G_s` drifts during test
     - Humidity control, immersion cell, shorter test duration
   * - Incomplete curing
     - :math:`G_s` increases over time
     - Post-cure, verify with DSC, discard if curing during test
   * - Stress overshoot in relaxation
     - :math:`G(t)` > :math:`G_s + G_p` at :math:`t=0`
     - Inertia artifact, ignore first 1-2 data points
   * - No :math:`G''` peak visible
     - :math:`\tau` outside frequency window
     - Expand frequency range, use TTS, or estimate from :math:`G(t)`
   * - Creep doesn't plateau
     - Uncrosslinked material (flows)
     - Check cure, use Maxwell/Burgers model instead

Fitting Guidance
----------------

Parameter Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From oscillatory frequency sweep**

**Step 1**: Estimate :math:`G_s` from low-frequency plateau
   :math:`G_s \approx \lim_{\omega \to 0} G'(\omega)` (use lowest 3-5 data points, average)

**Step 2**: Estimate :math:`G_s + G_p` from high-frequency plateau
   :math:`G_0 = G_s + G_p \approx \lim_{\omega \to \infty} G'(\omega)`

   Then :math:`G_p = G_0 - G_s`

**Step 3**: Estimate :math:`\tau_\epsilon` from :math:`G''(\omega)` peak
   :math:`\tau_\epsilon \approx 1 / \omega_{\max}` where :math:`G''` is maximum

**Step 4**: Calculate :math:`\eta_p`
   :math:`\eta_p = G_p \tau_\epsilon`

**Method 2: From stress relaxation data**

**Step 1**: Extract plateau modulus
   :math:`G_s = G(\infty)` (average last 20% of data after equilibration)

**Step 2**: Extract initial modulus
   :math:`G_0 = G(0)` → :math:`G_p = G_0 - G_s`

**Step 3**: Exponential fit to relaxation curve
   Fit :math:`G(t) = G_s + G_p e^{-t/\tau}` → extract :math:`\tau_\epsilon`

**Step 4**: Calculate :math:`\eta_p`
   :math:`\eta_p = G_p \tau_\epsilon`

**Method 3: From creep-recovery data**

**Step 1**: Equilibrium compliance
   :math:`J_e = 1/G_s` from creep plateau

**Step 2**: Instantaneous compliance
   :math:`J_0 = 1/(G_s + G_p)` from initial creep response

**Step 3**: Retardation time from recovery
   Fit recovery curve to :math:`\gamma(t) = \gamma_{\text{rec}}(1 - e^{-t/\tau_\sigma})`

**Step 4**: Calculate parameters
   :math:`G_s = 1/J_e`, :math:`G_p = 1/J_0 - G_s`, :math:`\eta_p = (G_s + G_p) \tau_\sigma`

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for Zener (3 parameters, well-conditioned if initialized correctly)
   - **Critical**: Good initialization required (especially :math:`G_s` vs :math:`G_p` separation)
   - 5-270× faster than scipy.optimize

**Alternative: Bayesian inference (NUTS)**
   - **Highly recommended** for Zener to quantify parameter uncertainty
   - :math:`G_s` and :math:`G_p` can be **correlated** → credible intervals important
   - Warm-start from NLSQ for faster convergence
   - See :doc:`../../examples/bayesian/01-bayesian-basics`

**Bounds** (adjust based on material class):
   - :math:`G_s`: [1e2, 1e10] Pa
   - :math:`G_p`: [1e2, 1e10] Pa
   - :math:`\eta_p`: [1e0, 1e12] Pa·s
   - **Constraint**: Ensure :math:`G_s > 0` (otherwise not a solid)

Troubleshooting Common Fitting Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics and solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Diagnostic
     - Solution
   * - :math:`G_s` fits to zero or negative
     - Material is liquid, not solid
     - Use Maxwell or Burgers model instead (no equilibrium modulus)
   * - :math:`G_p` much larger than :math:`G_s`
     - Weak network (mostly transient)
     - Check cross-linking, may need fractional model (FZSS)
   * - :math:`G''` peak not fit well
     - :math:`\tau` poorly initialized
     - Reinitialize :math:`\tau = 1/\omega_{\max}` from data peak
   * - Fitted :math:`\eta_p` unrealistic (> 10^{15})
     - :math:`G_p` too small or :math:`\tau` too large
     - Check frequency range covers relaxation, verify data quality
   * - Good fit but :math:`R^2 < 0.95`
     - Single relaxation inadequate
     - Try multi-mode Zener or Fractional Zener (FZSS)
   * - :math:`G_s` and :math:`G_p` swap values
     - Poor initialization or local minimum
     - Initialize with :math:`G_s < G_p`, add inequality constraint

**Special challenge: Gs vs Gp identifiability**

When :math:`\tau` is outside the experimental window (too fast or too slow):
   - **Too fast** (:math:`\omega \ll 1/\tau`): Only see :math:`G' \approx G_s` → :math:`G_p` unidentifiable
   - **Too slow** (:math:`\omega \gg 1/\tau`): Only see :math:`G' \approx G_s + G_p` → cannot separate :math:`G_s`, :math:`G_p`

**Solution**: Expand frequency range via time-temperature superposition (TTS) or use complementary relaxation data.

Validation Strategies
~~~~~~~~~~~~~~~~~~~~~

**1. Residual Analysis**

**Visual check**:
   - Plot residuals for :math:`G'` and :math:`G''` separately
   - Should be **random**, no systematic curvature
   - Check near :math:`G''` peak (most sensitive region)

**Statistical metrics**:
   - :math:`R^2 > 0.98` for :math:`G'`, :math:`R^2 > 0.95` for :math:`G''` (G'' is noisier)
   - RMSE in log-space < 0.1

**2. Physical Plausibility Checks**

.. math::

   \text{Check 1: } \tau_\epsilon = \eta_p / G_p \quad (\text{should match } 1/\omega_{\max})

   \text{Check 2: } G_s > 0 \quad (\text{viscoelastic solid})

   \text{Check 3: } G_p > 0 \quad (\text{relaxation strength positive})

   \text{Check 4: } \tau_\sigma = \frac{\eta_p}{G_s + G_p} < \tau_\epsilon \quad (\text{always true})

**3. Cross-validation Between Test Modes**

.. math::

   G_s^{\text{SAOS}} \stackrel{?}{=} G(\infty)^{\text{relaxation}} \stackrel{?}{=} (J_e^{\text{creep}})^{-1}

Discrepancies > 20% indicate:
   - Nonlinearity (strain amplitude too high)
   - Time-dependent structure (cross-linking, aging)
   - Multi-mode relaxation (need more parameters)

**4. Kramers-Kronig Validation**

For Zener model, :math:`G'` and :math:`G''` **automatically** satisfy Kramers-Kronig (causal model). Use to validate experimental data before fitting.

Worked Example with Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Material**: Crosslinked PDMS elastomer

**Experimental data** (SAOS):
   - Low-frequency plateau: :math:`G'(0.01 \text{ rad/s}) \approx 5.0 \times 10^4` Pa
   - High-frequency plateau: :math:`G'(100 \text{ rad/s}) \approx 2.5 \times 10^5` Pa
   - :math:`G''` peak at :math:`\omega = 10` rad/s, :math:`G''_{\max} \approx 1.0 \times 10^5` Pa

**Initialization**:
   - :math:`G_s = 5.0 \times 10^4` Pa (low-:math:`\omega` plateau)
   - :math:`G_s + G_p = 2.5 \times 10^5` Pa → :math:`G_p = 2.0 \times 10^5` Pa
   - :math:`\tau_\epsilon = 1 / 10 = 0.1` s
   - :math:`\eta_p = G_p \tau_\epsilon = 2.0 \times 10^5 \times 0.1 = 2.0 \times 10^4` Pa·s

**Optimization** (NLSQ, 150 iterations):
   - Fitted: :math:`G_s = 4.8 \times 10^4` Pa, :math:`G_p = 2.1 \times 10^5` Pa, :math:`\eta_p = 2.05 \times 10^4` Pa·s
   - :math:`R^2 = 0.996` (excellent)
   - Validation: :math:`\tau_\epsilon = 2.05 \times 10^4 / 2.1 \times 10^5 = 0.098` s ≈ 0.1 s ✓

**Physical interpretation**:
   - Cross-link density: :math:`\nu_c = G_s / RT = 4.8 \times 10^4 / (8.314 \times 298) = 19.4` mol/:math:`m^3`
   - Molecular weight between cross-links: :math:`M_c = \rho / (2\nu_c) \approx 25000` g/mol (typical for PDMS)
   - Relaxation time: 0.1 s suggests moderate chain dynamics

Model Comparison
----------------

When to Use Zener vs Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model selection decision tree
   :header-rows: 1
   :widths: 26 35 39

   * - Use Zener when...
     - Use alternative when...
     - Recommended model
   * - Viscoelastic **solid** (:math:`G_e > 0`)
     - Viscoelastic **liquid** (:math:`G_e = 0`)
     - Maxwell, Burgers
   * - **Single** relaxation time
     - **Broad** relaxation spectrum
     - FZSS, Generalized Maxwell
   * - **Creep recovery** observed
     - **No recovery** (pure flow)
     - Maxwell (liquid)
   * - **Exponential** :math:`G(t)` decay to plateau
     - **Power-law** relaxation
     - FZSS, FMG
   * - 3 parameters sufficient
     - Need creep + relaxation accuracy
     - Burgers (4 params)
   * - Simple network structure
     - Complex filled/composite
     - Fractional models

Model Hierarchy (Simpler → More Complex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Level 1: Maxwell**
   - 2 parameters
   - Viscoelastic **liquid** (no equilibrium modulus)
   - See :doc:`maxwell`

**Level 2: Zener (this model)**
   - 3 parameters
   - Viscoelastic **solid** (finite :math:`G_e`)
   - Simplest model capturing both relaxation and creep

**Level 3: Burgers (4-element)**
   - 4 parameters
   - Combines Maxwell + Kelvin-Voigt
   - Better creep modeling than Zener
   - See :doc:`../advanced/burgers` (if available)

**Level 4: Generalized Maxwell**
   - :math:`2N+1` parameters (N modes + equilibrium)
   - Discrete relaxation spectrum
   - High accuracy but many parameters

**Level 5: Fractional Zener Solid-Solid (FZSS)**
   - 4-5 parameters
   - Power-law relaxation via fractional elements
   - Captures broad spectra with fewer parameters than Generalized Maxwell
   - See :doc:`../fractional/fractional_zener_ss`

Diagnostic Tests to Discriminate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test 1: Creep-recovery**
   - **Full recovery** → Purely elastic (Hookean spring, not Zener)
   - **Partial recovery** → Zener (or FZSS)
   - **No recovery** → Maxwell (liquid)

**Test 2: Plot log** :math:`G(t) - G_s` **vs** :math:`t`
   - **Linear** → Zener (exponential relaxation)
   - **Curved** (concave down) → Multi-mode or fractional

**Test 3: Loss tangent vs frequency**
   - **Peak then decrease** → Zener
   - **Monotonic decrease** → Maxwell
   - **Flat (constant)** → Critical gel (FMG)

**Test 4: Low-frequency** :math:`G'` **behavior**
   - **Plateau** (:math:`G' \to G_s`) → Zener
   - **Decreases** (:math:`G' \sim \omega^2`) → Maxwell
   - **Power-law** (:math:`G' \sim \omega^\alpha`, :math:`0 < \alpha < 1`) → Fractional

Connection to Advanced Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Thermorheological complexity**:
   For Zener, time-temperature superposition (TTS) requires **single shift factor** :math:`a_T` if material is thermorheologically simple. If :math:`G_s` and :math:`\eta_p` have different temperature dependences, material is **thermorheologically complex** → need fractional models or multi-mode.

**Relationship to fractional models**:
   Fractional Zener (FZSS) generalizes Zener by replacing springs with SpringPots:

   .. math::

      \text{Zener: } G(t) = G_s + G_p e^{-t/\tau}

      \text{FZSS: } G(t) = G_s E_{\alpha_s}(-(t/\tau)^{\alpha_s}) + G_p E_{\alpha_p}(-(t/\tau)^{\alpha_p})

   where :math:`E_\alpha` is Mittag-Leffler function. Zener is special case with :math:`\alpha_s = 0`, :math:`\alpha_p = 1`.

**Percolation and gel point**:
   At the **gel point** (incipient infinite network), Winter-Chambon criterion predicts power-law behavior inconsistent with Zener. Use FMG or FZSS near gel point.

Usage
-----

Basic Fitting Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import Zener

   omega = jnp.logspace(-1, 3, 160)
   G_data = measured_modulus(omega)

   model = Zener()
   model.parameters.set_value('Ge', 2.5e4)
   model.parameters.set_value('Gm', 9.0e4)
   model.parameters.set_value('eta', 4.2e3)
   model.fit(omega, G_data)
   prediction = model.predict(omega)

Advanced: Stress Relaxation Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Zener
   import numpy as np

   # Stress relaxation data
   time = np.logspace(-2, 2, 100)  # 0.01 to 100 s
   G_relaxation = measured_relaxation_modulus(time)

   # Initialize from data
   G_s_init = np.mean(G_relaxation[-10:])  # Average last 10 points
   G_0_init = G_relaxation[0]
   G_p_init = G_0_init - G_s_init

   # Fit
   model = Zener()
   model.parameters.set_value('Ge', G_s_init)
   model.parameters.set_value('Gm', G_p_init)
   model.parameters.set_value('eta', 1e4)
   model.fit(time, G_relaxation, test_mode='relaxation')

   print(f"Equilibrium modulus: {model.parameters.get_value('Ge'):.2e} Pa")
   print(f"Relaxation time: {model.parameters.get_value('eta') / model.parameters.get_value('Gm'):.3f} s")

Bayesian Inference with Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Zener

   # 1. NLSQ point estimation
   model = Zener()
   model.fit(omega, G_star)

   # 2. Bayesian inference (warm-start)
   result = model.fit_bayesian(
       omega, G_star,
       num_warmup=1500,
       num_samples=3000
   )

   # 3. Check parameter correlations (Ge and Gm often correlated)
   posterior_Ge = result.posterior_samples['Ge']
   posterior_Gm = result.posterior_samples['Gm']

   import matplotlib.pyplot as plt
   plt.scatter(posterior_Ge, posterior_Gm, alpha=0.3)
   plt.xlabel('Ge (Pa)')
   plt.ylabel('Gm (Pa)')
   plt.title('Parameter correlation')

   # 4. Credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"Ge: [{intervals['Ge'][0]:.2e}, {intervals['Ge'][1]:.2e}] Pa")

Tips & Pitfalls
---------------

- Normalize frequency data so that :math:`\omega \tau` spans both < 1 and > 1; otherwise
  ``eta`` and ``Gm`` become unidentifiable.
- If the data show residual flow, augment with a dashpot in series
  (:class:`rheojax.models.Maxwell`) instead of forcing ``Ge`` to zero.
- Use log-scale residuals when fitting :math:`G'` and :math:`G''` together so high-
  frequency points do not dominate.
- Keep ``Ge`` and ``Gm`` strictly positive; unconstrained optimizers can otherwise dip
  below zero and destabilize the ODE.
- Seed :math:`\tau` from the peak of :math:`G''` or crossover of :math:`G'` and
  :math:`G''` to reduce optimizer iterations.
- **Critical**: Ensure frequency range captures :math:`G''` peak. If :math:`\tau` is outside experimental window, use TTS or relaxation tests.
- For filled polymers or composites, consider fractional models (FZSS) if Zener fit is poor.
- Check creep recovery: If no recovery observed, material is liquid → use Maxwell/Burgers instead.

See Also
--------

**Classical Models:**

- :doc:`maxwell` — series spring-dashpot limit for purely viscous relaxation
- :doc:`springpot` — replaces the dashpot with a fractional element for power-law decay

**Fractional Models:**

- :doc:`../fractional/fractional_zener_sl` — fractional generalization with SpringPot behavior plus an equilibrium branch
- :doc:`../fractional/fractional_zener_ss` — both springs replaced with SpringPots (power-law relaxation)

**Transforms:**

- :doc:`../../transforms/fft` — convert relaxation data to :math:`G^*(\omega)` before fitting
- :doc:`../../transforms/mastercurve` — time-temperature superposition to extend frequency range

**Examples:**

- :doc:`../../examples/basic/02-zener-creep` — notebook that estimates :math:`G_s`, :math:`G_p`, and :math:`\eta_p` from creep-recovery experiments

**User Guides:**

- :doc:`../../user_guide/model_selection` — decision flowcharts for model selection

References
----------

.. [1] Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd Edition. Wiley (1980).
   ISBN: 978-0471048947. Comprehensive treatment of linear viscoelasticity.

.. [2] Tschoegl, N. W. *The Phenomenological Theory of Linear Viscoelastic Behavior*.
   Springer, Berlin (1989). https://doi.org/10.1007/978-3-642-73602-5

.. [3] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758

.. [4] Lakes, R. S. *Viscoelastic Solids*. CRC Press (1999).
   ISBN: 978-0849396588. Focused on solid viscoelasticity with engineering applications.

.. [5] Christensen, R. M. *Theory of Viscoelasticity*, 2nd Edition. Dover (1982).
   ISBN: 978-0486428802. Mathematical foundations with composite materials applications.

.. [6] Zener, C. *Elasticity and Anelasticity of Metals*. University of Chicago
   Press (1948). Original work on anelastic relaxation in metals.

.. [7] Flory, P. J. *Principles of Polymer Chemistry*. Cornell University Press (1953).
   ISBN: 978-0801401343. Affine network theory and equilibrium modulus derivations.

.. [8] Rubinstein, M., and Colby, R. H. *Polymer Physics*. Oxford University Press (2003).
   ISBN: 978-0198520597. Modern treatment of polymer network elasticity and dynamics.

.. [9] Findley, W. N., Lai, J. S., and Onaran, K. *Creep and Relaxation of Nonlinear
   Viscoelastic Materials*. Dover (1989). ISBN: 978-0486660165

.. [10] Menard, K. P. *Dynamic Mechanical Analysis: A Practical Introduction*,
   2nd Edition. CRC Press (2008). https://doi.org/10.1201/9781420053135
