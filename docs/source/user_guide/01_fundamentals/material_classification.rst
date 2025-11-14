.. _material_classification:

Material Classification
========================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Classify materials as liquid-like, solid-like, or gel-like from rheological data
   2. Identify key signatures in G' and G" frequency sweeps
   3. Distinguish between physical gels, chemical gels, and viscoelastic liquids
   4. Predict material behavior from modulus frequency dependence
   5. Select appropriate models based on material classification

.. admonition:: Prerequisites
   :class: important

   - :doc:`what_is_rheology` — Core rheology concepts
   - Understanding of G' (storage modulus) and G" (loss modulus)

The Classification Problem
---------------------------

When you measure a material's rheological properties, the first question is:

**"What type of material is this?"**

The answer determines:

- Which mathematical models are appropriate
- What test modes to use
- How to interpret parameters physically
- What processing or application behavior to expect

Classification is based on observing **how moduli change with frequency** (or equivalently, how stress relaxes with time).

Three Main Categories
---------------------

Materials fall into three broad rheological classes:

1. **Viscoelastic Liquids** — Flow at long times (zero equilibrium modulus)
2. **Viscoelastic Solids** — Finite equilibrium modulus (network structure)
3. **Gels** — Power-law relaxation (critical or near-critical structure)

Let's explore each in detail.

1. Viscoelastic Liquids
-----------------------

Definition
~~~~~~~~~~

**Liquids flow** — they have no equilibrium modulus. At long times (low frequencies), all stress relaxes to zero.

Rheological Signature
~~~~~~~~~~~~~~~~~~~~~~

In a **frequency sweep** (Small-Amplitude Oscillatory Shear, SAOS):

**Low frequencies (ω → 0)**:

- G" > G' (loss dominates)
- G" ~ ω¹ (linear with frequency)
- G' ~ ω² (quadratic with frequency)

**High frequencies**:

- G' > G" (storage dominates — elastic-like on short timescales)
- Both G' and G" increase

**Crossover frequency (ω_c)**:

- Point where G' = G" (transition from liquid-like to solid-like)
- Related to characteristic relaxation time: τ ~ 1/ω_c

**Physical interpretation**: At low frequencies (long observation times), the material has time to flow—viscous dissipation (G") dominates. At high frequencies (short observation times), molecular rearrangements can't keep up—elastic storage (G') dominates.

Mathematical Models
~~~~~~~~~~~~~~~~~~~

Appropriate models for viscoelastic liquids:

- **Maxwell model**: Single relaxation time (simplest)
- **Fractional Maxwell Liquid (FML)**: Distribution of relaxation times
- **Generalized Maxwell model**: Discrete spectrum of relaxation times

Examples
~~~~~~~~

- **Polymer melts**: Polystyrene, polyethylene above T_g
- **Polymer solutions**: DNA solutions, xanthan gum solutions
- **Micellar solutions**: Surfactant solutions above CMC

Code Example: Identifying a Viscoelastic Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Synthetic frequency sweep data for a viscoelastic liquid
   omega = np.logspace(-2, 2, 50)  # Angular frequency (rad/s)

   # Maxwell-like behavior
   tau = 1.0  # Relaxation time (s)
   G0 = 1e5   # Plateau modulus (Pa)

   G_prime = G0 * (omega * tau)**2 / (1 + (omega * tau)**2)
   G_double_prime = G0 * (omega * tau) / (1 + (omega * tau)**2)

   # Plot
   plt.figure(figsize=(8, 5))
   plt.loglog(omega, G_prime, 'o-', label="G' (storage)")
   plt.loglog(omega, G_double_prime, 's-', label='G" (loss)')
   plt.xlabel('Angular Frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.title('Viscoelastic Liquid: G" > G\' at low ω')
   plt.grid(True, alpha=0.3)

   # Identify crossover
   crossover_idx = np.argmin(np.abs(G_prime - G_double_prime))
   omega_c = omega[crossover_idx]
   print(f"Crossover frequency: {omega_c:.3f} rad/s")
   print(f"Characteristic time: {1/omega_c:.3f} s")

**Classification criteria**:

.. code-block:: python

   # Check low-frequency behavior
   low_freq_idx = 0
   if G_double_prime[low_freq_idx] > G_prime[low_freq_idx]:
       print("Classification: VISCOELASTIC LIQUID")
       print("(G\" > G' at low frequencies)")

2. Viscoelastic Solids
----------------------

Definition
~~~~~~~~~~

**Solids have a finite equilibrium modulus** — they can support stress indefinitely without flowing.
The network structure (chemical crosslinks or strong physical interactions) prevents flow.

Rheological Signature
~~~~~~~~~~~~~~~~~~~~~~

In a **frequency sweep**:

**Low frequencies (ω → 0)**:

- G' > G" (storage dominates)
- G' → G_e (plateau at equilibrium modulus)
- G" → 0 or small constant

**High frequencies**:

- Both G' and G" increase
- G' remains larger than G"

**No terminal flow**: Unlike liquids, there is NO crossover where G" exceeds G' at low frequencies.

**Physical interpretation**: The crosslinked network can store elastic energy indefinitely. Even at long timescales, the material does not flow—it remains solid-like.

Mathematical Models
~~~~~~~~~~~~~~~~~~~

Appropriate models for viscoelastic solids:

- **Zener model** (Standard Linear Solid): Two elastic elements + one viscous
- **Fractional Zener Solid-Solid (FZSS)**: Two elastic elements + fractional damping
- **Kelvin-Voigt model**: Elastic + viscous in parallel

Examples
~~~~~~~~

- **Crosslinked rubbers**: Vulcanized rubber, silicone elastomers
- **Chemical gels**: Covalently crosslinked hydrogels
- **Strong physical gels**: Agar, gelatin (below melting temperature)

Code Example: Identifying a Viscoelastic Solid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Zener model (solid)
   Ge = 1e4   # Equilibrium modulus (Pa)
   Gm = 5e4   # Arm modulus (Pa)
   tau = 1.0  # Relaxation time (s)

   G_prime_solid = Ge + Gm * (omega * tau)**2 / (1 + (omega * tau)**2)
   G_double_prime_solid = Gm * (omega * tau) / (1 + (omega * tau)**2)

   plt.figure(figsize=(8, 5))
   plt.loglog(omega, G_prime_solid, 'o-', label="G' (storage)")
   plt.loglog(omega, G_double_prime_solid, 's-', label='G" (loss)')
   plt.axhline(Ge, color='gray', linestyle='--', label=f'G_e = {Ge} Pa')
   plt.xlabel('Angular Frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.title('Viscoelastic Solid: G\' plateaus at G_e, G\' > G" everywhere')
   plt.grid(True, alpha=0.3)

**Classification criteria**:

.. code-block:: python

   # Check low-frequency behavior
   low_freq_idx = 0
   if G_prime_solid[low_freq_idx] > G_double_prime_solid[low_freq_idx]:
       # Check if G' plateaus
       G_prime_slope = np.gradient(np.log(G_prime_solid), np.log(omega))
       if abs(G_prime_slope[low_freq_idx]) < 0.1:  # Nearly flat
           print("Classification: VISCOELASTIC SOLID")
           print(f"Equilibrium modulus G_e ≈ {G_prime_solid[low_freq_idx]:.1e} Pa")

3. Gels and Power-Law Materials
--------------------------------

Definition
~~~~~~~~~~

**Gels exhibit power-law relaxation** — neither fully solid nor fully liquid. They sit at or near a critical gelation point, with a broad distribution of relaxation times.

Rheological Signature
~~~~~~~~~~~~~~~~~~~~~~

In a **frequency sweep**:

**Across all frequencies**:

- G' ~ ω^α (power-law scaling)
- G" ~ ω^α (parallel scaling)
- G' ≈ G" (often within same order of magnitude)
- Exponent α between 0 and 1 (typically 0.1-0.5)

**Log-log plot**: Both G' and G" are nearly parallel straight lines

**Physical interpretation**: Gels have a fractal or near-critical network structure with relaxation times spanning many decades. The power-law exponent α characterizes the breadth of the relaxation spectrum.

Types of Gels
~~~~~~~~~~~~~

**Physical gels**:

- Weak transient networks (thermoreversible)
- Examples: Gelatin (warm), wormlike micelles, clay suspensions

**Chemical gels** (at gelation point):

- Crosslinking in progress (critical gel)
- Power-law behavior only AT gelation transition

**Weak gels / soft solids**:

- G' slightly > G" across frequencies
- Small equilibrium modulus
- Examples: Yogurt, mayonnaise, soft pastes

Mathematical Models
~~~~~~~~~~~~~~~~~~~

Appropriate models for gels:

- **Fractional Maxwell Gel (FMG)**: Pure power-law (critical gel)
- **Fractional Burgers**: Gel-like with weak plateau
- **SpringPot**: Simplest fractional element

Examples
~~~~~~~~

- **Weak gels**: Yogurt, soft cheese, mayonnaise
- **Colloidal gels**: Fumed silica dispersions, carbon black gels
- **Biopolymer networks**: Fibrin, collagen (weak crosslinking)

Code Example: Identifying a Gel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fractional power-law gel
   alpha = 0.3  # Fractional exponent
   S = 1e4      # Quasi-modulus (Pa·s^α)

   G_prime_gel = S * omega**alpha * np.cos(np.pi * alpha / 2)
   G_double_prime_gel = S * omega**alpha * np.sin(np.pi * alpha / 2)

   plt.figure(figsize=(8, 5))
   plt.loglog(omega, G_prime_gel, 'o-', label="G' (storage)")
   plt.loglog(omega, G_double_prime_gel, 's-', label='G" (loss)')
   plt.xlabel('Angular Frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.title(f'Gel: Power-law scaling G\', G" ~ ω^{alpha}')
   plt.grid(True, alpha=0.3)

**Classification criteria**:

.. code-block:: python

   # Check for power-law scaling
   log_omega = np.log(omega)
   log_Gprime = np.log(G_prime_gel)

   # Linear fit in log-log space
   slope, _ = np.polyfit(log_omega, log_Gprime, 1)

   if 0.1 < slope < 0.9:  # Power-law exponent
       print("Classification: GEL / POWER-LAW MATERIAL")
       print(f"Power-law exponent α ≈ {slope:.2f}")

Visual Summary: Frequency Sweep Signatures
-------------------------------------------

.. code-block:: text

   VISCOELASTIC LIQUID
   ─────────────────────
   log(G', G")
        │     G'
        │    /
        │   /
        │  /  G' > G" (high ω)
        │ /
   ─────┼───────── ω_c (crossover)
        │\
        │ \  G" > G' (low ω)
        │  \
        │   \ G"
        └─────────── log(ω)

   Low ω: G" > G' (liquid-like)
   High ω: G' > G" (solid-like)
   Crossover: τ ~ 1/ω_c


   VISCOELASTIC SOLID
   ──────────────────
   log(G', G")
        │
        │    G'
        │   ╱
   G_e  ├──╱─────── G' plateau
        │ ╱
        │╱  G' > G" everywhere
        │  G"
        └─────────── log(ω)

   Low ω: G' → G_e (equilibrium modulus)
   High ω: Both increase
   No terminal flow


   GEL (POWER-LAW)
   ───────────────
   log(G', G")
        │
        │    ╱ G'
        │   ╱
        │  ╱  Parallel lines
        │ ╱
        │╱ G"
        │
        └─────────── log(ω)

   All ω: G', G" ~ ω^α (parallel)
   G' ≈ G" (similar magnitude)
   No single relaxation time

Classification Flowchart
------------------------

.. code-block:: text

   START: Frequency sweep data (G' vs G" vs ω)
      │
      ▼
   [Low frequency: G' or G" larger?]
      │
      ├─→ G" > G' ──→ VISCOELASTIC LIQUID
      │                   │
      │                   ├─→ Models: Maxwell, FML
      │                   └─→ Examples: Polymer melts
      │
      ├─→ G' > G" ──→ [Does G' plateau?]
      │                   │
      │                   ├─→ YES ──→ VISCOELASTIC SOLID
      │                   │              │
      │                   │              ├─→ Models: Zener, FZSS
      │                   │              └─→ Examples: Rubbers, chemical gels
      │                   │
      │                   └─→ NO ──→ Check power-law
      │
      └─→ G' ≈ G" ──→ [Parallel lines in log-log?]
                          │
                          ├─→ YES ──→ GEL / POWER-LAW
                          │              │
                          │              ├─→ Models: FMG, SpringPot
                          │              └─→ Examples: Yogurt, soft gels
                          │
                          └─→ NO ──→ Re-examine data quality

Practical Classification Algorithm
-----------------------------------

Use this code to automatically classify materials:

.. code-block:: python

   def classify_material(omega, G_prime, G_double_prime):
       """
       Classify material from frequency sweep data.

       Parameters
       ----------
       omega : array
           Angular frequency (rad/s)
       G_prime : array
           Storage modulus (Pa)
       G_double_prime : array
           Loss modulus (Pa)

       Returns
       -------
       classification : str
           'liquid', 'solid', or 'gel'
       details : dict
           Classification metrics
       """
       # Low-frequency behavior (first 10% of data)
       low_freq_idx = len(omega) // 10

       G_prime_low = G_prime[:low_freq_idx]
       G_double_prime_low = G_double_prime[:low_freq_idx]

       # 1. Check low-frequency dominance
       if np.mean(G_double_prime_low) > np.mean(G_prime_low):
           return 'liquid', {'note': 'G" > G\' at low frequencies'}

       # 2. Check for plateau (solid)
       log_omega = np.log(omega[:low_freq_idx])
       log_Gprime = np.log(G_prime_low)
       slope, _ = np.polyfit(log_omega, log_Gprime, 1)

       if abs(slope) < 0.15:  # Nearly flat
           Ge = np.mean(G_prime_low)
           return 'solid', {'Ge': Ge, 'note': 'G\' plateau detected'}

       # 3. Check for power-law (gel)
       log_omega_all = np.log(omega)
       log_Gprime_all = np.log(G_prime)
       slope_all, _ = np.polyfit(log_omega_all, log_Gprime_all, 1)

       if 0.1 < slope_all < 0.9:
           return 'gel', {'alpha': slope_all, 'note': 'Power-law scaling'}

       # Default
       return 'unknown', {'note': 'Re-examine data or try different tests'}

   # Example usage
   material_type, info = classify_material(omega, G_prime, G_double_prime)
   print(f"Material classification: {material_type}")
   print(f"Details: {info}")

Key Concepts
------------

.. admonition:: Main Takeaways
   :class: tip

   1. **Viscoelastic liquids**: G" > G' at low frequencies, flow at long times (Maxwell, FML)

   2. **Viscoelastic solids**: G' > G" everywhere, G' plateaus at G_e (Zener, FZSS)

   3. **Gels**: G' ≈ G" with power-law scaling G' ~ ω^α (FMG, SpringPot)

   4. **Classification determines model selection** and expected behavior

   5. **Frequency sweeps** (SAOS) are the primary tool for material classification

.. admonition:: Self-Check Questions
   :class: tip

   1. **A material has G' = 100 Pa and G" = 1000 Pa at 0.01 rad/s. What type of material is it likely to be?**

      Hint: Compare G' and G" at low frequency

   2. **You observe G' = 5000 Pa at all frequencies from 0.01 to 100 rad/s, while G" varies from 100 to 500 Pa. What classification?**

      Hint: Look for a plateau in G'

   3. **Both G' and G" scale as ω^0.4 over 3 decades of frequency. What material type?**

      Hint: Parallel power-law scaling

   4. **Why can't you use a single-frequency measurement to classify a material?**

      Hint: Need to see frequency dependence to distinguish types

   5. **A material shows G" > G' at low frequencies but also has a small plateau in G'. What might this indicate?**

      Hint: Could be intermediate case or weak gel

Further Reading
---------------

**Within this documentation**:

- :doc:`test_modes` — Experimental methods for probing material behavior
- :doc:`../02_model_usage/model_families` — Mathematical models for each material type
- :doc:`../02_model_usage/model_selection` — Flowcharts for choosing models

**External resources**:

- Winter, H.H. & Chambon, F. "Analysis of Linear Viscoelasticity of a Crosslinking Polymer at the Gel Point" *J. Rheol.* 30, 367 (1986) — Critical gels
- Ferry, J.D. *Viscoelastic Properties of Polymers*, Chapter 3 — Material classification

Summary
-------

Materials are classified rheologically into **viscoelastic liquids** (G" > G' at low ω, terminal flow), **viscoelastic solids** (G' plateaus at G_e, no flow), and **gels** (power-law scaling G' ~ ω^α). Classification is based on frequency sweep signatures and determines appropriate models and expected behavior.

Next Steps
----------

Proceed to: :doc:`test_modes`

Learn about the four major experimental techniques (SAOS, relaxation, creep, flow) used to characterize materials.
