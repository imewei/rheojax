.. _vitrimer_models:

========================================
Vitrimer and Nanocomposite Models
========================================

Overview
========

Vitrimers represent a revolutionary class of polymers that combine the mechanical robustness of thermosets with the processability of thermoplastics. Unlike traditional crosslinked networks, vitrimers contain exchangeable bonds that can swap partners through bond exchange reactions (BER) while maintaining constant network connectivity. This unique architecture creates materials with:

- **Permanent topology** at low temperatures (thermoset-like)
- **Network rearrangement** at elevated temperatures (thermoplastic-like)
- **Self-healing** and **recyclability** properties
- **Tunable relaxation** via exchange kinetics

RheoJAX provides two constitutive models for vitrimers:

1. **HVM (Hybrid Vitrimer Model)**: Three-subnetwork architecture for neat vitrimers
2. **HVNM (Hybrid Vitrimer Nanocomposite Model)**: Extends HVM with a fourth interphase subnetwork for nanoparticle-filled vitrimers

.. admonition:: Key Insight

   The vitrimer hallmark is that the exchangeable stress :math:`\sigma_E \to 0` at steady state, even though the network remains connected. The natural state tensor :math:`\boldsymbol{\mu}_{\text{nat}}` tracks deformation through BER, eliminating driving forces for stress relaxation.

When to Use These Models
=========================

HVM: Hybrid Vitrimer Model
---------------------------

Use **HVM** for:

- Neat vitrimer materials (epoxy, polyester, polyurethane vitrimers)
- Covalent adaptable networks (CANs) with transesterification/transamination
- Systems requiring temperature-dependent relaxation (Arrhenius)
- Materials with combined permanent and exchangeable crosslinks
- Applications needing stress-dependent or stretch-dependent exchange kinetics

.. warning::
   HVM is an 11-component ODE system. Bayesian inference (NUTS) requires significant memory (8-16 GB). Use FAST_MODE for exploratory analysis.

HVNM: Hybrid Vitrimer Nanocomposite Model
------------------------------------------

Use **HVNM** for:

- Vitrimer nanocomposites (CNT, silica, graphene-filled)
- Materials with distinct matrix vs interphase kinetics
- Systems requiring Guth-Gold strain amplification :math:`X(\phi)`
- Nanoparticle-reinforced vitrimers (:math:`\phi < 0.3`)
- Applications needing to study interfacial exchange dynamics

.. note::
   HVNM reduces to HVM exactly at :math:`\phi = 0`, verified to machine precision. This makes it suitable for studying reinforcement effects systematically.

Theoretical Foundations
========================

What Are Vitrimers?
-------------------

Vitrimers are **covalent adaptable networks (CANs)** where chemical bonds can exchange through associative mechanisms:

1. **Transesterification**: Exchange of ester linkages (R-O-C=O + R'-OH ↔ R-OH + R'-O-C=O)
2. **Transamination**: Exchange of amine bonds
3. **Olefin metathesis**: Exchange of C=C bonds
4. **Disulfide exchange**: S-S bond swapping

Unlike dissociative CANs (where bonds break before reforming), vitrimers maintain **constant crosslink density** during exchange — no transient dangling chains.

.. admonition:: Key Insight

   The vitrimer "sweet spot" combines:

   - **Slow exchange** (:math:`\tau_{\text{BER}} \gg \tau_{\text{obs}}`) --- thermoset behavior
   - **Fast exchange** (:math:`\tau_{\text{BER}} \ll \tau_{\text{obs}}`) --- thermoplastic behavior
   - **Arrhenius temperature dependence** --- processable above :math:`T_v`

HVM Architecture: Three Subnetworks
------------------------------------

The HVM decomposes total stress into three parallel contributions:

.. math::

   \boldsymbol{\sigma} = \boldsymbol{\sigma}_P + \boldsymbol{\sigma}_E + \boldsymbol{\sigma}_D

**1. Permanent Subnetwork (P)**

- Represents covalent crosslinks that never exchange
- Purely elastic (neo-Hookean rubber):

.. math::

   \boldsymbol{\sigma}_P = G_P \, \boldsymbol{\mu}

where :math:`\boldsymbol{\mu}` is the left Cauchy-Green strain tensor.

**2. Exchangeable Subnetwork (E)**

- Vitrimer bonds with BER kinetics
- Evolves via two coupled ODEs:

.. math::

   \frac{d\boldsymbol{\mu}^E}{dt} &= \dot{\boldsymbol{\mu}} - 2 k_{\text{BER}} \, \boldsymbol{\mu}^E \\
   \frac{d\boldsymbol{\mu}^E_{\text{nat}}}{dt} &= - 2 k_{\text{BER}} \, \boldsymbol{\mu}^E_{\text{nat}}

- Stress: :math:`\boldsymbol{\sigma}_E = G_E (\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{\text{nat}})`
- **Factor-of-2**: Both :math:`\boldsymbol{\mu}^E` and :math:`\boldsymbol{\mu}^E_{\text{nat}}` relax toward each other, doubling the effective rate

**3. Dissociative Subnetwork (D)**

- Physical bonds (entanglements, H-bonds, van der Waals)
- Standard Maxwell relaxation:

.. math::

   \frac{d\boldsymbol{\mu}^D}{dt} = \dot{\boldsymbol{\mu}} - 2 k_{d,D} \, \boldsymbol{\mu}^D

- Stress: :math:`\boldsymbol{\sigma}_D = G_D \, \boldsymbol{\mu}^D`

Transition State Theory (TST) Kinetics
---------------------------------------

The BER rate follows TST with stress or stretch activation:

**Stress-Dependent (default)**

.. math::

   k_{\text{BER}} = \nu_0 \exp\left(-\frac{E_a}{RT}\right) \cosh\left(\frac{V_{\text{act}} \sigma_{\text{VM}}}{RT}\right)

where:

- :math:`\nu_0` = attempt frequency (Hz)
- :math:`E_a` = activation energy (J/mol)
- :math:`V_{\text{act}}` = activation volume (:math:`\text{m}^3`)
- :math:`\sigma_{\text{VM}}` = von Mises stress

**Stretch-Dependent (alternative)**

.. math::

   k_{\text{BER}} = \nu_0 \exp\left(-\frac{E_a}{RT}\right) \exp\left(\alpha \text{tr}(\boldsymbol{\mu}^E)\right)

.. note::
   The :math:`\cosh` form allows both tensile and compressive stresses to accelerate exchange (symmetric activation).

HVNM Extension: The Interphase Subnetwork
------------------------------------------

HVNM adds a **fourth subnetwork (I)** to capture polymer-nanoparticle interfacial dynamics:

.. math::

   \boldsymbol{\sigma} = \boldsymbol{\sigma}_P \, X(\phi) + \boldsymbol{\sigma}_E \, X(\phi) + \boldsymbol{\sigma}_D \, X(\phi) + \boldsymbol{\sigma}_I \, X(\phi) \, \phi \, \beta_I

**Guth-Gold Strain Amplification**

.. math::

   X(\phi) = 1 + 2.5\phi + 14.1\phi^2

accounts for hydrodynamic reinforcement (valid for :math:`\phi < 0.3`).

**Interphase Kinetics**

The I-subnetwork follows the same BER form as E, but with independent parameters:

- :math:`\nu_{0,\text{int}}` = interfacial attempt frequency
- :math:`E_{a,\text{int}}` = interfacial activation energy
- :math:`G_I` = interphase modulus
- :math:`\beta_I` = reinforcement factor

This allows modeling scenarios where:

- **Slow interfacial exchange** (:math:`E_{a,\text{int}} \gg E_a`) --- constrained interphase
- **Fast interfacial exchange** (:math:`E_{a,\text{int}} \ll E_a`) --- mobile interphase
- **Frozen interphase** (:math:`E_{a,\text{int}} \to \infty`) --- permanent reinforcement

Practical Implementation
=========================

HVM: Basic Usage
----------------

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np

   # Create full vitrimer model (P + E + D subnetworks)
   model = HVMLocal(kinetics="stress", include_dissociative=True)

   # Set parameters
   model.parameters.set_value("G_P", 5000.0)    # Permanent modulus (Pa)
   model.parameters.set_value("G_E", 3000.0)    # Exchangeable modulus (Pa)
   model.parameters.set_value("G_D", 1000.0)    # Dissociative modulus (Pa)
   model.parameters.set_value("nu_0", 1e10)     # Attempt frequency (Hz)
   model.parameters.set_value("E_a", 80e3)      # Activation energy (J/mol)
   model.parameters.set_value("V_act", 1e-5)    # Activation volume (m³)
   model.parameters.set_value("k_d_D", 10.0)    # Dissociative rate (1/s)
   model.parameters.set_value("T", 373.15)      # Temperature (K)

   # SAOS: Two Maxwell modes + G_P plateau
   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega)

At low frequencies, :math:`G' \to G_P` (permanent network dominates). At high frequencies, both E and D subnetworks contribute elasticity.

HVM Factory Methods
-------------------

RheoJAX provides five factory methods for common limiting cases:

**1. Neo-Hookean Rubber (P only)**

.. code-block:: python

   model = HVMLocal.neo_hookean(G_P=5000.0)
   # Pure elastomer: no relaxation

**2. Maxwell Liquid (D only)**

.. code-block:: python

   model = HVMLocal.maxwell(G_D=1000.0, k_d_D=10.0)
   # Standard viscoelastic liquid

**3. Zener Solid (P + D)**

.. code-block:: python

   model = HVMLocal.zener(G_P=5000.0, G_D=1000.0, k_d_D=10.0)
   # Standard linear solid (Voigt + Maxwell parallel)

**4. Pure Vitrimer (P + E)**

.. code-block:: python

   model = HVMLocal.pure_vitrimer(
       G_P=5000.0,
       G_E=3000.0,
       nu_0=1e10,
       E_a=80e3,
       V_act=1e-5,
       T=373.15,
       kinetics="stress"
   )
   # Vitrimer without physical bonds

**5. Partial Vitrimer (P + E + D)**

.. code-block:: python

   model = HVMLocal.partial_vitrimer(
       G_P=5000.0,
       G_E=3000.0,
       nu_0=1e10,
       E_a=80e3,
       V_act=1e-5,
       k_d_D=10.0,
       G_D=1000.0,
       T=373.15,
       kinetics="stress"
   )
   # Full HVM with all subnetworks active

.. admonition:: Key Insight

   Use factory methods for rapid prototyping. They enforce physical constraints (e.g., zeroing inactive subnetworks) and provide sensible defaults.

HVNM: Nanocomposite Usage
--------------------------

.. code-block:: python

   from rheojax.models import HVNMLocal
   import numpy as np

   # Create nanocomposite vitrimer
   model = HVNMLocal(kinetics="stress", include_dissociative=True)

   # Set matrix parameters (same as HVM)
   model.parameters.set_value("G_P", 5000.0)
   model.parameters.set_value("G_E", 3000.0)
   model.parameters.set_value("G_D", 1000.0)
   model.parameters.set_value("nu_0", 1e10)
   model.parameters.set_value("E_a", 80e3)

   # Add nanoparticle parameters
   model.parameters.set_value("phi", 0.1)        # Volume fraction
   model.parameters.set_value("beta_I", 3.0)     # Interphase reinforcement
   model.parameters.set_value("nu_0_int", 1e8)   # Slower interfacial exchange
   model.parameters.set_value("E_a_int", 120e3)  # Higher interfacial barrier

   # Predict reinforced modulus
   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega)
   # Expect G' plateau ~ G_P * X(0.1) ~ 5000 * 1.391 ~ 6955 Pa

HVNM Factory Methods
---------------------

**1. Unfilled Vitrimer** (:math:`\phi = 0`, recovers HVM)

.. code-block:: python

   model = HVNMLocal.unfilled_vitrimer(
       G_P=5000.0, G_E=3000.0, G_D=1000.0,
       nu_0=1e10, E_a=80e3
   )
   # Identical to HVM.partial_vitrimer()

**2. Filled Elastomer (no exchange)**

.. code-block:: python

   model = HVNMLocal.filled_elastomer(G_P=10000.0, phi=0.1)
   # Reinforced rubber, no BER (E, D, I zeroed)

**3. Vitrimer Nanocomposite**

.. code-block:: python

   model = HVNMLocal.partial_vitrimer_nc(
       G_P=5000.0, G_E=3000.0, phi=0.1, beta_I=3.0,
       nu_0=1e10, E_a=80e3
   )
   # P + E + I active, no D subnetwork

**4. Conventional Filled Rubber**

.. code-block:: python

   model = HVNMLocal.conventional_filled_rubber(
       G_P=5000.0, G_D=1000.0, phi=0.15
   )
   # P + D active, no vitrimer exchange

**5. Matrix-Only Exchange**

.. code-block:: python

   model = HVNMLocal.matrix_only_exchange(
       G_P=5000.0, G_E=3000.0, phi=0.1,
       nu_0=1e10, E_a=80e3
   )
   # P + E active, frozen interphase (E_a_int = 250e3)

Startup Shear: Stress Overshoot
--------------------------------

Vitrimers exhibit stress overshoot during startup due to BER relaxation:

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np
   import matplotlib.pyplot as plt

   model = HVMLocal.partial_vitrimer(
       G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3, T=373.15
   )

   # Simulate startup at constant shear rate
   t = np.linspace(0.01, 50, 300)
   result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)

   # Extract results
   stress = result["stress"][:, 2]  # σ_xy component
   sigma_E = result["sigma_E"][:, 2]
   sigma_P = result["sigma_P"][:, 2]

   # Plot
   plt.figure(figsize=(10, 6))
   plt.plot(t, stress, label="Total σ", linewidth=2)
   plt.plot(t, sigma_P, label="σ_P (permanent)", linestyle="--")
   plt.plot(t, sigma_E, label="σ_E (exchangeable)", linestyle="--")
   plt.xlabel("Time (s)")
   plt.ylabel("Shear Stress (Pa)")
   plt.legend()
   plt.title("HVM Startup: Stress Overshoot from BER")
   plt.show()

**Physical Interpretation:**

- **Early time**: :math:`\boldsymbol{\mu}^E` builds faster than :math:`\boldsymbol{\mu}^E_{\text{nat}}` --- :math:`\sigma_E` increases
- **Overshoot peak**: Maximum :math:`\sigma_E` when BER rate catches up
- **Steady state**: :math:`\boldsymbol{\mu}^E_{\text{nat}}` tracks deformation --- :math:`\sigma_E \to 0`, only :math:`\sigma_P` remains

Temperature Sweep: Arrhenius Behavior
--------------------------------------

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np
   import matplotlib.pyplot as plt

   model = HVMLocal.pure_vitrimer(
       G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3
   )

   omega = np.logspace(-3, 3, 100)
   temperatures = [300, 350, 400, 450]  # K

   plt.figure(figsize=(10, 6))
   for T in temperatures:
       model.parameters.set_value("T", T)
       G_prime, _ = model.predict_saos(omega)
       plt.loglog(omega, G_prime, label=f"T = {T} K")

   plt.axhline(5000, color="k", linestyle="--", label="G_P plateau")
   plt.xlabel("Angular Frequency ω (rad/s)")
   plt.ylabel("Storage Modulus G' (Pa)")
   plt.legend()
   plt.title("HVM Temperature Sweep: Arrhenius Shift")
   plt.show()

**Physical Interpretation:**

- **Low T**: :math:`k_{\text{BER}} \to 0` --- E subnetwork behaves elastically --- :math:`G'` plateau at :math:`G_P + G_E`
- **High T**: :math:`k_{\text{BER}} \to \infty` --- fast exchange --- :math:`G'` plateau at :math:`G_P` only
- **Topology freezing** :math:`T_v`: Temperature where :math:`k_{\text{BER}} \sim \omega_{\text{obs}}` (experiment-dependent)

Comparison: Neat vs Filled Vitrimers
-------------------------------------

.. code-block:: python

   from rheojax.models import HVMLocal, HVNMLocal
   import numpy as np
   import matplotlib.pyplot as plt

   # Neat vitrimer (HVM)
   hvm = HVMLocal.partial_vitrimer(G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3)

   # Filled vitrimer (HVNM, φ=0.15)
   hvnm = HVNMLocal.partial_vitrimer_nc(
       G_P=5000, G_E=3000, phi=0.15, beta_I=3.0,
       nu_0=1e10, E_a=80e3
   )

   omega = np.logspace(-3, 3, 100)

   G_prime_neat, _ = hvm.predict_saos(omega)
   G_prime_filled, _ = hvnm.predict_saos(omega)

   plt.figure(figsize=(10, 6))
   plt.loglog(omega, G_prime_neat, label="Neat (φ=0)", linewidth=2)
   plt.loglog(omega, G_prime_filled, label="Filled (φ=0.15)", linewidth=2)
   plt.xlabel("Angular Frequency ω (rad/s)")
   plt.ylabel("Storage Modulus G' (Pa)")
   plt.legend()
   plt.title("HVNM Reinforcement: Guth-Gold Amplification")
   plt.show()

   # Check X(φ) factor
   X_phi = 1 + 2.5*0.15 + 14.1*0.15**2
   print(f"X(φ=0.15) = {X_phi:.3f}")
   print(f"Expected G'_plateau (filled) ~ {5000 * X_phi:.0f} Pa")

Bayesian Inference
==================

HVM: Oscillatory Fitting
-------------------------

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np

   # Generate synthetic SAOS data
   model_true = HVMLocal.partial_vitrimer(
       G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3, T=373.15
   )
   omega = np.logspace(-2, 2, 50)
   G_prime_true, G_double_prime_true = model_true.predict_saos(omega)
   G_star_noisy = (G_prime_true + 1j*G_double_prime_true) * (1 + 0.05*np.random.randn(50))

   # Step 1: NLSQ warm-start
   model = HVMLocal(kinetics="stress", include_dissociative=True)
   model.fit(omega, G_star_noisy, test_mode='oscillation')

   # Step 2: Bayesian inference (4 chains for diagnostics)
   result = model.fit_bayesian(
       omega, G_star_noisy,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Step 3: Diagnostics
   print("R-hat values:")
   for name in result.posterior_samples.keys():
       samples = result.posterior_samples[name]
       print(f"  {name}: {result.diagnostics.get(f'{name}_rhat', np.nan):.4f}")

   # Step 4: Credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   for name, (lower, upper) in intervals.items():
       print(f"{name}: [{lower:.2e}, {upper:.2e}]")

.. warning::
   HVM has 8-9 parameters. Ensure sufficient data (N > 50) and tight priors to avoid posterior degeneracies (e.g., :math:`G_P` vs :math:`G_E` tradeoff).

HVNM: Startup Fitting
---------------------

.. code-block:: python

   from rheojax.models import HVNMLocal
   import numpy as np

   # Generate synthetic startup data
   model_true = HVNMLocal.partial_vitrimer_nc(
       G_P=5000, G_E=3000, phi=0.1, beta_I=3.0,
       nu_0=1e10, E_a=80e3
   )
   t = np.linspace(0.01, 50, 200)
   result_true = model_true.simulate_startup(t, gamma_dot=1.0, return_full=True)
   stress_noisy = result_true["stress"][:, 2] * (1 + 0.05*np.random.randn(200))

   # Fit with NLSQ
   model = HVNMLocal(kinetics="stress", include_dissociative=False)
   model.parameters.set_value("phi", 0.1)  # Fix volume fraction
   model.fit(t, stress_noisy, test_mode='startup', gamma_dot=1.0)

   # Bayesian inference (memory-intensive for ODE model)
   result = model.fit_bayesian(
       t, stress_noisy,
       test_mode='startup',
       gamma_dot=1.0,
       num_warmup=500,    # Reduced for ODE model
       num_samples=1000,
       num_chains=2,      # Reduced to avoid OOM
       seed=42
   )

.. note::
   HVNM startup uses 17-component ODE integration inside NUTS. Each leapfrog step requires forward + backward ODE solves. Reduce num_warmup/num_samples or use FAST_MODE for large datasets.

Visualization and Diagnostics
==============================

ArviZ Integration
-----------------

.. code-block:: python

   from rheojax.models import HVMLocal
   from rheojax.pipeline.bayesian import BayesianPipeline
   import arviz as az

   # Pipeline with ArviZ diagnostics
   pipeline = BayesianPipeline()
   (pipeline.load('vitrimer_saos.csv', x_col='omega', y_col='G_star')
            .fit_nlsq('hvm')
            .fit_bayesian(num_warmup=1000, num_samples=2000)
            .plot_pair(divergences=True, filter_degenerate=True)
            .plot_forest(hdi_prob=0.95)
            .plot_trace()
            .save('hvm_bayesian_results.hdf5'))

**Key Diagnostics:**

- **plot_pair()**: Posterior correlations (watch for :math:`G_P`-:math:`G_E` degeneracy)
- **plot_forest()**: 95% HDI intervals (ensure tight bounds)
- **plot_trace()**: MCMC chains (check mixing, no drift)
- **plot_energy()**: BFMI > 0.3 (good sampler efficiency)

Subnetwork Decomposition
-------------------------

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np
   import matplotlib.pyplot as plt

   model = HVMLocal.partial_vitrimer(G_P=5000, G_E=3000, G_D=1000, nu_0=1e10, E_a=80e3)

   t = np.linspace(0.01, 50, 300)
   result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)

   # Extract subnetwork stresses
   sigma_P = result["sigma_P"][:, 2]  # Permanent
   sigma_E = result["sigma_E"][:, 2]  # Exchangeable
   sigma_D = result["sigma_D"][:, 2]  # Dissociative
   sigma_total = result["stress"][:, 2]

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

   # Total stress
   ax1.plot(t, sigma_total, 'k', linewidth=2, label="Total σ")
   ax1.set_xlabel("Time (s)")
   ax1.set_ylabel("Shear Stress (Pa)")
   ax1.legend()
   ax1.set_title("HVM Startup: Total Response")

   # Subnetwork contributions
   ax2.plot(t, sigma_P, label="σ_P (permanent)", linewidth=2)
   ax2.plot(t, sigma_E, label="σ_E (exchangeable)", linewidth=2)
   ax2.plot(t, sigma_D, label="σ_D (dissociative)", linewidth=2)
   ax2.set_xlabel("Time (s)")
   ax2.set_ylabel("Stress (Pa)")
   ax2.legend()
   ax2.set_title("Subnetwork Decomposition")

   plt.tight_layout()
   plt.show()

.. admonition:: Key Insight

   At steady state:

   - :math:`\sigma_P` --- constant (elastic contribution)
   - :math:`\sigma_E \to 0` (vitrimer signature!)
   - :math:`\sigma_D \to 0` (physical bonds fully relaxed)

Comparison with Classical Models
=================================

HVM vs Zener Model
-------------------

.. list-table:: HVM vs Zener (Standard Linear Solid)
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - HVM
     - Zener
   * - Subnetworks
     - P + E (+ optional D)
     - Spring + Maxwell parallel
   * - Natural state
     - Evolves via BER
     - Fixed reference config
   * - Steady stress
     - :math:`\sigma_E \to 0`, only :math:`\sigma_P`
     - :math:`\sigma_{\text{spring}} + \sigma_{\text{Maxwell}}`
   * - Temperature
     - Arrhenius :math:`k_{\text{BER}}(T)`
     - Time-temp superposition
   * - Relaxation time
     - :math:`\tau_E = 1/(2 k_{\text{BER}})`
     - :math:`\tau = \eta/G`
   * - Best for
     - Vitrimers, CANs
     - Standard viscoelastics

**Code Comparison:**

.. code-block:: python

   from rheojax.models import HVMLocal, Zener
   import numpy as np
   import matplotlib.pyplot as plt

   # HVM (P + E, no D)
   hvm = HVMLocal.pure_vitrimer(G_P=5000, G_E=3000, nu_0=1e10, E_a=80e3, T=373.15)

   # Zener (G_e + G_eq in parallel with η)
   zener = Zener()
   zener.parameters.set_value("G_e", 5000)   # Elastic spring
   zener.parameters.set_value("G_eq", 3000)  # Maxwell spring
   zener.parameters.set_value("eta", 300)    # Maxwell dashpot (τ ~ 0.1s)

   t = np.linspace(0, 50, 300)

   # Relaxation after unit strain
   G_hvm = hvm.predict(t, test_mode='relaxation')
   G_zener = zener.predict(t, test_mode='relaxation')

   plt.figure(figsize=(10, 6))
   plt.plot(t, G_hvm, label="HVM (vitrimer)", linewidth=2)
   plt.plot(t, G_zener, label="Zener (standard)", linewidth=2, linestyle="--")
   plt.axhline(5000, color="k", linestyle=":", label="G_P (HVM) / G_e (Zener)")
   plt.xlabel("Time (s)")
   plt.ylabel("Relaxation Modulus G(t) (Pa)")
   plt.legend()
   plt.title("HVM vs Zener: Stress Relaxation")
   plt.show()

**Key Difference:** Zener relaxes to :math:`G_e + G_{eq}/(1 + t/\tau)`, while HVM relaxes to :math:`G_P` only (:math:`\sigma_E \to 0`).

HVNM vs Filled Rubber Models
-----------------------------

.. list-table:: HVNM vs Conventional Filled Rubber
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - HVNM
     - Filled Rubber (P + D)
   * - Subnetworks
     - P + E + D + I (4)
     - P + D (2)
   * - Interphase
     - Explicit I-subnetwork
     - Implicit in :math:`X(\phi)`
   * - Exchange
     - Matrix + interface BER
     - None (permanent + physical)
   * - Moduli amplification
     - Guth-Gold :math:`X(\phi)`
     - Guth-Gold :math:`X(\phi)`
   * - Relaxation
     - Dual TST kinetics
     - Single Maxwell mode
   * - Best for
     - Vitrimer nanocomposites
     - Rubber compounds

Limitations and Considerations
===============================

Computational Constraints
-------------------------

.. warning::
   **Memory Requirements:**

   - HVM: 11-component ODE --- 8-12 GB RAM for NUTS
   - HVNM: 17-18-component ODE --- 12-16 GB RAM for NUTS
   - LAOS with NUTS may OOM on 16 GB systems

**Mitigation strategies:**

1. Use **FAST_MODE** (50+100 warmup/samples) for exploratory analysis
2. Reduce **num_chains=2** instead of default 4
3. Subsample data for LAOS (1000 to 200 points)
4. Use **NLSQ only** for parameter estimation if posteriors not needed

Physical Limitations
--------------------

**1. TST Assumptions**

- Single energy barrier (no multi-step BER)
- Arrhenius temperature dependence (no WLF behavior)
- Mean-field kinetics (no spatial heterogeneity)

**2. Guth-Gold Validity**

- Valid for :math:`\phi < 0.3` (dilute to semi-dilute)
- Assumes spherical, non-interacting nanoparticles
- No percolation or agglomeration effects

**3. No Damage Evolution**

- Permanent bonds never break (:math:`G_P` constant)
- No fatigue or cyclic degradation
- No void nucleation under large strains

**4. Interphase Simplification**

- Single effective interphase thickness
- No gradient in properties near NP surface
- No polymer-NP entanglement dynamics

Parameter Sensitivity
---------------------

.. note::
   **Critical parameters** (small changes lead to large effects):

   - **V_act**: Bounds (1e-8, 0.01), default 1e-5. Never use 1e-28 (unphysical)
   - **E_a**: Typical 60--120 kJ/mol for vitrimers. Higher means slower exchange
   - **beta_I**: Typical 1--5. Higher means stronger interphase reinforcement
   - **phi**: Guth-Gold nonlinear above 0.2 (:math:`14.1\phi^2` term dominates)

Example Notebooks
=================

HVM Tutorials
-------------

RheoJAX provides 13 HVM example notebooks:

.. code-block:: text

   examples/hvm/
   ├── 01_hvm_saos.ipynb                    # SAOS with two Maxwell modes
   ├── 02_hvm_stress_relaxation.ipynb       # Relaxation to G_P plateau
   ├── 03_hvm_startup_shear.ipynb           # TST stress overshoot
   ├── 04_hvm_creep.ipynb                   # Delayed compliance
   ├── 05_hvm_flow_curve.ipynb              # Shear thinning from BER
   ├── 06_hvm_laos.ipynb                    # Nonlinear oscillatory
   ├── 07_hvm_overview.ipynb                # All protocols + theory
   ├── 08_hvm_flow_curve.ipynb              # Extended flow curve
   ├── 09_hvm_creep.ipynb                   # Extended creep
   ├── 10_hvm_relaxation.ipynb              # Extended relaxation
   ├── 11_hvm_startup.ipynb                 # Extended startup
   ├── 12_hvm_saos.ipynb                    # Extended SAOS
   └── 13_hvm_laos.ipynb                    # Extended LAOS

**Key notebooks:**

- **07**: Comprehensive overview with all 6 protocols
- **02**: Demonstrates :math:`\sigma_E \to 0` signature
- **03**: TST stress overshoot mechanism

HVNM Tutorials
--------------

.. code-block:: text

   examples/hvnm/
   ├── 01_hvnm_saos.ipynb                   # Reinforced SAOS
   ├── 02_hvnm_stress_relaxation.ipynb      # Relaxation with interphase
   ├── 03_hvnm_startup_shear.ipynb          # Startup with NPs
   ├── 04_hvnm_creep.ipynb                  # Creep compliance
   ├── 05_hvnm_flow_curve.ipynb             # Flow curves (neat vs filled)
   ├── 06_hvnm_laos.ipynb                   # LAOS with reinforcement
   ├── 07_hvnm_limiting_cases.ipynb         # Factory methods demo
   ├── 08_data_intake_and_qc.ipynb          # Data preprocessing
   ├── 09_flow_curve_nlsq_nuts.ipynb        # Bayesian flow curve
   ├── 10_creep_compliance_nlsq_nuts.ipynb  # Bayesian creep
   ├── 11_stress_relaxation_nlsq_nuts.ipynb # Bayesian relaxation
   ├── 12_startup_shear_nlsq_nuts.ipynb     # Bayesian startup
   ├── 13_saos_nlsq_nuts.ipynb              # Bayesian SAOS
   ├── 14_laos_nlsq_nuts.ipynb              # Bayesian LAOS
   └── 15_global_multi_protocol.ipynb       # Multi-protocol fitting

**Key notebooks:**

- **07**: Demonstrates all 5 factory methods and limiting cases
- **09-14**: Full NLSQ to NUTS workflow for each protocol
- **15**: Global fitting across multiple protocols (advanced)

References
==========

Foundational Papers
-------------------

1. **Vitrimers (original concept)**

   Montarnal, D., Capelot, M., Tournilhac, F., & Leibler, L. (2011). Silica-like malleable materials from permanent organic networks. *Science*, 334(6058), 965-968.

   DOI: `10.1126/science.1212648 <https://doi.org/10.1126/science.1212648>`_

2. **HVM constitutive model**

   Vernerey, F. J., Long, R., & Brighenti, R. (2017). A statistically-based continuum theory for polymers with transient networks. *Journal of the Mechanics and Physics of Solids*, 107, 1-20.

   DOI: `10.1016/j.jmps.2017.05.016 <https://doi.org/10.1016/j.jmps.2017.05.016>`_

3. **Epoxy vitrimers (transesterification)**

   Denissen, W., Winne, J. M., & Du Prez, F. E. (2016). Vitrimers: permanent organic networks with glass-like fluidity. *Chemical Science*, 7(1), 30-38.

   DOI: `10.1039/C5SC02223A <https://doi.org/10.1039/C5SC02223A>`_

4. **Guth-Gold reinforcement (empirical)**

   Guth, E., & Gold, O. (1938). On the hydrodynamical theory of the viscosity of suspensions. *Physical Review*, 53(2), 322.

   *Note: Conference abstract; no DOI registered with CrossRef.*

Advanced Topics
---------------

5. **Topology freezing transition**

   Capelot, M., Unterlass, M. M., Tournilhac, F., & Leibler, L. (2012). Catalytic control of the vitrimer glass transition. *ACS Macro Letters*, 1(7), 789-792.

   DOI: `10.1021/mz300239f <https://doi.org/10.1021/mz300239f>`_

6. **Stress-dependent exchange kinetics**

   Johlitz, M., Diebels, S., Possart, W., & Steinmann, P. (2008). Modelling of thermo-viscoelastic material behaviour of polyurethane close to the glass transition temperature. *ZAMM*, 88(8), 606-623.
   https://doi.org/10.1002/zamm.200900361

7. **Vitrimer nanocomposites (experimental)**

   Long, R., Qi, H. J., & Dunn, M. L. (2013). Modeling the mechanics of covalently adaptable polymer networks with temperature-dependent bond exchange reactions. *Soft Matter*, 9, 4083-4096. https://doi.org/10.1039/C3SM27945F

8. **Natural state formulation**

   Rajagopal, K. R., & Srinivasa, A. R. (2004). On thermomechanical restrictions of continua. *Proceedings of the Royal Society A*, 460(2042), 631-651.
   https://doi.org/10.1098/rspa.2002.1111

See Also
========

**Related User Guide Sections:**

- :ref:`ode_constitutive_models` — General ODE-based models
- :ref:`polymer_network_models` — TNT, VLB (related network theories)
- :doc:`bayesian_inference` — Bayesian workflow for complex models
- :doc:`/user_guide/02_model_usage/fitting_strategies` — Global fitting strategies

**API Documentation:**

- :doc:`/models/hvm/index` — HVM API reference
- :doc:`/models/hvnm/index` — HVNM API reference
- :doc:`/models/hvm/hvm` — HVM implementation details
- :doc:`/models/hvnm/hvnm` — HVNM implementation details
- :doc:`/models/hvm/hvm_knowledge` — HVM knowledge base
- :doc:`/models/hvnm/hvnm_knowledge` — HVNM knowledge base

**Example Notebooks:**

- ``examples/hvm/07_hvm_overview.ipynb`` — Comprehensive HVM tutorial
- ``examples/hvnm/07_hvnm_limiting_cases.ipynb`` — HVNM factory methods
- ``examples/hvnm/15_global_multi_protocol.ipynb`` — Advanced multi-protocol fitting
