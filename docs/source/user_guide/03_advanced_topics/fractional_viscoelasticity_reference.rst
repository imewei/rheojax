Fractional Viscoelasticity: Mathematical Reference
==================================================

.. note::
   This is the **definitive reference** for fractional calculus concepts in RheoJAX.
   All fractional model pages link here instead of duplicating this content.

Overview
--------

Fractional calculus generalizes differentiation and integration to non-integer orders, providing a powerful mathematical framework for describing complex viscoelastic behavior that cannot be captured by classical integer-order models. In rheology, fractional derivatives enable the modeling of **power-law relaxation**, **broad relaxation spectra**, and **self-similar microstructures** using fewer parameters than multi-mode classical models.

Why Fractional Calculus in Rheology?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most real materials exhibit viscoelastic behavior that deviates from simple exponential relaxation:

**Experimental observations:**

- **Power-law relaxation** :math:`G(t) \sim t^{-\alpha}` over multiple time decades
- **Broad relaxation spectra** arising from structural heterogeneity
- **Frequency-dependent moduli** with parallel slopes in log-log plots
- **Non-exponential creep** that cannot be fit with single relaxation times

**Classical model limitations:**

- Single relaxation time :math:`\tau` (Maxwell, Zener) insufficient for complex materials
- Multi-mode models require many parameters (5-20+) with limited physical insight
- Exponential functions cannot capture power-law dynamics

**Fractional model advantages:**

- Capture power-law behavior naturally with 3-5 parameters
- Fractional order :math:`\alpha` has clear physical meaning (spectrum breadth, microstructure)
- Fewer parameters than multi-mode models while maintaining accuracy
- Interpolate smoothly between elastic (:math:`\alpha=0`) and viscous (:math:`\alpha=1`) extremes

SpringPot Element
-----------------

The SpringPot (Scott-Blair element) is the **fundamental building block** of fractional rheology, generalizing both elastic springs and viscous dashpots into a single element.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

The SpringPot constitutive equation relates stress and strain through a fractional derivative:

.. math::

   \sigma(t) = E_0 \, D^\alpha \gamma(t)

where:
   - :math:`E_0`: quasi-property with units Pa·s :math:`^\alpha`
   - :math:`D^\alpha`: fractional derivative of order :math:`\alpha \in [0, 1]`
   - :math:`\gamma(t)`: strain as a function of time
   - :math:`\sigma(t)`: stress as a function of time

Limiting Cases
~~~~~~~~~~~~~~

The SpringPot smoothly interpolates between classical elements:

.. list-table:: SpringPot Limiting Behavior
   :header-rows: 1
   :widths: 15 35 50

   * - :math:`\alpha` Value
     - Element Type
     - Constitutive Equation
   * - :math:`\alpha = 0`
     - Pure elastic spring
     - :math:`\sigma = E_0 \gamma` (Hooke's law)
   * - :math:`0 < \alpha < 1`
     - Fractional viscoelastic
     - :math:`\sigma = E_0 D^\alpha \gamma` (intermediate behavior)
   * - :math:`\alpha = 1`
     - Pure viscous dashpot
     - :math:`\sigma = E_0 \, d\gamma/dt` (Newton's law)

Frequency-Domain Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In oscillatory shear (frequency domain), the SpringPot impedance is:

.. math::

   Z(\omega) = E_0 (i\omega)^\alpha = E_0 \omega^\alpha \left[\cos\left(\frac{\alpha\pi}{2}\right) + i\sin\left(\frac{\alpha\pi}{2}\right)\right]

This reveals that the SpringPot **simultaneously contributes to both storage and loss moduli** with a constant phase angle:

.. math::

   \delta = \frac{\alpha\pi}{2}

where :math:`\delta` is the loss angle (phase shift between stress and strain).

**Physical interpretation:**

- :math:`\alpha = 0`: :math:`\delta = 0^\circ` (purely elastic, no phase shift)
- :math:`\alpha = 0.5`: :math:`\delta = 45^\circ` (balanced viscoelasticity)
- :math:`\alpha = 1`: :math:`\delta = 90^\circ` (purely viscous, maximum phase shift)

The storage and loss moduli contributions scale as:

.. math::

   G'(\omega) &\sim \omega^\alpha \cos(\alpha\pi/2) \\
   G''(\omega) &\sim \omega^\alpha \sin(\alpha\pi/2)

**Key insight:** Both moduli have **parallel slopes** of :math:`\alpha` in log-log plots, which is the hallmark signature of fractional viscoelasticity.

Mittag-Leffler Functions
-------------------------

Mittag-Leffler functions play the same role in fractional viscoelasticity as exponential functions do in classical models. They provide the **exact analytical solutions** for fractional differential equations governing viscoelastic constitutive relations.

One-Parameter Mittag-Leffler Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one-parameter Mittag-Leffler function is defined by the infinite series:

.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

where :math:`\Gamma` is the gamma function (generalization of factorial to real numbers).

**Key Properties:**

1. **Recovers exponential**: :math:`E_1(z) = \exp(z)` (classical limit)
2. **Initial value**: :math:`E_\alpha(0) = 1` for all :math:`\alpha > 0`
3. **Asymptotic behavior**:

   - Short times: :math:`E_\alpha(-t^\alpha) \approx 1 - t^\alpha/\Gamma(\alpha+1)`
   - Intermediate times: :math:`E_\alpha(-t^\alpha) \sim t^{-\alpha}` (power-law decay)
   - Long times: :math:`E_\alpha(-t^\alpha) \sim \exp(-A \, t^{\alpha/(1-\alpha)})` (stretched exponential)

4. **Interpolation**: Smoothly interpolates between exponential (:math:`\alpha=1`) and power-law (:math:`0<\alpha<1`)

**Physical Meaning in Relaxation:**

The relaxation modulus for fractional models typically has the form:

.. math::

   G(t) = G_0 \, E_\alpha\left(-\left(\frac{t}{\tau_\alpha}\right)^\alpha\right)

This captures:

- **Initial plateau**: :math:`G(0) = G_0` (elastic response)
- **Power-law relaxation**: :math:`G(t) \sim G_0 (t/\tau_\alpha)^{-\alpha}` at intermediate times
- **Broad relaxation spectrum**: Continuous distribution of relaxation times
- **Characteristic time** :math:`\tau_\alpha`: Time scale for onset of power-law decay

Two-Parameter Mittag-Leffler Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-parameter generalization adds a second parameter :math:`\beta`:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

**Key Properties:**

1. **Reduces to one-parameter**: :math:`E_{\alpha,1}(z) = E_\alpha(z)`
2. **Initial value**: :math:`E_{\alpha,\beta}(0) = 1/\Gamma(\beta)`
3. **More flexible asymptotics**: Controls short-time behavior via :math:`\beta`

**Applications in Fractional Models:**

- **Creep compliance**: :math:`J(t)` often involves :math:`E_{\alpha,1+\alpha}(-t^\alpha)`
- **Complex constitutive equations**: Fractional Maxwell Liquid uses :math:`E_{1-\alpha,1-\alpha}`
- **General viscoelasticity**: Provides exact solutions for arbitrary fractional orders

Computational Note
~~~~~~~~~~~~~~~~~~

RheoJAX computes Mittag-Leffler functions using the ``mittag_leffler`` module (:mod:`rheojax.utils.mittag_leffler`), which implements:

- **One-parameter**: ``E_alpha(z, alpha)`` via series expansion + asymptotic approximations
- **Two-parameter**: ``E_alpha_beta(z, alpha, beta)`` via series expansion

These functions are JAX-compatible and GPU-accelerated for fast evaluation in optimization and Bayesian inference.

Physical Meaning of Fractional Order α
---------------------------------------

The fractional order :math:`\alpha` is **not an arbitrary fitting parameter** -- it has deep physical significance related to material microstructure and relaxation dynamics.

1. Relaxation Spectrum Width
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fractional order :math:`\alpha` quantifies the **breadth of the relaxation time distribution**:

.. list-table:: Spectrum Breadth Interpretation
   :header-rows: 1
   :widths: 15 35 50

   * - :math:`\alpha` Value
     - Spectrum Type
     - Physical Meaning
   * - :math:`\alpha = 1`
     - Narrow (Dirac delta)
     - Single relaxation time (classical exponential)
   * - :math:`0.7 < \alpha < 1`
     - Moderate breadth
     - Few dominant relaxation processes
   * - :math:`0.3 < \alpha < 0.7`
     - Broad distribution
     - Continuous spectrum over many decades
   * - :math:`\alpha \to 0`
     - Very broad (power-law)
     - Hierarchical or fractal structure, no characteristic time

**Mathematical connection:**

For fractional models, the relaxation time spectrum :math:`H(\tau)` is approximately:

.. math::

   H(\tau) \sim \tau^{-(1-\alpha)} \quad \text{for } \tau_{\text{min}} < \tau < \tau_{\text{max}}

where:
   - Narrow spectrum (:math:`\alpha \to 1`): :math:`H(\tau) \to \delta(\tau - \tau_0)` (Dirac delta)
   - Broad spectrum (:math:`\alpha \approx 0.5`): :math:`H(\tau) \sim \tau^{-0.5}` (power-law distribution)

2. Microstructural Heterogeneity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lower :math:`\alpha` values indicate greater **structural heterogeneity** at the molecular/microscopic level:

**For cross-linked networks (e.g., elastomers, hydrogels):**

- :math:`\alpha < 0.5`: Hierarchical structure with multiple length scales

  - Broad cross-link density distribution
  - Polydisperse mesh sizes
  - Fractal or self-similar network architecture

- :math:`\alpha \approx 0.5`: Critical gel-like behavior

  - Sol-gel transition point
  - Percolation threshold
  - Maximum structural disorder

- :math:`\alpha > 0.5`: More homogeneous networks

  - Narrow cross-link density distribution
  - Approaching regular lattice structure

**For polymer melts:**

- :math:`\alpha < 0.5`: Broad molecular weight distribution (polydispersity)

  - Significant chain length heterogeneity
  - Branched or star polymers
  - Complex intermolecular interactions

- :math:`\alpha \approx 0.7\text{--}0.9`: Relatively monodisperse linear polymers

  - Narrow molecular weight distribution
  - Simple chain dynamics (reptation)

3. Material Character (Solid vs. Liquid vs. Gel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fractional order :math:`\alpha` influences the **dominant viscoelastic character**:

.. list-table:: Material Character Classification
   :header-rows: 1
   :widths: 20 30 50

   * - :math:`\alpha` Range
     - Dominant Character
     - Typical Materials
   * - :math:`\alpha < 0.3`
     - Strong solid-like
     - Stiff gels, covalently cross-linked elastomers, biological tissues
   * - :math:`0.3 < \alpha < 0.5`
     - Solid-like viscoelastic
     - Soft gels, filled polymers, weak networks
   * - :math:`\alpha \approx 0.5`
     - Critical gel (balanced)
     - Gel point, percolation threshold, :math:`G' \approx G''` across all :math:`\omega`
   * - :math:`0.5 < \alpha < 0.7`
     - Liquid-like viscoelastic
     - Concentrated polymer solutions, weak gels
   * - :math:`\alpha > 0.7`
     - Strong liquid-like
     - Polymer melts, dilute solutions, approaching classical Maxwell

**Oscillatory shear signature:**

- :math:`\alpha < 0.5`: :math:`G'(\omega) > G''(\omega)` at low frequencies (elastic dominance)
- :math:`\alpha \approx 0.5`: :math:`G'(\omega) \approx G''(\omega)` across all frequencies (critical gel)
- :math:`\alpha > 0.5`: :math:`G''(\omega) > G'(\omega)` at low frequencies (viscous dominance)

4. Typical α Ranges by Material Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extensive experimental studies have established typical fractional order ranges for common materials:

.. list-table:: Fractional Order Ranges by Material
   :header-rows: 1
   :widths: 30 20 50

   * - Material Class
     - Typical :math:`\alpha`
     - Notes
   * - **Cross-linked polymer networks**
     - 0.3 - 0.6
     - Natural rubber, synthetic elastomers, cured epoxies
   * - **Filled elastomers**
     - 0.2 - 0.5
     - Carbon black or silica-filled rubber; lower :math:`\alpha` due to filler-polymer interactions
   * - **Hydrogels (chemical)**
     - 0.4 - 0.7
     - Covalently cross-linked PVA, alginate, PAA
   * - **Hydrogels (physical)**
     - 0.3 - 0.5
     - Non-covalent cross-links, weaker structure
   * - **Biological tissues (soft)**
     - 0.1 - 0.4
     - Skin, tendons, cartilage; very broad spectra from hierarchical collagen/elastin
   * - **Biological tissues (stiff)**
     - 0.3 - 0.5
     - Bone, dentin, cornea
   * - **Semi-crystalline polymers**
     - 0.3 - 0.5
     - Polyethylene, polypropylene; crystalline vs amorphous phase relaxation
   * - **Polymer melts (linear)**
     - 0.7 - 0.9
     - Linear homopolymers; approaching classical Maxwell behavior
   * - **Polymer melts (branched)**
     - 0.5 - 0.7
     - Long-chain branched polymers, star polymers
   * - **Concentrated polymer solutions**
     - 0.5 - 0.8
     - Above overlap concentration :math:`c^*`
   * - **Emulsions**
     - 0.4 - 0.7
     - Droplet size polydispersity and interfacial dynamics
   * - **Colloidal gels**
     - 0.2 - 0.4
     - Particle network with weak attractive interactions
   * - **Critical gels**
     - 0.45 - 0.55
     - Sol-gel transition, gelation point

Physical Interpretation Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key takeaway:** The fractional order :math:`\alpha` is a **structural fingerprint** that encodes:

1. **How broad the relaxation spectrum is** (spectrum width)
2. **How heterogeneous the microstructure is** (structural disorder)
3. **Whether the material is solid-like or liquid-like** (material character)
4. **What physical processes dominate relaxation** (molecular vs network dynamics)

Lower :math:`\alpha` values indicate:
   - Broader relaxation spectra
   - More heterogeneous microstructure
   - More solid-like character
   - Hierarchical or fractal organization

Higher :math:`\alpha` values indicate:
   - Narrower relaxation spectra
   - More homogeneous microstructure
   - More liquid-like character
   - Approaching classical exponential behavior

Fractional Models in RheoJAX
-----------------------------

RheoJAX implements **11 fractional models** organized into families based on their mechanical analogs:

**Fractional Maxwell Family (4 models):**

- :doc:`/models/fractional/fractional_maxwell_gel` — Gel-like with elastic component
- :doc:`/models/fractional/fractional_maxwell_liquid` — Liquid-like with memory
- :doc:`/models/fractional/fractional_maxwell_model` — Dual SpringPot series (general)
- :doc:`/models/fractional/fractional_kelvin_voigt` — Solid-like with slow relaxation

**Fractional Zener Family (4 models):**

- :doc:`/models/fractional/fractional_zener_ss` — **Most common**: Dual elastic plateaus
- :doc:`/models/fractional/fractional_zener_sl` — Solid + fractional liquid
- :doc:`/models/fractional/fractional_zener_ll` — Fractional liquid-liquid
- :doc:`/models/fractional/fractional_kv_zener` — FKV + series spring

**Advanced Fractional Models (3 models):**

- :doc:`/models/fractional/fractional_burgers` — Maxwell + FKV (creep + relaxation)
- :doc:`/models/fractional/fractional_jeffreys` — Two dashpots + SpringPot
- :doc:`/models/fractional/fractional_poynting_thomson` — Multi-plateau solid

See :doc:`/models/index` for detailed model documentation.

Key References
--------------

**Foundational Theory:**

- **Mainardi, F. (2010)**. *Fractional Calculus and Waves in Linear Viscoelasticity*. Imperial College Press.
  ISBN: 978-1-84816-329-4

  *The definitive reference on fractional calculus in viscoelasticity.*

- **Schiessel, H., Metzler, R., Blumen, A., Nonnenmacher, T.F. (1995)**. "Generalized viscoelastic models: their fractional equations with solutions." *J. Phys. A* 28, 6567–6584.
  https://doi.org/10.1088/0305-4470/28/23/012

  *Original derivation of fractional viscoelastic models.*

- **Gorenflo, R., Kilbas, A.A., Mainardi, F., Rogosin, S.V. (2014)**. *Mittag-Leffler Functions, Related Topics and Applications*. Springer.
  https://doi.org/10.1007/978-3-662-43930-2

  *Comprehensive treatment of Mittag-Leffler functions.*

**Physical Interpretation:**

- **Mainardi, F., Spada, G. (2011)**. "Creep, Relaxation and Viscosity Properties for Basic Fractional Models in Rheology." *European Physical Journal Special Topics*, 193, 133-160.
  https://doi.org/10.1140/epjst/e2011-01387-1

  *Physical meaning of fractional parameters in rheology.*

- **Friedrich, C., Braun, H. (1992)**. "Generalized Cole-Cole Behavior and its Rheological Relevance." *Rheologica Acta*, 31, 309-322.
  https://doi.org/10.1007/BF00418328

  *Connection between fractional order and relaxation spectrum width.*

**Applications:**

- **Koeller, R.C. (1984)**. "Applications of fractional calculus to the theory of viscoelasticity." *J. Appl. Mech.* 51, 299–307.
  https://doi.org/10.1115/1.3167616

  *Early application of fractional calculus to viscoelasticity.*

- **Metzler, R., Klafter, J. (2000)**. "The Random Walk's Guide to Anomalous Diffusion: A Fractional Dynamics Approach." *Physics Reports*, 339(1), 1-77.
  https://doi.org/10.1016/S0370-1573(00)00070-3

  *Broader context: fractional dynamics in physics.*

Further Reading
---------------

**Within RheoJAX Documentation:**

- :doc:`/user_guide/model_selection` — Decision flowcharts for choosing fractional vs classical models
- :doc:`/developer/architecture` — Template Method pattern for smart initialization
- :doc:`/examples/advanced/04-fractional-models-deep-dive` — Jupyter notebook with case studies

**External Resources:**

- **Podlubny, I. (1999)**. *Fractional Differential Equations*. Academic Press.
  ISBN: 978-0-12-558840-9
- **Hilfer, R. (Ed.) (2000)**. *Applications of Fractional Calculus in Physics*. World Scientific.
  ISBN: 978-981-02-3457-7
- **Tarasov, V.E. (2010)**. *Fractional Dynamics: Applications of Fractional Calculus to Dynamics of Particles, Fields and Media*. Springer.
  https://doi.org/10.1007/978-3-642-14003-7

See Also
--------

- :doc:`/models/index` — Complete model catalog with governing equations
- :doc:`/user_guide/core_concepts` — RheoData, parameters, and test modes
- :doc:`/user_guide/modular_api` — Direct model API usage
- :doc:`/user_guide/bayesian_inference` — Bayesian inference for fractional models
