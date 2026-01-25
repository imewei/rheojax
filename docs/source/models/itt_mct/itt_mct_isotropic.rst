.. _model-itt-mct-isotropic:

ITT-MCT Isotropic (ISM)
=======================

Quick Reference
---------------

- **Use when:** Quantitative predictions needed, S(k) available, wave-vector-dependent
dynamics important

- **Parameters:** 5 (φ, σ_d, D₀, k_BT, γ_c) + S(k) input

- **Key equation:** k-resolved correlator Φ(k,t) with MCT vertex from S(k)

- **Test modes:** Flow curve, oscillation, startup, creep, relaxation, LAOS

- **Material examples:** Dense colloidal suspensions, hard-sphere glasses, silica particles,
PMMA colloids, concentrated emulsions

**Data required:** Structure factor S(k) from experiment or Percus-Yevick

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\Phi(k,t)`
     - k-resolved density correlator (one function per wave vector)
   * - :math:`S(k)`
     - Static structure factor (equilibrium pair correlation)
   * - :math:`c(k)`
     - Direct correlation function, :math:`c(k) = 1 - 1/S(k)`
   * - :math:`V(k,q,p)`
     - MCT vertex function (mode-coupling kernel)
   * - :math:`\phi`
     - Volume fraction (control parameter for glass transition)
   * - :math:`\phi_g`
     - Glass transition volume fraction (:math:`\approx 0.516` for hard spheres)
   * - :math:`\sigma_d`
     - Particle diameter (m)
   * - :math:`D_0`
     - Bare short-time diffusion coefficient (m²/s)
   * - :math:`k_B T`
     - Thermal energy (J)
   * - :math:`\Gamma(k)`
     - k-dependent bare relaxation rate, :math:`\Gamma(k) = k^2 D_0 / S(k)`
   * - :math:`\gamma_c`
     - Critical strain for cage breaking (dimensionless)
   * - :math:`n`
     - Number density (particles/m³)

Overview
--------

The Isotropically Sheared Model (ISM) is the full k-resolved MCT for nonlinear
rheology. Unlike the F₁₂ schematic model, ISM tracks correlators at each wave
vector k, using the static structure factor S(k) to compute the memory kernel.

**Key differences from F₁₂:**

- k-resolved correlators Φ(k,t)
- Memory kernel from S(k) via MCT vertex V(k,q,|k-q|)
- Quantitative predictions without empirical parameters
- Higher computational cost

**When to use ISM:**

- S(k) is known (from scattering experiments or simulation)
- Wave-vector-dependent relaxation is important
- Quantitative comparison with microscopic measurements
- Systems where F₁₂ simplifications are too severe

Physical Foundations
--------------------

The ISM model extends the schematic F₁₂ theory (see :doc:`itt_mct_schematic`) to
include full k-dependence. All physical concepts from the schematic model apply:
cage effect, β-relaxation, α-relaxation, glass transition. The key addition is
**wave-vector resolution** of the dynamics.

**Why k-dependence matters:**

1. **Length-scale-dependent relaxation**: Small k (long wavelengths) relax slower
   than large k (short wavelengths)
2. **Structure factor weighting**: Peaks in S(k) indicate preferred length scales
   that dominate dynamics
3. **Quantitative stress predictions**: Integration over k-space with S(k) weighting
   gives absolute stress values without empirical modulus

The ISM model is the most faithful representation of MCT for colloidal glasses,
but requires S(k) as input and is computationally more expensive than F₁₂.

Structure Factor Input
----------------------

Percus-Yevick (Default)
~~~~~~~~~~~~~~~~~~~~~~~

For hard spheres, the analytic Percus-Yevick solution provides S(k):

.. code-block:: python

   model = ITTMCTIsotropic(phi=0.55)  # Uses Percus-Yevick automatically

The glass transition occurs at φ_MCT ≈ 0.516 for hard spheres.

User-Provided S(k)
~~~~~~~~~~~~~~~~~~

For real experimental data:

.. code-block:: python

   # From light scattering or X-ray experiments
   k_data = np.array([...])  # Wave vectors
   sk_data = np.array([...])  # Structure factor
   
   model = ITTMCTIsotropic(
       sk_source="user_provided",
       k_data=k_data,
       sk_data=sk_data
   )

Parameters
----------

.. list-table::
   :widths: 15 15 15 15 40
   :header-rows: 1

   * - Name
     - Default
     - Bounds
     - Units
     - Physical Meaning
   * - φ
     - 0.55
     - (0.1, 0.64)
     - —
     - Volume fraction (glass at φ ≈ 0.516)
   * - σ_d
     - 10⁻⁶
     - (10⁻⁹, 10⁻³)
     - m
     - Particle diameter
   * - D₀
     - 10⁻¹²
     - (10⁻¹⁸, 10⁻⁶)
     - m²/s
     - Bare short-time diffusion coefficient
   * - k_BT
     - 4.1×10⁻²¹
     - (10⁻²⁴, 10⁻¹⁸)
     - J
     - Thermal energy
   * - γ_c
     - 0.1
     - (0.01, 0.5)
     - —
     - Critical strain for cage breaking

Isotropic Shear Approximation (ISM)
-----------------------------------

The ISM simplifies the full anisotropic MCT equations by assuming that the
correlator depends only on the **magnitude** of the advected wavevector.

Wavevector Advection Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For simple shear with rate :math:`\dot{\gamma}`, the advected wavevector is:

.. math::

   \mathbf{k}(t,t') = \big(k_x,\; k_y - \gamma(t,t')k_x,\; k_z\big)

The advected magnitude squared is:

.. math::

   k(t,t')^2 = k_x^2 + (k_y - \gamma k_x)^2 + k_z^2
   = k^2 - 2\gamma k_x k_y + \gamma^2 k_x^2

**Orientational averaging**: Averaging over all initial orientations of
:math:`\mathbf{k}` on a sphere:

.. math::

   \langle k_x^2 \rangle = \langle k_y^2 \rangle = \langle k_z^2 \rangle = k^2/3

   \langle k_x k_y \rangle = 0

This gives the **isotropically sheared** wavevector magnitude:

.. math::

   \boxed{
   k(t,t') \approx k\sqrt{1 + \frac{1}{3}\gamma(t,t')^2}
   }

**Physical interpretation**:

- At :math:`\gamma = 0`: :math:`k(t,t') = k` (no advection)
- At :math:`\gamma \sim 1`: :math:`k(t,t') \approx 1.15k` (moderate stretch)
- At :math:`\gamma \gg 1`: :math:`k(t,t') \propto k\gamma/\sqrt{3}` (strong stretch)

The increased wavevector magnitude accelerates relaxation via the bare decay rate
:math:`\Gamma(k) = k^2 D_0/S(k)`.

Governing Equations
-------------------

k-Resolved Correlator Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each wave vector k has its own correlator equation:

.. math::

   \partial_t \Phi(k,t) + \Gamma(k) \left[ \Phi(k,t) + \int_0^t m(k,t-s) \partial_s \Phi(k,s) ds \right] = 0

with the k-dependent bare relaxation rate:

.. math::

   \Gamma(k) = \frac{k^2 D_0}{S(k)}

This shows that:

- Modes with large S(k) (strong correlations) relax slower
- Short-wavelength modes (large k) have faster bare rates
- The memory kernel m(k,t) couples all k-modes together

MCT Vertex Function
~~~~~~~~~~~~~~~~~~~

The memory kernel at wave vector k involves coupling to all other wave vectors:

.. math::

   m(k,t) = \sum_q V(k,q,|\mathbf{k}-\mathbf{q}|) \Phi(q,t) \Phi(|\mathbf{k}-\mathbf{q}|,t)

The vertex V depends on S(k) and its derivatives:

.. math::

   V(k,q,p) \propto n S(k) S(q) S(p) \left[ \frac{\mathbf{k} \cdot \mathbf{q}}{k^2} c(q) + \frac{\mathbf{k} \cdot \mathbf{p}}{k^2} c(p) \right]^2

where c(k) = 1 - 1/S(k) is the direct correlation function.

k-Resolved Correlators
~~~~~~~~~~~~~~~~~~~~~~

Each wave vector has its own relaxation dynamics:

.. math::

   \partial_t \Phi(k,t) + \Gamma(k) \left[ \Phi(k,t) + \int_0^t m(k,t-s) \partial_s \Phi(k,s) ds \right] = 0

with k-dependent relaxation rate:

.. math::

   \Gamma(k) = \frac{k^2 D_0}{S(k)}

Stress from k-Space Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress tensor involves integration over all wave vectors:

.. math::

   \sigma = \frac{k_B T}{6\pi^2} \int_0^\infty dk \, k^4 S(k)^2 \left[\frac{\partial \ln S}{\partial \ln k}\right]^2 \int_0^\infty d\tau \, \Phi(k,\tau)^2 h(\dot{\gamma}\tau)

Microscopic Stress Formula Detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full generalized Green-Kubo expression for the shear modulus is:

.. math::

   G(t,t') = \frac{k_B T}{60\pi^2} \int_0^{\infty} dk\; k^4
   \left[\frac{S'(k)}{S(k)^2}\right]^2\,\Phi_k(t,t')^2

**Physical interpretation of the weighting factors**:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Factor
     - Physical Meaning
   * - :math:`k^4`
     - Short wavelengths contribute more to stress (local rearrangements)
   * - :math:`[S'(k)]^2`
     - Modes where S(k) varies rapidly (near the peak) dominate
   * - :math:`[S(k)]^{-4}`
     - Modes with strong correlations contribute less (collective, slow)
   * - :math:`\Phi_k^2`
     - Only correlated (unrelaxed) modes carry stress

**Quantitative predictions without adjustable modulus**: Unlike the schematic
model where :math:`G_\infty` is fitted, ISM computes the stress magnitude
directly from :math:`k_BT`, :math:`S(k)`, and :math:`\Phi_k`. This provides a
first-principles prediction of yield stress and flow curves.

Equilibrium vs Driven Correlators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correlator dynamics differ between quiescent and driven states:

**Quiescent MCT** (no shear, for SAOS):

.. math::

   \partial_t \Phi_k^{\text{eq}}(t) + \Gamma_k \left[\Phi_k^{\text{eq}}(t) +
   \int_0^t ds\; m_k(t-s)\;\partial_s \Phi_k^{\text{eq}}(s)\right] = 0

where :math:`\Gamma_k = k^2 D_0 / S(k)` is constant.

**Driven ITT-MCT** (with shear):

.. math::

   \partial_t \Phi_k(t,t') + \Gamma_k(t,t') \left[\Phi_k(t,t') +
   \int_{t'}^t ds\; m_k(t,s,t')\;\partial_s \Phi_k(s,t')\right] = 0

where :math:`\Gamma_k(t,t') = D_0 k(t,t')^2 / S(k(t,t'))` depends on the
advected wavevector.

The key difference is that shear:

1. **Accelerates initial decay** via increased :math:`\Gamma_k(t,t')`
2. **Decorrelates the memory kernel** via :math:`h[\gamma(t,s)]`
3. **Creates two-time dependence** in the correlator

This microscopic stress formula requires:

- :math:`S(k)` and its derivative (from structure factor)
- :math:`\Phi(k,\tau)` for all k (from solving the k-resolved MCT equations)
- :math:`h(\dot{\gamma}\tau)` (strain decorrelation function)

The integral weights contributions by :math:`k^4 S(k)^2 [S'(k)]^2`, meaning:

- Modes near the S(k) peak contribute most
- Both large S(k) and large S'(k) enhance stress contribution
- Short-wavelength modes (large k) have higher weight due to :math:`k^4`

Validity and Assumptions
-------------------------

**When ISM works well:**

- Dense colloidal suspensions (φ > 0.4 for hard spheres)
- Monodisperse or narrow size distribution
- No attractive interactions (or weak compared to entropic caging)
- Brownian dynamics (not granular or inertial)
- S(k) accurately known (from scattering or theory)

**Limitations:**

- Computationally expensive (O(n_k² × N) vs O(N) for F₁₂)
- Requires accurate S(k) input
- Assumes isotropic structure under shear (no shear-induced ordering)
- No hopping or activated processes (important deep in glass)
- Underestimates relaxation times far from transition

**When to simplify to F₁₂:**

If you don't have S(k) data or if qualitative trends are sufficient, use the
F₁₂ schematic model instead. ISM is for quantitative comparison with experiments
where S(k) is measured via light scattering, X-rays, or neutron scattering.

What You Can Learn
------------------

The ISM model extends the F₁₂ schematic with full k-resolution and quantitative predictions from the structure factor S(k). All parameters now have microscopic interpretation tied to colloidal physics.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**φ (Volume Fraction)**:
   The packing fraction of particles, controlling the glass transition.

   *For graduate students*: φ is the order parameter for the jamming/glass transition in hard spheres. The MCT glass transition occurs at φ_MCT ≈ 0.516, slightly below the random close packing φ_RCP ≈ 0.64. The Percus-Yevick structure factor S(k;φ) becomes singular at φ_MCT, where the self-consistent MCT equation develops a non-zero long-time limit f(k) > 0. The separation from the transition scales as ε ∼ (φ - φ_g)/φ_g.

   *For practitioners*: φ < 0.4 is dilute (fluid), 0.4 < φ < 0.516 is dense fluid (slow but ergodic), φ > 0.516 is glass (yield stress). Fitting φ from rheology requires knowing the particle size σ_d to convert number density to volume fraction. Typical calibration: measure φ gravimetrically or via osmotic pressure.

**σ_d (Particle Diameter)**:
   The hard-sphere diameter used to compute S(k) and set the k-grid resolution.

   *For graduate students*: σ_d sets the characteristic length scale for structural correlations. The S(k) peak occurs at k* ≈ 2π/σ_d (nearest-neighbor spacing). In the microscopic stress formula, σ_d appears implicitly through the k-grid: stress is dominated by modes near k* where S(k) is maximal and S'(k) is large.

   *For practitioners*: Use σ_d from microscopy (e.g., dynamic light scattering radius), not the hydrodynamic radius. For polydisperse systems, use the number-average diameter. Typical values: 10 nm - 10 μm for colloids.

**D₀ (Bare Diffusion Coefficient)**:
   The short-time (non-interacting) diffusion coefficient, D₀ = k_BT/(6πη_s a) for Stokes-Einstein.

   *For graduate students*: D₀ sets the bare relaxation rate Γ(k) = k²D₀/S(k). At high k (short wavelengths), S(k) → 1 and Γ(k) ≈ k²D₀ (free diffusion). At the S(k) peak, Γ(k) is strongly suppressed by the large S(k), leading to slow collective relaxation. The long-time diffusion coefficient D_L = D₀/S(0) accounts for thermodynamic slowing.

   *For practitioners*: Measure D₀ from dilute suspension DLS (φ → 0 limit) or calculate from Stokes-Einstein using solvent viscosity η_s. Typical values: 10⁻¹²-10⁻⁹ m²/s for colloids in water.

**k_BT (Thermal Energy)**:
   The thermal energy scale, k_B × temperature in Kelvin.

   *For graduate students*: k_BT sets the absolute stress scale in the microscopic formula σ ∼ (k_BT/60π²)∫dk k⁴[S'(k)]²Φ². For hard spheres, the stress is purely entropic (no potential energy), so k_BT is the only energy scale. At room temperature, k_BT ≈ 4.11 × 10⁻²¹ J.

   *For practitioners*: Use k_BT = 4.11×10⁻²¹ J at 25°C. For temperature-dependent studies, scale k_BT linearly with T. If fitted stress is off by a factor of 2, check if the effective temperature differs from solvent temperature (non-equilibrium heating).

**γ_c (Critical Strain)**:
   The cage-breaking strain scale (same as F₁₂ schematic).

   *For graduate students*: γ_c appears in the strain decorrelation h(γ) = exp[-(γ/γ_c)²]. For hard spheres, γ_c ≈ 0.05-0.1 corresponds to the Lindemann parameter: the ratio of thermal vibration amplitude to particle spacing. Unlike the schematic model, γ_c in ISM is the only remaining fit parameter—all other quantities are determined by φ, σ_d, D₀, k_BT, and S(k).

   *For practitioners*: Fit γ_c from the shear-thinning onset in flow curves. Smaller γ_c means easier cage breaking. Typical values: 0.05 (rigid hard spheres), 0.15 (soft microgels), 0.3 (polymeric cages).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from ITT-MCT ISM Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - φ Range
     - Glass State
     - Typical Materials
     - S(k) Characteristics
   * - **φ < 0.45**
     - Dilute fluid
     - Low-concentration PMMA colloids, silica sols
     - S(k) peak < 2, weak correlations, fast relaxation at all k
   * - **0.45 < φ < 0.516**
     - Dense fluid
     - Pre-jammed colloids, moderate emulsions
     - S(k) peak = 2-3, strong correlations, slow but ergodic, critical slowing as φ → φ_g
   * - **0.516 < φ < 0.55**
     - Marginal glass
     - Weakly jammed colloids, soft microgel pastes
     - S(k) peak > 3, non-ergodic Φ(k,t→∞) > 0, small yield stress
   * - **0.55 < φ < 0.58**
     - Moderate glass
     - Hard-sphere colloids, carbopol microgels
     - S(k) peak > 4, large f(k), clear yield stress, pronounced plateau
   * - **φ > 0.58**
     - Deep glass/jammed
     - Highly concentrated colloids, dense emulsions
     - S(k) peak > 5, near-complete arrest, large yield stress, approaching RCP

Wave-Vector-Dependent Relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By inspecting :math:`\Phi(k,t)` at different k:

- **Small k** (long wavelengths, :math:`k\sigma_d < 1`): Collective density fluctuations,
  slow relaxation, sensitive to hydrodynamic interactions
- **Peak k** (:math:`k \approx 2\pi/\sigma_d`): Nearest-neighbor cage length scale,
  dominates stress response
- **Large k** (:math:`k\sigma_d > 5`): Single-particle rattling, fast relaxation,
  nearly free diffusion

**Diagnostic use**: If experimental dynamic light scattering provides :math:`\Phi(k,t)`
at multiple k, fit the ISM model to all k simultaneously to validate MCT predictions.

Quantitative Stress Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike F₁₂ (which has a fitted modulus :math:`G_\infty`), ISM predicts stress
from first principles given:

- Volume fraction φ
- Particle size σ_d
- Thermal energy k_BT
- S(k) (from Percus-Yevick or experiment)

**No adjustable stress scale**: The only rheological fit parameter is γ_c
(critical strain). The absolute stress magnitude is predicted from S(k).

**Validation test**: Compare ISM predictions to experimental flow curves. If
the magnitude is wrong by a factor >2, check:

1. Is S(k) correct? (Use experimental scattering if available)
2. Are particles truly hard spheres? (Softness changes S(k))
3. Is temperature correct? (k_BT enters the prefactor)

Structure Factor Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the current ISM implementation uses static S(k), inspecting S(k) features
reveals:

- **S(k) peak position**: Nearest-neighbor distance :math:`2\pi / k_{\text{peak}}`
- **S(k) peak height**: Strength of structural correlations (higher = stronger caging)
- **S(0)**: Compressibility (diverges at jamming in hard spheres)

**Connection to φ_g**: The glass transition volume fraction φ_g ≈ 0.516 is where
S(k) becomes so large that :math:`\Phi(k, t \to \infty) > 0` for some k.

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From known colloid properties**

If you have a well-characterized colloidal suspension:

.. code-block:: python

   phi = 0.55              # Volume fraction (measured)
   sigma_d = 1e-6          # Particle diameter (1 μm)
   T = 298                 # Temperature (K)
   k_BT = 1.38e-23 * T     # Thermal energy
   eta_s = 1e-3            # Solvent viscosity (water, Pa·s)
   D0 = k_BT / (3 * np.pi * eta_s * sigma_d)  # Stokes-Einstein

   model = ITTMCTIsotropic(phi=phi, sigma_d=sigma_d, D0=D0, k_BT=k_BT)

**Method 2: Fit to rheological data**

If material properties are unknown, fit φ and γ_c to flow curve data, keeping
σ_d, D₀, k_BT as physically reasonable estimates.

Troubleshooting
~~~~~~~~~~~~~~~

**Problem: S(k) peak too sharp/broad**

- Solution: Check if Percus-Yevick is appropriate for your system. For soft
  particles, provide user S(k) from scattering.

**Problem: Predicted stress too high/low**

- Solution: Adjust k_BT (effective thermal energy may differ from room temperature
  in driven systems) or check if D₀ is correct (hydrodynamic interactions).

**Problem: Slow computation**

- Solution: Reduce k-grid resolution (``n_k_points`` parameter) or use F₁₂ schematic
  for initial exploration.

Usage
-----

Basic Prediction
~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.itt_mct import ITTMCTIsotropic
   import numpy as np
   
   # Hard-sphere glass
   model = ITTMCTIsotropic(phi=0.55)
   
   # Check glass state
   info = model.get_glass_transition_info()
   print(f"Glass: {info['is_glass']}")  # True for φ > 0.516
   
   # Flow curve
   gamma_dot = np.logspace(-2, 2, 30)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

Inspect S(k)
~~~~~~~~~~~~

.. code-block:: python

   # Get S(k) information
   sk_info = model.get_sk_info()
   print(f"S(k) peak at k = {sk_info['S_max_position']:.2f}")
   print(f"S(k) max = {sk_info['S_max']:.2f}")
   
   # Access k-grid and S(k) directly
   import matplotlib.pyplot as plt
   plt.loglog(model.k_grid, model.S_k)
   plt.xlabel('k')
   plt.ylabel('S(k)')

Update Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Change volume fraction and recalculate S(k)
   model.update_structure_factor(phi=0.52)
   
   # Or provide new experimental S(k)
   model.update_structure_factor(k_data=k_new, sk_data=sk_new)

Model Comparison
----------------

ISM vs F₁₂
~~~~~~~~~~

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - F₁₂ Schematic
     - ISM
   * - Correlators
     - Single scalar Φ(t)
     - Array Φ(k,t), n_k points
   * - S(k) input
     - Not needed
     - Required
   * - Parameters
     - ε, Γ, γ_c, G_∞
     - φ, D₀, σ_d, k_BT, γ_c
   * - Glass transition
     - At v₂ = 4
     - At φ ≈ 0.516
   * - Computation
     - O(N) per step
     - O(n_k² × N)
   * - Best for
     - Fitting, exploration
     - Quantitative predictions

See Also
--------

- :doc:`itt_mct_schematic` — Simplified F₁₂ schematic model (faster, no S(k) required)
- :doc:`../sgr/sgr_conventional` — Alternative glass transition framework (trap model)
- :doc:`../stz/stz_conventional` — Shear transformation zone theory (effective temperature)

**When to use ISM vs F₁₂:**

- **Use ISM** if: S(k) is known, quantitative predictions needed, validating MCT theory
- **Use F₁₂** if: Fitting rheological data, qualitative trends, faster computation

API Reference
-------------

.. autoclass:: rheojax.models.itt_mct.ITTMCTIsotropic
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

References
----------

.. [1] Brader, J. M., Cates, M. E., and Fuchs, M. "First-principles constitutive
   equation for suspension rheology." *Proceedings of the National Academy of
   Sciences*, 106, 15186 (2009). https://doi.org/10.1073/pnas.0905330106

.. [2] Fuchs, M. and Cates, M. E. "A mode coupling theory for Brownian particles
   in homogeneous steady shear flow." *Journal of Rheology*, 53, 957 (2009).
   https://doi.org/10.1122/1.3119084

.. [3] Hansen, J. P. and McDonald, I. R. *Theory of Simple Liquids*, 4th ed.
   Academic Press (2013). ISBN: 978-0123870322

.. [4] Götze, W. *Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory*.
   Oxford University Press (2009). https://doi.org/10.1093/acprof:oso/9780199235346.001.0001

.. [5] Voigtmann, T. "Nonlinear rheology of hard sphere suspensions near jamming."
   *Physical Review Letters*, 105, 248303 (2010).
   https://doi.org/10.1103/PhysRevLett.105.248303

.. [6] Götze, W. & Sjögren, L. "Relaxation processes in supercooled liquids."
   *Reports on Progress in Physics*, **55**, 241 (1992).
   https://doi.org/10.1088/0034-4885/55/3/001

.. [7] Fuchs, M. & Cates, M. E. "Theory of nonlinear rheology and yielding of dense colloidal suspensions."
   *Physical Review Letters*, **89**, 248304 (2002).
   https://doi.org/10.1103/PhysRevLett.89.248304

.. [8] Fuchs, M. "Nonlinear rheology of dense colloidal dispersions: A phenomenological model and its connection to mode coupling theory."
   *Journal of Non-Newtonian Fluid Mechanics*, **157**, 22-25 (2009).
   https://doi.org/10.1016/j.jnnfm.2008.08.010

.. [9] Brader, J. M. "Nonlinear rheology of colloidal dispersions."
   *Journal of Physics: Condensed Matter*, **22**, 363101 (2010).
   https://doi.org/10.1088/0953-8984/22/36/363101

.. [10] Zausch, J., Horbach, J., Laurati, M., Egelhaaf, S. U., Brader, J. M., Voigtmann, T., & Fuchs, M. "From equilibrium to steady state: The transient dynamics of colloidal liquids under shear."
   *Journal of Physics: Condensed Matter*, **20**, 404210 (2008).
   https://doi.org/10.1088/0953-8984/20/40/404210
