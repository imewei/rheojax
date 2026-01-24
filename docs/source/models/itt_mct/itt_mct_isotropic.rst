.. _model-itt-mct-isotropic:

ITT-MCT Isotropic (ISM)
=======================

Quick Reference
---------------

**Use when:** Quantitative predictions needed, S(k) available, wave-vector-dependent
dynamics important

**Parameters:** 5 (φ, σ_d, D₀, k_BT, γ_c) + S(k) input

**Key equation:** k-resolved correlator Φ(k,t) with MCT vertex from S(k)

**Test modes:** Flow curve, oscillation, startup, creep, relaxation, LAOS

**Data required:** Structure factor S(k) from experiment or Percus-Yevick

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

Governing Physics
-----------------

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

Usage Examples
--------------

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

API Reference
-------------

.. autoclass:: rheojax.models.itt_mct.ITTMCTIsotropic
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

.. [Brader2009] Brader J.M. et al. (2009) "First-principles constitutive equation
   for suspension rheology", Proc. Natl. Acad. Sci. 106, 15186.

.. [Fuchs2009] Fuchs M. & Cates M.E. (2009) "A mode coupling theory for Brownian
   particles in homogeneous steady shear flow", J. Rheol. 53, 957.

.. [Hansen2013] Hansen J.P. & McDonald I.R. (2013) "Theory of Simple Liquids",
   4th ed., Academic Press. *Percus-Yevick and structure factor theory.*
