.. _models-itt-mct:

ITT-MCT Models
==============

Integration Through Transients Mode-Coupling Theory (ITT-MCT) models describe
the nonlinear rheology of dense colloidal suspensions and glassy materials
through microscopic physics: the cage effect.

Overview
--------

Mode-Coupling Theory (MCT) provides a first-principles approach to understanding
the dynamics of dense particulate systems. The theory predicts:

- **Glass transition** at a critical volume fraction (φ ≈ 0.516 for hard spheres)
- **Two-step relaxation** with β (in-cage) and α (cage-breaking) processes
- **Yield stress** in the glass state from arrested structure
- **Shear thinning** from flow-induced cage breaking

ITT-MCT extends MCT to nonlinear deformations by tracking how flow "advects"
density fluctuations, destroying the cage structure above a critical strain.

Available Models
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Model
     - Description
   * - :ref:`ITTMCTSchematic <model-itt-mct-schematic>`
     - F₁₂ schematic model with scalar correlator. Fast computation, captures
       essential physics with ~6 parameters. Best for qualitative understanding
       and fitting experimental data.
   * - :ref:`ITTMCTIsotropic <model-itt-mct-isotropic>`
     - Full isotropically sheared model with k-resolved correlators Φ(k,t).
       Uses structure factor S(k) input. More quantitative but computationally
       expensive.

Model Selection Guide
---------------------

**Use ITTMCTSchematic when:**

- You need fast computations for fitting or exploration
- Qualitative understanding of glass/yield phenomena is sufficient
- You want to explore parameter space quickly
- Working with systems where S(k) is unknown

**Use ITTMCTIsotropic when:**

- Quantitative predictions are needed
- S(k) is available (measured or from simulation)
- Wave-vector-dependent relaxation is important
- Comparing with microscopic measurements (DLS, X-ray scattering)

Supported Protocols
-------------------

Both models support all six standard rheological protocols:

1. **Flow curve** (steady shear): σ(γ̇) - shows yield stress and shear thinning
2. **SAOS** (oscillation): G'(ω), G''(ω) - shows glass plateau and loss peak
3. **Startup**: σ(t) at constant γ̇ - shows stress overshoot
4. **Creep**: J(t) at constant σ - shows viscosity bifurcation
5. **Relaxation**: σ(t) after cessation - shows residual stress in glass
6. **LAOS**: σ(t) for γ = γ₀sin(ωt) - shows nonlinear harmonics

Physical Context
----------------

MCT is most applicable to:

- **Hard-sphere colloids** (PMMA, silica particles)
- **Dense emulsions** (mayonnaise, cosmetics)
- **Concentrated polymer solutions** near gelation
- **Soft glassy materials** (pastes, gels)

The theory captures the universal features of the glass transition that emerge
from the cage effect, independent of specific interparticle interactions.

Key Parameters
--------------

**F₁₂ Schematic Model:**

- **ε (epsilon)**: Separation parameter controlling distance from glass transition
  
  - ε < 0: Ergodic fluid
  - ε = 0: Critical point
  - ε > 0: Glass state

- **γ_c**: Critical strain for cage breaking (~0.05-0.2)
- **Γ**: Bare relaxation rate (microscopic timescale)
- **G_∞**: High-frequency modulus

**ISM Model:**

- **φ (phi)**: Volume fraction (glass at φ ≈ 0.516)
- **S(k)**: Structure factor (from Percus-Yevick or experiment)
- **D₀**: Bare diffusion coefficient

References
----------

.. [Gotze2009] Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids",
   Cambridge University Press.

.. [Fuchs2002] Fuchs M. & Cates M.E. (2002) "Theory of Nonlinear Rheology
   and Yielding of Dense Colloidal Suspensions", Phys. Rev. Lett. 89, 248304.

.. [Fuchs2009] Fuchs M. & Cates M.E. (2009) "A mode coupling theory for
   Brownian particles in homogeneous steady shear flow", J. Rheol. 53, 957.

.. [Brader2008] Brader J.M. et al. (2008) "Glass rheology: From mode-coupling
   theory to a dynamical yield criterion", J. Phys.: Condens. Matter 20, 494243.

.. toctree::
   :maxdepth: 2
   :caption: Models

   itt_mct_schematic
   itt_mct_isotropic
