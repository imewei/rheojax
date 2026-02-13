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

- **Glass transition** at a critical volume fraction (:math:`\phi \approx 0.516` for hard spheres)
- **Two-step relaxation** with :math:`\beta` (in-cage) and :math:`\alpha` (cage-breaking) processes
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
     - :math:`F_{12}` schematic model with scalar correlator. Fast computation, captures
       essential physics with ~6 parameters. Best for qualitative understanding
       and fitting experimental data.
   * - :ref:`ITTMCTIsotropic <model-itt-mct-isotropic>`
     - Full isotropically sheared model with k-resolved correlators :math:`\Phi(k,t)`.
       Uses structure factor :math:`S(k)` input. More quantitative but computationally
       expensive.

Model Selection Guide
---------------------

**Use ITTMCTSchematic when:**

- You need fast computations for fitting or exploration
- Qualitative understanding of glass/yield phenomena is sufficient
- You want to explore parameter space quickly
- Working with systems where :math:`S(k)` is unknown

**Use ITTMCTIsotropic when:**

- Quantitative predictions are needed
- :math:`S(k)` is available (measured or from simulation)
- Wave-vector-dependent relaxation is important
- Comparing with microscopic measurements (DLS, X-ray scattering)

Supported Protocols
-------------------

Both models support all six standard rheological protocols:

1. **Flow curve** (steady shear): :math:`\sigma(\dot{\gamma})` - shows yield stress and shear thinning
2. **SAOS** (oscillation): :math:`G'(\omega)`, :math:`G''(\omega)` - shows glass plateau and loss peak
3. **Startup**: :math:`\sigma(t)` at constant :math:`\dot{\gamma}` - shows stress overshoot
4. **Creep**: :math:`J(t)` at constant :math:`\sigma` - shows viscosity bifurcation
5. **Relaxation**: :math:`\sigma(t)` after cessation - shows residual stress in glass
6. **LAOS**: :math:`\sigma(t)` for :math:`\gamma = \gamma_0 \sin(\omega t)` - shows nonlinear harmonics

For detailed mathematical formulation of each protocol including governing
equations and physical interpretation, see :doc:`itt_mct_protocols`.

Theoretical Framework
---------------------

The ITT-MCT formalism consists of three key components:

1. **ITT Stress Functional**: A history integral over past deformations weighted
   by a generalized shear modulus built from transient density correlators. This
   is the microscopic generalization of the Green-Kubo relation for driven systems.

2. **MCT Correlator Dynamics**: The Zwanzig-Mori integro-differential equation
   with a mode-coupling memory kernel. This describes how density fluctuations
   decorrelate under the combined influence of Brownian motion and shear advection.

3. **Wavevector Advection**: Flow "advects" density fluctuations, causing the
   wavevector :math:`\mathbf{k}` to evolve as :math:`\mathbf{k}(t,t') = \mathbf{k}
   \cdot \mathbf{E}^{-1}(t,t')` where :math:`\mathbf{E}` is the deformation gradient.
   This advection destroys the cage structure above a critical accumulated strain.

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

:math:`F_{12}` **Schematic Model:**

- :math:`\varepsilon` **(epsilon)**: Separation parameter controlling distance from glass transition

  - :math:`\varepsilon < 0`: Ergodic fluid
  - :math:`\varepsilon = 0`: Critical point
  - :math:`\varepsilon > 0`: Glass state

- :math:`\gamma_c`: Critical strain for cage breaking (~0.05-0.2)
- :math:`\Gamma`: Bare relaxation rate (microscopic timescale)
- :math:`G_\infty`: High-frequency modulus

**ISM Model:**

- :math:`\phi` **(phi)**: Volume fraction (glass at :math:`\phi \approx 0.516`)
- :math:`S(k)`: Structure factor (from Percus-Yevick or experiment)
- :math:`D_0`: Bare diffusion coefficient

References
----------

.. [Gotze2009] Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids: A
   Mode-Coupling Theory", Oxford University Press.
   https://doi.org/10.1093/acprof:oso/9780199235346.001.0001

.. [Fuchs2002] Fuchs M. & Cates M.E. (2002) "Theory of Nonlinear Rheology
   and Yielding of Dense Colloidal Suspensions", Phys. Rev. Lett. 89, 248304.
   https://doi.org/10.1103/PhysRevLett.89.248304

.. [Fuchs2009] Fuchs M. & Cates M.E. (2009) "A mode coupling theory for
   Brownian particles in homogeneous steady shear flow", J. Rheol. 53, 957.
   https://doi.org/10.1122/1.3119084

.. [Brader2008] Brader J.M., Voigtmann T., Fuchs M., Larson R.G. & Cates M.E.
   (2009) "Glass rheology: From mode-coupling theory to a dynamical yield
   criterion", Proc. Natl. Acad. Sci. USA 106, 15186-15191.
   https://doi.org/10.1073/pnas.0905330106

.. toctree::
   :maxdepth: 2
   :caption: Models

   itt_mct_schematic
   itt_mct_isotropic
   itt_mct_protocols
